// p194_moe_fused128.cuh — P194 moe_fused128: P066 / P190's one persistent launch for 65 ≤ M ≤ 128 rows (DSpark
// verify / draft steps of 9–16 sequences: 72–128 rows) instead of the default path at M > 64 (moe_route, act_quant,
// P065 / P086's prefill experts + combine). Included by moe.cu after p190_moe_fused_mid.cuh (P066's route_row, loads,
// gbar; P035's Sel, sel_slot, ld_na64, PDL helpers).
//
// A 128-row ring does not fit (2 × 36 KB a warp group + ~30 KB static > 99 KB), so the rows are split in two halves
// (rows 0 … 63, 64 … 127) and each half is P190's 64-pair item: an expert u with rows in both halves has two
// sub-experts (u, 0), (u, 1), each with its own up items (32 inter blocks), up counter and down items (32 latent
// blocks). The two halves' items of the same weight tile take ADJACENT tickets ((u, c, 0), (u, c, 1), (u, c + 1, 0),
// …), so two warp groups stream the same 16-row tile at the same time and the second read is served by L2: each
// expert's weights come from DRAM about once. Shared memory: P190's ring (2 groups × 2 stages × 64 pairs = 72 KB
// dynamic; the prologue's 128-row masks and routed ids live in it until the items start) + ~20 KB static.
// Everything else is P190's code: the same items, the same latent quantization, the same per-pair MMA chains (an MMA
// column is one pair's chain whatever the other columns hold), SiTU / MXFP8 of h, and combine (Σ_slot w·y in slot
// order, the last down item of a latent block over all sub-experts): bitwise P066 / P190 / the base per row
// (test_moe.py --only moe_fused128: vs the base at 65 … 128 rows and vs P066 on the rows' 8-row pieces).
#pragma once

namespace p194 {

constexpr int kMaxM = 128, kHalf = 64, kMaxP = kMaxM * kTopK, kS = p066::kS;
constexpr int kMaxU = kMaxP < kExperts ? kMaxP : kExperts;  // union slots (distinct experts of a call)
constexpr int kMaxV = 2 * kMaxU;                            // sub-experts (union slot, half)
constexpr int kMW = kMaxM / 32;                             // 32-bit words of an expert's row mask (words 0, 1: half 0)
constexpr int kNT = 256;                                    // threads a CTA (2 warp groups)
constexpr int kNJ = (kMaxP + kNT - 1) / kNT;                // pair indices a thread builds the union from (5)
constexpr int kBlk = p066::kBlk, kRS = p066::kRS;
constexpr int kRing = kHalf * kRS + kHalf * 16;             // one ring stage: a half's B rows + their scales
constexpr int kGroupSmem = 2 * kRing;
constexpr int kGroups = 2;
constexpr int kSmem = kGroups * kGroupSmem;                 // dynamic shared memory (73,728 B)
constexpr int kRowsOff = 0, kIdOff = kMaxU * kMW * 4;       // prologue-only arrays in the (not yet used) ring
static_assert(kIdOff + kMaxP * 4 <= kSmem, "the prologue's masks and ids fit the ring");
static_assert(2 * 32 * kHalf * 4 <= kRing, "the up epilogue's g / u exchange fits ring stage 0");
static_assert(kSmem + 24 * 1024 <= 101376, "dynamic + static shared memory");

struct Cnt {           // scratch counters (zero between calls; dcnt holds 8 KB)
  int tick, exit, qdone;
  int up[kMaxV];       // per sub-expert: finished up items (32 = done)
  int blk[kBlk];       // per latent block: finished down items (of all sub-experts)
};
static_assert(sizeof(Cnt) <= 8192, "dcnt holds 8 KB");

struct Item {          // an up / down ticket (relative to its range) decoded
  int u, cb, vi, np, pl;  // union slot, inter block c / latent block b, sub-expert, pairs, first pair in s_plist
};

// P190's sel_build for P ≤ kNJ · blockDim pairs
__device__ __forceinline__ void sel_build(p035::Sel& S, const int* __restrict__ ids, int P, int (&id)[kNJ]) {
  const int tid = threadIdx.x;
  if (tid < 16) S.mask[tid] = 0;
#pragma unroll
  for (int j = 0; j < kNJ; j++) {
    const int i = tid + j * blockDim.x;
    id[j] = i < P ? ids[i] : -1;
  }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < kNJ; j++)
    if (id[j] >= 0) atomicOr(&S.mask[id[j] >> 5], 1u << (id[j] & 31));
  __syncthreads();
  if (tid < 32) {
    const int c = tid < 16 ? __popc(S.mask[tid]) : 0;
    int incl = c;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, incl, o);
      if (tid >= o) incl += v;
    }
    if (tid < 16) S.pre[tid] = incl - c;
    if (tid == 31) S.n = incl;
  }
  __syncthreads();
}

template <class OutT>
__global__ void __launch_bounds__(kNT, 1) k_fused194(
    const uint2* __restrict__ Wu, const uint8_t* __restrict__ WSu, const uint2* __restrict__ Wd,
    const uint8_t* __restrict__ WSd, const float* __restrict__ lg, const float* __restrict__ bias,
    const float* __restrict__ scale, const bf16* __restrict__ lat, int ldl, const uint8_t* __restrict__ lq,
    const uint8_t* __restrict__ ls, int M, int* ids_out, float* w_out, uint8_t* xq, uint8_t* xs, uint8_t* hq,
    uint8_t* hs, float* y, Cnt* cn, OutT* part, int ld) {
  constexpr int kWarps = 4 * kGroups, NT = kHalf / 8;
  __shared__ float sa[kWarps][16];
  __shared__ float s_w[kMaxP];
  __shared__ p035::Sel S;
  __shared__ int s_e[kMaxU], s_plist[kMaxP];
  __shared__ ushort4 s_ux[kMaxU];   // per union slot: first sub-expert, first pair, pairs in half 0, in half 1
  __shared__ uint16_t s_vu[kMaxV];  // sub-expert → union slot
  __shared__ int s_nv;
  __shared__ int s_t[kGroups], s_last[kGroups];
  __shared__ int s_r[kGroups][kHalf];  // an up item's latent rows (pair / 10)
  extern __shared__ __align__(16) uint8_t dsm[];
  uint32_t(*s_rows)[kMW] = reinterpret_cast<uint32_t(*)[kMW]>(dsm + kRowsOff);  // prologue only
  int* s_id = reinterpret_cast<int*>(dsm + kIdOff);                                // prologue only
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wg = warp >> 2, w4 = warp & 3, gt = tid & 127;
  const int P = M * kTopK;
  uint8_t* ring = dsm + wg * kGroupSmem;

  p035::pdl_wait();  // logits (router GEMM) and latent (fc1) complete
  // ---- routing: warp w routes rows w, w + kWarps, …
  for (int m = warp; m < M; m += kWarps) {
    p066::route_row(lg, bias, scale, m, lane, sa[warp], s_id + m * kTopK, s_w + m * kTopK);
    __syncwarp();
  }
  for (int i = tid; i < kMaxU * kMW; i += blockDim.x) (&s_rows[0][0])[i] = 0;
  __syncthreads();
  if (blockIdx.x == 0)
    for (int i = tid; i < P; i += blockDim.x) ids_out[i] = s_id[i], w_out[i] = s_w[i];
  // ---- the union of the experts (ascending id) and each one's pairs in row order (compact lists: half 0's pairs,
  // then half 1's)
  int id2[kNJ];
  sel_build(S, s_id, P, id2);
  int slot2[kNJ];
#pragma unroll
  for (int j = 0; j < kNJ; j++)
    if (id2[j] >= 0) {
      const int i = tid + j * blockDim.x, m = i / kTopK;
      slot2[j] = p035::sel_slot(S, id2[j]);
      atomicOr(&s_rows[slot2[j]][m >> 5], 1u << (m & 31));
      s_e[slot2[j]] = id2[j];
    }
  __syncthreads();
  const int ne = S.n;
  if (warp == 0) {  // sub-expert and pair offsets: exclusive scans of the halves present and of the row counts
    int vbase = 0, pbase = 0;
    for (int u0 = 0; u0 < ne; u0 += 32) {
      const int u = u0 + lane;
      int c0 = 0, c1 = 0;
      if (u < ne) {
        c0 = __popc(s_rows[u][0]) + __popc(s_rows[u][1]);
        c1 = __popc(s_rows[u][2]) + __popc(s_rows[u][3]);
      }
      const int nh = (c0 > 0) + (c1 > 0), c = c0 + c1;
      int iv = nh, ip = c;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int a = __shfl_up_sync(0xffffffffu, iv, o), b = __shfl_up_sync(0xffffffffu, ip, o);
        if (lane >= o) iv += a, ip += b;
      }
      if (u < ne)
        s_ux[u] = make_ushort4((unsigned short)(vbase + iv - nh), (unsigned short)(pbase + ip - c),
                               (unsigned short)c0, (unsigned short)c1);
      vbase += __shfl_sync(0xffffffffu, iv, 31);
      pbase += __shfl_sync(0xffffffffu, ip, 31);
    }
    if (lane == 0) s_nv = vbase;
  }
  __syncthreads();
  for (int u = tid; u < ne; u += blockDim.x) {
    const ushort4 x = s_ux[u];
    s_vu[x.x] = uint16_t(u);
    if (x.z && x.w) s_vu[x.x + 1] = uint16_t(u);
  }
#pragma unroll
  for (int j = 0; j < kNJ; j++)
    if (id2[j] >= 0) {
      const int i = tid + j * blockDim.x, m = i / kTopK, u = slot2[j], mw = m >> 5;
      int below = __popc(s_rows[u][mw] & ((1u << (m & 31)) - 1u));  // the expert's rows before m
#pragma unroll
      for (int k = 0; k < kMW; k++)
        if (k < mw) below += __popc(s_rows[u][k]);
      s_plist[s_ux[u].y + below] = i;
    }
  __syncthreads();  // s_rows / s_id (in the ring) are not read past here
  p035::pdl_launch();
  const int nv = s_nv;
  const int q_end = lq ? 0 : M, n_up = nv * 32, u_end = q_end + n_up, n_all = q_end + 2 * n_up;
  const uint8_t* xqr = lq ? lq : xq;  // where the up items read the quantized latent
  const uint8_t* xsr_ = lq ? ls : xs;

  // Relative ticket r of the up (or down) range → its item. Sub-experts are numbered by union slot, then half, and an
  // expert's 32 · (halves) tickets start at 32 · (its first sub-expert), so r / 32 is one of the expert's sub-experts;
  // with two halves the tickets alternate halves: (c, 0), (c, 1), (c + 1, 0), …
  auto item = [&](int r) {
    Item it;
    it.u = s_vu[r >> 5];
    const ushort4 x = s_ux[it.u];
    const int i = r - 32 * int(x.x);
    const bool two = x.z && x.w;
    const int h = two ? (i & 1) : (x.z ? 0 : 1);
    it.cb = two ? i >> 1 : i;
    it.vi = int(x.x) + (two ? h : 0);
    it.np = h ? x.w : x.z;
    it.pl = int(x.y) + (h ? int(x.z) : 0);
    return it;
  };
  // Item tk → this warp's weight tile (words / scales): up item (u, c, ·) → tile c·4 + w4 of expert u; down item
  // (u, b, ·) → latent tile 4b + w4 of expert u.
  auto wptr = [&](int tk, const uint2*& wt, const uint8_t*& wst) {
    if (tk < u_end) {
      const Item it = item(tk - q_end);
      const int e = s_e[it.u], c = it.cb;
      wt = Wu + (size_t)e * kUpWords + ((size_t)(c * 4 + w4) * kUpKB) * 32 + lane;
      wst = WSu + (size_t)e * kUpScales + ((size_t)(c * 4 + w4) * kUpKB) * 16;
    } else {
      const Item it = item(tk - u_end);
      const int e = s_e[it.u], T = 4 * it.cb + w4;
      wt = Wd + (size_t)e * kDnWords + ((size_t)T * kDnKB) * 32 + lane;
      wst = WSd + (size_t)e * kDnScales + ((size_t)T * kDnKB) * 16;
    }
  };
  uint2 wf[2][kS];
  // weight scales (P066): a stage's 8 k32 steps × 16 rows = 128 contiguous bytes, one 32-bit load a lane
  static_assert(kS == 8, "one 128-byte scale load a stage");
  uint32_t sf[2];
  const int s_src = 2 * (lane & 1) + (lane >> 4), s_sh = 8 * (g & 3);
  auto sfa = [&](uint32_t raw, int i) { return (__shfl_sync(0xffffffffu, raw, 4 * i + s_src) >> s_sh) & 0xFFu; };
  auto ldw = [&](const uint2* wt, const uint8_t* wst, int st, uint2 (&bw)[kS], uint32_t& s) {
#pragma unroll
    for (int i = 0; i < kS; i++) bw[i] = p035::ld_na64(wt + (size_t)(st * kS + i) * 32);
    s = p066::ld_na32(wst + (size_t)st * kS * 16 + 4 * lane);
  };

  // one ring stage of the B rows: pair j's k32 steps [st·kS, +kS) (256 B) and the 16 scale bytes holding them
  auto fill = [&](int buf, const uint8_t* src, int rowb, const uint8_t* ssrc, int srowb, const int* rows, int np,
                  int st) {
    uint8_t* rb = ring + buf * kRing;
    for (int i = gt; i < np * 17; i += 128) {
      const int j = i / 17, c = i % 17, r = rows[j];
      if (c < 16) cp_async16(rb + j * kRS + c * 16, src + (size_t)r * rowb + st * (kS * 32) + c * 16, true);
      else cp_async16(rb + kHalf * kRS + j * 16, ssrc + (size_t)r * srowb + ((st * kS) & ~15), true);
    }
    cp_async_commit();
  };
  auto mma_stage = [&](int buf, int st, float (&acc)[NT][4], int ntv) {
    const uint8_t* rb = ring + buf * kRing;
    const uint8_t* sb = rb + kHalf * kRS + ((st * kS) & 15);
#pragma unroll
    for (int i = 0; i < kS; i++) {
      const uint2 w = wf[st & 1][i];
      const uint32_t sa_ = sfa(sf[st & 1], i);
      const uint32_t a0 = e2m1x4_expand(w.x & 0xFFFF), a1 = e2m1x4_expand(w.x >> 16);
      const uint32_t a2 = e2m1x4_expand(w.y & 0xFFFF), a3 = e2m1x4_expand(w.y >> 16);
#pragma unroll
      for (int nt = 0; nt < NT; nt++)
        if (nt < ntv) {
          const int j = 8 * nt + g;
          const uint8_t* p = rb + j * kRS + i * 32 + 4 * t;
          mma_e2m1_e4m3(acc[nt], a0, a1, a2, a3, *reinterpret_cast<const uint32_t*>(p),
                        *reinterpret_cast<const uint32_t*>(p + 16), sa_, sb[j * 16 + i]);
        }
    }
  };
  auto lead_wait = [&](const int* c, uint32_t target) {
    if (w4 == 0 && lane == 0)
      while (p066::ld_acq(c) < target) __nanosleep(32);
    p066::gbar(wg);
  };

  if (w4 == 0 && lane == 0) s_t[wg] = atomicAdd(&cn->tick, 1);
  p066::gbar(wg);
  int tk = s_t[wg];
  p066::gbar(wg);  // read by all before next() rewrites it
  const uint2* wt = nullptr;
  const uint8_t* wst = nullptr;
  if (tk >= q_end && tk < n_all) {
    wptr(tk, wt, wst);
    ldw(wt, wst, 0, wf[0], sf[0]);
  }
  while (tk < n_all) {
    int nxt = 0;
    if (w4 == 0 && lane == 0) nxt = atomicAdd(&cn->tick, 1);  // used at the item's last stage (no stall here)
    int tn = n_all;
    const uint2* nwt = nullptr;
    const uint8_t* nwst = nullptr;
    auto next = [&] {  // at the last stage: publish the next ticket, start its first weight stage
      if (w4 == 0 && lane == 0) s_t[wg] = nxt;
      p066::gbar(wg);
      tn = s_t[wg];
      if (tn >= q_end && tn < n_all) {
        wptr(tn, nwt, nwst);
        ldw(nwt, nwst, 0, wf[0], sf[0]);
      }
    };
    if (tk < q_end) {
      // ---------------------------------------------------------------- latent row tk as MXFP8 (act_quant's arithmetic)
      if (gt < kLatent / 32) {
        const uint4* src = reinterpret_cast<const uint4*>(lat + (size_t)tk * ldl + gt * 32);
        float f[32];
#pragma unroll
        for (int q = 0; q < 4; q++) {
          const uint4 u = src[q];
          const uint32_t uv[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
          for (int i = 0; i < 4; i++) f[8 * q + 2 * i] = bf16_lo(uv[i]), f[8 * q + 2 * i + 1] = bf16_hi(uv[i]);
        }
        float mx = 0.f;
#pragma unroll
        for (int i = 0; i < 32; i++) mx = fmaxf(mx, fabsf(f[i]));
        const uint32_t ex = act_scale_e8(mx, 1e-10f);
        uint32_t o[8];
#pragma unroll
        for (int i = 0; i < 8; i++)
          o[i] = quant_e4m3(f[4 * i], ex) | quant_e4m3(f[4 * i + 1], ex) << 8 | quant_e4m3(f[4 * i + 2], ex) << 16 |
                 quant_e4m3(f[4 * i + 3], ex) << 24;
        uint4* d = reinterpret_cast<uint4*>(xq + (size_t)tk * kLatent + gt * 32);
        d[0] = make_uint4(o[0], o[1], o[2], o[3]), d[1] = make_uint4(o[4], o[5], o[6], o[7]);
        xs[tk * (kLatent / 32) + gt] = uint8_t(ex);
      }
      __threadfence();
      next();
      if (w4 == 0 && lane == 0) atomicAdd(&cn->qdone, 1);
    } else if (tk < u_end) {
      // ---------------------------------------------------------------- up item: sub-expert vi, inter dims [32c, +32)
      const Item it = item(tk - q_end);
      const int c = it.cb, np = it.np, ntv = (np + 7) >> 3;
      const int* pl = s_plist + it.pl;
      constexpr int GS = kHalf;  // g / u exchange row stride
      float acc[NT][4] = {};
      constexpr int NS = kUpKB / kS;
      if (q_end) lead_wait(&cn->qdone, uint32_t(M));
      for (int j = gt; j < np; j += 128) s_r[wg][j] = pl[j] / kTopK;
      p066::gbar(wg);
      fill(0, xqr, kLatent, xsr_, kLatent / 32, s_r[wg], np, 0);
#pragma unroll
      for (int st = 0; st < NS; st++) {
        cp_async_wait<0>();
        p066::gbar(wg);
        if (st + 1 < NS) {
          fill((st + 1) & 1, xqr, kLatent, xsr_, kLatent / 32, s_r[wg], np, st + 1);
          ldw(wt, wst, st + 1, wf[(st + 1) & 1], sf[(st + 1) & 1]);
        } else {
          next();
        }
        mma_stage(st & 1, st, acc, ntv);
      }
      // gate (warps 0, 1) / up (2, 3) of dims 16·(w4 & 1) + g / + 8, pairs 8nt + 2t, +1 → [2][32][GS] floats in ring
      // stage 0 (free: last read at stage NS − 2, before this stage's barrier)
      float* gu = reinterpret_cast<float*>(ring);
      float* dst = gu + (w4 >> 1) * 32 * GS;
      const int ra = 16 * (w4 & 1) + g;
#pragma unroll
      for (int nt = 0; nt < NT; nt++)
        if (nt < ntv) {
          const int q = 8 * nt + 2 * t;
          dst[ra * GS + q] = acc[nt][0], dst[ra * GS + q + 1] = acc[nt][1];
          dst[(ra + 8) * GS + q] = acc[nt][2], dst[(ra + 8) * GS + q + 1] = acc[nt][3];
        }
      p066::gbar(wg);
      for (int j = w4; j < np; j += 4) {
        const float h = situ(gu[lane * GS + j], gu[32 * GS + lane * GS + j]);
        const uint32_t ex = act_scale_e8(warp_max(fabsf(h)), 1e-10f);
        const size_t p = (size_t)pl[j];
        hq[p * kInterR + 32 * c + lane] = uint8_t(quant_e4m3(h, ex));
        if (lane == 0) hs[p * (kInterR / 32) + c] = uint8_t(ex);
      }
      __threadfence();
      p066::gbar(wg);
      if (w4 == 0 && lane == 0) atomicAdd(&cn->up[it.vi], 1);
    } else {
      // ---------------------------------------------------------------- down item: sub-expert vi, latent rows [64b, +64)
      const Item it = item(tk - u_end);
      const int b = it.cb, np = it.np, ntv = (np + 7) >> 3, T = 4 * b + w4;
      const int* pl = s_plist + it.pl;
      lead_wait(&cn->up[it.vi], 32u);
      float acc[NT][4] = {};
      constexpr int NS = kDnKB / kS;
      fill(0, hq, kInterR, hs, kInterR / 32, pl, np, 0);
#pragma unroll
      for (int st = 0; st < NS; st++) {
        cp_async_wait<0>();
        p066::gbar(wg);
        if (st + 1 < NS) {
          fill((st + 1) & 1, hq, kInterR, hs, kInterR / 32, pl, np, st + 1);
          ldw(wt, wst, st + 1, wf[(st + 1) & 1], sf[(st + 1) & 1]);
        } else {
          next();
        }
        mma_stage(st & 1, st, acc, ntv);
      }
      // y rows 16T + g / + 8 (latent), pairs 8nt + 2t, +1
#pragma unroll
      for (int nt = 0; nt < NT; nt++)
        if (nt < ntv)
#pragma unroll
          for (int i = 0; i < 4; i++) {
            const int j = 8 * nt + 2 * t + (i & 1);
            if (j < np) y[(size_t)pl[j] * kLatent + 16 * T + g + (i < 2 ? 0 : 8)] = acc[nt][i];
          }
      __threadfence();
      p066::gbar(wg);
      if (w4 == 0 && lane == 0) s_last[wg] = atomicAdd(&cn->blk[b], 1) == nv - 1;
      p066::gbar(wg);
      if (s_last[wg]) {  // every sub-expert's y of block b is in: combine it (k_combine's expression, slot order)
        __threadfence();
        for (int i = gt; i < M * 64; i += 128) {
          const int m = i >> 6, col = 64 * b + (i & 63);
          float a = 0.f;
#pragma unroll
          for (int k = 0; k < kTopK; k++)
            a = __fmaf_rn(s_w[m * kTopK + k], p066::ld_cgf(y + (size_t)(m * kTopK + k) * kLatent + col), a);
          put_out(part + (size_t)m * ld + col, a);
        }
      }
    }
    tk = tn, wt = nwt, wst = nwst;
  }
  // ---- the last CTA out resets the counters
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    if (atomicAdd(&cn->exit, 1) == int(gridDim.x) - 1) {
      __threadfence();
      for (int i = 0; i < nv; i++) cn->up[i] = 0;
      for (int i = 0; i < kBlk; i++) cn->blk[i] = 0;
      cn->tick = 0, cn->qdone = 0, cn->exit = 0;
      __threadfence();
    }
  }
}

template <class OutT>
void run(const MoeExperts* E, MoeScratch* S, const float* lg, const float* bias, const float* scale, const bf16* lat,
         int ldl, const uint8_t* lq, const uint8_t* ls, int M, int* ids, float* w, OutT* part, int ld, cudaStream_t s) {
  if (M < 1 || M > kMaxM || M > S->max_rows) qlog::fail("moe", "moe_fused128: M %d (scratch %d)", M, S->max_rows);
  static const int grid = [] {
    int dev = 0, sms = 0, occ = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    CUDA_CHECK(cudaFuncSetAttribute(k_fused194<OutT>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k_fused194<OutT>, kNT, kSmem));
    if (occ < 1) qlog::fail("moe", "moe_fused128: no CTA fits an SM (%d B dynamic shared)", kSmem);
    return sms * occ;
  }();
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(kNT);
  cfg.dynamicSmemBytes = kSmem;
  cfg.stream = s;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, k_fused194<OutT>, E->up, (const uint8_t*)E->ups, E->dn, (const uint8_t*)E->dns,
                                lg, bias, scale, lat, ldl, lq, ls, M, ids, w, S->dlat, S->dlat + kMaxM * kLatent, S->hq,
                                S->hs, S->y, reinterpret_cast<Cnt*>(S->dcnt), part, ld));
}

}  // namespace p194
