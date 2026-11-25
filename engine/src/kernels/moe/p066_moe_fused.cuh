// p066_moe_fused.cuh — P066 moe_fused: a decode / verify MoE call (M ≤ 32 rows) as ONE persistent launch — the
// routing (P036's k_route36 per row), the latent's MXFP8 quantization (act_quant's group-32 arithmetic) and the
// experts with their combine — instead of moe_route + act_quant + the expert launches. Included by moe.cu after P035
// (its Sel, loads and PDL helpers).
//   prologue (every CTA, after griddepcontrol.wait): route the M rows (a warp per row; CTA 0 writes ids / w out), the
//            union of the selected experts (ascending id) and each expert's pairs (row order); M ≤ 8: the latent as
//            MXFP8 into shared memory (quantized, or copied when the caller passes fc1's MXFP8 copy).
//   items   : taken from one atomic ticket by warp groups of 4 warps, in this order: (M > 8, bf16 latent) the M latent
//            rows' MXFP8 quantization (→ global, then a counter); every (expert, 32 inter dims) up item (k_up35's
//            tile: gate rows on warps 0, 1, up rows on 2, 3; h → hq / hs at pair row·10 + slot, then the expert's up
//            counter); every (expert, 64 latent rows) down item (k_dn35's tile per warp; y → global, L2-resident), whose last finisher
//            per latent block combines it: part = Σ_slot w·y in slot order (k_combine's FFMA chain). An item waits
//            only for items taken before it (quantization, its expert's up items), so it always progresses.
//   weights : each warp streams its 16-row tile with the next stage's words and scales in flight (P035's loads) and
//            the next item's first stage loaded during the current item's last; a stage's weight scales (128 B) as one
//            32-bit load a lane, spread by shuffles. B operand: M ≤ 8 from shared memory (up) / global into registers a
//            stage ahead (down), one n8 tile; M > 8 the expert's latent or h rows (≤ 32 pairs = 4 n8 tiles) through a
//            2-stage cp.async ring per warp group. An expert's weights are read once per call whatever its number of
//            rows (no per-row re-reads).
//   The last CTA out resets the counters for the next call. Launched as a programmatic dependent.
// Bitwise the base: the same routing expressions, quantization, MMA chains (MMA columns are independent), SiTU /
// MXFP8 and combine expressions.
namespace p066 {

constexpr int kMaxM = 32, kMaxP = kMaxM * kTopK, kS = 8;
constexpr int kBlk = kLatent / 64;          // down latent blocks of 64 rows (4 tiles)
constexpr int kRS = kS * 32 + 16;           // ring row stride (bytes): one stage of a pair's B row, conflict-free
constexpr int kRing = kMaxM * kRS + kMaxM * 16;  // one ring stage: B rows + their scales (16 B a pair)
constexpr int kGroupSmem = 2 * kRing;       // per warp group (the up epilogue's g / u exchange reuses stage 0)
constexpr int kSmM = 8, kSmRB = kLatent + 16;  // kSm: rows, shared latent row stride (conflict-free B fragments)

struct Cnt {           // scratch counters (zero between calls; dcnt holds 4 KB)
  int tick, exit, qdone;
  int up[kMaxP];       // per union slot: finished up items (32 = done)
  int blk[kBlk];       // per latent block: finished down items
};

__device__ __forceinline__ uint32_t ld_acq(const int* p) {
  uint32_t v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint32_t ld_na32(const void* p) {
  uint32_t r;
  asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(r) : "l"(p));
  return r;
}
__device__ __forceinline__ uint32_t ld_cg32(const void* p) {
  uint32_t r;
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(r) : "l"(p));
  return r;
}
__device__ __forceinline__ uint32_t ld_cg8(const void* p) {
  uint32_t r;
  asm volatile("ld.global.cg.u8 %0, [%1];" : "=r"(r) : "l"(p));
  return r;
}
__device__ __forceinline__ float ld_cgf(const float* p) {
  float r;
  asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(r) : "l"(p));
  return r;
}
__device__ __forceinline__ void gbar(int wg) {  // the 128 threads of warp group wg
  asm volatile("bar.sync %0, 128;" ::"r"(1 + wg) : "memory");
}

// P036's k_route36 for one row (one warp): the same expressions, outputs to oid / ow (slot order)
__device__ __forceinline__ void route_row(const float* __restrict__ lg, const float* __restrict__ bias,
                                          const float* __restrict__ scale, int row, int lane, float* sa, int* oid,
                                          float* ow) {
  const float4* l0 = reinterpret_cast<const float4*>(lg + (size_t)row * 1024 + 16 * lane);
  const float4* l1 = reinterpret_cast<const float4*>(lg + (size_t)row * 1024 + 512 + 16 * lane);
  const float4* bb = reinterpret_cast<const float4*>(bias + 16 * lane);
  float s[16];
  uint32_t key[16];
#pragma unroll
  for (int q = 0; q < 4; q++) {
    const float4 a = l0[q], b = l1[q], c = bb[q];
    const float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w}, cv[4] = {c.x, c.y, c.z, c.w};
#pragma unroll
    for (int i = 0; i < 4; i++) {
      s[4 * q + i] = sigmoid_ieee(av[i]) + sigmoid_ieee(bv[i]);
      key[4 * q + i] = p036::okey(s[4 * q + i] + cv[i]);
    }
  }
  auto local_best = [&](uint32_t& bk, int& bj) {
    bk = 0, bj = 0;
#pragma unroll
    for (int j = 0; j < 16; j++)
      if (key[j] > bk) bk = key[j], bj = j;
  };
  uint32_t bk;
  int bj;
  local_best(bk, bj);
  int my_id = 0;
  uint32_t my_v = 0;
  float my_s = 0.f;
#pragma unroll 1
  for (int k = 0; k < kTopK; k++) {
    const uint32_t vmax = __reduce_max_sync(0xffffffffu, bk);
    const int id = int(__reduce_min_sync(0xffffffffu, bk == vmax ? uint32_t(16 * lane + bj) : 0xffffu));
    float park = 0.f;
    if (lane == (id >> 4)) {
#pragma unroll
      for (int j = 0; j < 16; j++)
        if (j == bj) park = s[j], key[j] = 0;
      local_best(bk, bj);
    }
    const float sv = __shfl_sync(0xffffffffu, park, id >> 4);
    if (lane == k) my_id = id, my_v = vmax, my_s = sv;
  }
  const uint32_t v10 = __shfl_sync(0xffffffffu, my_v, kTopK - 1);
  const int kk = my_v > v10 ? my_id : my_id + kExperts;
  int slot = 0;
#pragma unroll
  for (int j = 0; j < kTopK; j++) {
    const int kj = __shfl_sync(0xffffffffu, kk, j);
    slot += kj < kk;
  }
  if (lane < 16) sa[lane] = 0.f;
  __syncwarp();
  if (lane < kTopK) sa[slot] = my_s;
  __syncwarp();
  const int sid = __shfl_sync(0xffffffffu, my_id, [&] {
    int src = 0;
#pragma unroll
    for (int j = 0; j < kTopK; j++)
      if (__shfl_sync(0xffffffffu, slot, j) == lane) src = j;
    return src;
  }());
  const float a = sa[lane & 15];
  float t = a + __shfl_xor_sync(0xffffffffu, a, 8);
  t = t + __shfl_xor_sync(0xffffffffu, t, 4);
  t = t + __shfl_xor_sync(0xffffffffu, t, 2);
  t = t + __shfl_xor_sync(0xffffffffu, t, 1);
  const float tot = fmaxf(t, 1e-20f);
  if (lane < kTopK) oid[lane] = sid, ow[lane] = __fdiv_rn(a, tot) * scale[sid];
}

// kSm (M ≤ 8): the latent quantized by every CTA into shared memory and the B operands straight from it (up) or from
// global into registers a stage ahead (down), one n8 tile — no per-stage group barrier. Otherwise (M ≤ 32): the latent
// quantized once (the first M items) into global, the B operands through the group's cp.async ring, ≤ 4 n8 tiles.
template <bool kSm>
constexpr int dyn_smem(int groups) {
  return kSm ? kSmM * kSmRB + kSmM * 64 + groups * 2 * 32 * kSmM * 4 : groups * kGroupSmem;
}
template <class OutT, int kGroups, bool kSm>
__global__ void __launch_bounds__(128 * kGroups, 1) k_fused66(
    const uint2* __restrict__ Wu, const uint8_t* __restrict__ WSu, const uint2* __restrict__ Wd,
    const uint8_t* __restrict__ WSd, const float* __restrict__ lg, const float* __restrict__ bias,
    const float* __restrict__ scale, const bf16* __restrict__ lat, int ldl, const uint8_t* __restrict__ lq,
    const uint8_t* __restrict__ ls, int M, int* ids_out, float* w_out, uint8_t* xq, uint8_t* xs, uint8_t* hq,
    uint8_t* hs, float* y, Cnt* cn, OutT* part, int ld) {
  constexpr int kWarps = 4 * kGroups, NT = kSm ? 1 : kMaxM / 8;
  __shared__ float sa[kWarps][16];
  __shared__ int s_id[kMaxP];
  __shared__ float s_w[kMaxP];
  __shared__ p035::Sel S;
  __shared__ int s_e[kMaxP], s_np[kMaxP], s_poff[kMaxP], s_plist[kMaxP];
  __shared__ uint32_t s_rows[kMaxP];
  __shared__ int s_t[kGroups], s_last[kGroups];
  __shared__ int s_r[kGroups][kMaxM];  // an up item's latent rows (pair / 10)
  extern __shared__ __align__(16) uint8_t dsm[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wg = warp >> 2, w4 = warp & 3, gt = tid & 127;
  const int P = M * kTopK;
  uint8_t* ring = dsm + wg * kGroupSmem;  // !kSm
  uint8_t(*sx)[kSmRB] = reinterpret_cast<uint8_t(*)[kSmRB]>(dsm);  // kSm: the latent, MXFP8
  uint8_t(*ssx)[64] = reinterpret_cast<uint8_t(*)[64]>(dsm + kSmM * kSmRB);
  float* gsm = reinterpret_cast<float*>(dsm + kSmM * kSmRB + kSmM * 64) + wg * 2 * 32 * kSmM;  // kSm: g / u exchange

  p035::pdl_wait();  // logits (router GEMM) and latent (fc1) complete
  // ---- routing: warp w routes rows w, w + kWarps, …
  for (int m = warp; m < M; m += kWarps) {
    route_row(lg, bias, scale, m, lane, sa[warp], s_id + m * kTopK, s_w + m * kTopK);
    __syncwarp();
  }
  for (int i = tid; i < kMaxP; i += blockDim.x) s_rows[i] = 0;
  if (kSm && lq) {  // the latent already as MXFP8 (fc1's epilogue): copy it in
    for (int i = tid; i < M * (kLatent / 16); i += blockDim.x) {
      const int m = i / (kLatent / 16), c = i % (kLatent / 16);
      *reinterpret_cast<uint4*>(&sx[m][16 * c]) = reinterpret_cast<const uint4*>(lq + (size_t)m * kLatent)[c];
    }
    for (int i = tid; i < M * 4; i += blockDim.x)
      *reinterpret_cast<uint4*>(&ssx[i >> 2][16 * (i & 3)]) = reinterpret_cast<const uint4*>(ls + (size_t)(i >> 2) * 64)[i & 3];
  } else if constexpr (kSm) {  // the latent as MXFP8 into shared memory, a 32-group a thread (act_quant's arithmetic)
    for (int gi = tid; gi < M * (kLatent / 32); gi += blockDim.x) {
      const int m = gi / (kLatent / 32), gr = gi % (kLatent / 32);
      const uint4* src = reinterpret_cast<const uint4*>(lat + (size_t)m * ldl + gr * 32);
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
      *reinterpret_cast<uint4*>(&sx[m][gr * 32]) = make_uint4(o[0], o[1], o[2], o[3]);
      *reinterpret_cast<uint4*>(&sx[m][gr * 32 + 16]) = make_uint4(o[4], o[5], o[6], o[7]);
      ssx[m][gr] = uint8_t(ex);
    }
  }
  __syncthreads();
  if (blockIdx.x == 0)
    for (int i = tid; i < P; i += blockDim.x) ids_out[i] = s_id[i], w_out[i] = s_w[i];
  // ---- the union of the experts (ascending id) and each one's pairs in row order (compact lists)
  int id2[2];
  p035::sel_build(S, s_id, P, id2);
  int slot2[2];
#pragma unroll
  for (int j = 0; j < 2; j++)
    if (id2[j] >= 0) {
      const int i = tid + j * blockDim.x;
      slot2[j] = p035::sel_slot(S, id2[j]);
      atomicOr(&s_rows[slot2[j]], 1u << (i / kTopK));
      s_e[slot2[j]] = id2[j];
    }
  __syncthreads();
  const int ne = S.n;
  if (warp == 0) {  // pair offsets: exclusive scan of the experts' row counts
    int base = 0;
    for (int u0 = 0; u0 < ne; u0 += 32) {
      const int u = u0 + lane, c = u < ne ? __popc(s_rows[u]) : 0;
      int incl = c;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int v = __shfl_up_sync(0xffffffffu, incl, o);
        if (lane >= o) incl += v;
      }
      if (u < ne) s_np[u] = c, s_poff[u] = base + incl - c;
      base += __shfl_sync(0xffffffffu, incl, 31);
    }
  }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 2; j++)
    if (id2[j] >= 0) {
      const int i = tid + j * blockDim.x, m = i / kTopK, u = slot2[j];
      s_plist[s_poff[u] + __popc(s_rows[u] & ((1u << m) - 1u))] = i;
    }
  __syncthreads();
  p035::pdl_launch();
  const int q_end = kSm || lq ? 0 : M, n_up = ne * 32, u_end = q_end + n_up, n_all = q_end + 2 * n_up;
  const uint8_t* xqr = lq ? lq : xq;  // !kSm: where the up items read the quantized latent
  const uint8_t* xsr_ = lq ? ls : xs;

  // Item tk → this warp's weight tile (words / scales): up item u·32 + c → tile c·4 + w4 of expert u; down item
  // u·32 + b → latent tile 4b + w4 of expert u.
  auto wptr = [&](int tk, const uint2*& wt, const uint8_t*& wst) {
    if (tk < u_end) {
      const int r = tk - q_end, e = s_e[r >> 5], c = r & 31;
      wt = Wu + (size_t)e * kUpWords + ((size_t)(c * 4 + w4) * kUpKB) * 32 + lane;
      wst = WSu + (size_t)e * kUpScales + ((size_t)(c * 4 + w4) * kUpKB) * 16;
    } else {
      const int r = tk - u_end, e = s_e[r >> 5], T = 4 * (r & 31) + w4;
      wt = Wd + (size_t)e * kDnWords + ((size_t)T * kDnKB) * 32 + lane;
      wst = WSd + (size_t)e * kDnScales + ((size_t)T * kDnKB) * 16;
    }
  };
  uint2 wf[2][kS];
  // weight scales: a stage's 8 k32 steps × 16 rows = 128 contiguous bytes, one 32-bit load a lane (lane L: step L / 4,
  // bytes 4·(L % 4) …); the byte of row g + 8·(lane & 1) at step i comes from lane 4i + its word by one shuffle
  static_assert(kS == 8, "one 128-byte scale load a stage");
  uint32_t sf[2];
  const int s_src = 2 * (lane & 1) + (lane >> 4), s_sh = 8 * (g & 3);
  auto sfa = [&](uint32_t raw, int i) { return (__shfl_sync(0xffffffffu, raw, 4 * i + s_src) >> s_sh) & 0xFFu; };
  auto ldw = [&](const uint2* wt, const uint8_t* wst, int st, uint2 (&bw)[kS], uint32_t& s) {
#pragma unroll
    for (int i = 0; i < kS; i++) bw[i] = p035::ld_na64(wt + (size_t)(st * kS + i) * 32);
    s = ld_na32(wst + (size_t)st * kS * 16 + 4 * lane);
  };

  // one ring stage of the B rows: pair j's k32 steps [st·kS, +kS) (256 B) and the 16 scale bytes holding them
  auto fill = [&](int buf, const uint8_t* src, int rowb, const uint8_t* ssrc, int srowb, const int* rows, int np,
                  int st) {
    uint8_t* rb = ring + buf * kRing;
    for (int i = gt; i < np * 17; i += 128) {
      const int j = i / 17, c = i % 17, r = rows[j];
      if (c < 16) cp_async16(rb + j * kRS + c * 16, src + (size_t)r * rowb + st * (kS * 32) + c * 16, true);
      else cp_async16(rb + kMaxM * kRS + j * 16, ssrc + (size_t)r * srowb + ((st * kS) & ~15), true);
    }
    cp_async_commit();
  };
  auto mma_stage = [&](int buf, int st, float (&acc)[NT][4], int ntv) {
    const uint8_t* rb = ring + buf * kRing;
    const uint8_t* sb = rb + kMaxM * kRS + ((st * kS) & 15);
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
      while (ld_acq(c) < target) __nanosleep(32);
    gbar(wg);
  };

  if (w4 == 0 && lane == 0) s_t[wg] = atomicAdd(&cn->tick, 1);
  gbar(wg);
  int tk = s_t[wg];
  gbar(wg);  // read by all before next() rewrites it
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
      gbar(wg);
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
      // ---------------------------------------------------------------- up item: expert u, inter dims [32c, +32)
      const int r = tk - q_end, u = r >> 5, c = r & 31, np = s_np[u], ntv = (np + 7) >> 3;
      const int* pl = s_plist + s_poff[u];
      constexpr int GS = kSm ? kSmM : kMaxM;  // g / u exchange row stride
      float acc[NT][4] = {};
      constexpr int NS = kUpKB / kS;
      if constexpr (kSm) {
        const bool live = g < np;
        const uint8_t* xr = live ? &sx[pl[g] / kTopK][4 * t] : &sx[0][0];
        const uint8_t* xsr = live ? &ssx[pl[g] / kTopK][0] : &ssx[0][0];
#pragma unroll
        for (int st = 0; st < NS; st++) {
          if (st + 1 < NS) ldw(wt, wst, st + 1, wf[(st + 1) & 1], sf[(st + 1) & 1]);
          else next();
#pragma unroll
          for (int i = 0; i < kS; i++) {
            const int kb = st * kS + i;
            const uint2 w = wf[st & 1][i];
            uint32_t b0 = 0, b1 = 0, sb = 0;
            if (live) {
              b0 = *reinterpret_cast<const uint32_t*>(xr + kb * 32);
              b1 = *reinterpret_cast<const uint32_t*>(xr + kb * 32 + 16);
              sb = xsr[kb];
            }
            mma_e2m1_e4m3(acc[0], e2m1x4_expand(w.x & 0xFFFF), e2m1x4_expand(w.x >> 16), e2m1x4_expand(w.y & 0xFFFF),
                          e2m1x4_expand(w.y >> 16), b0, b1, sfa(sf[st & 1], i), sb);
          }
        }
      } else {
        if (q_end) lead_wait(&cn->qdone, uint32_t(M));
        for (int j = gt; j < np; j += 128) s_r[wg][j] = pl[j] / kTopK;
        gbar(wg);
        fill(0, xqr, kLatent, xsr_, kLatent / 32, s_r[wg], np, 0);
#pragma unroll
        for (int st = 0; st < NS; st++) {
          cp_async_wait<0>();
          gbar(wg);
          if (st + 1 < NS) {
            fill((st + 1) & 1, xqr, kLatent, xsr_, kLatent / 32, s_r[wg], np, st + 1);
            ldw(wt, wst, st + 1, wf[(st + 1) & 1], sf[(st + 1) & 1]);
          } else {
            next();
          }
          mma_stage(st & 1, st, acc, ntv);
        }
      }
      // gate (warps 0, 1) / up (2, 3) of dims 16·(w4 & 1) + g / + 8, pairs 8nt + 2t, +1 → [2][32][GS] floats (!kSm:
      // in ring stage 0 — free: last read at stage NS − 2, before this stage's barrier)
      float* gu = kSm ? gsm : reinterpret_cast<float*>(ring);
      float* dst = gu + (w4 >> 1) * 32 * GS;
      const int ra = 16 * (w4 & 1) + g;
#pragma unroll
      for (int nt = 0; nt < NT; nt++)
        if (nt < ntv) {
          const int q = 8 * nt + 2 * t;
          dst[ra * GS + q] = acc[nt][0], dst[ra * GS + q + 1] = acc[nt][1];
          dst[(ra + 8) * GS + q] = acc[nt][2], dst[(ra + 8) * GS + q + 1] = acc[nt][3];
        }
      gbar(wg);
      for (int j = w4; j < np; j += 4) {
        const float h = situ(gu[lane * GS + j], gu[32 * GS + lane * GS + j]);
        const uint32_t ex = act_scale_e8(warp_max(fabsf(h)), 1e-10f);
        const size_t p = (size_t)pl[j];
        hq[p * kInterR + 32 * c + lane] = uint8_t(quant_e4m3(h, ex));
        if (lane == 0) hs[p * (kInterR / 32) + c] = uint8_t(ex);
      }
      __threadfence();
      gbar(wg);
      if (w4 == 0 && lane == 0) atomicAdd(&cn->up[u], 1);
    } else {
      // ---------------------------------------------------------------- down item: expert u, latent rows [64b, +64)
      const int r = tk - u_end, u = r >> 5, b = r & 31, np = s_np[u], ntv = (np + 7) >> 3, T = 4 * b + w4;
      const int* pl = s_plist + s_poff[u];
      lead_wait(&cn->up[u], 32u);
      float acc[NT][4] = {};
      constexpr int NS = kDnKB / kS;
      if constexpr (kSm) {  // B (h of the ≤ 8 pairs) from global into registers, a stage ahead
        uint32_t bx[2][kS][3];
        const bool live = g < np;
        const size_t pg = live ? (size_t)pl[g] : 0;
        const uint8_t* hp = hq + pg * kInterR + 4 * t;
        const uint8_t* sp = hs + pg * (kInterR / 32);
        auto ldx = [&](int st, uint32_t (&x)[kS][3]) {
#pragma unroll
          for (int i = 0; i < kS; i++) {
            const int kb = st * kS + i;
            x[i][0] = live ? ld_cg32(hp + kb * 32) : 0u, x[i][1] = live ? ld_cg32(hp + kb * 32 + 16) : 0u;
            x[i][2] = live ? ld_cg8(sp + kb) : 0u;
          }
        };
        ldx(0, bx[0]);
#pragma unroll
        for (int st = 0; st < NS; st++) {
          if (st + 1 < NS) {
            ldw(wt, wst, st + 1, wf[(st + 1) & 1], sf[(st + 1) & 1]);
            ldx(st + 1, bx[(st + 1) & 1]);
          } else {
            next();
          }
#pragma unroll
          for (int i = 0; i < kS; i++) {
            const uint2 w = wf[st & 1][i];
            mma_e2m1_e4m3(acc[0], e2m1x4_expand(w.x & 0xFFFF), e2m1x4_expand(w.x >> 16), e2m1x4_expand(w.y & 0xFFFF),
                          e2m1x4_expand(w.y >> 16), bx[st & 1][i][0], bx[st & 1][i][1], sfa(sf[st & 1], i),
                          bx[st & 1][i][2]);
          }
        }
      } else {
        fill(0, hq, kInterR, hs, kInterR / 32, pl, np, 0);
#pragma unroll
        for (int st = 0; st < NS; st++) {
          cp_async_wait<0>();
          gbar(wg);
          if (st + 1 < NS) {
            fill((st + 1) & 1, hq, kInterR, hs, kInterR / 32, pl, np, st + 1);
            ldw(wt, wst, st + 1, wf[(st + 1) & 1], sf[(st + 1) & 1]);
          } else {
            next();
          }
          mma_stage(st & 1, st, acc, ntv);
        }
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
      gbar(wg);
      if (w4 == 0 && lane == 0) s_last[wg] = atomicAdd(&cn->blk[b], 1) == ne - 1;
      gbar(wg);
      if (s_last[wg]) {  // every expert's y of block b is in: combine it (k_combine's expression, slot order)
        __threadfence();
        for (int i = gt; i < M * 64; i += 128) {
          const int m = i >> 6, col = 64 * b + (i & 63);
          float a = 0.f;
#pragma unroll
          for (int k = 0; k < kTopK; k++)
            a = __fmaf_rn(s_w[m * kTopK + k], ld_cgf(y + (size_t)(m * kTopK + k) * kLatent + col), a);
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
      for (int i = 0; i < ne; i++) cn->up[i] = 0;
      for (int i = 0; i < kBlk; i++) cn->blk[i] = 0;
      cn->tick = 0, cn->qdone = 0, cn->exit = 0;
      __threadfence();
    }
  }
}

template <class OutT, int kGroups, bool kSm>
void run2(const MoeExperts* E, MoeScratch* S, const float* lg, const float* bias, const float* scale, const bf16* lat,
          int ldl, const uint8_t* lq, const uint8_t* ls, int M, int* ids, float* w, OutT* part, int ld, cudaStream_t s) {
  constexpr int smem = dyn_smem<kSm>(kGroups);
  static const int grid = [] {
    int dev = 0, sms = 0, occ = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    CUDA_CHECK(cudaFuncSetAttribute(k_fused66<OutT, kGroups, kSm>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k_fused66<OutT, kGroups, kSm>, 128 * kGroups, smem));
    return sms * std::max(1, occ);
  }();
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(128 * kGroups);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = s;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, k_fused66<OutT, kGroups, kSm>, E->up, (const uint8_t*)E->ups, E->dn,
                                (const uint8_t*)E->dns, lg, bias, scale, lat, ldl, lq, ls, M, ids, w, S->dlat,
                                S->dlat + kMaxM * kLatent, S->hq, S->hs, S->y, reinterpret_cast<Cnt*>(S->dcnt), part,
                                ld));
}

template <class OutT>
void run(const MoeExperts* E, MoeScratch* S, const float* lg, const float* bias, const float* scale, const bf16* lat,
         int ldl, const uint8_t* lq, const uint8_t* ls, int M, int* ids, float* w, OutT* part, int ld, cudaStream_t s) {
  if (M <= kSmM) return run2<OutT, 2, true>(E, S, lg, bias, scale, lat, ldl, lq, ls, M, ids, w, part, ld, s);
  run2<OutT, 2, false>(E, S, lg, bias, scale, lat, ldl, lq, ls, M, ids, w, part, ld, s);
}

}  // namespace p066
