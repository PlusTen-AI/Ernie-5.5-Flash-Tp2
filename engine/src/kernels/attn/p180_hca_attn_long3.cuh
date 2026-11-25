// p180_hca_attn_long3.cuh — P180 hca_attn_long3 (default off: not faster than P172): the long-context hca_attn_fused
// (rows of 5 … 48 key chunks) as a warp-specialized pipeline. Included by hca.cu after p172_hca_attn_long2.cuh (uses
// P088's Row / MMA / ldmatrix / barrier wrappers and P150's dq8 / cache-hint helpers).
//
// Design — per CTA (one per SM) four warpgroups with their own register budgets (setmaxnreg, works on sm_121a; ptxas
// allocates each role's code to its budget):
//   · 4 producer warps (120 registers): work-item tickets one item ahead, the K tile gather (4 threads a key, 32-byte
//     runs; the next tile's raw rows in flight while the current one waits dequantized for its buffer) and the exact
//     dequantization into the 2 shared K buffers (P150's tile: dq8 nope, rope as stored, zero rows);
//   · 4 score warps (184) in 2 groups taking the even / odd tiles of the CTA's tile stream, each warp 16 heads: S = Q·Kᵀ
//     with the prepped q of its heads as A fragments in registers (base k_attn's operand roles), 4 independent MMA
//     chains a tile, K as B fragments (half P172's shared-memory reads); the tile's online softmax; each group preps an
//     item's q itself between its tiles (Σ x² in P088's slice layout, the fragments loaded and prepped in place);
//   · 8 P·V warps (104): P172's P·V, chunk partials, row counter, merge (passes of 8 heads) and VHA.
// The tiles of consecutive items form one stream per CTA (no pipeline drain at an item boundary).
//
// Measured (one GPU, 2048 rows × 32 heads, CUDA events, median of 25): 131,072 / 524,288 / 1,040,000 4.48 / 15.72 /
// 39.35 ms (P172) → 4.79 / 16.52 / 38.97 ms (2.90 / 2.73 / 3.30× the bf16 MMA floor); bitwise equal to P172.
// Limit (clock64 events of CTA 0 that a barrier orders; a clock read after bar.sync is not ordered against the wait):
// per two tiles ~10k cycles = P·V start → K buffer free → refilled 4.5k + score chain 3.7k + softmax hand-off 1.8k —
// the 2-buffer cycle (smem: 2 × 33 KB bf16 K tiles + P / e / merge scratch = 96.3 KB of the 99 KB a CTA may use).
// Estimated, not run: shortening it needs registers the 64K file lacks beside q (2 × 32 KB) and the P·V accumulators
// (64 KB) — K released at the P·V tile's start (+32 a P·V thread) together with a producer holding a dequantized tile
// (budgets 560 > 512 a thread on average). Measured in dev builds (trace on, timing only): no K tile fill at all (no
// dequant, no K stores) 17.97 → 13.5 ms at 524,288; per-call bf16 key copies (dequantized once per key; 1.75× the K
// bytes) refilled by cp.async after the buffer frees or staged one tile ahead in registers: no gain. The merge keeps
// half of P172's loads in flight (passes of 8 heads to fit 104 registers) — part of the gap at 524,288.
//
// Bitwise contract — P172's (= P150 = P088 = the base): the same keys, 32-key tiles, 256-key chunks, scores (the
// m16n8k16 chain over the 32 k-steps in order, × 512^-0.5; A = q / B = K as the base, P088's K·Qᵀ is the same bits),
// online softmax per tile (max, expf, bf16 P, the tile sum in key order, l = l·fac + sum), P·V MMAs, chunk partials,
// chunk merge order, q prep (per head Σ x² over the same 8-dim slices, warp sums, two halves, × rsqrt, RoPE) and VHA
// chains. Deterministic: a row's arithmetic does not depend on the CTA, the item split, M or the other rows.
#pragma once

namespace e55 {
namespace p180 {

using p088::bar_arrive;
using p088::bar_sync;
using p088::CS;
using p088::kCh;
using p088::kD;
using p088::kKS;
using p088::kKT;
using p088::kLH;
using p088::ldsm4;
using p088::ldsm4t;
using p088::mma;
using p088::Row;
using p088::TPC;
using p150::discard_l2;
using p150::dq8;
using p150::ld2;
using p150::pol_first;
using p150::pol_last;
using p150::st1;
using p150::st2;
using p150::stu;

constexpr int kTh = 512;                    // warpgroups: 0 score, 1–2 P·V, 3 producers
constexpr int kRS = 184, kRV = 104, kRG = 120;  // registers a thread (setmaxnreg): kRS + 2·kRV + kRG = 4 · 128
// named barriers: K ready [2] (producers → score group), K free [2] (P·V → producers), P ready [2] (score group →
// P·V), running max / sum to the other group [2 + 2], P·V warps' own, producers' own
enum { kBarKr = 1, kBarKf = 3, kBarP = 5, kBarM = 7, kBarL = 9, kBarV = 11, kBarG = 12 };
constexpr int kNKr = 128 + 64, kNKf = 256 + 128, kNP = 64 + 256, kNML = 128, kNV = 256, kNG = 128;
constexpr int kMaxRows = 16384;  // rows per pass (per-row counters)
constexpr int kMinCh = 5;        // dispatch: max_keys of 5 … 48 chunks (P172's range)
constexpr int kMaxCh = 48;
constexpr int kRing = 8;         // item ring (the producers' tickets run ≤ 5 items ahead of the P·V warps)

__device__ int g_cnt[kMaxRows];  // per row of the pass: finished items (zero between launches)
__device__ int g_next, g_done;   // ticket counter, CTAs done (zero between launches)

struct Item {
  int id, r, j0, nt;  // ticket (< 0: the stream's end), row, first tile, tiles
};

struct Smem {
  bf16 k[2][kKT][kKS];  // K tiles (stream tile n in k[n & 1])
  bf16 p[2][kLH][kKT + 8];
  float e[2][kLH][kKT + 1];
  float fac[2][kLH];
  float m[kLH], l[kLH];  // running max / sum of the current chunk
  float den2[kLH];       // merge: the denominator per head
  float mf[kMaxCh][kLH];  // merge: m_c per chunk and head, then expf(m_c − M)
  float lf[kMaxCh][kLH];  // merge: l_c
  __align__(16) float U[kLH * kVhaRank];
  float rcs[4][32], rsn[4][32];  // score warp w: its item's RoPE angles (q prep)
  Item it[kRing];  // item k of the CTA's stream in it[k % kRing]
  int lastf;       // P·V warps: this CTA merges the row
};

template <int N>
__device__ __forceinline__ void reg_inc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(N));
}
template <int N>
__device__ __forceinline__ void reg_dec() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(N));
}
__device__ __forceinline__ void ldsm4r(uint32_t* r, const void* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}
__device__ __forceinline__ uint32_t ldnc32(const void* p) {
  uint32_t v;
  asm("ld.global.nc.b32 %0, [%1];\n" : "=r"(v) : "l"(p));
  return v;
}

__device__ __forceinline__ int row_nk(int p) { return (p + 1) / kRatio + min(p + 1, kWindow); }

// the next non-empty work item (row-major tickets; P150's chunk groups)
__device__ __forceinline__ Item fetch(const RowMeta* meta, int nitems, int ng) {
  int id = atomicAdd(&g_next, 1);
  while (id < nitems) {
    const int r = id / ng, gi = id - r * ng, nk = row_nk(meta[r].pos);
    const int nch = (nk + kCh - 1) / kCh, ntile = (nk + kKT - 1) / kKT;
    const int c0 = gi * nch / ng, c1 = (gi + 1) * nch / ng;
    if (c0 < c1) return Item{id, r, c0 * TPC, min(ntile, c1 * TPC) - c0 * TPC};
    id = atomicAdd(&g_next, 1);
  }
  return Item{-1, 0, 0, 0};
}

// one K tile's raw rows, 128 producer threads: thread (key kq, piece pp) holds nope words pp + 4i (8 e4m3 codes each,
// i < 14), rope words 56 + pp + 4i (4 bf16, i < 4) and the scale word (7 UE8M0) — a key's 4 pieces read 32 contiguous
// bytes a load
struct Raw {
  uint2 nw[14], rw[4], sc;
  bool ok;  // the key exists (else a zero row)
  __device__ __forceinline__ void load(const uint8_t* row, int pp, uint64_t pol) {
    ok = row != nullptr;
    if (row) {
      const uint2* src = reinterpret_cast<const uint2*>(row);
#pragma unroll
      for (int i = 0; i < 14; i++) nw[i] = ld2(src + pp + 4 * i, pol);
#pragma unroll
      for (int i = 0; i < 4; i++) rw[i] = ld2(src + kNope / 8 + pp + 4 * i, pol);
      sc = ld2(src + 72, pol);
    }
  }
};

// grid: persistent CTAs; items id = row · ng + group (row-major tickets); ws: per (row, chunk) slab of CS floats
__global__ void __launch_bounds__(kTh, 1)
    k_attn(const bf16* __restrict__ q, const uint8_t* __restrict__ rows, const RowMeta* __restrict__ meta, HcaCache c,
           const float* __restrict__ sink, const float* __restrict__ inv, const float* __restrict__ U, int M, int ng,
           int nchmax, bf16* __restrict__ o, float* __restrict__ zp, float* __restrict__ ws) {
  extern __shared__ __align__(16) uint8_t smraw[];
  Smem& S = *reinterpret_cast<Smem*>(smraw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int nitems = M * ng;
  for (int i = tid; i < kLH * kVhaRank; i += kTh) S.U[i] = U[i];
  if (tid == 0) {  // items 0 and 1 (then the producers take tickets ahead of their loads)
    S.it[0] = fetch(meta, nitems, ng);
    S.it[1] = S.it[0].id < 0 ? S.it[0] : fetch(meta, nitems, ng);
  }
  __syncthreads();

  if (warp < 4) {
    // ================= score warps: group grp takes the stream tiles n ≡ grp (mod 2); warp wi: heads 16·wi … =================
    reg_inc<kRS>();
    const int grp = warp >> 1, wi = warp & 1, hA = 16 * wi + g, hB = hA + 8;
    const float scale = rsqrtf(float(kD));
    const uint64_t keep = pol_last();
    uint32_t qf[kD / 16][4];  // A fragments: [kk] = {head hA dims 16kk+2t.., hB, hA dims +8, hB +8}
    int n = grp, k = 0, jj = grp, qk = -1;
    Item I = S.it[0];
    while (I.id >= 0 && jj >= I.nt) jj -= I.nt, k++, I = S.it[k % kRing];
    while (I.id >= 0) {
      const int r = I.r;
      const Row R(meta, r);
      if (qk != k) {
        // ---- q prep of the item's row (k_q_prep / P088's chains): Σ x² per 8-dim slice (thread i of a head's half:
        // dims 8i …), warp sums, the two halves; then this lane's fragment pairs × rsqrt and RoPE, bf16
        const bf16* qrow = q + (size_t)r * kLH * kD + (size_t)(16 * wi) * kD;
        {  // the row's RoPE angles (pair i = lane), this warp's table (its loads overlap the q loads below)
          float sn, cs;
          sincosf(__fmul_rn(float(R.p), inv[lane]), &sn, &cs);
          __syncwarp();  // (the previous item's table has been read)
          S.rcs[warp][lane] = cs, S.rsn[warp][lane] = sn;
        }
        float rsa = 0.f, rsb = 0.f;
        {
          uint4 u[16][2];  // every slice of the warp's 16 heads requested before the first use (one L2 round trip)
#pragma unroll
          for (int hl = 0; hl < 16; hl++)
#pragma unroll
            for (int x = 0; x < 2; x++) u[hl][x] = __ldg(reinterpret_cast<const uint4*>(qrow + hl * kD + 256 * x + 8 * lane));
#pragma unroll
          for (int hl = 0; hl < 16; hl++) {
            float ss2[2];
#pragma unroll
            for (int x = 0; x < 2; x++) {
              const uint32_t w4[4] = {u[hl][x].x, u[hl][x].y, u[hl][x].z, u[hl][x].w};
              float v[8];
#pragma unroll
              for (int jq = 0; jq < 4; jq++) v[2 * jq] = bf16_lo(w4[jq]), v[2 * jq + 1] = bf16_hi(w4[jq]);
              float ss = 0.f;
#pragma unroll
              for (int jq = 0; jq < 8; jq++) ss += v[jq] * v[jq];
              ss2[x] = warp_sum(ss);
            }
            const float rs = rsqrtf(__fdiv_rn(ss2[0] + ss2[1], float(kD)) + kEps);
            if (hl == g) rsa = rs;
            if (hl == g + 8) rsb = rs;
          }
        }
        // this lane's fragment pairs, all requested first, then × rsqrt, RoPE, bf16 in place
#pragma unroll
        for (int kk = 0; kk < kD / 16; kk++)
#pragma unroll
          for (int x = 0; x < 4; x++) qf[kk][x] = ldnc32(qrow + (g + 8 * (x & 1)) * kD + 16 * kk + 8 * (x >> 1) + 2 * t);
        __syncwarp();  // the angle table
#pragma unroll
        for (int kk = 0; kk < kD / 16; kk++)
#pragma unroll
          for (int x = 0; x < 4; x++) {
            const float rs = (x & 1) ? rsb : rsa;
            float v0 = bf16_lo(qf[kk][x]), v1 = bf16_hi(qf[kk][x]);
            v0 *= rs, v1 *= rs;
            if (kk >= kNope / 16) {  // pair (d − 448) / 2
              const int a = 8 * (kk - kNope / 16) + 4 * (x >> 1) + t;
              const float e = v0, od = v1, cs = S.rcs[warp][a], sn = S.rsn[warp][a];
              v0 = __fsub_rn(__fmul_rn(e, cs), __fmul_rn(od, sn));
              v1 = __fadd_rn(__fmul_rn(od, cs), __fmul_rn(e, sn));
            }
            qf[kk][x] = pack_bf16x2(v0, v1);
          }
        qk = k;
      }
      const int j = I.j0 + jj, k0 = j * kKT, nk = R.nk, ntile = R.ntile;
      const bool first = j % TPC == 0, lastt = j % TPC == TPC - 1 || j == ntile - 1;
      bar_sync(kBarKr + grp, kNKr);  // K tile n (the ring holds the items of tiles ≤ n + 2 from here)
      const bool more = jj + 1 < I.nt || S.it[(k + 1) % kRing].id >= 0;  // stream tile n + 1 exists
      float sacc[4][4];  // key block nb: S[hA][k0 + 8nb + 2t (+1)], S[hB][…]
#pragma unroll
      for (int nb = 0; nb < 4; nb++) sacc[nb][0] = sacc[nb][1] = sacc[nb][2] = sacc[nb][3] = 0.f;
      {
        uint32_t bf[2][8];  // B fragments of k-step kk: keys 0–15 (nb 0, 1), 16–31 (nb 2, 3)
        const bf16* kt = &S.k[grp][(lane >> 4) * 8 + (lane & 7)][((lane >> 3) & 1) * 8];
        ldsm4r(&bf[0][0], kt), ldsm4r(&bf[0][4], kt + 16 * kKS);
#pragma unroll
        for (int kk = 0; kk < kD / 16; kk++) {
          if (kk + 1 < kD / 16)
            ldsm4r(&bf[(kk + 1) & 1][0], kt + 16 * (kk + 1)), ldsm4r(&bf[(kk + 1) & 1][4], kt + 16 * kKS + 16 * (kk + 1));
#pragma unroll
          for (int nb = 0; nb < 4; nb++) mma(sacc[nb], qf[kk], bf[kk & 1][2 * nb], bf[kk & 1][2 * nb + 1]);
        }
      }
      // softmax of the tile: sacc[nb][i] = head (i < 2 ? hA : hB), key k0 + 8nb + 2t + (i & 1)
#pragma unroll
      for (int nb = 0; nb < 4; nb++)
#pragma unroll
        for (int i = 0; i < 4; i++) sacc[nb][i] = k0 + 8 * nb + 2 * t + (i & 1) < nk ? sacc[nb][i] * scale : -INFINITY;
      float ta = fmaxf(fmaxf(sacc[0][0], sacc[0][1]), fmaxf(sacc[1][0], sacc[1][1]));
      float tb = fmaxf(fmaxf(sacc[0][2], sacc[0][3]), fmaxf(sacc[1][2], sacc[1][3]));
      ta = fmaxf(ta, fmaxf(fmaxf(sacc[2][0], sacc[2][1]), fmaxf(sacc[3][0], sacc[3][1])));
      tb = fmaxf(tb, fmaxf(fmaxf(sacc[2][2], sacc[2][3]), fmaxf(sacc[3][2], sacc[3][3])));
#pragma unroll
      for (int off = 1; off < 4; off <<= 1)
        ta = fmaxf(ta, __shfl_xor_sync(0xffffffffu, ta, off)), tb = fmaxf(tb, __shfl_xor_sync(0xffffffffu, tb, off));
      if (n > 0) bar_sync(kBarM + (grp ^ 1), kNML);  // the running max after stream tile n − 1
      const float moa = first ? -INFINITY : S.m[hA], mob = first ? -INFINITY : S.m[hB];
      const float mxa = fmaxf(moa, ta), mxb = fmaxf(mob, tb);
      __syncwarp();
      if (t == 0) S.m[hA] = mxa, S.m[hB] = mxb;
      __syncwarp();
      if (more) bar_arrive(kBarM + grp, kNML);
#pragma unroll
      for (int nb = 0; nb < 4; nb++) {
        const int key = 8 * nb + 2 * t;
        const float e0 = expf(sacc[nb][0] - mxa), e1 = expf(sacc[nb][1] - mxa);
        const float e2 = expf(sacc[nb][2] - mxb), e3 = expf(sacc[nb][3] - mxb);
        S.e[grp][hA][key] = e0, S.e[grp][hA][key + 1] = e1, S.e[grp][hB][key] = e2, S.e[grp][hB][key + 1] = e3;
        *reinterpret_cast<uint32_t*>(&S.p[grp][hA][key]) = pack_bf16x2(e0, e1);
        *reinterpret_cast<uint32_t*>(&S.p[grp][hB][key]) = pack_bf16x2(e2, e3);
      }
      if (t == 0) S.fac[grp][hA] = expf(moa - mxa), S.fac[grp][hB] = expf(mob - mxb);
      __syncwarp();
      float sum = 0.f;
      if (lane < 16) {  // the tile sum in key order (base)
        const int h = 16 * wi + lane;
#pragma unroll
        for (int x = 0; x < kKT; x++) sum += S.e[grp][h][x];
      }
      float* w = ws + ((size_t)r * nchmax + j / TPC) * CS;  // the chunk's partial (every chunk)
      if (n > 0) bar_sync(kBarL + (grp ^ 1), kNML);  // the running sum after stream tile n − 1
      if (lane < 16) {
        const int h = 16 * wi + lane;
        const float l = (first ? 0.f : S.l[h]) * S.fac[grp][h] + sum;
        S.l[h] = l;
        if (lastt) st1(w + kLH + h, l, keep);
      }
      if (lastt && t == 0) st1(w + hA, mxa, keep), st1(w + hB, mxb, keep);
      __syncwarp();
      if (more) bar_arrive(kBarL + grp, kNML);
      bar_arrive(kBarP + grp, kNP);
      // ---- this group's next stream tile n + 2 (its item is in the ring: see bar Kr above)
      n += 2, jj += 2;
      while (I.id >= 0 && jj >= I.nt) jj -= I.nt, k++, I = S.it[k % kRing];
    }
  } else if (warp < 12) {
    // ================= P·V warps: dims 64·vw …, all 32 heads; partials, the row's merge =================
    reg_dec<kRV>();
    const int vw = warp - 4, vt = tid - 128;
    const int d0 = 64 * vw + 8 * g + 2 * t;  // the merge's two dims of this lane
    const uint64_t keep = pol_last(), once = pol_first();
    int n = 0;
    for (int k = 0;; k++) {
      const Item I = S.it[k % kRing];  // (written before bar Kr of stream tile n − 1 or the start)
      if (I.id < 0) break;
      const int r = I.r, j0 = I.j0, nt = I.nt;
      const Row R(meta, r);
      const int nch = R.nch, ntile = R.ntile, nit = min(ng, nch);
      float* wsl = ws + (size_t)r * nchmax * CS;
      {
        float acc[2][8][4];  // heads 16a + g (+8), dims 64·vw + 8n + 2t (+1)
#pragma unroll
        for (int a = 0; a < 2; a++)
#pragma unroll
          for (int nn = 0; nn < 8; nn++) acc[a][nn][0] = acc[a][nn][1] = acc[a][nn][2] = acc[a][nn][3] = 0.f;
        for (int jj = 0; jj < nt; jj++, n++) {
          const int j = j0 + jj, b = n & 1;
          bar_sync(kBarP + b, kNP);
#pragma unroll
          for (int a = 0; a < 2; a++) {
            const float f0 = S.fac[b][16 * a + g], f1 = S.fac[b][16 * a + g + 8];
#pragma unroll
            for (int nn = 0; nn < 8; nn++)
              acc[a][nn][0] *= f0, acc[a][nn][1] *= f0, acc[a][nn][2] *= f1, acc[a][nn][3] *= f1;
          }
#pragma unroll
          for (int ks = 0; ks < kKT / 16; ks++) {
            uint32_t pa[2][4];
            ldsm4(pa[0], &S.p[b][lane & 15][16 * ks + (lane >> 4) * 8]);
            ldsm4(pa[1], &S.p[b][16 + (lane & 15)][16 * ks + (lane >> 4) * 8]);
#pragma unroll
            for (int bp = 0; bp < 4; bp++) {
              uint32_t vv[4];
              ldsm4t(vv, &S.k[b][16 * ks + (lane & 15)][64 * vw + 16 * bp + (lane >> 4) * 8]);
              if (ks == 1 && bp == 3) bar_arrive(kBarKf + b, kNKf);  // this warp is done with k[b], p[b], fac[b]
              mma(acc[0][2 * bp], pa[0], vv[0], vv[1]);
              mma(acc[1][2 * bp], pa[1], vv[0], vv[1]);
              mma(acc[0][2 * bp + 1], pa[0], vv[2], vv[3]);
              mma(acc[1][2 * bp + 1], pa[1], vv[2], vv[3]);
            }
          }
          if (j % TPC == TPC - 1 || j == ntile - 1) {  // the chunk's O → ws; fresh state
            float* wo = wsl + (size_t)(j / TPC) * CS + 2 * kLH;
#pragma unroll
            for (int a = 0; a < 2; a++)
#pragma unroll
              for (int nn = 0; nn < 8; nn++) {
                const int hr = 16 * a + g, d = 64 * vw + 8 * nn + 2 * t;
                st2(wo + hr * kD + d, acc[a][nn][0], acc[a][nn][1], keep);
                st2(wo + (hr + 8) * kD + d, acc[a][nn][2], acc[a][nn][3], keep);
                acc[a][nn][0] = acc[a][nn][1] = acc[a][nn][2] = acc[a][nn][3] = 0.f;
              }
          }
        }
      }
      __threadfence();  // this thread's partial stores, GPU-wide, before the row's count
      bar_sync(kBarV, kNV);
      if (vt == 0) {  // the score warps' m / l stores came before bar P (cumulative fence)
        int last = 1;
        if (nit > 1) {
          __threadfence();
          last = atomicAdd(&g_cnt[r], 1) + 1 == nit;
          if (last) g_cnt[r] = 0;
        }
        S.lastf = last;
      }
      bar_sync(kBarV, kNV);
      if (!S.lastf) continue;
      __threadfence();
      // ---- the row's merge, every chunk from the workspace (k_merge / P088's order and expressions)
      for (int i = vt; i < nch * kLH; i += kNV) {  // every chunk's m / l of the 32 heads, loads in flight together
        const int cc = i / kLH, h = i % kLH;
        S.mf[cc][h] = __ldcg(wsl + cc * CS + h), S.lf[cc][h] = __ldcg(wsl + cc * CS + kLH + h);
      }
      bar_sync(kBarV, kNV);
      if (vt < kLH) {  // P088's coefficients, the same chains (M by fmaxf in chunk order, den in chunk order + the sink)
        const int h = vt;
        float Mx = sink[h];
        for (int cc = 0; cc < nch; cc++) Mx = fmaxf(Mx, S.mf[cc][h]);
        float den = 0.f;
        for (int cc = 0; cc < nch; cc++) den += expf(S.mf[cc][h] - Mx) * S.lf[cc][h];
        den += expf(sink[h] - Mx);
        S.den2[h] = den;
        for (int cc = 0; cc < nch; cc++) S.mf[cc][h] = expf(S.mf[cc][h] - Mx);  // P088's per-chunk factor
      }
      bar_sync(kBarV, kNV);
      // o of this lane's two dims d0, d0 + 1 for all 32 heads (P088's post-transpose layout): per head and dim the
      // chunk chain x = Σ_c expf(m_c − M) · o_c from 0 in chunk order, then x / den — four passes of 8 heads, every
      // warp load 256 contiguous bytes, two chunks' loads in flight. W[n][i] = bf16 pair of head n + 8i.
      uint32_t W[8][4];
#pragma unroll
      for (int hq = 0; hq < 4; hq++) {
        float x[8][2];
#pragma unroll
        for (int i = 0; i < 8; i++) x[i][0] = x[i][1] = 0.f;
#pragma unroll 2
        for (int cc = 0; cc < nch; cc++) {
          const float* w = wsl + (size_t)cc * CS + 2 * kLH + (8 * hq) * kD + d0;
          float2 v[8];
#pragma unroll
          for (int i = 0; i < 8; i++) v[i] = __ldcg(reinterpret_cast<const float2*>(w + i * kD));
#pragma unroll
          for (int i = 0; i < 8; i++) {
            const float f = S.mf[cc][8 * hq + i];
            x[i][0] += f * v[i].x, x[i][1] += f * v[i].y;
          }
        }
#pragma unroll
        for (int i = 0; i < 8; i++) {
          const float dh = S.den2[8 * hq + i];
          W[i][hq] = pack_bf16x2(__fdiv_rn(x[i][0], dh), __fdiv_rn(x[i][1], dh));
        }
      }
      // piece pc: o of heads 8pc … 8pc+7, and zp ranks 4pc … 4pc+3 — the VHA chains of k_vha_partial (per rank and dim
      // Σ_h U[h][rank] · o[h][dim], h ascending, from 0)
#pragma unroll
      for (int pc = 0; pc < 4; pc++) {
        bf16* orow = o + (size_t)r * kLH * kD + d0;
#pragma unroll
        for (int nn = 0; nn < 8; nn++) stu(orow + (8 * pc + nn) * kD, W[nn][pc], once);
        float z[4][2];
#pragma unroll
        for (int kq = 0; kq < 4; kq++) z[kq][0] = z[kq][1] = 0.f;
#pragma unroll
        for (int h = 0; h < kLH; h++) {
          const uint32_t wv = W[h & 7][h >> 3];
          const float x0 = bf16_lo(wv), x1 = bf16_hi(wv);
          const float4 u = *reinterpret_cast<const float4*>(&S.U[h * kVhaRank + 4 * pc]);
          z[0][0] += u.x * x0, z[0][1] += u.x * x1;
          z[1][0] += u.y * x0, z[1][1] += u.y * x1;
          z[2][0] += u.z * x0, z[2][1] += u.z * x1;
          z[3][0] += u.w * x0, z[3][1] += u.w * x1;
        }
        float* zrow = zp + ((size_t)r * kVhaRank + 4 * pc) * kD + d0;
#pragma unroll
        for (int kq = 0; kq < 4; kq++) st2(zrow + kq * kD, z[kq][0], z[kq][1], once);
      }
      bar_sync(kBarV, kNV);  // every P·V warp has read the partials
      for (int cc = 0; cc < nch; cc++) {  // the row's slabs: 514 lines of 128 B each
        const char* wb = reinterpret_cast<const char*>(wsl + (size_t)cc * CS);
        for (int i = vt; i < CS * 4 / 128; i += kNV) discard_l2(wb + (size_t)i * 128);
      }
    }
  } else {
    // ================= producers: tickets, the K tile gather and dequantization =================
    reg_dec<kRG>();
    const int pt = tid - 384, kq = pt >> 2, pp = pt & 3;  // key kq of a tile, its pieces pp
    const uint64_t keep = pol_last();
    // load cursor: the next tile to load is tile lj of item lk (LI; its row's key layout cached); entering item lk:
    // item lk + 1's ticket. Raw rows of tile n + 1 in flight while tile n waits dequantized for its buffer.
    int lk = 0, lj = 0, n = 0;
    Item LI = S.it[0];
    RowMeta lm = meta[LI.id >= 0 ? LI.r : 0];
    int lp = lm.pos, lnc = (lp + 1) / kRatio, lnw = min(lp + 1, kWindow), lnk = lnc + lnw;
    auto load = [&](Raw& X) {
      if (LI.id < 0) {
        X.ok = false;
        return false;
      }
      const int key = (LI.j0 + lj) * kKT + kq;
      X.load(key < lnk ? p041::key_row(key, lnc, lnw, lp, lm, rows, c) : nullptr, pp, keep);
      if (++lj >= LI.nt) {
        lj = 0, lk++;
        LI = S.it[lk % kRing];
        if (LI.id >= 0) {
          if (pt == 0) S.it[(lk + 1) % kRing] = fetch(meta, nitems, ng);
          bar_sync(kBarG, kNG);
          lm = meta[LI.r];
          lp = lm.pos, lnc = (lp + 1) / kRatio, lnw = min(lp + 1, kWindow), lnk = lnc + lnw;
        }
      }
      return true;
    };
    Raw X;
    uint4 dv[14];
    uint2 dr[4];
    auto deq = [&]() {  // P150's tile row of X: dq8 nope, rope as stored, zero past the row's last key
#pragma unroll
      for (int i = 0; i < 14; i++) {
        const int grp = (pp + 4 * i) >> 3;
        dv[i] = X.ok ? dq8(X.nw[i], ((grp < 4 ? X.sc.x : X.sc.y) >> (8 * (grp & 3))) & 0xFF) : make_uint4(0, 0, 0, 0);
      }
#pragma unroll
      for (int i = 0; i < 4; i++) dr[i] = X.ok ? X.rw[i] : make_uint2(0, 0);
    };
    bool v = load(X);
    if (v) deq();
    bool vn = load(X);
    while (v) {
      const int b = n & 1;
      if (n >= 2) bar_sync(kBarKf + b, kNKf);  // the P·V warps are done with tile n − 2
      bf16* dst = &S.k[b][kq][0];
#pragma unroll
      for (int i = 0; i < 14; i++) *reinterpret_cast<uint4*>(dst + 8 * (pp + 4 * i)) = dv[i];
#pragma unroll
      for (int i = 0; i < 4; i++) *reinterpret_cast<uint2*>(dst + kNope + 4 * (pp + 4 * i)) = dr[i];
      bar_arrive(kBarKr + b, kNKr);
      n++;
      v = vn;
      if (!v) break;
      deq();          // tile n (its rows requested a tile ago)
      vn = load(X);   // tile n + 1
    }
    // n tiles: the K-free arrivals of the last two (never waited for by a fill) close their barrier phases
    for (int m = max(0, n - 2); m < n; m++) bar_sync(kBarKf + (m & 1), kNKf);
    if (pt == 0) {  // the last CTA resets the ticket counters (every CTA's tickets were taken before its count)
      __threadfence();
      if (atomicAdd(&g_done, 1) == (int)gridDim.x - 1) {
        g_next = 0, g_done = 0;
        __threadfence();
      }
    }
  }
}

// rows of kMinCh … kMaxCh chunks (host: max_keys); G: most chunks per item (0: 2 from 24 chunks, else 3)
inline void launch(const bf16* q, const uint8_t* rows, const RowMeta* meta, HcaCache c, const float* sink,
                   const float* inv, const float* U, int M, int max_keys, bf16* o, float* zp, float* ws, size_t ws_bytes,
                   int G, cudaStream_t s) {
  static int nsm = 0;
  if (!nsm) {
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
    CUDA_CHECK(cudaFuncSetAttribute(k_attn, cudaFuncAttributeMaxDynamicSharedMemorySize, int(sizeof(Smem))));
  }
  const int nch = std::max(1, (max_keys + kCh - 1) / kCh);
  if (G <= 0) G = nch >= 24 ? 2 : 3;
  const int ng = (nch + G - 1) / G;
  const size_t per_row = size_t(nch) * CS * 4;
  const int most = int(std::min<size_t>({size_t(M), size_t(kMaxRows), ws_bytes / per_row}));
  if (most < 1) qlog::fail("hca", "attention workspace %zu B < one row's %zu B", ws_bytes, per_row);
  const int npass = (M + most - 1) / most, pass = (M + npass - 1) / npass;  // equal passes
  for (int r0 = 0; r0 < M; r0 += pass) {
    const int n = std::min(pass, M - r0);
    const int grid = int(std::min<long>(nsm, long(n) * ng));
    k_attn<<<grid, kTh, sizeof(Smem), s>>>(q + (size_t)r0 * kLH * kD, rows, meta + r0, c, sink, inv, U, n, ng, nch,
                                           o + (size_t)r0 * kLH * kD, zp + (size_t)r0 * kVhaRank * kD, ws);
    CUDA_CHECK(cudaGetLastError());
  }
}

}  // namespace p180
}  // namespace e55
