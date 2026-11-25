// p172_hca_attn_long2.cuh — P172 hca_attn_long2: P150's long-context hca_attn_fused with its tile loop's gather
// rescheduled. Included by hca.cu after p150_hca_attn_long.cuh (uses P088's Row / MMA / ldmatrix / barrier wrappers and
// P150's dq8 / cache-hint helpers).
//
// Why (measured, one GPU, 2048 rows × 32 heads at 524,288; clock64 traces of CTA 0 and timing-only knock-out variants):
// a 32-key tile took ~5,400 cycles against ~2,100 of tensor work; without the K dequantization the score warps' chains
// ran 4.6–6.6k cycles instead of 8.5–11.5k — the P·V warps dequantized the next tile in one burst between their MMA
// phases, while the other score group's chain was running. At 5 chunks (131,072) P088's chunk slots go through DRAM
// (4.5–5.5 ms run to run; the engine's step profile: 5.75 ms a call), P150 / P172 keep them in L2 (4.5–4.6 ms).
// Result: 524,288 16.22 → 15.73 ms. Left: the time scales almost linearly with the CTAs (4 lanes at 524,288 on 16 / 24
// / 32 / 48 SMs: 184.9 / 122.8 / 93.7 / 64.7 ms; ~10 GB of L2 traffic a call is ~0.64 TB/s against the 1.5–1.7 TB/s
// an L2 gather test reaches) — a per-SM limit: the tile loop's pipeline under the 128-register cap of 512 threads.
// Measured slower here (spills): both first tiles requested before the handshake (17.66 ms), the q prep staged
// through S.e (17.16), slice-owned gathers without the P·V barrier (24.1).
//
// Bitwise contract — P150's (= P088's = the base): the same keys, tiles, chunks, score chains, online softmax, P·V MMAs,
// chunk partials, chunk merge, q prep and VHA chains. Changed (nothing that touches a value):
//   · the P·V warps dequantize the next-but-one tile word by word between their k-step MMAs (one 8-code word after
//     each ldmatrix of V, held in registers) and only store it after every P·V warp has read the buffer;
//   · a gather piece reads 8-byte words vp + 8i of the key (the 8 pieces of a key: 64 contiguous bytes a load);
//   · an item's RoPE angles computed before waiting for the P·V warps;
//   · dispatch from 5 chunks (P150: 6), rows in equal passes (P150: full passes and a short last one);
//   · its own ticket / row counters.
#pragma once

namespace e55 {
namespace p172 {

using p088::bar_arrive;
using p088::bar_sync;
using p088::CS;
using p088::kCh;
using p088::kD;
using p088::kKS;
using p088::kKT;
using p088::kLH;
using p088::ldsm2;
using p088::ldsm4;
using p088::ldsm4t;
using p088::mma;
using p088::Row;
using p088::TPC;

constexpr int kSW = 8, kVW = 8, kST = 32 * kSW, kVT = 32 * kVW, kTh = kST + kVT;  // 512: score groups 0 / 1, P·V
// named barriers: P ready [2], K tile ready [2], P·V warps' own, running max / sum to the other group [2 + 2], item
// handshake, score warps' own
enum { kBarP = 1, kBarK = 3, kBarV = 5, kBarM = 6, kBarL = 8, kBarA = 10, kBarS = 11 };
constexpr int kBarPK = 32 * (kVW + 4);  // one score group + the P·V warps
constexpr int kMaxRows = 16384;         // rows per pass (per-row counters)
constexpr int kMinCh = 5;               // dispatch: max_keys of ≥ 5 chunks (fewer: P150 / P088)
constexpr int kMaxCh = 48;              // … and ≤ 48 chunks (12,288 keys; more: P088)

__device__ int g_cnt[kMaxRows];  // per row of the pass: finished items (zero between launches)
__device__ int g_next, g_done;   // ticket counter, CTAs done (zero between launches)

struct Smem {
  bf16 k[2][kKT][kKS];  // K tiles (tile j in k[j & 1]); k[1] also stages the prepped q at an item's start
  bf16 p[2][kLH][kKT + 8];
  float e[2][kLH][kKT + 1];
  float fac[2][kLH];
  float m[kLH], l[kLH];      // running max / sum of the current chunk
  float den2[kLH];          // merge: the denominator per head
  float mf[kMaxCh][kLH];    // merge: m_c per chunk and head, then expf(m_c − M)
  float lf[kMaxCh][kLH];    // merge: l_c
  float red[64];
  float rcs[32], rsn[32];
  __align__(16) float U[kLH * kVhaRank];
  int ids[2];  // the item of ordinal k for the P·V warps (−1: no more)
  int tick;    // the score warps' ticket broadcast
  int lastf;   // P·V warps: this CTA merges the row
};

using p150::discard_l2;
using p150::dq8;
using p150::ld2;
using p150::pol_first;
using p150::pol_last;
using p150::st1;
using p150::st2;
using p150::stu;

// one K tile's gather by 256 threads: thread (key i / 8, piece vp = i % 8) moves nope words vp + 8i (i < 7: 8 codes of
// scale group i) and rope words 56 + vp, 64 + vp of the 584-byte row (the key's 8 pieces read 64 contiguous bytes a
// load), and the scale word; P150's tile: dequantized nope (dq8), rope as stored, zero rows past the row's last key.
// conv(i) dequantizes word i into registers, put() stores the tile (P·V warps: words converted between their MMAs);
// store() does both (an item's first two tiles).
struct Gather {
  static constexpr int NW = 7, NR = 2;
  uint2 nw[NW], rw[NR], rsc;
  uint4 dq[NW];
  __device__ __forceinline__ void load(const uint8_t* row, int vp, uint64_t pol) {
    if (row) {
      const uint2* src = reinterpret_cast<const uint2*>(row);
#pragma unroll
      for (int i = 0; i < NW; i++) nw[i] = ld2(src + vp + 8 * i, pol);
#pragma unroll
      for (int i = 0; i < NR; i++) rw[i] = ld2(src + kNope / 8 + vp + 8 * i, pol);
      rsc = ld2(src + 72, pol);
    }
  }
  __device__ __forceinline__ void conv(int i) { dq[i] = dq8(nw[i], ((i < 4 ? rsc.x : rsc.y) >> (8 * (i & 3))) & 0xFF); }
  __device__ __forceinline__ void put(bf16* dst, bool ok, int vp) const {
#pragma unroll
    for (int i = 0; i < NW; i++) *reinterpret_cast<uint4*>(dst + 8 * (vp + 8 * i)) = ok ? dq[i] : make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int i = 0; i < NR; i++)
      *reinterpret_cast<uint2*>(dst + kNope + 4 * (vp + 8 * i)) = ok ? rw[i] : make_uint2(0, 0);
  }
  __device__ __forceinline__ void store(bf16* dst, bool ok, int vp) {
#pragma unroll
    for (int i = 0; i < NW; i++) conv(i);
    put(dst, ok, vp);
  }
};

__device__ __forceinline__ int row_nch(const RowMeta* meta, int r) {
  const int p = meta[r].pos;
  return ((p + 1) / kRatio + min(p + 1, kWindow) + kCh - 1) / kCh;
}

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
  if (tid == 0) S.tick = atomicAdd(&g_next, 1);
  __syncthreads();

  if (warp < kSW) {
    // ================= score warps: group grp takes the tiles jj ≡ grp (mod 2) of an item =================
    const int st = tid, grp = warp >> 2, hw = 8 * (warp & 3), h0 = hw + 2 * t;
    const int gk = st >> 3, gp = st & 7;  // gather job at an item's start: key gk of the tile, piece gp
    const float scale = rsqrtf(float(kD));
    const uint64_t keep = pol_last();
    int id = S.tick, nxt = 0, k = 0;
    while (true) {
      // ---- the next non-empty item (uniform over the score warps)
      int r = 0, nch = 0, c0 = 0, c1 = 0;
      while (id < nitems) {
        r = id / ng;
        const int gi = id - r * ng;
        nch = row_nch(meta, r);
        c0 = gi * nch / ng, c1 = (gi + 1) * nch / ng;
        if (c0 < c1) break;
        bar_sync(kBarS, kST);  // every score thread has read S.tick
        if (st == 0) S.tick = atomicAdd(&g_next, 1);
        bar_sync(kBarS, kST);
        id = S.tick;
      }
      if (id >= nitems) break;
      if (st == 0) nxt = atomicAdd(&g_next, 1);  // the next ticket, used at this item's end
      const Row R(meta, r);
      const int j0 = c0 * TPC, j1 = min(R.ntile, c1 * TPC), nt = j1 - j0;
      float* wsl = ws + (size_t)r * nchmax * CS;
      // ---- q (raw) and tile j0 requested before the handshake
      uint4 qr[8];  // pass i: head 4i + warp / 2, dims 256·(warp % 2) + 8·lane … (P088's warp slices)
      const bf16* qrow = q + (size_t)r * kLH * kD;
      const int qd = 256 * (warp & 1) + 8 * lane;
#pragma unroll
      for (int i = 0; i < 8; i++) qr[i] = *reinterpret_cast<const uint4*>(qrow + (4 * i + (warp >> 1)) * kD + qd);
      Gather G;
      G.load(R.key(j0 * kKT + gk, rows, c), gp, keep);
      if (st < 32) {  // the RoPE angles (only the score warps use them)
        float sn, cs;
        sincosf(__fmul_rn(float(R.p), inv[st]), &sn, &cs);
        S.rcs[st] = cs, S.rsn[st] = sn;
      }
      if (k) bar_sync(kBarA, kTh);  // the P·V warps are done with the previous item's buffers
      if (st == 0) S.ids[k & 1] = id;
      // ---- q prep (k_q_prep / P070 / P088: Σ x² per thread over its 8 dims, warp sums, the head's two halves)
#pragma unroll
      for (int i = 0; i < 8; i++) {
        float v[8];
        const uint32_t w4[4] = {qr[i].x, qr[i].y, qr[i].z, qr[i].w};
#pragma unroll
        for (int jq = 0; jq < 4; jq++) v[2 * jq] = bf16_lo(w4[jq]), v[2 * jq + 1] = bf16_hi(w4[jq]);
        float ss = 0.f;
#pragma unroll
        for (int jq = 0; jq < 8; jq++) ss += v[jq] * v[jq];
        ss = warp_sum(ss);
        if (lane == 0) S.red[2 * (4 * i + (warp >> 1)) + (warp & 1)] = ss;
        *reinterpret_cast<uint4*>(&S.k[1][4 * i + (warp >> 1)][qd]) = qr[i];  // raw, prepped in place below
      }
      G.store(&S.k[0][gk][0], j0 * kKT + gk < R.nk, gp);  // tile j0 → k[0]
      if (nt > 1) G.load(R.key((j0 + 1) * kKT + gk, rows, c), gp, keep);
      bar_sync(kBarS, kST);
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const int hh = 4 * i + (warp >> 1);
        float v[8];
        p041::ld_bf16x8(&S.k[1][hh][qd], v);  // this thread's own raw q
        const float rs = rsqrtf(__fdiv_rn(S.red[2 * hh] + S.red[2 * hh + 1], float(kD)) + kEps);
#pragma unroll
        for (int jq = 0; jq < 8; jq++) v[jq] *= rs;
        if (qd >= kNope) p041::rope8t(v, (qd - kNope) / 2, S.rcs, S.rsn);
        p041::st_bf16x8(&S.k[1][hh][qd], v);
      }
      bar_sync(kBarS, kST);
      uint32_t qb[kD / 16][2];
#pragma unroll
      for (int kk = 0; kk < kD / 16; kk++)
        ldsm2(qb[kk][0], qb[kk][1], &S.k[1][hw + (lane & 7)][16 * kk + ((lane >> 3) & 1) * 8]);
      bar_sync(kBarS, kST);  // k[1] free
      if (nt > 1) G.store(&S.k[1][gk][0], (j0 + 1) * kKT + gk < R.nk, gp);  // tile j0+1 → k[1]
      bar_sync(kBarS, kST);
      // ---- the tiles (P071's score groups on P088's arithmetic)
      const int nk = R.nk, ntile = R.ntile;
      for (int jj = grp; jj < nt; jj += 2) {
        const int j = j0 + jj, b = jj & 1, k0 = j * kKT;
        const bool first = j % TPC == 0, lastt = j % TPC == TPC - 1 || j == ntile - 1, more = jj + 1 < nt;
        if (jj > 1) bar_sync(kBarK + b, kBarPK);
        if (st == 0 && jj == (nt > 2 ? 2 : 0) && nxt < nitems)  // the next item's q → L2 (one bulk prefetch)
          asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(q + (size_t)(nxt / ng) * kLH * kD),
                       "r"(kLH * kD * 2)
                       : "memory");
        float sacc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        {
          uint32_t af[2][2][4];
          const bf16* kb0 = &S.k[b][lane & 15][(lane >> 4) * 8];
          const bf16* kb1 = &S.k[b][16 + (lane & 15)][(lane >> 4) * 8];
          ldsm4(af[0][0], kb0), ldsm4(af[0][1], kb1);
#pragma unroll
          for (int kk = 0; kk < kD / 16; kk++) {
            if (kk + 1 < kD / 16)
              ldsm4(af[(kk + 1) & 1][0], kb0 + 16 * (kk + 1)), ldsm4(af[(kk + 1) & 1][1], kb1 + 16 * (kk + 1));
            mma(sacc[0], af[kk & 1][0], qb[kk][0], qb[kk][1]);
            mma(sacc[1], af[kk & 1][1], qb[kk][0], qb[kk][1]);
          }
        }
        // softmax of the tile: sacc[m][i] = key 16m + g + 8(i >> 1), head h0 + (i & 1)
        float sv[2][4];
#pragma unroll
        for (int m = 0; m < 2; m++)
#pragma unroll
          for (int i = 0; i < 4; i++) sv[m][i] = k0 + 16 * m + g + (i >> 1) * 8 < nk ? sacc[m][i] * scale : -INFINITY;
        float ta = fmaxf(fmaxf(sv[0][0], sv[0][2]), fmaxf(sv[1][0], sv[1][2]));
        float tb = fmaxf(fmaxf(sv[0][1], sv[0][3]), fmaxf(sv[1][1], sv[1][3]));
#pragma unroll
        for (int off = 4; off < 32; off <<= 1)
          ta = fmaxf(ta, __shfl_xor_sync(0xffffffffu, ta, off)), tb = fmaxf(tb, __shfl_xor_sync(0xffffffffu, tb, off));
        if (jj > 0) bar_sync(kBarM + (b ^ 1), kST);  // the running max after tile jj − 1
        const float moa = first ? -INFINITY : S.m[h0], mob = first ? -INFINITY : S.m[h0 + 1];
        const float mxa = fmaxf(moa, ta), mxb = fmaxf(mob, tb);
        __syncwarp();
        if (g == 0) S.m[h0] = mxa, S.m[h0 + 1] = mxb;
        __syncwarp();
        if (more) bar_arrive(kBarM + b, kST);
#pragma unroll
        for (int m = 0; m < 2; m++)
#pragma unroll
          for (int i = 0; i < 4; i++) {
            const int key = 16 * m + g + (i >> 1) * 8, h = h0 + (i & 1);
            const float e = expf(sv[m][i] - ((i & 1) ? mxb : mxa));
            S.e[b][h][key] = e;
            S.p[b][h][key] = __float2bfloat16_rn(e);
          }
        if (g == 0) S.fac[b][h0] = expf(moa - mxa), S.fac[b][h0 + 1] = expf(mob - mxb);
        __syncwarp();
        float sum = 0.f;
        if (lane < 8) {  // the tile sum in key order (base)
          const int h = hw + lane;
#pragma unroll
          for (int x = 0; x < kKT; x++) sum += S.e[b][h][x];
        }
        float* w = wsl + (size_t)(j / TPC) * CS;  // the chunk's partial (every chunk)
        if (jj > 0) bar_sync(kBarL + (b ^ 1), kST);  // the running sum after tile jj − 1
        if (lane < 8) {
          const int h = hw + lane;
          const float l = (first ? 0.f : S.l[h]) * S.fac[b][h] + sum;
          S.l[h] = l;
          if (lastt) st1(w + kLH + h, l, keep);
        }
        if (lastt && g == 0) st2(w + h0, mxa, mxb, keep);
        __syncwarp();
        if (more) bar_arrive(kBarL + b, kST);
        bar_arrive(kBarP + b, kBarPK);
      }
      // ---- the next ticket (issued at this item's start)
      if (st == 0) S.tick = nxt;
      bar_sync(kBarS, kST);
      id = S.tick;
      k++;
    }
    if (k) bar_sync(kBarA, kTh);
    if (st == 0) S.ids[k & 1] = -1;  // the P·V warps' end
    if (grp == 0) bar_arrive(kBarP, kBarPK);
    if (st == 0) {  // the last CTA resets the ticket counters (every CTA's tickets were taken before its count)
      __threadfence();
      if (atomicAdd(&g_done, 1) == (int)gridDim.x - 1) {
        g_next = 0, g_done = 0;
        __threadfence();
      }
    }
  } else {
    // ================= P·V warps: dims 64·vw …, all 32 heads; the K tile gather =================
    const int vw = warp - kSW, vt = tid - kST;
    const int gk = vt >> 3, gp = vt & 7;     // gather job: key gk of the tile, piece gp
    const int d0 = 64 * vw + 8 * g + 2 * t;  // the merge's two dims of this lane
    const uint64_t keep = pol_last(), once = pol_first();
    for (int k = 0;; k++) {
      bar_sync(kBarP, kBarPK);  // item k's first tile (or the end)
      const int id = S.ids[k & 1];
      if (id < 0) break;
      const int r = id / ng, gi = id - r * ng;
      const Row R(meta, r);
      const int nk = R.nk, nch = R.nch, ntile = R.ntile;
      const int c0 = gi * nch / ng, c1 = (gi + 1) * nch / ng, nit = min(ng, nch);
      const int j0 = c0 * TPC, j1 = min(ntile, c1 * TPC), nt = j1 - j0;
      float* wsl = ws + (size_t)r * nchmax * CS;
      Gather G;
      if (nt > 2) G.load(R.key((j0 + 2) * kKT + gk, rows, c), gp, keep);
      {
        float acc[2][8][4];  // heads 16a + g (+8), dims 64·vw + 8n + 2t (+1)
#pragma unroll
        for (int a = 0; a < 2; a++)
#pragma unroll
          for (int n = 0; n < 8; n++) acc[a][n][0] = acc[a][n][1] = acc[a][n][2] = acc[a][n][3] = 0.f;
        for (int jj = 0; jj < nt; jj++) {
          const int j = j0 + jj, b = jj & 1;
          if (jj > 0) bar_sync(kBarP + b, kBarPK);
#pragma unroll
          for (int a = 0; a < 2; a++) {
            const float f0 = S.fac[b][16 * a + g], f1 = S.fac[b][16 * a + g + 8];
#pragma unroll
            for (int n = 0; n < 8; n++) acc[a][n][0] *= f0, acc[a][n][1] *= f0, acc[a][n][2] *= f1, acc[a][n][3] *= f1;
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
              if (4 * ks + bp < Gather::NW && jj + 2 < nt) G.conv(4 * ks + bp);  // tile j+2's word, in registers
              if (ks == 1 && bp == 3 && jj + 2 < nt) bar_sync(kBarV, kVT);  // every P·V warp has read k[b]
              mma(acc[0][2 * bp], pa[0], vv[0], vv[1]);
              mma(acc[1][2 * bp], pa[1], vv[0], vv[1]);
              mma(acc[0][2 * bp + 1], pa[0], vv[2], vv[3]);
              mma(acc[1][2 * bp + 1], pa[1], vv[2], vv[3]);
            }
          }
          if (jj + 2 < nt) {  // tile j+2 → k[b] (dequantized between the MMAs above)
            G.put(&S.k[b][gk][0], (j + 2) * kKT + gk < nk, gp);
            bar_arrive(kBarK + b, kBarPK);
            if (jj + 3 < nt) G.load(R.key((j + 3) * kKT + gk, rows, c), gp, keep);
          }
          if (j % TPC == TPC - 1 || j == ntile - 1) {  // the chunk's O → ws; fresh state
            float* wo = wsl + (size_t)(j / TPC) * CS + 2 * kLH;
#pragma unroll
            for (int a = 0; a < 2; a++)
#pragma unroll
              for (int n = 0; n < 8; n++) {
                const int hr = 16 * a + g, d = 64 * vw + 8 * n + 2 * t;
                st2(wo + hr * kD + d, acc[a][n][0], acc[a][n][1], keep);
                st2(wo + (hr + 8) * kD + d, acc[a][n][2], acc[a][n][3], keep);
                acc[a][n][0] = acc[a][n][1] = acc[a][n][2] = acc[a][n][3] = 0.f;
              }
          }
        }
      }
      __threadfence();         // this thread's partial stores, GPU-wide, before the row's count
      bar_arrive(kBarA, kTh);  // done with the K / P buffers
      bar_sync(kBarV, kVT);
      if (vt == 0) {  // the score warps' m / l stores came before bar P (cumulative fence)
        int last = 1;
        if (nit > 1) {
          __threadfence();
          last = atomicAdd(&g_cnt[r], 1) + 1 == nit;
          if (last) g_cnt[r] = 0;
        }
        S.lastf = last;
      }
      bar_sync(kBarV, kVT);
      if (!S.lastf) continue;
      __threadfence();
      // ---- the row's merge, every chunk from the workspace (k_merge / P088's order and expressions)
      for (int i = vt; i < nch * kLH; i += kVT) {  // every chunk's m / l of the 32 heads, loads in flight together
        const int cc = i / kLH, h = i % kLH;
        S.mf[cc][h] = __ldcg(wsl + cc * CS + h), S.lf[cc][h] = __ldcg(wsl + cc * CS + kLH + h);
      }
      bar_sync(kBarV, kVT);
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
      bar_sync(kBarV, kVT);
      // o of this lane's two dims d0, d0 + 1 for all 32 heads (the layout P088 reaches after its transpose): per head
      // and dim the chunk chain x = Σ_c expf(m_c − M) · o_c from 0 in chunk order, then x / den — two passes of 16
      // heads, every warp load 256 contiguous bytes, two chunks' loads in flight. W[n][i] = bf16 pair of head n + 8i.
      uint32_t W[8][4];
#pragma unroll
      for (int hp = 0; hp < 2; hp++) {
        float x[16][2];
#pragma unroll
        for (int i = 0; i < 16; i++) x[i][0] = x[i][1] = 0.f;
#pragma unroll 2
        for (int cc = 0; cc < nch; cc++) {
          const float* w = wsl + (size_t)cc * CS + 2 * kLH + (16 * hp) * kD + d0;
          float2 v[16];
#pragma unroll
          for (int i = 0; i < 16; i++) v[i] = __ldcg(reinterpret_cast<const float2*>(w + i * kD));
#pragma unroll
          for (int i = 0; i < 16; i++) {
            const float f = S.mf[cc][16 * hp + i];
            x[i][0] += f * v[i].x, x[i][1] += f * v[i].y;
          }
        }
#pragma unroll
        for (int i = 0; i < 16; i++) {
          const int h = 16 * hp + i;
          const float dh = S.den2[h];
          W[h & 7][h >> 3] = pack_bf16x2(__fdiv_rn(x[i][0], dh), __fdiv_rn(x[i][1], dh));
        }
      }
      // piece pc: o of heads 8pc … 8pc+7, and zp ranks 4pc … 4pc+3 — the VHA chains of k_vha_partial (per rank and dim
      // Σ_h U[h][rank] · o[h][dim], h ascending, from 0)
#pragma unroll
      for (int pc = 0; pc < 4; pc++) {
        bf16* orow = o + (size_t)r * kLH * kD + d0;
#pragma unroll
        for (int n = 0; n < 8; n++) stu(orow + (8 * pc + n) * kD, W[n][pc], once);
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
      bar_sync(kBarV, kVT);  // every P·V warp has read the partials
      for (int cc = 0; cc < nch; cc++) {  // the row's slabs: 514 lines of 128 B each
        const char* wb = reinterpret_cast<const char*>(wsl + (size_t)cc * CS);
        for (int i = vt; i < CS * 4 / 128; i += kVT) discard_l2(wb + (size_t)i * 128);
      }
    }
  }
}

// rows of kMinCh … kMaxCh chunks (host: max_keys); G: most chunks per item (0: 2 from 24 chunks, else 3); cap > 0: at
// most cap CTAs (tests: the overlap gate's SM split)
inline void launch(const bf16* q, const uint8_t* rows, const RowMeta* meta, HcaCache c, const float* sink,
                   const float* inv, const float* U, int M, int max_keys, bf16* o, float* zp, float* ws, size_t ws_bytes,
                   int G, int cap, cudaStream_t s) {
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
  const int ctas = cap > 0 ? std::min(cap, nsm) : nsm;
  for (int r0 = 0; r0 < M; r0 += pass) {
    const int n = std::min(pass, M - r0);
    const int grid = int(std::min<long>(ctas, long(n) * ng));
    k_attn<<<grid, kTh, sizeof(Smem), s>>>(q + (size_t)r0 * kLH * kD, rows, meta + r0, c, sink, inv, U, n, ng, nch,
                                           o + (size_t)r0 * kLH * kD, zp + (size_t)r0 * kVhaRank * kD, ws);
    CUDA_CHECK(cudaGetLastError());
  }
}

}  // namespace p172
}  // namespace e55
