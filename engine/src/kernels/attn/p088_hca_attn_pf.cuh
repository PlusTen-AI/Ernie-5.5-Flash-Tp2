// p088_hca_attn_pf.cuh — P088 hca_attn_pf: hca_attn_fused at prefill sizes (M ≥ 64). Included by hca.cu after
// p041_hca_attn.cuh (uses its key_row / rope / bf16 helpers).
//
// Bitwise contract — P070's (= P041's = the base k_q_prep + k_attn + k_merge + k_vha_partial): per row and 256-key
// chunk the same 32-key tiles, the same scores (Sᵀ = K·Qᵀ, the m16n8k16 chain over the 32 k-steps in order, × 512^-0.5),
// the same online softmax per tile, the same two k16 P·V MMAs after o·f, the same chunk merge with the sink (earlier
// chunks through the per-CTA workspace slot, in chunk order), the same q prep (64 threads a head, Σ x² per thread, warp
// sums, two halves) and the same VHA chain (per rank and dim Σ_h U[h][rank] · bf16 o[h][dim], h = 0 … 31 in order,
// fused multiply-adds from 0). The tile loop is P070's (4 score warps, 8 P·V warps, named barriers). Changed:
//   · row boundary: the score warps run ahead into the next row while the P·V warps finish the last one — after the
//     last tile they only compute the merge coefficients, then load the next row's q (bulk-prefetched into L2 during
//     the previous row), stage it raw in shared memory and prep it in place (P070 held it in registers and spilled),
//     gather tiles 0 and 1 themselves and start scoring;
//   · epilogue in the P·V warps' registers: merge, an 8 × 8 shuffle transpose (32 heads × 16 dims a lane → 32 heads ×
//     2 dims), o stored straight from registers (128 B a warp store), the VHA chains in registers (U from shared
//     memory), zp stored straight to global — no shared staging, no bulk-copy wait, no 384-thread VHA pass;
//   · K dequantization with one cvt.rn.f16x2.e4m3x2 per two codes (exact, then × 2^e and bf16 RN as before);
//   · the P·V warps request tile 2 of a row before its first P·V.
#pragma once

namespace e55 {
namespace p088 {

constexpr int kLH = 32, kD = kHeadDim, kCh = 256, kKT = 32, TPC = kCh / kKT;
constexpr int kKS = kD + 8;  // K tile / q staging row, bf16 (1040 B: conflict-free ldmatrix)
constexpr int kSW = 4, kVW = 8, kST = 32 * kSW, kVT = 32 * kVW, kTh = kST + kVT;  // score / P·V warps, threads
constexpr int kRB = kKvRowBytes;
constexpr int CS = 2 * kLH + kLH * kD;  // floats per chunk partial in the workspace
constexpr int kMinM = 64;                // smaller calls (drafts, verify, short chunks) stay on P070
enum { kBarP = 1, kBarK = 3, kBarV = 5, kBarA = 6, kBarB = 7, kBarS = 8 };

struct Smem {
  bf16 k[2][kKT][kKS];  // K tiles; k[1] also stages the prepped q at a row's start
  bf16 p[2][kLH][kKT + 8];
  float e[2][kLH][kKT + 1];
  float fac[2][kLH];
  float m[kLH], l[kLH];
  float M[kLH], den[kLH], flast[kLH];
  float red[64];
  float rcs[32], rsn[32];
  __align__(16) float U[kLH * kVhaRank];
};

__device__ __forceinline__ void mma(float d[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}
__device__ __forceinline__ void ldsm4t(uint32_t (&r)[4], const void* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}
__device__ __forceinline__ void ldsm2(uint32_t& r0, uint32_t& r1, const void* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(a));
}
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_arrive(int id, int n) {
  asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "r"(n) : "memory");
}

// two e4m3 codes (low byte first) · s → bf16 pair: P041's e4m3x2_bf16x2 values (float(e4m3) · 2^e, exact, then bf16 RN of
// an exact value) with one cvt for both codes (e4m3x2 → f16x2 is exact) instead of two
__device__ __forceinline__ uint32_t dq2(uint32_t two, float s) {
  uint32_t h2;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(h2) : "h"(static_cast<unsigned short>(two & 0xFFFF)));
  const float lo = __half2float(__ushort_as_half(static_cast<unsigned short>(h2 & 0xFFFF)));
  const float hi = __half2float(__ushort_as_half(static_cast<unsigned short>(h2 >> 16)));
  return pack_bf16x2(lo * s, hi * s);
}

// one K tile's gather by NT threads: thread (key vk = i / NPC, piece vp = i % NPC) moves W = 72 / NPC words (8 B) of the
// row — words 0–55 nope (8 e4m3 codes each, the scale of their 64-dim group), 56–71 rope (4 bf16) — and the 7 scales;
// store: P070's dequantization (float(e4m3) · 2^e → bf16, exact; rope as stored), zero rows past the row's last key
template <int NPC>
struct Gather {
  static constexpr int W = 72 / NPC;
  uint2 rw[W], rsc;
  __device__ __forceinline__ void load(const uint8_t* row, int vp) {
    if (row) {
      const uint2* src = reinterpret_cast<const uint2*>(row);
#pragma unroll
      for (int i = 0; i < W; i++) rw[i] = src[W * vp + i];
      rsc = src[72];
    }
  }
  __device__ __forceinline__ void store(bf16* dst, bool ok, int vp) const {
#pragma unroll
    for (int i = 0; i < W; i++) {
      const int u = W * vp + i;
      if (u < kNope / 8) {
        uint4 v = make_uint4(0, 0, 0, 0);
        if (ok) {
          const int grp = u >> 3;
          const float s = ue8m0_f32(((grp < 4 ? rsc.x : rsc.y) >> (8 * (grp & 3))) & 0xFF);
          v = make_uint4(dq2(rw[i].x, s), dq2(rw[i].x >> 16, s), dq2(rw[i].y, s), dq2(rw[i].y >> 16, s));
        }
        *reinterpret_cast<uint4*>(dst + 8 * u) = v;
      } else {
        *reinterpret_cast<uint2*>(dst + kNope + 4 * (u - kNope / 8)) = ok ? rw[i] : make_uint2(0, 0);
      }
    }
  }
};

struct Row {
  RowMeta mt;
  int p, ncomp, nwin, nk, nch, ntile;
  __device__ __forceinline__ Row(const RowMeta* meta, int r) : mt(meta[r]) {
    p = mt.pos, ncomp = (p + 1) / kRatio, nwin = min(p + 1, kWindow), nk = ncomp + nwin;
    nch = (nk + kCh - 1) / kCh, ntile = (nk + kKT - 1) / kKT;
  }
  __device__ __forceinline__ const uint8_t* key(int k, const uint8_t* rows, const HcaCache& c) const {
    return k < nk ? p041::key_row(k, ncomp, nwin, p, mt, rows, c) : nullptr;
  }
};

// grid: persistent CTAs over rows (CTA x: rows x, x + gridDim.x, …); ws: per CTA slot of earlier chunks' partials
__global__ void __launch_bounds__(kTh, 1)
    k_attn(const bf16* __restrict__ q, const uint8_t* __restrict__ rows, const RowMeta* __restrict__ meta, HcaCache c,
           const float* __restrict__ sink, const float* __restrict__ inv, const float* __restrict__ U, int M,
           bf16* __restrict__ o, float* __restrict__ zp, float* __restrict__ ws, size_t ws_slot) {
  extern __shared__ __align__(16) uint8_t smraw[];
  Smem& S = *reinterpret_cast<Smem*>(smraw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  float* wsl = ws + ws_slot * blockIdx.x;
  for (int i = tid; i < kLH * kVhaRank; i += kTh) S.U[i] = U[i];
  __syncthreads();

  if (warp < kSW) {
    // ================= score warps =================
    const int st = tid, hw = 8 * warp, h0 = hw + 2 * t;
    const int gk = st >> 2, gp = st & 3;  // gather job: key gk of the tile, piece gp
    const float scale = rsqrtf(float(kD));
    int prev_nch = 0;
    // the merge coefficients of the previous row (k_merge / P041 / P070): after every P·V warp is done with its tiles
    auto coefs = [&](int nch) {
      bar_sync(kBarA, kTh);
      if (st < kLH) {
        const int h = st;
        float Mx = sink[h];
        for (int cc = 0; cc + 1 < nch; cc++) Mx = fmaxf(Mx, wsl[cc * CS + h]);
        Mx = fmaxf(Mx, S.m[h]);
        float den = 0.f;
        for (int cc = 0; cc + 1 < nch; cc++) den += expf(wsl[cc * CS + h] - Mx) * wsl[cc * CS + kLH + h];
        den += expf(S.m[h] - Mx) * S.l[h];
        den += expf(sink[h] - Mx);
        S.M[h] = Mx, S.den[h] = den, S.flast[h] = expf(S.m[h] - Mx);
      }
      bar_arrive(kBarB, kTh);
    };
    for (int r = blockIdx.x; r < M; r += gridDim.x) {
      const Row R(meta, r);
      // ---- q (raw) and tile 0 requested before the previous row's coefficients
      uint4 qr[16];  // pass it: head 2·it + st / 64, dims 8·(st % 64) …
      const bf16* qrow = q + (size_t)r * kLH * kD;
#pragma unroll
      for (int it = 0; it < 16; it++)
        qr[it] = *reinterpret_cast<const uint4*>(qrow + (2 * it + (st >> 6)) * kD + 8 * (st & 63));
      Gather<4> G;
      G.load(R.key(gk, rows, c), gp);
      if (r != (int)blockIdx.x) coefs(prev_nch);
      // ---- q prep (k_q_prep / P070: thread i of a head's 64 → dims 8i …; Σ x² per thread, warp sums, two halves)
      if (st < 32) {
        float sn, cs;
        sincosf(__fmul_rn(float(R.p), inv[st]), &sn, &cs);
        S.rcs[st] = cs, S.rsn[st] = sn;
      }
#pragma unroll
      for (int it = 0; it < 16; it++) {
        float v[8];
        const uint32_t w4[4] = {qr[it].x, qr[it].y, qr[it].z, qr[it].w};
#pragma unroll
        for (int i = 0; i < 4; i++) v[2 * i] = bf16_lo(w4[i]), v[2 * i + 1] = bf16_hi(w4[i]);
        float ss = 0.f;
#pragma unroll
        for (int j = 0; j < 8; j++) ss += v[j] * v[j];
        ss = warp_sum(ss);
        if (lane == 0) S.red[2 * (2 * it + (st >> 6)) + (warp & 1)] = ss;
        *reinterpret_cast<uint4*>(&S.k[1][2 * it + (st >> 6)][8 * (st & 63)]) = qr[it];  // raw, prepped in place below
      }
      G.store(&S.k[0][gk][0], gk < R.nk, gp);  // tile 0 → k[0] (the P·V warps are done with it: bar A)
      if (R.ntile > 1) G.load(R.key(kKT + gk, rows, c), gp);
      bar_sync(kBarS, kST);
      {
        const int d = 8 * (st & 63);
#pragma unroll
        for (int it = 0; it < 16; it++) {
          const int hh = 2 * it + (st >> 6);
          float v[8];
          p041::ld_bf16x8(&S.k[1][hh][d], v);  // this thread's own raw q
          const float rs = rsqrtf(__fdiv_rn(S.red[2 * hh] + S.red[2 * hh + 1], float(kD)) + kEps);
#pragma unroll
          for (int j = 0; j < 8; j++) v[j] *= rs;
          if (d >= kNope) p041::rope8t(v, (d - kNope) / 2, S.rcs, S.rsn);
          p041::st_bf16x8(&S.k[1][hh][d], v);
        }
      }
      bar_sync(kBarS, kST);
      uint32_t qb[kD / 16][2];
#pragma unroll
      for (int kk = 0; kk < kD / 16; kk++)
        ldsm2(qb[kk][0], qb[kk][1], &S.k[1][8 * warp + (lane & 7)][16 * kk + ((lane >> 3) & 1) * 8]);
      if (st == 0 && r + (int)gridDim.x < M)  // the next row's q → L2 (one bulk prefetch)
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(q + (size_t)(r + gridDim.x) * kLH * kD),
                     "r"(kLH * kD * 2)
                     : "memory");
      bar_sync(kBarS, kST);  // k[1] free
      if (R.ntile > 1) {
        G.store(&S.k[1][gk][0], kKT + gk < R.nk, gp);
        bar_sync(kBarS, kST);
      }
      // ---- the tiles (P070's score warps, one group)
      const int nk = R.nk, nch = R.nch, ntile = R.ntile;
      for (int j = 0; j < ntile; j++) {
        const int b = j & 1, k0 = j * kKT;
        const bool first = j % TPC == 0, lastt = j % TPC == TPC - 1 || j == ntile - 1;
        if (j > 1) bar_sync(kBarK + b, 32 * (kVW + kSW));
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
        const float moa = first ? -INFINITY : S.m[h0], mob = first ? -INFINITY : S.m[h0 + 1];
        const float mxa = fmaxf(moa, ta), mxb = fmaxf(mob, tb);
        __syncwarp();
        if (g == 0) S.m[h0] = mxa, S.m[h0 + 1] = mxb;
        __syncwarp();
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
          for (int jj = 0; jj < kKT; jj++) sum += S.e[b][h][jj];
        }
        const bool early = lastt && j / TPC + 1 < nch;  // the end of a chunk before the row's last: partial → ws
        float* w = wsl + (size_t)(j / TPC) * CS;
        if (lane < 8) {
          const int h = hw + lane;
          const float l = (first ? 0.f : S.l[h]) * S.fac[b][h] + sum;
          S.l[h] = l;
          if (early) w[kLH + h] = l;
        }
        if (early && g == 0) w[h0] = mxa, w[h0 + 1] = mxb;
        __syncwarp();
        bar_arrive(kBarP + b, 32 * (kVW + kSW));
      }
      prev_nch = nch;
    }
    if (prev_nch) coefs(prev_nch);
  } else {
    // ================= P·V warps =================
    const int vw = warp - kSW, vt = tid - kST;
    const int gk = vt >> 3, gp = vt & 7;  // gather job: key gk of the tile, piece gp
    const int d0 = 64 * vw + 8 * g + 2 * t;  // after the transpose: this lane's two dims
    // the merged o after the transpose: W[n][i] = bf16 pair of head n + 8i, dims d0, d0 + 1
    uint32_t W[8][4];
    // piece pc of row r: o of heads 8pc … 8pc+7, and zp ranks 4pc … 4pc+3 — the VHA chains of k_vha_partial (per rank
    // and dim Σ_h U[h][rank] · o[h][dim], h ascending, from 0)
    auto piece = [&](int r, int pc) {
      bf16* orow = o + (size_t)r * kLH * kD + d0;
#pragma unroll
      for (int n = 0; n < 8; n++) *reinterpret_cast<uint32_t*>(orow + (8 * pc + n) * kD) = W[n][pc];
      float z[4][2];
#pragma unroll
      for (int k = 0; k < 4; k++) z[k][0] = z[k][1] = 0.f;
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
      for (int k = 0; k < 4; k++) *reinterpret_cast<float2*>(zrow + k * kD) = make_float2(z[k][0], z[k][1]);
    };
    for (int r = blockIdx.x; r < M; r += gridDim.x) {
      const Row R(meta, r);
      const int nk = R.nk, nch = R.nch, ntile = R.ntile;
      Gather<8> G;
      if (ntile > 2) G.load(R.key(2 * kKT + gk, rows, c), gp);
      float acc[2][8][4];  // heads 16a + g (+8), dims 64·vw + 8n + 2t (+1)
#pragma unroll
      for (int a = 0; a < 2; a++)
#pragma unroll
        for (int n = 0; n < 8; n++) acc[a][n][0] = acc[a][n][1] = acc[a][n][2] = acc[a][n][3] = 0.f;
      for (int j = 0; j < ntile; j++) {
        const int b = j & 1;
        bar_sync(kBarP + b, 32 * (kVW + kSW));
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
            if (ks == 1 && bp == 3 && j + 2 < ntile) bar_sync(kBarV, kVT);  // every P·V warp has read k[b]
            mma(acc[0][2 * bp], pa[0], vv[0], vv[1]);
            mma(acc[1][2 * bp], pa[1], vv[0], vv[1]);
            mma(acc[0][2 * bp + 1], pa[0], vv[2], vv[3]);
            mma(acc[1][2 * bp + 1], pa[1], vv[2], vv[3]);
          }
        }
        if (j + 2 < ntile) {  // tile j+2 → k[b], beside k-step 1's MMAs
          G.store(&S.k[b][gk][0], (j + 2) * kKT + gk < nk, gp);
          bar_arrive(kBarK + b, 32 * (kVW + kSW));
          if (j + 3 < ntile) G.load(R.key((j + 3) * kKT + gk, rows, c), gp);
        }
        if ((j % TPC == TPC - 1 || j == ntile - 1) && j / TPC + 1 < nch) {  // an earlier chunk's O → ws; fresh state
          float* wo = wsl + (size_t)(j / TPC) * CS + 2 * kLH;
#pragma unroll
          for (int a = 0; a < 2; a++)
#pragma unroll
            for (int n = 0; n < 8; n++) {
              const int hr = 16 * a + g, d = 64 * vw + 8 * n + 2 * t;
              *reinterpret_cast<float2*>(wo + hr * kD + d) = make_float2(acc[a][n][0], acc[a][n][1]);
              *reinterpret_cast<float2*>(wo + (hr + 8) * kD + d) = make_float2(acc[a][n][2], acc[a][n][3]);
              acc[a][n][0] = acc[a][n][1] = acc[a][n][2] = acc[a][n][3] = 0.f;
            }
        }
      }
      bar_arrive(kBarA, kTh);  // done with the K / P buffers; partials written
      bar_sync(kBarB, kTh);    // M, den, flast ready
      // ---- merge (P070's code): W[n][i] = bf16 pair of head g + 8i, dims 64·vw + 8n + 2t (+1)
#pragma unroll
      for (int a = 0; a < 2; a++)
#pragma unroll
        for (int n0 = 0; n0 < 8; n0 += 4) {
          const int h = 16 * a + g;
          float x[4][4];
#pragma unroll
          for (int n = 0; n < 4; n++) x[n][0] = x[n][1] = x[n][2] = x[n][3] = 0.f;
          for (int cc = 0; cc + 1 < nch; cc++) {
            const float* w = wsl + (size_t)cc * CS;
            float2 v[4][2];
#pragma unroll
            for (int n = 0; n < 4; n++) {
              const int d = 64 * vw + 8 * (n0 + n) + 2 * t;
              v[n][0] = *reinterpret_cast<const float2*>(&w[2 * kLH + h * kD + d]);
              v[n][1] = *reinterpret_cast<const float2*>(&w[2 * kLH + (h + 8) * kD + d]);
            }
            const float f0 = expf(w[h] - S.M[h]), f1 = expf(w[h + 8] - S.M[h + 8]);
#pragma unroll
            for (int n = 0; n < 4; n++)
              x[n][0] += f0 * v[n][0].x, x[n][1] += f0 * v[n][0].y, x[n][2] += f1 * v[n][1].x, x[n][3] += f1 * v[n][1].y;
          }
          const float fa = S.flast[h], fb = S.flast[h + 8], da = S.den[h], db = S.den[h + 8];
#pragma unroll
          for (int n = 0; n < 4; n++) {
            const float* ac = acc[a][n0 + n];
            W[n0 + n][2 * a] = pack_bf16x2(__fdiv_rn(x[n][0] + fa * ac[0], da), __fdiv_rn(x[n][1] + fa * ac[1], da));
            W[n0 + n][2 * a + 1] = pack_bf16x2(__fdiv_rn(x[n][2] + fb * ac[2], db), __fdiv_rn(x[n][3] + fb * ac[3], db));
          }
        }
      // ---- 8 × 8 transpose over the lanes of equal t (g ↔ n): then W[n][i] = head n + 8i, dims d0, d0 + 1
#pragma unroll
      for (int mb = 1; mb < 8; mb <<= 1) {
        const bool hi = g & mb;
#pragma unroll
        for (int n = 0; n < 8; n++) {
          if (n & mb) continue;
#pragma unroll
          for (int i = 0; i < 4; i++) {
            const uint32_t snd = hi ? W[n][i] : W[n | mb][i];
            const uint32_t rcv = __shfl_xor_sync(0xffffffffu, snd, mb << 2);
            if (hi) W[n][i] = rcv;
            else W[n | mb][i] = rcv;
          }
        }
      }
#pragma unroll
      for (int pc = 0; pc < 4; pc++) piece(r, pc);  // constant piece numbers: W stays in registers
    }
  }
}

inline void launch(const bf16* q, const uint8_t* rows, const RowMeta* meta, HcaCache c, const float* sink,
                   const float* inv, const float* U, int M, int max_keys, bf16* o, float* zp, float* ws, size_t ws_bytes,
                   cudaStream_t s) {
  static int nsm = 0;
  if (!nsm) {
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
    CUDA_CHECK(cudaFuncSetAttribute(k_attn, cudaFuncAttributeMaxDynamicSharedMemorySize, int(sizeof(Smem))));
  }
  const int nch = std::max(1, (max_keys + kCh - 1) / kCh);
  const size_t slot = size_t(nch - 1) * CS;  // floats per CTA
  int nx = std::min(M, nsm);
  if (slot) nx = std::min<long>(nx, long(ws_bytes / 4 / slot));
  if (nx < 1) qlog::fail("hca", "attention workspace %zu B < one CTA's %zu B", ws_bytes, slot * 4);
  k_attn<<<nx, kTh, sizeof(Smem), s>>>(q, rows, meta, c, sink, inv, U, M, o, zp, ws, slot);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace p088
}  // namespace e55
