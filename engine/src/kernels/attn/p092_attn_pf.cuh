// p092_attn_pf.cuh — P092 dsa_attn_pf: prefill dsa_attn = P071 (dsa_attn_ws, copied: its tile pipeline unchanged)
// with the chunk partials kept in L2 and the row's merge reworked. Included by dsa.cu inside its anonymous namespace
// after p071_dsa_ws.cuh, selected by patch::on(dsa_attn_pf) (P071's launches: M · chunks > SMs / 2).
//
// Bitwise contract — P071's (= the base's): the same tiles, scores, online softmax, chunk partials, and the same merge
// arithmetic (M = max(sink, m_c) in chunk order, den = Σ_c expf(m_c − M)·l_c in chunk order + expf(sink − M),
// out = Σ_c expf(m_c − M)·o_c in chunk order / den, bf16). Only memory behaviour differs:
//   · the 64 KB fp32 partial of each (row, 256-key chunk) is stored with an L2 evict_last hint: at long context the
//     gathered MLA rows (> L2) pushed P071's partials to DRAM before the row's merge read them (and discarded them);
//   · P071's last CTA of a row walked the partials with a run-time chunk loop (a few loads in flight, ~3 L2 round
//     trips per 512 outputs, 8 times per row, on an SM it holds alone); here all ≤ 9 chunk partials of two 512-output
//     passes are requested at once, before the weights are known (4 round trips per row instead of ~24), and the
//     per-head m / l in one batch.
#pragma once

namespace p092 {

// stores of data read back soon (then discarded): L2 evict_last, so the gathers' stream does not push them to DRAM
__device__ __forceinline__ uint64_t keep_policy() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(p));
  return p;
}
__device__ __forceinline__ void st_keep(float* a, float x, float y, uint64_t pol) {
  asm volatile("st.global.L2::cache_hint.v2.f32 [%0], {%1, %2}, %3;\n" ::"l"(a), "f"(x), "f"(y), "l"(pol) : "memory");
}

constexpr int kSW = 8, kVW = 8, kThreads = 32 * (kSW + kVW);  // 512: score groups 0 / 1 (even / odd tiles), P·V
constexpr int kCpc = 3;  // most chunks per CTA (P071's; 4 or 5 overflow L2 with the partials at long context)
constexpr int NH = 32, TPC = kChunk / kKT, SL = 2 * NH + NH * kKvLora;
// named barriers: P ready [2], K tile ready [2], P·V warps' own, running max / sum handed to the other score group [2]
enum { kBarP = 1, kBarK = 3, kBarV = 5, kBarM = 6, kBarL = 8 };

struct Smem {
  bf16 k[2][kKT][kQS];  // K tiles (k[1] stages q at the start)
  bf16 p[2][NH][kKT + 8];
  float e[2][NH][kKT + 1];
  float fac[2][NH];
  float m[NH], l[NH];
  int key[kCpc * kChunk];
  float mf[10][NH];
  float den[NH];
  int last;
};

// a register-only MMA (not volatile: the compiler may interleave independent chains; each chain keeps its order)
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
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_arrive(int id, int n) {
  asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "r"(n) : "memory");
}

// grid: n rows × ⌈nch / kCpc⌉ chunk groups (row-major); ws per (row, chunk) slab [m 32][l 32][o 32 × 512] fp32;
// ctr per row (zero at launch, zero again after the merge)
// EP (P110 dsa_attn_ep): each score warp publishes P / f before its tile sum (the base order of the sum kept)
template <bool EP>
__global__ void __launch_bounds__(kThreads, 1)
    k_attn(const bf16* __restrict__ q, const int* __restrict__ list, const int* __restrict__ cnt,
           const RowMeta* __restrict__ meta, DsaCache c, const float* __restrict__ sink, int row0, int nch, int cpc,
           float* ws, int* ctr, bf16* __restrict__ olat) {
  extern __shared__ __align__(16) uint8_t smraw[];
  Smem& S = *reinterpret_cast<Smem*>(smraw);
  const int ngrp = (nch + cpc - 1) / cpc;
  const int cg = blockIdx.x % ngrp, ri = blockIdx.x / ngrp, r = row0 + ri;
  const int nk = cnt[r], kbeg = cg * cpc * kChunk;
  if (kbeg >= nk) return;
  const int kend = min(nk, kbeg + cpc * kChunk), ntile = (kend - kbeg + kKT - 1) / kKT;
  const int nchr = (nk + kChunk - 1) / kChunk;
  const RowMeta mt = meta[r];
  const uint8_t* base = c.mla + (size_t)mt.slot * c.ctx * kMR;
  const int* lst = list + (size_t)r * kMaxKeys;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const bool sw = warp < kSW;  // warps 0–7: score groups 0 / 1; 8–15: P·V
  const int si = warp, vw = warp - kSW, vt = tid - 32 * kSW, u8 = vt & 7, k8 = vt >> 3;  // score / P·V index, thread
  float* wrow = ws + (size_t)ri * nch * SL;

  // ---- P·V warps' K tile gather (P045 Raw: latent bytes 16·lane … of keys vw + 8i, k_pe unit u8 of key k8) ---------
  uint4 rkv[4], rpe;
  float rsc[4];
  auto load = [&](int k0, const int* keys) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int kk = k0 + vw + 8 * i;
      if (kk < kend) {
        const uint8_t* row = base + (size_t)keys[kk - kbeg] * kMR;
        rkv[i] = __ldg(reinterpret_cast<const uint4*>(row) + lane);
        rsc[i] = __ldg(reinterpret_cast<const float*>(row + kKvLora) + (lane >> 3));
      }
    }
    const int kk = k0 + k8;
    if (kk < kend) rpe = __ldg(reinterpret_cast<const uint4*>(base + (size_t)keys[kk - kbeg] * kMR + kKvLora + 16) + u8);
  };
  auto store = [&](int k0, int buf) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int kl = vw + 8 * i;
      uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;
      if (k0 + kl < kend) {
        const float sc = rsc[i];
        const uint4 x = rkv[i];
        v0 = make_uint4(p045::e4m3x2_bf16x2(x.x & 0xFFFF, sc), p045::e4m3x2_bf16x2(x.x >> 16, sc),
                        p045::e4m3x2_bf16x2(x.y & 0xFFFF, sc), p045::e4m3x2_bf16x2(x.y >> 16, sc));
        v1 = make_uint4(p045::e4m3x2_bf16x2(x.z & 0xFFFF, sc), p045::e4m3x2_bf16x2(x.z >> 16, sc),
                        p045::e4m3x2_bf16x2(x.w & 0xFFFF, sc), p045::e4m3x2_bf16x2(x.w >> 16, sc));
      }
      uint4* d = reinterpret_cast<uint4*>(&S.k[buf][kl][16 * lane]);
      d[0] = v0, d[1] = v1;
    }
    *reinterpret_cast<uint4*>(&S.k[buf][k8][kKvLora + 8 * u8]) = k0 + k8 < kend ? rpe : make_uint4(0, 0, 0, 0);
  };

  // ---- prologue: tile 0's loads, the key list and q (staged in k[1]) in flight together; tile 0 → k[0] ------------
  if (!sw) load(kbeg, lst + kbeg);
  {
    constexpr int NQ8 = NH * kQK / 8, NQV = (NQ8 + kThreads - 1) / kThreads, NKV = (kCpc * kChunk + kThreads - 1) / kThreads;
    uint4 qv[NQV];
    int kv[NKV];
#pragma unroll
    for (int j = 0; j < NKV; j++) {
      const int i = tid + j * kThreads;
      kv[j] = i < kend - kbeg ? __ldg(lst + kbeg + i) : 0;
    }
#pragma unroll
    for (int j = 0; j < NQV; j++) {
      const int i = tid + j * kThreads, h = i / (kQK / 8), c8 = i % (kQK / 8);
      if (i < NQ8) qv[j] = __ldg(reinterpret_cast<const uint4*>(q + ((size_t)r * kLH + h) * kQK) + c8);
    }
#pragma unroll
    for (int j = 0; j < NKV; j++)
      if (tid + j * kThreads < kCpc * kChunk) S.key[tid + j * kThreads] = kv[j];
#pragma unroll
    for (int j = 0; j < NQV; j++) {
      const int i = tid + j * kThreads, h = i / (kQK / 8), c8 = i % (kQK / 8);
      if (i < NQ8) *reinterpret_cast<uint4*>(&S.k[1][h][c8 * 8]) = qv[j];
    }
  }
  if (!sw) store(kbeg, 0);
  __syncthreads();
  uint32_t qb[kQK / 16][2];
  if (sw) {
#pragma unroll
    for (int kk = 0; kk < kQK / 16; kk++)
      ldsm_x2(qb[kk][0], qb[kk][1], &S.k[1][8 * (si & 3) + (lane & 7)][16 * kk + ((lane >> 3) & 1) * 8]);
  } else if (ntile > 1) {
    load(kbeg + kKT, S.key);
  }
  __syncthreads();  // k[1] free
  if (sw) {
    // ======== score warps: group grp takes tiles grp, grp+2, …; heads h0 = 8·(warp & 3) + 2t, h0 + 1 ========
    const float scale = 0.0625f;  // 256^-0.5
    const int grp = si >> 2, hw = 8 * (si & 3), h0 = hw + 2 * t;
    for (int j = grp; j < ntile; j += 2) {
      const int b = j & 1, k0 = kbeg + j * kKT;
      const bool first = j % TPC == 0, lastt = j % TPC == TPC - 1 || j == ntile - 1, more = j + 1 < ntile;
      if (j > 0) bar_sync(kBarK + b, 32 * (kVW + 4));
      float sacc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
      {  // the two k-chains, K fragments loaded one k-step ahead
        uint32_t af[2][2][4];
        const bf16* kb0 = &S.k[b][lane & 15][(lane >> 4) * 8];
        const bf16* kb1 = &S.k[b][16 + (lane & 15)][(lane >> 4) * 8];
        ldsm4(af[0][0], kb0), ldsm4(af[0][1], kb1);
#pragma unroll
        for (int kk = 0; kk < kQK / 16; kk++) {
          if (kk + 1 < kQK / 16) ldsm4(af[(kk + 1) & 1][0], kb0 + 16 * (kk + 1)), ldsm4(af[(kk + 1) & 1][1], kb1 + 16 * (kk + 1));
          mma(sacc[0], af[kk & 1][0], qb[kk][0], qb[kk][1]);
          mma(sacc[1], af[kk & 1][1], qb[kk][0], qb[kk][1]);
        }
      }
      // softmax of the tile: sacc[m][i] = key 16m + g + 8(i >> 1), head h0 + (i & 1)
      float sv[2][4];
#pragma unroll
      for (int m = 0; m < 2; m++)
#pragma unroll
        for (int i = 0; i < 4; i++) sv[m][i] = k0 + 16 * m + g + (i >> 1) * 8 < kend ? sacc[m][i] * scale : -INFINITY;
      float ta = fmaxf(fmaxf(sv[0][0], sv[0][2]), fmaxf(sv[1][0], sv[1][2]));
      float tb = fmaxf(fmaxf(sv[0][1], sv[0][3]), fmaxf(sv[1][1], sv[1][3]));
#pragma unroll
      for (int off = 4; off < 32; off <<= 1)
        ta = fmaxf(ta, __shfl_xor_sync(0xffffffffu, ta, off)), tb = fmaxf(tb, __shfl_xor_sync(0xffffffffu, tb, off));
      if (j > 0) bar_sync(kBarM + (b ^ 1), 32 * kSW);  // the running max after tile j−1
      const float moa = first ? -INFINITY : S.m[h0], mob = first ? -INFINITY : S.m[h0 + 1];
      const float mxa = fmaxf(moa, ta), mxb = fmaxf(mob, tb);
      __syncwarp();
      if (g == 0) S.m[h0] = mxa, S.m[h0 + 1] = mxb;
      __syncwarp();
      if (more) bar_arrive(kBarM + b, 32 * kSW);
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
      if constexpr (EP) bar_arrive(kBarP + b, 32 * (kVW + 4));
      float* w = wrow + (size_t)(kbeg / kChunk + j / TPC) * SL;
      float sum = 0.f;
      if (lane < 8) {  // the tile sum in key order (base)
        const int h = hw + lane;
#pragma unroll
        for (int jj = 0; jj < kKT; jj++) sum += S.e[b][h][jj];
      }
      if (j > 0) bar_sync(kBarL + (b ^ 1), 32 * kSW);  // the running sum after tile j−1
      if (lane < 8) {  // l·f + sum
        const int h = hw + lane;
        const float l = (first ? 0.f : S.l[h]) * S.fac[b][h] + sum;
        S.l[h] = l;
        if (lastt) w[NH + h] = l;
      }
      if (lastt && g == 0) w[h0] = mxa, w[h0 + 1] = mxb;
      __syncwarp();
      if (more) bar_arrive(kBarL + b, 32 * kSW);
      if constexpr (!EP) bar_arrive(kBarP + b, 32 * (kVW + 4));
    }
  } else {
    // ======== P·V warps: dims 64·vw …, all 32 heads; K tile gather ========
    float o[2][8][4];
#pragma unroll
    for (int a = 0; a < 2; a++)
#pragma unroll
      for (int n = 0; n < 8; n++) o[a][n][0] = o[a][n][1] = o[a][n][2] = o[a][n][3] = 0.f;
    if (ntile > 1) {
      store(kbeg + kKT, 1);
      bar_arrive(kBarK + 1, 32 * (kVW + 4));
      if (ntile > 2) load(kbeg + 2 * kKT, S.key);
    }
    for (int j = 0; j < ntile; j++) {
      const int b = j & 1;
      const bool lastt = j % TPC == TPC - 1 || j == ntile - 1;
      bar_sync(kBarP + b, 32 * (kVW + 4));
#pragma unroll
      for (int a = 0; a < 2; a++) {
        const float f0 = S.fac[b][16 * a + g], f1 = S.fac[b][16 * a + g + 8];
#pragma unroll
        for (int n = 0; n < 8; n++) o[a][n][0] *= f0, o[a][n][1] *= f0, o[a][n][2] *= f1, o[a][n][3] *= f1;
      }
      {  // k-step 0's MMAs; then k-step 1's fragments, the barrier that frees k[b] (every P·V warp has read it), k-step
         // 1's MMAs and — while they run — tile j+2's dequantization into k[b]
        uint32_t pa[2][4], vv[4][4];
#pragma unroll
        for (int ks = 0; ks < kKT / 16; ks++) {
          ldsm4(pa[0], &S.p[b][lane & 15][16 * ks + (lane >> 4) * 8]);
          ldsm4(pa[1], &S.p[b][16 + (lane & 15)][16 * ks + (lane >> 4) * 8]);
#pragma unroll
          for (int bp = 0; bp < 4; bp++) ldsm4t(vv[bp], &S.k[b][16 * ks + (lane & 15)][64 * vw + 16 * bp + (lane >> 4) * 8]);
          if (ks == 1 && j + 2 < ntile) bar_sync(kBarV, 32 * kVW);
#pragma unroll
          for (int bp = 0; bp < 4; bp++) {
            mma(o[0][2 * bp], pa[0], vv[bp][0], vv[bp][1]);
            mma(o[1][2 * bp], pa[1], vv[bp][0], vv[bp][1]);
            mma(o[0][2 * bp + 1], pa[0], vv[bp][2], vv[bp][3]);
            mma(o[1][2 * bp + 1], pa[1], vv[bp][2], vv[bp][3]);
          }
        }
      }
      if (j + 2 < ntile) {  // tile j+2 → k[b] (freed by the barrier above), beside k-step 1's MMAs
        store(kbeg + (j + 2) * kKT, b);
        bar_arrive(kBarK + b, 32 * (kVW + 4));
        if (j + 3 < ntile) load(kbeg + (j + 3) * kKT, S.key);
      }
      if (lastt) {  // this chunk's o → its slab; fresh state for the next chunk
        const uint64_t pol = keep_policy();
        float* wo = wrow + (size_t)(kbeg / kChunk + j / TPC) * SL + 2 * NH;
#pragma unroll
        for (int a = 0; a < 2; a++)
#pragma unroll
          for (int n = 0; n < 8; n++) {
            const int hr = 16 * a + g, d = 64 * vw + 8 * n + 2 * t;
            st_keep(wo + hr * kKvLora + d, o[a][n][0], o[a][n][1], pol);
            st_keep(wo + (hr + 8) * kKvLora + d, o[a][n][2], o[a][n][3], pol);
            o[a][n][0] = o[a][n][1] = o[a][n][2] = o[a][n][3] = 0.f;
          }
      }
    }
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    const int mine = (ntile + TPC - 1) / TPC;
    S.last = atomicAdd(ctr + ri, mine) + mine == nchr;
  }
  __syncthreads();
  if (!S.last) return;
  __threadfence();
  // ---- the row's last CTA: merge the chunk partials (base k_mla_merge arithmetic), every load of a pass in flight
  constexpr int MC = (kMaxKeys + kChunk - 1) / kChunk;  // ≤ 9 chunks a row
  constexpr int NI = NH * kKvLora / 4 / kThreads;      // float4 outputs per thread (8)
  const float* wr = wrow;
  float4 x[2][MC];
  auto fetch = [&](int it) {  // outputs it·2, it·2 + 1 of this thread: every chunk's partial
#pragma unroll
    for (int u = 0; u < 2; u++) {
      const int i = tid + (2 * it + u) * kThreads;
#pragma unroll
      for (int cc = 0; cc < MC; cc++)
        x[u][cc] = cc < nchr ? __ldcg(reinterpret_cast<const float4*>(wr + cc * SL + 2 * NH) + i) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
  };
  fetch(0);
  if (tid < NH) {
    const int h = tid;
    const float sk = sink[h];
    float mv[MC], lv[MC];
#pragma unroll
    for (int cc = 0; cc < MC; cc++) {
      mv[cc] = cc < nchr ? __ldcg(wr + cc * SL + h) : 0.f;
      lv[cc] = cc < nchr ? __ldcg(wr + cc * SL + NH + h) : 0.f;
    }
    float Mx = sk;
#pragma unroll
    for (int cc = 0; cc < MC; cc++)
      if (cc < nchr) Mx = fmaxf(Mx, mv[cc]);
    float den = 0.f;
#pragma unroll
    for (int cc = 0; cc < MC; cc++)
      if (cc < nchr) den += expf(mv[cc] - Mx) * lv[cc];
    den += expf(sk - Mx);
#pragma unroll
    for (int cc = 0; cc < MC; cc++)
      if (cc < nchr) S.mf[cc][h] = expf(mv[cc] - Mx);
    S.den[h] = den;
  }
  __syncthreads();
#pragma unroll 1
  for (int it = 0; it < NI / 2; it++) {
    if (it > 0) fetch(it);
#pragma unroll
    for (int u = 0; u < 2; u++) {
      const int i = tid + (2 * it + u) * kThreads, h = (4 * i) / kKvLora;
      float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
      for (int cc = 0; cc < MC; cc++)
        if (cc < nchr) {
          const float f = S.mf[cc][h];
          const float4 xv = x[u][cc];
          a0 += f * xv.x, a1 += f * xv.y, a2 += f * xv.z, a3 += f * xv.w;
        }
      const float den = S.den[h];
      *reinterpret_cast<uint2*>(olat + (size_t)r * kLH * kKvLora + 4 * i) =
          make_uint2(pack_bf16x2(__fdiv_rn(a0, den), __fdiv_rn(a1, den)), pack_bf16x2(__fdiv_rn(a2, den), __fdiv_rn(a3, den)));
    }
  }
  __syncthreads();
  for (int cc = 0; cc < nchr; cc++) {
    const char* wb = reinterpret_cast<const char*>(wr + cc * SL);
    for (int i = tid; i < SL * 4 / 128; i += kThreads) p045::discard_l2(wb + (size_t)i * 128);
  }
  if (tid == 0) ctr[ri] = 0;
}

// P071's launches
inline bool wanted(int M, int max_keys, int nsm) { return 2 * M * std::max(1, (max_keys + kChunk - 1) / kChunk) > nsm; }
template <bool EP = false>
inline void run(const bf16* q576, const int* list, const int* cnt, const RowMeta* meta, DsaCache c, const float* sink,
                int M, int max_keys, bf16* olat, float* ws, size_t ws_bytes, cudaStream_t s) {
  static int* ctr = nullptr;
  static size_t nctr = 0;
  if ((size_t)M > nctr) {
    if (ctr) CUDA_CHECK(cudaFree(ctr));
    nctr = std::max<size_t>((size_t)M, 1 << 14);
    CUDA_CHECK(cudaMalloc(&ctr, nctr * 4));
    CUDA_CHECK(cudaMemset(ctr, 0, nctr * 4));
  }
  static bool attr = false;
  if (!attr) {
    CUDA_CHECK(cudaFuncSetAttribute(k_attn<EP>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(sizeof(Smem))));
    attr = true;
  }
  const int nch = std::max(1, (max_keys + kChunk - 1) / kChunk);
  const size_t per_row = (size_t)nch * SL * 4;
  if (ws_bytes < per_row) qlog::fail("dsa", "attention workspace too small");
  const int pass = int(std::min<size_t>(M, ws_bytes / per_row));
  static int nsm = 0;
  if (!nsm) CUDA_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  const int cpc = M * nch >= 8 * nsm ? kCpc : 1;
  const int ngrp = (nch + cpc - 1) / cpc;
  for (int r0 = 0; r0 < M; r0 += pass) {
    const int n = std::min(pass, M - r0);
    k_attn<EP><<<n * ngrp, kThreads, sizeof(Smem), s>>>(q576, list, cnt, meta, c, sink, r0, nch, cpc, ws, ctr, olat);
    CUDA_CHECK(cudaGetLastError());
  }
}

}  // namespace p092
