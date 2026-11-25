// p043_hca_attn_dec.cuh — P043 hca_attn_dec (included by hca.cu after p041): decode attention (M ≤ 10) as one 8-CTA
// cluster per row. CTA j owns output dims 64j … 64j+63 (as P041's decode slices), but the Q·Kᵀ chains of a 256-key
// chunk are split over the 8 CTAs (n-tile n → CTA n % 8, one (n-tile, m-tile) chain per warp); the scores meet in an
// L2 exchange buffer behind one cluster barrier (release / acquire; distributed shared memory measured far slower on
// GB10) — no CTA computes the scores of the whole row alone.
// Then each CTA runs the chunk's softmax tiles, P·V over its slice, the merge and the VHA partial exactly as P041.
// Arithmetic per score / tile / output identical to k_attn + k_merge → bit-identical.
#pragma once

namespace e55 {
namespace p043 {

using p041::kLH, p041::kD, p041::kCh, p041::kKT, p041::kBK, p041::kTh, p041::kQS;
using p041::ld_bf16x8, p041::st_bf16x8, p041::rope8t, p041::ldsm4, p041::ldsm2t, p041::e4m3x2_bf16x2, p041::key_row;
constexpr int kMaxM = 10;        // rows up to which this beats P041 (measured: M = 8 24–27 µs vs 39; M = 16 45 vs 33)
constexpr int kCl = 8;           // CTAs per row (cluster), 64 dims each
constexpr int kDS = kD / kCl;    // 64
constexpr int kSR = kCh + 4;     // score row (fp32)
constexpr int kVR = kDS + 8;     // V row (bf16)

struct Smem {
  union {
    bf16 q[kLH][kQS];      // Q·Kᵀ of the chunk
    bf16 v[kCh][kVR];      // then V of the chunk (this CTA's 64 dims)
  } qv;
  float s[kLH][kSR];       // scores of the chunk (all CTAs' chains, read back from the L2 exchange)
  bf16 p[kLH][kCh + 8];
  float fac[kLH][8];
  float m[kLH], l[kLH];
  float M[kLH], den[kLH], flast[kLH];
  float red[64];
  float rcs[32], rsn[32];  // the row's RoPE angles (q prep)
  float U[kLH * kVhaRank];
};

__device__ __forceinline__ void cl_arrive() { asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void cl_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory"); }

// the softmax statistics of 4 tiles (keys k0 … k0+127 of the chunk); as P041
__device__ __forceinline__ void softmax4(Smem& S, int k0, int ntl, int warp, int lane) {
  const int h = 2 * warp + (lane >> 4), tt = (lane >> 2) & 3, q4 = lane & 3, hb = lane & 16;
  const bool tok = tt < ntl;
  float sv[8], tm = -INFINITY;
  {
    const float4* sp = reinterpret_cast<const float4*>(&S.s[h][k0 + 32 * tt + 8 * q4]);
    const float4 x0 = sp[0], x1 = sp[1];
    sv[0] = x0.x, sv[1] = x0.y, sv[2] = x0.z, sv[3] = x0.w, sv[4] = x1.x, sv[5] = x1.y, sv[6] = x1.z, sv[7] = x1.w;
  }
#pragma unroll
  for (int i = 0; i < 8; i++) tm = fmaxf(tm, sv[i]);
  tm = fmaxf(tm, __shfl_xor_sync(0xffffffffu, tm, 1));
  tm = fmaxf(tm, __shfl_xor_sync(0xffffffffu, tm, 2));
  const float m_in = S.m[h], l_in = S.l[h];
  float mprev = m_in, mx = m_in;
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const float x = __shfl_sync(0xffffffffu, tm, hb + 4 * j);
    if (j < ntl && j <= tt) mprev = mx, mx = fmaxf(mx, x);
  }
  float e[8];
#pragma unroll
  for (int i = 0; i < 8; i++) e[i] = expf(sv[i] - mx);
  float sum = 0.f;
#pragma unroll
  for (int st = 0; st < 4; st++) {
    if (q4 == st)
#pragma unroll
      for (int i = 0; i < 8; i++) sum += e[i];
    sum = __shfl_sync(0xffffffffu, sum, (lane & ~3) | st);
  }
  const float f = expf(mprev - mx);
  float l = l_in, mlast = m_in;
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const float fj = __shfl_sync(0xffffffffu, f, hb + 4 * j), sj = __shfl_sync(0xffffffffu, sum, hb + 4 * j);
    const float mj = __shfl_sync(0xffffffffu, mx, hb + 4 * j);
    if (j < ntl) l = l * fj + sj, mlast = mj;
  }
  if (tok) {
    *reinterpret_cast<uint4*>(&S.p[h][k0 + 32 * tt + 8 * q4]) =
        make_uint4(pack_bf16x2(e[0], e[1]), pack_bf16x2(e[2], e[3]), pack_bf16x2(e[4], e[5]), pack_bf16x2(e[6], e[7]));
    if (q4 == 0) S.fac[h][k0 / kKT + tt] = f;
  }
  if ((lane & 15) == 0) S.m[h] = mlast, S.l[h] = l;
  __syncwarp();
}

// grid (8, M), cluster (8, 1, 1): CTA x = dims slice and cluster rank, y = row
template <bool PREP, bool VHA>
__global__ void __launch_bounds__(kTh, 1)
    k_dec(const bf16* __restrict__ q, const uint8_t* __restrict__ rows, const RowMeta* __restrict__ meta, HcaCache c,
          const float* __restrict__ sink, const float* __restrict__ inv, const float* __restrict__ U,
          bf16* __restrict__ o, float* __restrict__ zp, float* __restrict__ ws, size_t ws_slot) {
  extern __shared__ __align__(16) uint8_t smraw[];
  Smem& S = *reinterpret_cast<Smem*>(smraw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
  const int rank = blockIdx.x, r = blockIdx.y, d0 = rank * kDS;
  float* xs = ws + (size_t)r * kLH * kCh;                             // the row's score exchange [32][256] (L2)
  float* wsl = ws + (size_t)gridDim.y * kLH * kCh + ws_slot * (r * kCl + rank);  // chunk partials
  if (VHA)
    for (int i = tid; i < kLH * kVhaRank; i += kTh) S.U[i] = U[i];
  const float scale = rsqrtf(float(kD));
  const RowMeta mt = meta[r];
  const int p = mt.pos, ncomp = (p + 1) / kRatio, nwin = min(p + 1, kWindow), nk = ncomp + nwin;
  const int nch = (nk + kCh - 1) / kCh;
  float acc[4];
  bool first = true;
#pragma unroll 1
  for (int ch = 0; ch < nch; ch++) {
    const int kc0 = ch * kCh, kc1 = min(nk, kc0 + kCh), nkc = kc1 - kc0;
    const int ntl = (nkc + kKT - 1) / kKT;
    // ---- Q → shared (prepped like k_q_prep when PREP); the V region aliases it, so again for every chunk ----------
    {
      const bf16* qr = q + (size_t)r * kLH * kD;
      const int i = tid & 63, d = 8 * i;
      float v[4][8];
#pragma unroll
      for (int it = 0; it < 4; it++) ld_bf16x8(qr + (it * 8 + (tid >> 6)) * kD + d, v[it]);
      if (PREP) {
        if (tid < 32) {  // the base's angles fl(pos · inv[i]), once per row instead of once per head
          float sn, cs;
          sincosf(__fmul_rn(float(p), inv[tid]), &sn, &cs);
          S.rcs[tid] = cs, S.rsn[tid] = sn;
        }
#pragma unroll
        for (int it = 0; it < 4; it++) {
          float ss = 0.f;
#pragma unroll
          for (int j = 0; j < 8; j++) ss += v[it][j] * v[it][j];
          ss = warp_sum(ss);
          if (lane == 0) S.red[it * 16 + warp] = ss;
        }
      }
      __syncthreads();  // red ready; the previous chunk's P·V is done with the V region
      if (PREP) {
#pragma unroll
        for (int it = 0; it < 4; it++) {
          const float rs =
              rsqrtf(__fdiv_rn(S.red[it * 16 + (warp & ~1)] + S.red[it * 16 + (warp | 1)], float(kD)) + kEps);
#pragma unroll
          for (int j = 0; j < 8; j++) v[it][j] *= rs;
          if (d >= kNope) rope8t(v[it], (d - kNope) / 2, S.rcs, S.rsn);
        }
      }
#pragma unroll
      for (int it = 0; it < 4; it++) st_bf16x8(&S.qv.q[it * 8 + (tid >> 6)][d], v[it]);
    }
    // V of the chunk: 256 keys × this CTA's 64 dims, two segments of 16 dims per thread (job = key · 4 + segment)
    uint2 vu[2][4];
    float vsc[2];
#pragma unroll
    for (int j = 0; j < 2; j++) {
      const int job = tid + j * kTh, key = kc0 + (job >> 2), d = d0 + 16 * (job & 3);
      if (key < kc1) {
        const uint8_t* row = key_row(key, ncomp, nwin, p, mt, rows, c);
        if (d < kNope) {
          const uint2* src = reinterpret_cast<const uint2*>(row + d);
          vu[j][0] = src[0], vu[j][1] = src[1];
          vsc[j] = ue8m0_f32(row[576 + d / 64]);
        } else {
          const uint2* src = reinterpret_cast<const uint2*>(row + kNope + 2 * (d - kNope));
          vu[j][0] = src[0], vu[j][1] = src[1], vu[j][2] = src[2], vu[j][3] = src[3];
        }
      }
    }
    if (tid < kLH) S.m[tid] = -INFINITY, S.l[tid] = 0.f;
    acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
    __syncthreads();  // Q ready
    // ---- S = Q·Kᵀ: chain (n-tile n, m-tile a) on CTA n % 8, warp 2·(n / 8) + a; scores stored to all 8 CTAs --------
    {
      const int a = warp & 1, n = (warp >> 1) * kCl + rank;
      float sacc[4] = {0.f, 0.f, 0.f, 0.f};
      const bool act = warp < 8 && n < 4 * ntl;  // n-tiles past the chunk's keys (≥ nnt) store −inf
      if (act) {
        const int key = kc0 + 8 * n + g;
        p041::KRaw kr;
        kr.load(key < kc1 ? key_row(key, ncomp, nwin, p, mt, rows, c) : nullptr, t4);
        float sc1[1][4] = {{0.f, 0.f, 0.f, 0.f}};
        p041::qk_chain<1>(S.qv.q, a, kr, lane, sc1);
#pragma unroll
        for (int i = 0; i < 4; i++) sacc[i] = sc1[0][i];
      }
      // the chain's scores → the row's exchange buffer in L2; later chunks: first every peer must be done reading
      // the previous chunk's
      if (!first) cl_wait();
      first = false;
      if (act) {
        const int col = 8 * n + 2 * t4;
        const bool ok0 = kc0 + col < kc1, ok1 = kc0 + col + 1 < kc1;
        __stcg(reinterpret_cast<float2*>(&xs[(16 * a + g) * kCh + col]),
               make_float2(ok0 ? sacc[0] * scale : -INFINITY, ok1 ? sacc[1] * scale : -INFINITY));
        __stcg(reinterpret_cast<float2*>(&xs[(16 * a + g + 8) * kCh + col]),
               make_float2(ok0 ? sacc[2] * scale : -INFINITY, ok1 ? sacc[3] * scale : -INFINITY));
      }
    }
    cl_arrive();
    cl_wait();  // every CTA's scores of the chunk are in L2; every CTA is done with its Q
    {  // the chunk's scores (32 heads × 32·ntl keys) → shared, L2 only
      const int nq = 8 * ntl;  // 16-B pieces per head row
      float4 v[4];
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const int idx = tid + j * kTh, h = idx >> 6, c4 = idx & 63;
        if (c4 < nq) v[j] = __ldcg(reinterpret_cast<const float4*>(&xs[h * kCh + 4 * c4]));
      }
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const int idx = tid + j * kTh, h = idx >> 6, c4 = idx & 63;
        if (c4 < nq) *reinterpret_cast<float4*>(&S.s[h][4 * c4]) = v[j];
      }
    }
    cl_arrive();  // done reading the exchange buffer (waited for before the next chunk's stores)
    // V (registers) → shared over Q
#pragma unroll
    for (int j = 0; j < 2; j++) {
      const int job = tid + j * kTh, kk = job >> 2, seg = job & 3, d = d0 + 16 * seg;
      uint4* dst = reinterpret_cast<uint4*>(&S.qv.v[kk][16 * seg]);
      if (kc0 + kk >= kc1) {
        dst[0] = dst[1] = make_uint4(0, 0, 0, 0);
      } else if (d < kNope) {
        const float s = vsc[j];
        float v[16];
#pragma unroll
        for (int hh = 0; hh < 2; hh++)
#pragma unroll
          for (int i = 0; i < 4; i++) {
            __nv_fp8_e4m3 x, y;
            x.__x = (vu[j][hh].x >> (8 * i)) & 0xFF, y.__x = (vu[j][hh].y >> (8 * i)) & 0xFF;
            v[8 * hh + i] = float(x) * s, v[8 * hh + 4 + i] = float(y) * s;
          }
        dst[0] = make_uint4(pack_bf16x2(v[0], v[1]), pack_bf16x2(v[2], v[3]), pack_bf16x2(v[4], v[5]),
                            pack_bf16x2(v[6], v[7]));
        dst[1] = make_uint4(pack_bf16x2(v[8], v[9]), pack_bf16x2(v[10], v[11]), pack_bf16x2(v[12], v[13]),
                            pack_bf16x2(v[14], v[15]));
      } else {
        dst[0] = make_uint4(vu[j][0].x, vu[j][0].y, vu[j][1].x, vu[j][1].y);
        dst[1] = make_uint4(vu[j][2].x, vu[j][2].y, vu[j][3].x, vu[j][3].y);
      }
    }
    __syncthreads();  // scores and V in shared memory
    // ---- softmax statistics, 4 tiles at a time (in order: the running max / sum carry over) ----------------------
    softmax4(S, 0, min(ntl, 4), warp, lane);
    if (ntl > 4) softmax4(S, kBK, ntl - 4, warp, lane);
    __syncthreads();
    // ---- O = O·fac + P·V: warp → m-tile warp & 1, dims 8·(warp >> 1) … of the slice ---------------------------------
    {
      const int a = warp & 1, b = warp >> 1;
      const float* fr0 = S.fac[16 * a + g];
      const float* fr1 = S.fac[16 * a + g + 8];
#pragma unroll 1
      for (int tt = 0; tt < ntl; tt++) {
        const float f0 = fr0[tt], f1 = fr1[tt];
        acc[0] *= f0, acc[1] *= f0, acc[2] *= f1, acc[3] *= f1;
#pragma unroll
        for (int ks = 0; ks < 2; ks++) {
          uint32_t a0, a1, a2, a3, b0, b1;
          ldsm4(a0, a1, a2, a3, &S.p[16 * a + (lane & 15)][32 * tt + 16 * ks + (lane >> 4) * 8]);
          ldsm2t(b0, b1, &S.qv.v[32 * tt + 16 * ks + (lane & 15)][8 * b]);
          mma_bf16(acc, a0, a1, a2, a3, b0, b1);
        }
      }
    }
    __syncthreads();
    if (ch + 1 < nch) {  // an earlier chunk's partial → this CTA's workspace slot
      float* w = wsl + (size_t)ch * (2 * kLH + kLH * kDS);
      if (tid < kLH) w[tid] = S.m[tid], w[kLH + tid] = S.l[tid];
      const int a = warp & 1, b = warp >> 1, h0 = 16 * a + g, d = 8 * b + 2 * t4;
      float* wo = w + 2 * kLH;
      *reinterpret_cast<float2*>(&wo[h0 * kDS + d]) = make_float2(acc[0], acc[1]);
      *reinterpret_cast<float2*>(&wo[(h0 + 8) * kDS + d]) = make_float2(acc[2], acc[3]);
    }
  }
  // ---- merge (k_merge) ---------------------------------------------------------------------------------------------
  __syncthreads();
  if (tid < kLH) {
    const int h = tid;
    const size_t cs = 2 * kLH + kLH * kDS;
    float Mx = sink[h];
    for (int cc = 0; cc + 1 < nch; cc++) Mx = fmaxf(Mx, wsl[cc * cs + h]);
    Mx = fmaxf(Mx, S.m[h]);
    float den = 0.f;
    for (int cc = 0; cc + 1 < nch; cc++) den += expf(wsl[cc * cs + h] - Mx) * wsl[cc * cs + kLH + h];
    den += expf(S.m[h] - Mx) * S.l[h];
    den += expf(sink[h] - Mx);
    S.M[h] = Mx, S.den[h] = den, S.flast[h] = expf(S.m[h] - Mx);
  }
  __syncthreads();
  bf16* ob = &S.qv.q[0][0];  // merged output [32][kQS] over the V region (V is done)
  {
    const int a = warp & 1, b = warp >> 1, d = 8 * b + 2 * t4;
    float x[4] = {0.f, 0.f, 0.f, 0.f};
    for (int cc = 0; cc + 1 < nch; cc++) {
      const float* w = wsl + (size_t)cc * (2 * kLH + kLH * kDS);
      const int h = 16 * a + g;
      const float2 v0 = *reinterpret_cast<const float2*>(&w[2 * kLH + h * kDS + d]);
      const float2 v1 = *reinterpret_cast<const float2*>(&w[2 * kLH + (h + 8) * kDS + d]);
      const float f0 = expf(w[h] - S.M[h]), f1 = expf(w[h + 8] - S.M[h + 8]);
      x[0] += f0 * v0.x, x[1] += f0 * v0.y, x[2] += f1 * v1.x, x[3] += f1 * v1.y;
    }
#pragma unroll
    for (int hh = 0; hh < 2; hh++) {
      const int h = 16 * a + g + 8 * hh;
      const float f = S.flast[h];
      const float xx = x[2 * hh] + f * acc[2 * hh], yy = x[2 * hh + 1] + f * acc[2 * hh + 1];
      *reinterpret_cast<uint32_t*>(&ob[h * kQS + d]) = pack_bf16x2(__fdiv_rn(xx, S.den[h]), __fdiv_rn(yy, S.den[h]));
    }
  }
  __syncthreads();
  for (int i = tid; i < kLH * kDS / 8; i += kTh) {
    const int h = i / (kDS / 8), c8 = i % (kDS / 8);
    *reinterpret_cast<uint4*>(o + ((size_t)r * kLH + h) * kD + d0 + 8 * c8) =
        *reinterpret_cast<const uint4*>(&ob[h * kQS + 8 * c8]);
  }
  if (VHA) {  // zp[r][rk][d] = Σ_h U[h][rk] · o[h][d]: 64 dims × 16 ranks, two ranks per thread
    const int dd = tid % kDS, rk0 = (tid / kDS) * 2;
    float s0 = 0.f, s1 = 0.f;
#pragma unroll 8
    for (int h = 0; h < kLH; h++) {
      const float x = __bfloat162float(ob[h * kQS + dd]);
      s0 += S.U[h * kVhaRank + rk0] * x;
      s1 += S.U[h * kVhaRank + rk0 + 1] * x;
    }
    zp[((size_t)r * kVhaRank + rk0) * kD + d0 + dd] = s0;
    zp[((size_t)r * kVhaRank + rk0 + 1) * kD + d0 + dd] = s1;
  }
  __syncthreads();
  cl_wait();  // pairs the last arrive
}

template <bool PREP, bool VHA>
void launch(const bf16* q, const uint8_t* rows, const RowMeta* meta, HcaCache c, const float* sink, const float* inv,
            const float* U, int M, int max_keys, bf16* o, float* zp, float* ws, size_t ws_bytes, cudaStream_t s) {
  static bool attr = false;
  if (!attr) {
    CUDA_CHECK(cudaFuncSetAttribute(k_dec<PREP, VHA>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(sizeof(Smem))));
    attr = true;
  }
  const int nch = std::max(1, (max_keys + kCh - 1) / kCh);
  const size_t slot = size_t(nch - 1) * (2 * kLH + kLH * kDS);
  const size_t need = (size_t(M) * kLH * kCh + slot * kCl * M) * 4;
  if (need > ws_bytes) qlog::fail("hca", "attention workspace %zu B < %zu B", ws_bytes, need);
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = kCl, at[0].val.clusterDim.y = 1, at[0].val.clusterDim.z = 1;
  cudaLaunchConfig_t cfg{dim3(kCl, M), dim3(kTh), sizeof(Smem), s, at, 1};
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, k_dec<PREP, VHA>, q, rows, meta, c, sink, inv, U, o, zp, ws, slot));
}

}  // namespace p043
}  // namespace e55
