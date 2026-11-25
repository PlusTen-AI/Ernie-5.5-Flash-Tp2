// p089_hca_pre_pf.cuh — P089 hca_pre_pf: hca_pre at prefill sizes (M > 64). Included by hca.cu after
// p042_hca_fuse.cuh (uses P044's shared-memory layout and its 8-lane KV-row store).
//
// The base pre stage (P042 k_pre) gives every row a 512-thread CTA; the few rows that close a group (one per 128 rows)
// walk their group's 128 compressor rows three times with one load in flight per thread (0.07 / 0.12 ms at M = 1024 /
// 4096, almost all of it those walks). Here: 8-CTA clusters, cluster y owns rows [128y, 128y + 128): every CTA writes
// 16 of its KV rows (k_kv_rows), then the cluster finds the range's group-closing rows (meta, the same on all 8 CTAs) and
// runs P044's compressor on each — CTA x owns dims 64x …, stages the group's 128 scores / kv of its dims in shared
// memory (all loads in flight), keeps every per-dim chain of the base in order (max, Σ expf sequential, Σ (e / sum) · kv
// sequential), the 16 RMS warp sums meet in L2 behind one cluster barrier, then CTA x writes its 64 dims of the entry.
// Same bits as k_compress / k_kv_rows (P044's arithmetic, checked there against the base).
#pragma once

namespace e55 {
namespace p089 {

constexpr int kRG = 128;  // rows per cluster
constexpr int kMinM = 65;

__global__ void __launch_bounds__(512) k_pre(const bf16* __restrict__ kv, int ld, const float* __restrict__ ckv,
                                             const float* __restrict__ ape, const bf16* __restrict__ nw,
                                             const RowMeta* __restrict__ meta, const float* __restrict__ inv, int M,
                                             uint8_t* rows, HcaCache c, float* __restrict__ xs) {
  const int x = blockIdx.x, y = blockIdx.y, tid = threadIdx.x;
  const int rb = y * kRG;
  // ---- KV rows rb + 16x … rb + 16x + 15: 8 rows a pass, 64 threads a row (k_kv_rows)
#pragma unroll
  for (int ps = 0; ps < kRG / 8 / 8; ps++) {
    const int r = rb + (kRG / 8) * x + 8 * ps + (tid >> 6), d0 = 8 * (tid & 63);
    if (r < M) {
      const int p = meta[r].pos;
      float v[8];
      load_bf16x8(kv + (size_t)r * ld + d0, v);
      if (d0 >= kNope) rope_pairs8(v, d0 - kNope, p, inv, false);
      store_kv_row8(rows + (size_t)r * kKvRowBytes, d0, v);
    }
  }
  // ---- the range's group-closing rows (bit i: row rb + i)
  __shared__ uint32_t cl[kRG / 32];
  if (tid < kRG) {
    const int r = rb + tid;
    const bool close = r < M && (meta[r].pos + 1) % kRatio == 0;
    const uint32_t bm = __ballot_sync(0xffffffffu, close);
    if ((tid & 31) == 0) cl[tid >> 5] = bm;
  }
  __syncthreads();
  extern __shared__ __align__(16) uint8_t smraw[];
  p042::CompSmem& S = *reinterpret_cast<p042::CompSmem*>(smraw);
  __shared__ float pm[8][64];
  const int dx = 64 * x;
  for (int wi = 0; wi < kRG / 32; wi++) {
    uint32_t bits = cl[wi];
    while (bits) {
      const int r = rb + 32 * wi + __ffs(bits) - 1;
      bits &= bits - 1;
      const RowMeta mt = meta[r];
      const int p = mt.pos, g = p / kRatio, t0 = g * kRatio;
      const float* st = c.state + (size_t)mt.slot * kRatio * 1024;
      __syncthreads();  // the previous closing row is done with S
      // stage tokens t0 … p (all 128: the group closes at p) for dims dx …: 128 × 16 float4 per array, 4 per thread
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const int i = tid + 512 * j, tt = i >> 4, c4 = (i & 15) * 4, t = t0 + tt, d = dx + c4;
        float4 s4, k4;
        if (t >= mt.p0) {
          const float* src = ckv + (size_t)(mt.r0 + t - mt.p0) * 1024;
          const float4 a = *reinterpret_cast<const float4*>(src + 512 + d);
          const float4 e = *reinterpret_cast<const float4*>(ape + (t % kRatio) * 512 + d);
          s4 = make_float4(a.x + e.x, a.y + e.y, a.z + e.z, a.w + e.w);
          k4 = *reinterpret_cast<const float4*>(src + d);
        } else {
          s4 = *reinterpret_cast<const float4*>(st + (t % kRatio) * 1024 + 512 + d);
          k4 = *reinterpret_cast<const float4*>(st + (t % kRatio) * 1024 + d);
        }
        *reinterpret_cast<float4*>(&S.sc[tt][c4]) = s4;
        *reinterpret_cast<float4*>(&S.kv[tt][c4]) = k4;
      }
      __syncthreads();
      const int dd = tid & 63, q8 = tid >> 6;  // 8 threads per dim, 16 tokens each
      {  // max over the group (exact in any order)
        float m = -INFINITY;
#pragma unroll
        for (int k = 0; k < 16; k++) m = fmaxf(m, S.sc[16 * q8 + k][dd]);
        pm[q8][dd] = m;
        __syncthreads();
        if (tid < 64) {
          float mm = -INFINITY;
          for (int k = 0; k < 8; k++) mm = fmaxf(mm, pm[k][tid]);
          S.mx[tid] = mm;
        }
        __syncthreads();
      }
      {  // e = expf(score − max), all threads
        const float m = S.mx[dd];
#pragma unroll
        for (int k = 0; k < 16; k++) S.sc[16 * q8 + k][dd] = expf(S.sc[16 * q8 + k][dd] - m);
      }
      __syncthreads();
      if (tid < 64) {  // Σ e in token order (k_compress's sum)
        float sum = 0.f;
#pragma unroll 16
        for (int t = 0; t < kRatio; t++) sum += S.sc[t][tid];
        S.sum[tid] = sum;
      }
      __syncthreads();
      {
        const float sm = S.sum[dd];
#pragma unroll
        for (int k = 0; k < 16; k++) S.sc[16 * q8 + k][dd] = __fdiv_rn(S.sc[16 * q8 + k][dd], sm);
      }
      __syncthreads();
      if (tid < 64) {  // Σ (e / sum) · kv in token order, then the base warps' Σ cv² (lanes = dims of warp 2x + tid / 32)
        float cv = 0.f;
#pragma unroll 16
        for (int t = 0; t < kRatio; t++) cv += S.sc[t][tid] * S.kv[t][tid];
        S.cv[tid] = cv;
        const float ss = warp_sum(cv * cv);
        if ((tid & 31) == 0) __stcg(xs + (size_t)r * 16 + 2 * x + (tid >> 5), ss);
      }
      asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
      asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
      if (tid < 32) {  // k_compress's second level over the 16 warp sums, every CTA alike
        float v = tid < 16 ? __ldcg(xs + (size_t)r * 16 + tid) : 0.f;
        v = warp_sum(v);
        if (tid == 0) S.red[0] = v;
      }
      __syncthreads();
      if (tid < 8) {  // part x of the compressed row: dims dx + 8·tid …
        const float rs = rsqrtf(__fdiv_rn(S.red[0], 512.f) + kEps);
        const int d0 = dx + 8 * tid;
        float v[8];
#pragma unroll
        for (int j = 0; j < 8; j++) v[j] = S.cv[8 * tid + j] * rs * __bfloat162float(nw[d0 + j]);
        if (d0 >= kNope) rope_pairs8(v, d0 - kNope, t0, inv, false);
        p042::store_kv_row8_l8(c.comp + ((size_t)mt.slot * c.max_comp + g) * kKvRowBytes, d0, v);
      }
    }
  }
}

inline void launch(const bf16* kv, int ldkv, const float* ckv, const float* ape, const bf16* normw, const RowMeta* meta,
                   const float* inv, int M, uint8_t* rows, HcaCache c, float* ws, cudaStream_t s) {
  static bool attr = false;
  if (!attr) {
    CUDA_CHECK(cudaFuncSetAttribute(k_pre, cudaFuncAttributeMaxDynamicSharedMemorySize, int(sizeof(p042::CompSmem))));
    attr = true;
  }
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = 8, at[0].val.clusterDim.y = 1, at[0].val.clusterDim.z = 1;
  cudaLaunchConfig_t cfg{dim3(8, (M + kRG - 1) / kRG), dim3(512), sizeof(p042::CompSmem), s, at, 1};
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, k_pre, kv, ldkv, ckv, ape, normw, meta, inv, M, rows, c, ws));
}

}  // namespace p089
}  // namespace e55
