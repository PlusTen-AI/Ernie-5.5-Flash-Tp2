// p047_vha.cuh — P047 dsa_vha_fast: dsa_vha_gate (base: dsa.cu k_vha_gate), selected by patch::on(dsa_vha_fast).
// Included by dsa.cu inside its anonymous namespace.
//
// Bitwise contract — every output element is the base's: a = Σ_r V[h][r]·z[r][d] as the same fma chain over r = 0 …
// 15 from 0, o' = bf16(o + a), y = bf16(o'·gate). Only scheduling differs: the base's grid (rows fastest, 32 head blocks
// per row far apart) re-reads a row's 16 KB z from DRAM for every head; here a CTA takes one row (or, for few rows, a
// slice of its heads), keeps z[·][d] in registers and V in shared memory, and runs over its heads. Few rows (decode,
// M ≤ 16) keep the base kernel.
#pragma once

namespace p047 {

// grid: M rows × HS head slices (32 / HS heads each); 256 threads = dims
__global__ void __launch_bounds__(256) k_vha_gate(const bf16* __restrict__ o, const float* __restrict__ z,
                                                  const float* __restrict__ V, const bf16* __restrict__ gate, bf16* y,
                                                  int HS) {
  __shared__ float Vs[kLH * kVhaRank];
  const int m = blockIdx.x / HS, hs = blockIdx.x % HS, d = threadIdx.x;
  const int nh = kLH / HS, h0 = hs * nh;
  for (int i = d; i < nh * kVhaRank; i += 256) Vs[i] = V[h0 * kVhaRank + i];
  float zr[kVhaRank];
#pragma unroll
  for (int r = 0; r < kVhaRank; r++) zr[r] = z[((size_t)m * kVhaRank + r) * kMlaV + d];
  __syncthreads();
#pragma unroll 4
  for (int hh = 0; hh < nh; hh++) {
    const int h = h0 + hh;
    float a = 0.f;
#pragma unroll
    for (int r = 0; r < kVhaRank; r++) a += Vs[hh * kVhaRank + r] * zr[r];
    const size_t i = ((size_t)m * kLH + h) * kMlaV + d;
    const float v = __bfloat162float(__float2bfloat16_rn(__bfloat162float(o[i]) + a));
    y[i] = __float2bfloat16_rn(v * __bfloat162float(gate[i]));
  }
}

inline void run(const bf16* o, const float* z, const float* V, const bf16* gate, int M, bf16* y, cudaStream_t s) {
  const int HS = M < 48 ? 8 : 1;
  k_vha_gate<<<M * HS, 256, 0, s>>>(o, z, V, gate, y, HS);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace p047
