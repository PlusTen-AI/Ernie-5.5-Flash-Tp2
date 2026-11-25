// p048_bmm.cuh — P048 dsa_bmm_dec: the per-head W_UK / W_UV GEMMs of dsa_bmm for few rows (M ≤ 16; base: dsa.cu
// k_bmm<16>), selected by patch::on(dsa_bmm_dec). Included by dsa.cu inside its anonymous namespace.
//
// Bitwise contract — every output element is the base's: the same m16n8k16 bf16 MMA with the same operands (A = the
// rows of x, rows ≥ M zero; B = W[h][n][k] columns), the k16 steps chained in ascending order from 0, bf16 out. Only
// the data path differs: decode reads every weight once from DRAM, so one warp owns 8 output columns of one head and
// loads all of its weight rows (8 × K) into registers up front with coalesced 16-byte loads — the whole weight set in
// flight at once — instead of k chunks of 64 staged through shared memory one after another.
#pragma once

namespace p048 {

// grid: H·N/64 CTAs × 8 warps (warp: 8 columns n0 …); K = KC·64. The warp's 8 weight rows arrive with coalesced
// 16-byte loads (all in flight), then pass through a 1.1 KB per-warp staging tile in 64-wide chunks into ldmatrix.
template <int KC>
__global__ void __launch_bounds__(256) k_bmm(const bf16* __restrict__ x, long xm, long xh, const bf16* __restrict__ w,
                                             int N, int M, bf16* y, long ym, long yh) {
  __shared__ __align__(16) bf16 buf[8][8][72];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int tiles = N / 8, wt = blockIdx.x * 8 + warp, h = wt / tiles, n0 = (wt % tiles) * 8;
  constexpr int K = KC * 64;
  const bf16* wb = w + ((size_t)h * N + n0) * K;
  // register 2c + j: row 4j + lane / 8, k 64c + 8·(lane % 8) …
  uint4 wr[KC][2];
#pragma unroll
  for (int c = 0; c < KC; c++)
#pragma unroll
    for (int j = 0; j < 2; j++)
      wr[c][j] = __ldg(reinterpret_cast<const uint4*>(wb + (size_t)(4 * j + (lane >> 3)) * K + 64 * c + 8 * (lane & 7)));
  const bool r0 = g < M, r1 = g + 8 < M;
  const bf16* x0 = x + g * xm + h * xh + 2 * t;
  const bf16* x1 = x + (g + 8) * xm + h * xh + 2 * t;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  bf16(*bw)[72] = buf[warp];
#pragma unroll
  for (int c = 0; c < KC; c++) {
#pragma unroll
    for (int j = 0; j < 2; j++) *reinterpret_cast<uint4*>(&bw[4 * j + (lane >> 3)][8 * (lane & 7)]) = wr[c][j];
    __syncwarp();
#pragma unroll
    for (int kq = 0; kq < 4; kq++) {
      const int kk = 4 * c + kq;
      uint32_t b0, b1;
      ldsm_x2(b0, b1, &bw[lane & 7][16 * kq + ((lane >> 3) & 1) * 8]);
      const uint32_t a0 = r0 ? *reinterpret_cast<const uint32_t*>(x0 + 16 * kk) : 0u;
      const uint32_t a1 = r1 ? *reinterpret_cast<const uint32_t*>(x1 + 16 * kk) : 0u;
      const uint32_t a2 = r0 ? *reinterpret_cast<const uint32_t*>(x0 + 16 * kk + 8) : 0u;
      const uint32_t a3 = r1 ? *reinterpret_cast<const uint32_t*>(x1 + 16 * kk + 8) : 0u;
      mma_bf16(acc, a0, a1, a2, a3, b0, b1);
    }
    __syncwarp();
  }
  const int n = n0 + 2 * t;
  if (r0) *reinterpret_cast<uint32_t*>(y + g * ym + h * yh + n) = pack_bf16x2(acc[0], acc[1]);
  if (r1) *reinterpret_cast<uint32_t*>(y + (g + 8) * ym + h * yh + n) = pack_bf16x2(acc[2], acc[3]);
}

// false: shape not covered (the base runs)
inline bool run(const bf16* x, long xm, long xh, const bf16* w, int H, int N, int K, int M, bf16* y, long ym, long yh,
                cudaStream_t s) {
  if (M > 16 || (H * N / 8) % 8) return false;
  const int grid = H * N / 64;
  if (K == 192) k_bmm<3><<<grid, 256, 0, s>>>(x, xm, xh, w, N, M, y, ym, yh);
  else if (K == 512) k_bmm<8><<<grid, 256, 0, s>>>(x, xm, xh, w, N, M, y, ym, yh);
  else return false;
  CUDA_CHECK(cudaGetLastError());
  return true;
}

}  // namespace p048
