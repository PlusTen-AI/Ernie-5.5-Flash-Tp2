// p073_bmm_f8.cuh — P073 dsa_bmm_f8: decode W_UK / W_UV (dsa_bmm, M ≤ 16) reading kv_b_proj's own FP8 bytes (e4m3 +
// its 128×128 block scales, half the bytes of the bf16 copy) instead of the bf16 copy. Included by dsa.cu inside its
// anonymous namespace after p048_bmm.cuh.
//
// Bitwise contract — P048's (= the base's): the bf16 B operand is the same value the host's copy holds (float(e4m3) ·
// scale, exact in bf16 — the scales are powers of two; the host copy truncates, RN here, equal when exact), the same
// m16n8k16 chain over the k16 steps in ascending order, A = the rows of x (rows ≥ M zero), bf16 out. Layout as P048:
// a warp owns 8 output columns of one head, loads all of its weight bytes up front (coalesced 8-byte loads, all in
// flight), converts each 8-element run with its block scale into a per-warp staging tile, then ldmatrix.
// w8 [H][N][K] e4m3: W_UV = kv_b_proj rows 448h + 192 + n as stored; W_UK = the transposed bytes ([h][k_lat][i] =
// kv_b_proj[448h + i][k_lat]). sc: kv_b_proj.weight_scale_inv of this rank's rows, fp32 [rows / 128][4].
// Launched as a programmatic dependent: the weight bytes and scales (constant) are requested before
// griddepcontrol.wait, x is read and y written after it.
#pragma once

namespace p073 {

enum Mode { kUK = 0, kUV = 1 };

__device__ __forceinline__ uint32_t f8x2_bf16x2(uint32_t two, float s) {  // P041's dequantization of two codes
  __nv_fp8_e4m3 a, b;
  a.__x = two & 0xFF, b.__x = (two >> 8) & 0xFF;
  return pack_bf16x2(float(a) * s, float(b) * s);
}

template <int KC, int MODE>
__global__ void __launch_bounds__(256) k_bmm(const bf16* __restrict__ x, long xm, long xh, const uint8_t* __restrict__ w,
                                             const float* __restrict__ sc, int N, int M, bf16* y, long ym, long yh) {
  __shared__ __align__(16) bf16 buf[8][8][72];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int tiles = N / 8, wt = blockIdx.x * 8 + warp, h = wt / tiles, n0 = (wt % tiles) * 8;
  constexpr int K = KC * 64;
  const uint8_t* wb = w + ((size_t)h * N + n0) * K;
  // register (c, j): row 4j + lane / 8, k 64c + 8·(lane % 8) … (as P048)
  uint2 wr[KC][2];
#pragma unroll
  for (int c = 0; c < KC; c++)
#pragma unroll
    for (int j = 0; j < 2; j++)
      wr[c][j] = __ldg(reinterpret_cast<const uint2*>(wb + (size_t)(4 * j + (lane >> 3)) * K + 64 * c + 8 * (lane & 7)));
  float scv[KC][2];
#pragma unroll
  for (int c = 0; c < KC; c++)
#pragma unroll
    for (int j = 0; j < 2; j++) {
      const int n = n0 + 4 * j + (lane >> 3), k = 64 * c + 8 * (lane & 7);
      scv[c][j] = MODE == kUV ? __ldg(sc + ((448 * h + 192 + n) >> 7) * 4 + (k >> 7))
                              : __ldg(sc + ((448 * h + k) >> 7) * 4 + (n >> 7));
    }
  asm volatile("griddepcontrol.wait;\n" ::: "memory");  // x comes from the previous kernels
  const bool r0 = g < M, r1 = g + 8 < M;
  const bf16* x0 = x + g * xm + h * xh + 2 * t;
  const bf16* x1 = x + (g + 8) * xm + h * xh + 2 * t;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  bf16(*bw)[72] = buf[warp];
#pragma unroll
  for (int c = 0; c < KC; c++) {
#pragma unroll
    for (int j = 0; j < 2; j++) {
      const float s = scv[c][j];
      const uint2 v = wr[c][j];
      *reinterpret_cast<uint4*>(&bw[4 * j + (lane >> 3)][8 * (lane & 7)]) =
          make_uint4(f8x2_bf16x2(v.x & 0xFFFF, s), f8x2_bf16x2(v.x >> 16, s), f8x2_bf16x2(v.y & 0xFFFF, s),
                     f8x2_bf16x2(v.y >> 16, s));
    }
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

// false: not covered (M > 16 or another shape) — the caller runs dsa_bmm on the bf16 copy
inline bool run(const bf16* x, long xm, long xh, const uint8_t* w8, const float* sc, int mode, int H, int N, int K,
                int M, bf16* y, long ym, long yh, cudaStream_t s) {
  if (M > 16 || (H * N / 8) % 8) return false;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(H * N / 64);
  cfg.blockDim = dim3(256);
  cfg.stream = s;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  if (mode == kUK && K == 192) CUDA_CHECK(cudaLaunchKernelEx(&cfg, k_bmm<3, kUK>, x, xm, xh, w8, sc, N, M, y, ym, yh));
  else if (mode == kUV && K == 512) CUDA_CHECK(cudaLaunchKernelEx(&cfg, k_bmm<8, kUV>, x, xm, xh, w8, sc, N, M, y, ym, yh));
  else return false;
  return true;
}

}  // namespace p073
