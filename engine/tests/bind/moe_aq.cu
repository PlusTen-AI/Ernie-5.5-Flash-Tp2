// moe_aq.cu — test-only: the engine's act_quant (kernels/gemm/fp8.cu k_act_quant, copied verbatim) so the MoE test
// can quantize the latent exactly as model.cc does without building the GEMM family.
#include <cuda_runtime.h>

#include "core/common.h"
#include "kernels/gemm/mma.cuh"

namespace e55 {
namespace {
// one thread per 4 consecutive elements; a group of G/4 threads (a whole warp for G = 128) shares the scale
__global__ void k_act_quant(const bf16* x, int ldx, int M, int k, int G, uint8_t* q, uint8_t* sc, int ld) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int q4 = k >> 2;
  const bool ok = i < (size_t)M * q4;
  const int row = ok ? int(i / q4) : 0, c = ok ? int(i % q4) * 4 : 0;
  float v[4] = {0.f, 0.f, 0.f, 0.f};
  if (ok) {
    const uint2 u = *reinterpret_cast<const uint2*>(x + (size_t)row * ldx + c);
    v[0] = bf16_lo(u.x), v[1] = bf16_hi(u.x), v[2] = bf16_lo(u.y), v[3] = bf16_hi(u.y);
  }
  float m = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
  for (int o = (G >> 3); o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  const uint32_t e = act_scale_e8(m, 1e-10f);
  if (!ok) return;
  uint32_t out = 0;
#pragma unroll
  for (int j = 0; j < 4; j++) out |= quant_e4m3(v[j], e) << (8 * j);
  *reinterpret_cast<uint32_t*>(q + (size_t)row * ld + c) = out;
  if ((c % G) == 0) sc[(size_t)row * (ld / G) + c / G] = uint8_t(e);
}
}  // namespace
void test_act_quant32(const bf16* x, int M, uint8_t* q, uint8_t* s, cudaStream_t st) {
  const size_t thr = (size_t)M * (2048 / 4);
  k_act_quant<<<unsigned((thr + 255) / 256), 256, 0, st>>>(x, 2048, M, 2048, 32, q, s, 2048);
}
}  // namespace e55

