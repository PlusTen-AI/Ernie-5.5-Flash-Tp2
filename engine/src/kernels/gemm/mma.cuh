// mma.cuh — tensor-core wrappers and number-format helpers for GB10 (sm_121a), shared by the GEMM, MoE and attention
// kernels. The block-scaled operand layouts below are the hardware's (mma.sync m16n8k32 kind::mxf8f6f4, scale_vec::1X,
// selectors 0), as measured on GB10 by DeepSeekV4.1-Flash-Pico's micro/mma.cu (MIT) and checked again by our tests.
//
// m16n8k32 8-bit operands, lane (g = lane>>2, t = lane&3):
//   A (16 × 32, row-major bytes): a0 = A[g][4t…4t+3], a1 = A[g+8][4t…], a2 = A[g][16+4t…], a3 = A[g+8][16+4t…]
//   B (32 × 8, column j = 32 bytes along k): b0 = B[col g][4t…4t+3], b1 = B[col g][16+4t…]
//   D (16 × 8 fp32): d0, d1 = D[g][2t, 2t+1]; d2, d3 = D[g+8][2t, 2t+1]
//   sfa = UE8M0 scale of A row g + 8·(lane & 1), sfb = that of B column g (each per 32 k)
//   e2m1 operands (kind::mxf8f6f4): one value per byte, in bits 5:2
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace e55 {

__device__ __forceinline__ void mma_e4m3_e4m3(float d[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                              uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0),
        "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
}

// A in e2m1 (weights of the W4A8 experts), B in e4m3 (MXFP8 activations)
__device__ __forceinline__ void mma_e2m1_e4m3(float d[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                              uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e4m3.f32.ue8m0 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0),
        "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
}

// both operands e2m1 (the DSA indexer: MXFP4 query × MXFP4 keys)
__device__ __forceinline__ void mma_e2m1_e2m1(float d[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                              uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e2m1.f32.ue8m0 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0),
        "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
}

// D += A(16×16 bf16) · B(16×8 bf16): a0 = A[g][2t, 2t+1], a1 = A[g+8][…], a2 = A[g][2t+8, 2t+9], a3 = A[g+8][…];
// b0 = B[col g][2t, 2t+1], b1 = B[col g][2t+8, 2t+9]
__device__ __forceinline__ void mma_bf16(float d[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// four packed e2m1 codes (16 bits, low nibble = lower k) → four bytes, code in bits 5:2
__device__ __forceinline__ uint32_t e2m1x4_expand(uint32_t v16) {
  const uint32_t p = __byte_perm(v16, 0, 0x1100);  // b0 b0 b1 b1
  return ((p << 2) & 0x003C003Cu) | ((p >> 2) & 0x3C003C00u);
}

// UE8M0 byte of 2^ceil(log2(max(amax, eps) / 448)) — IEEE division as the reference, exact ceil from the bits
__device__ __forceinline__ uint32_t act_scale_e8(float amax, float eps) {
  const uint32_t b = __float_as_uint(fmaxf(amax, eps) / 448.0f);
  return ((b >> 23) & 0xFF) + ((b & 0x7FFFFF) != 0);
}

// x / 2^(e−127) to e4m3 (round to nearest even, saturating at ±448); 1 ≤ e ≤ 253
__device__ __forceinline__ uint32_t quant_e4m3(float x, uint32_t e) {
  const float q = fminf(fmaxf(x * __uint_as_float((254u - e) << 23), -448.0f), 448.0f);
  return (uint32_t)__nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ float ue8m0_f32(uint32_t e) { return __uint_as_float(e << 23); }
__device__ __forceinline__ float bf16_lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t v) { return __uint_as_float(v & 0xFFFF0000u); }
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}

// cp.async 16 B global → shared; ok == false fills zeros
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool ok) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

}  // namespace e55
