// attn_common.cuh — device helpers of the attention kernels: bf16 vectors, RoPE pairs, the 584-byte HCA KV row,
// ldmatrix.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "core/common.h"
#include "kernels/gemm/mma.cuh"

namespace e55 {

__device__ __forceinline__ void load_bf16x8(const bf16* p, float v[8]) {
  const uint4 u = *reinterpret_cast<const uint4*>(p);
  const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
  for (int i = 0; i < 4; i++) v[2 * i] = bf16_lo(w[i]), v[2 * i + 1] = bf16_hi(w[i]);
}
__device__ __forceinline__ void store_bf16x8(bf16* p, const float v[8]) {
  *reinterpret_cast<uint4*>(p) =
      make_uint4(pack_bf16x2(v[0], v[1]), pack_bf16x2(v[2], v[3]), pack_bf16x2(v[4], v[5]), pack_bf16x2(v[6], v[7]));
}

// interleaved RoPE on 8 consecutive dims of the rope part (offset r0 = 0, 8, … 56 inside the 64 rope dims) at pos:
// pair i: angle = fl(pos · inv[i]), (e, o) → (e·c − o·s, o·c + e·s); inverse: s → −s. No fused multiply-add (the
// reference's separate products).
__device__ __forceinline__ void rope_pairs8(float v[8], int r0, int pos, const float* inv, bool inverse) {
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const float a = __fmul_rn(float(pos), inv[r0 / 2 + j]);
    float sn, cs;
    sincosf(a, &sn, &cs);
    if (inverse) sn = -sn;
    const float e = v[2 * j], o = v[2 * j + 1];
    v[2 * j] = __fsub_rn(__fmul_rn(e, cs), __fmul_rn(o, sn));
    v[2 * j + 1] = __fadd_rn(__fmul_rn(o, cs), __fmul_rn(e, sn));
  }
}

// NeoX RoPE on the leading 64 dims (indexer): pair (i, i+32); thread holds dims d0 … d0+7 of the first half and
// d0+32 … of the second half
__device__ __forceinline__ void rope_neox_pair(float& a, float& b, int i, int pos, const float* inv) {
  const float ang = __fmul_rn(float(pos), inv[i]);
  float sn, cs;
  sincosf(ang, &sn, &cs);
  const float x = a, y = b;
  a = __fsub_rn(__fmul_rn(x, cs), __fmul_rn(y, sn));
  b = __fadd_rn(__fmul_rn(y, cs), __fmul_rn(x, sn));
}

// one of the 64 threads that write a 584-byte KV row: dims d0 … d0+7 (fp32, after RoPE). Values are rounded to bf16;
// nope groups of 64 (8 threads) get an e4m3 code each and one UE8M0 scale 2^ceil(log2(max(amax, 1e-4)/448)); the
// rope dims stay bf16. Layout [448 e4m3][64 bf16][7 UE8M0][pad]. All 64 threads must call it (shuffles). A row is
// only 8-byte aligned (584 = 8 · 73): no 16-byte accesses on it.
__device__ __forceinline__ void store_kv_row8(uint8_t* row, int d0, const float in[8]) {
  float v[8], amax = 0.f;
#pragma unroll
  for (int j = 0; j < 8; j++) v[j] = __bfloat162float(__float2bfloat16_rn(in[j])), amax = fmaxf(amax, fabsf(v[j]));
#pragma unroll
  for (int off = 4; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
  if (d0 < kNope) {
    const uint32_t e = act_scale_e8(amax, 1e-4f);
    uint32_t lo = 0, hi = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) lo |= quant_e4m3(v[j], e) << (8 * j), hi |= quant_e4m3(v[4 + j], e) << (8 * j);
    *reinterpret_cast<uint2*>(row + d0) = make_uint2(lo, hi);
    if ((d0 & 63) == 0) row[576 + d0 / 64] = uint8_t(e);
    if (d0 == 0) row[583] = 0;
  } else {  // rows are 584 B apart: only 8-byte aligned
    uint2* dst = reinterpret_cast<uint2*>(row + kNope + 2 * (d0 - kNope));
    dst[0] = make_uint2(pack_bf16x2(v[0], v[1]), pack_bf16x2(v[2], v[3]));
    dst[1] = make_uint2(pack_bf16x2(v[4], v[5]), pack_bf16x2(v[6], v[7]));
  }
}

// part j < 7: nope dims 64j … +63 dequantized (exact in bf16); part 7: the 64 rope dims
__device__ __forceinline__ void dequant_kv_part(const uint8_t* row, int part, bf16* dst) {
  if (part < 7) {
    const float s = ue8m0_f32(row[576 + part]);
    const uint8_t* src = row + part * 64;
#pragma unroll
    for (int j = 0; j < 64; j += 8) {
      const uint2 u = *reinterpret_cast<const uint2*>(src + j);
      float v[8];
#pragma unroll
      for (int i = 0; i < 4; i++) {
        __nv_fp8_e4m3 a, b;
        a.__x = (u.x >> (8 * i)) & 0xFF, b.__x = (u.y >> (8 * i)) & 0xFF;
        v[i] = float(a) * s, v[4 + i] = float(b) * s;
      }
      store_bf16x8(dst + j, v);
    }
  } else {  // 8-byte loads: rows are 584 B apart
    const uint2* src = reinterpret_cast<const uint2*>(row + kNope);
#pragma unroll
    for (int j = 0; j < 16; j++) reinterpret_cast<uint2*>(dst)[j] = src[j];
  }
}

__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(a));
}
__device__ __forceinline__ void ldsm_x2(uint32_t& r0, uint32_t& r1, const void* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(a));
}
__device__ __forceinline__ void ldsm_x2_t(uint32_t& r0, uint32_t& r1, const void* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(a));
}

}  // namespace e55
