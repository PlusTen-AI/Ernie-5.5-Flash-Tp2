// p040_hca_vha_out.cuh — P040 hca_vha_row (included by hca.cu): hca_vha_out with one CTA per row instead of one per
// (row, head). The base re-reads the row's 32 KB of z for each of its 32 heads (4 GB per layer at M = 4096); here a
// thread keeps z[·][its 8 dims] in registers and runs all 32 heads. Arithmetic unchanged (same FMA chain over the 16
// ranks, bf16 rounding, inverse RoPE with the same angles, FP8 1×128 over the same 16 lanes) → bit-identical.
// With POST (P042 hca_post) the same CTA also stores the row's ring entry and compressor state (hca_ring_store, k_state).
#pragma once

namespace e55 {
namespace p040 {

__device__ __forceinline__ void ld8(const bf16* p, float v[8]) {
  const uint4 u = *reinterpret_cast<const uint4*>(p);
  const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
  for (int i = 0; i < 4; i++) v[2 * i] = bf16_lo(w[i]), v[2 * i + 1] = bf16_hi(w[i]);
}

// grid (M, 32 / HPB [+ 1 with POST]), 64 threads; CTA: heads HPB·y … of row x; thread i: dims 8i … 8i+7. HPB = 32 at
// large M (z read once per row), fewer at decode (more CTAs per row; V read straight from global). With POST the last
// CTA of a row stores its ring entry and compressor state.
template <bool POST, int HPB>
__global__ void __launch_bounds__(64) k_vha_out(const bf16* __restrict__ o, const float* __restrict__ z,
                                                const float* __restrict__ V, const RowMeta* __restrict__ meta,
                                                const float* __restrict__ inv, int M, uint8_t* __restrict__ q,
                                                uint8_t* __restrict__ sc, const uint8_t* __restrict__ rows,
                                                const float* __restrict__ ckv, const float* __restrict__ ape,
                                                HcaCache c) {
  constexpr int kLH = kHeads / 2;
  constexpr bool kSmV = HPB == kLH;
  __shared__ float Vs[kSmV ? kLH * kVhaRank : 1];
  const int m = blockIdx.x, i = threadIdx.x, d0 = 8 * i;
  const RowMeta mt = meta[m];
  if (POST && blockIdx.y == kLH / HPB) {
    if (mt.pos > mt.pend - kWindow) {  // k_ring
      const uint2* src = reinterpret_cast<const uint2*>(rows + (size_t)m * kKvRowBytes);
      uint2* dst = reinterpret_cast<uint2*>(c.ring + ((size_t)mt.slot * kWindow + mt.pos % kWindow) * kKvRowBytes);
      for (int k = i; k < kKvRowBytes / 8; k += 64) dst[k] = src[k];
    }
    if (mt.pos >= (mt.pend + 1) / kRatio * kRatio) {  // k_state
      float* st = c.state + ((size_t)mt.slot * kRatio + mt.pos % kRatio) * 1024;
      const float* src = ckv + (size_t)m * 1024;
      for (int d = i; d < 1024; d += 64) st[d] = d < 512 ? src[d] : src[d] + ape[(mt.pos % kRatio) * 512 + d - 512];
    }
    return;
  }
  if (kSmV)
    for (int k = i; k < kLH * kVhaRank; k += 64) Vs[k] = V[k];
  float zr[kVhaRank][8];
#pragma unroll
  for (int r = 0; r < kVhaRank; r++) {
    const float4* zp = reinterpret_cast<const float4*>(z + ((size_t)m * kVhaRank + r) * kHeadDim + d0);
    const float4 a = zp[0], b = zp[1];
    zr[r][0] = a.x, zr[r][1] = a.y, zr[r][2] = a.z, zr[r][3] = a.w;
    zr[r][4] = b.x, zr[r][5] = b.y, zr[r][6] = b.z, zr[r][7] = b.w;
  }
  float cs[4] = {1.f, 1.f, 1.f, 1.f}, sn[4] = {0.f, 0.f, 0.f, 0.f};
  if (d0 >= kNope) {  // the base's angles: fl(pos · inv), sincosf, inverse → −sin
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const float a = __fmul_rn(float(mt.pos), inv[(d0 - kNope) / 2 + j]);
      sincosf(a, &sn[j], &cs[j]);
      sn[j] = -sn[j];
    }
  }
  if (kSmV) __syncthreads();
#pragma unroll 1
  for (int h = blockIdx.y * HPB; h < (blockIdx.y + 1) * HPB; h++) {
    float v[8], vr[kVhaRank];
    ld8(o + ((size_t)m * kLH + h) * kHeadDim + d0, v);
#pragma unroll
    for (int r = 0; r < kVhaRank; r++) vr[r] = kSmV ? Vs[h * kVhaRank + r] : __ldg(V + h * kVhaRank + r);
#pragma unroll
    for (int j = 0; j < 8; j++) {
      float a = 0.f;
#pragma unroll
      for (int r = 0; r < kVhaRank; r++) a += vr[r] * zr[r][j];
      v[j] = __bfloat162float(__float2bfloat16_rn(v[j] + a));
    }
    if (d0 >= kNope) {
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const float e = v[2 * j], od = v[2 * j + 1];
        v[2 * j] = __fsub_rn(__fmul_rn(e, cs[j]), __fmul_rn(od, sn[j]));
        v[2 * j + 1] = __fadd_rn(__fmul_rn(od, cs[j]), __fmul_rn(e, sn[j]));
      }
    }
    float amax = 0.f;
#pragma unroll
    for (int j = 0; j < 8; j++) amax = fmaxf(amax, fabsf(v[j]));
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    const uint32_t e = act_scale_e8(amax, 1e-10f);
    const int gi = h >> 3, col = (h & 7) * kHeadDim + d0;
    uint32_t lo = 0, hi = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) lo |= quant_e4m3(v[j], e) << (8 * j), hi |= quant_e4m3(v[4 + j], e) << (8 * j);
    *reinterpret_cast<uint2*>(q + ((size_t)gi * M + m) * 4096 + col) = make_uint2(lo, hi);
    if ((d0 & 127) == 0) sc[((size_t)gi * M + m) * 32 + col / 128] = uint8_t(e);
  }
}

template <bool POST>
void launch(const bf16* o, const float* z, const float* V, const RowMeta* meta, const float* inv, int M, uint8_t* q,
            uint8_t* sc, const uint8_t* rows, const float* ckv, const float* ape, HcaCache c, cudaStream_t s) {
  const int x = POST ? 1 : 0;
  if (M >= 64)
    k_vha_out<POST, 32><<<dim3(M, 1 + x), 64, 0, s>>>(o, z, V, meta, inv, M, q, sc, rows, ckv, ape, c);
  else if (M >= 16)
    k_vha_out<POST, 4><<<dim3(M, 8 + x), 64, 0, s>>>(o, z, V, meta, inv, M, q, sc, rows, ckv, ape, c);
  else
    k_vha_out<POST, 1><<<dim3(M, 32 + x), 64, 0, s>>>(o, z, V, meta, inv, M, q, sc, rows, ckv, ape, c);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace p040
}  // namespace e55
