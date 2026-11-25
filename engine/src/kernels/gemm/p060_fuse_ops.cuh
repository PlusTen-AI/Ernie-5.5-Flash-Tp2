// p060_fuse_ops.cuh — P060 f8_fuse_ops: the step's small ops around the FP8 GEMMs (misc rmsnorm / mul_bf16 / situ_bf16
// before a GEMM, sigmoid_inplace / add_bf16 / act_quant (group 32) after one) without their own launches and round
// trips, selected by patch::on(f8_fuse_ops) in model.cc; the base kernels stay. Included by fp8.cu inside its anonymous
// namespace before p030 (whose kernel uses the device helpers here).
//   M ≤ 8 (P032's fused activation): the op in the GEMM's prologue (p030 k_dec, PRO), the side RMSNorm (kv norm) as an
//         extra unit of the GEMM launch (Aux), the epilogue in the GEMM's store (put_out).
//   M > 8: one launch computes the op and quantizes the result (k_prep: the bytes act_quant would have written from the
//         misc kernel's output; the side RMSNorm in the same launch); the epilogue in p030's store where p030 runs
//         (M ≤ 32), else one elementwise launch after the GEMM (k_epi).
// Bitwise: every value is the separate kernels' — k_rms's reduction tree reproduced exactly (rms_scale: one warp per
// row; rms_rows: one CTA over ≤ 8 rows), each op misc's expression with its bf16 rounding, then act_quant's amax, scale
// and e4m3 conversion. Both new kernels are programmatic dependents (griddepcontrol.wait before any read or write).
#pragma once

namespace p060 {

// k_rms (misc.cu): 256 threads, thread i sums x[i + 256·s]² in order s (ss += v·v), warp_sum, then one warp_sum over
// the 8 warp sums in lanes 0–7 (lanes 8–31 zero), rs = rsqrt(sum / n + eps). The zero lanes of that last butterfly drop
// out exactly (x + 0 = x; sums of squares are never −0), leaving ((w0 + w4) + (w2 + w6)) + ((w1 + w5) + (w3 + w7)).
__device__ __forceinline__ float rs_of(const float* w8, int n) {
  return rsqrtf(__fdiv_rn(((w8[0] + w8[4]) + (w8[2] + w8[6])) + ((w8[1] + w8[5]) + (w8[3] + w8[7])), float(n)) + kEps);
}

// One warp per row (n = 256·NS): lane l plays thread 32·v + l of each virtual warp v — the same chains and butterflies.
// xv: the row's values x[256·s + 32·v + l].
template <int NS>
__device__ __forceinline__ float rms_scale(const bf16* x, int lane, float (&xv)[NS][8]) {
#pragma unroll
  for (int s = 0; s < NS; s++)
#pragma unroll
    for (int v = 0; v < 8; v++) xv[s][v] = __bfloat162float(x[256 * s + 32 * v + lane]);
  float p[8];
#pragma unroll
  for (int v = 0; v < 8; v++) {
    float ss = 0.f;
#pragma unroll
    for (int s = 0; s < NS; s++) ss += xv[s][v] * xv[s][v];
    p[v] = warp_sum(ss);
  }
  return rs_of(p, NS * 256);
}

// rs of rows 0 … mv − 1 (mv ≤ 8, row stride ldx) by the whole CTA (128 threads): threads 0–63 take one row, 64–127 the
// next (4 row pairs, every load in flight); thread j = tid & 63 plays k_rms threads 4j … 4j+3 (one 8-byte load per
// 256-column slab s, chains in order s) — lanes 4(j & 7) + b of virtual warp j >> 3, so the virtual butterfly's steps
// 16, 8, 4 are shuffles xor 4, 2, 1 and steps 2, 1 are local: warp total = (p0 + p2) + (p1 + p3).
// red: shared [8][8], rs: shared [mv]; ends with __syncthreads.
template <int NS>
__device__ __forceinline__ void rms_rows(const bf16* x, int ldx, int mv, float* red, float* rs) {
  const int tid = threadIdx.x, j = tid & 63, half = tid >> 6, lane = tid & 31, w = tid >> 5;
  uint2 u[4][NS];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const int r = 2 * i + half;
#pragma unroll
    for (int s = 0; s < NS; s++)
      u[i][s] = r < mv ? *reinterpret_cast<const uint2*>(x + (size_t)r * ldx + 256 * s + 4 * j) : make_uint2(0, 0);
  }
#pragma unroll
  for (int i = 0; i < 4; i++) {
    if (2 * i >= mv) break;  // CTA-uniform
    float p[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int s = 0; s < NS; s++) {
      const float v0 = bf16_lo(u[i][s].x), v1 = bf16_hi(u[i][s].x), v2 = bf16_lo(u[i][s].y), v3 = bf16_hi(u[i][s].y);
      p[0] += v0 * v0, p[1] += v1 * v1, p[2] += v2 * v2, p[3] += v3 * v3;
    }
#pragma unroll
    for (int o = 4; o > 0; o >>= 1)
#pragma unroll
      for (int b = 0; b < 4; b++) p[b] += __shfl_xor_sync(0xffffffffu, p[b], o);
    const int r = 2 * i + half;
    if ((lane & 7) == 0 && r < mv) red[r * 8 + 4 * (w & 1) + (lane >> 3)] = (p[0] + p[2]) + (p[1] + p[3]);
  }
  __syncthreads();
  if (tid < mv) rs[tid] = rs_of(red + tid * 8, NS * 256);
  __syncthreads();
}
__device__ __forceinline__ void rms_rows_n(const bf16* x, int ldx, int n, int mv, float* red, float* rs) {
  if (n == 1024) rms_rows<4>(x, ldx, mv, red, rs);
  else if (n == 2048) rms_rows<8>(x, ldx, mv, red, rs);
  else rms_rows<2>(x, ldx, mv, red, rs);
}

__device__ __forceinline__ float bf16r(float f) { return __bfloat162float(__float2bfloat16_rn(f)); }
// misc k_situ's expression (vLLM situ_and_mul), unchanged
__device__ __forceinline__ float situ_f(float g, float u) {
  const float gate = (0.5f * 4.0f) * tanhf(g * 0.25f) * (1.0f + tanhf(g * 0.5f));
  const float up = 25.0f * tanhf(u * (1.0f / 25.0f));
  return gate * up;
}
// misc k_sigmoid's expression on a bf16 value
__device__ __forceinline__ bf16 sigmoid_bf16(bf16 h) {
  return __float2bfloat16_rn(__fdiv_rn(1.0f, 1.0f + expf(-__bfloat162float(h))));
}

// the side RMSNorm (F8Aux), row r by one warp: y = bf16(x · rs · w) (k_rms's bits)
template <int NS>
__device__ __forceinline__ void aux_row(const F8Aux& x, int r, int lane) {
  const bf16* xr = x.x + (size_t)r * x.ldx;
  float xv[NS][8];
  const float rs = rms_scale<NS>(xr, lane, xv);
  bf16* yr = x.y + (size_t)r * x.ldy;
#pragma unroll
  for (int s = 0; s < NS; s++)
#pragma unroll
    for (int v = 0; v < 8; v++) {
      const int i = 256 * s + 32 * v + lane;
      yr[i] = __float2bfloat16_rn(xv[s][v] * rs * __bfloat162float(x.w[i]));
    }
}
__device__ __forceinline__ void aux_rows(const F8Aux& x, int M, int r0, int step, int lane) {
  for (int r = r0; r < M; r += step) {
    if (x.n == 512) aux_row<2>(x, r, lane);
    else if (x.n == 1024) aux_row<4>(x, r, lane);
    else aux_row<8>(x, r, lane);
  }
}

// ---- M > 8: op + quantization in one launch -------------------------------------------------------------------------
struct PrepArgs {
  const bf16* x;
  int ldx;
  const bf16* pw;
  int pld;
  int M, k;
  uint8_t* q;   // e4m3 [M][ld]
  uint8_t* s;   // UE8M0 [M][ld / 128]
  int ld;
  F8Aux aux;    // aux.x == nullptr: none
  int nrow;     // blockIdx.x < nrow: rows 2·x, 2·x + 1, groups kGC·y …; then (y = 0) the side RMSNorm, 4 rows a CTA
};
constexpr int kGC = 4;  // 128-groups per CTA and row

// 128 threads: rows 2x, 2x+1 × groups [kGC·y, kGC·y + kGC); a warp per (row, 128-group), 4 elements a lane —
// act_quant's (k_act_quant, G = 128). kProRms: each CTA takes its rows' rs over the whole row first.
template <int OP>
__global__ void __launch_bounds__(128) k_prep(PrepArgs a) {
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;\n" :::);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (int(blockIdx.x) >= a.nrow) {
    if (blockIdx.y == 0) aux_rows(a.aux, a.M, (blockIdx.x - a.nrow) * 4 + warp, 1 << 30, lane);
    return;
  }
  const int r0 = blockIdx.x * 2, mv = min(2, a.M - r0);
  [[maybe_unused]] __shared__ float red[8 * 8], rs[8];
  if constexpr (OP == kProRms) rms_rows_n(a.x + (size_t)r0 * a.ldx, a.ldx, a.k, mv, red, rs);
  const int g0 = blockIdx.y * kGC, gn = min(kGC, (a.k >> 7) - g0);
  for (int u = warp; u < mv * gn; u += 4) {  // warp-uniform
    const int rr = u / gn, g = g0 + u - rr * gn, c = g * 128 + lane * 4, row = r0 + rr;
    const uint2 xv = *reinterpret_cast<const uint2*>(a.x + (size_t)row * a.ldx + c);
    float v[4] = {bf16_lo(xv.x), bf16_hi(xv.x), bf16_lo(xv.y), bf16_hi(xv.y)};
    if constexpr (OP != kProNone) {
      const bf16* op = OP == kProRms ? a.pw + c : OP == kProMul ? a.pw + (size_t)row * a.pld + c
                                                                : a.x + (size_t)row * a.ldx + a.k + c;
      const uint2 wv = *reinterpret_cast<const uint2*>(op);
      const float w[4] = {bf16_lo(wv.x), bf16_hi(wv.x), bf16_lo(wv.y), bf16_hi(wv.y)};
#pragma unroll
      for (int i = 0; i < 4; i++) {
        if constexpr (OP == kProRms) v[i] = bf16r(v[i] * rs[rr] * w[i]);  // k_rms: x · rs · w
        else if constexpr (OP == kProMul) v[i] = bf16r(v[i] * w[i]);      // k_mul: a · b
        else v[i] = bf16r(situ_f(v[i], w[i]));                            // k_situ
      }
    }
    const float m = warp_max(fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3]))));
    const uint32_t e = act_scale_e8(m, 1e-10f);
    uint32_t out = 0;
#pragma unroll
    for (int i = 0; i < 4; i++) out |= quant_e4m3(v[i], e) << (8 * i);
    *reinterpret_cast<uint32_t*>(a.q + (size_t)row * a.ld + c) = out;
    if (lane == 0) a.s[(size_t)row * (a.ld >> 7) + g] = uint8_t(e);
  }
}

// ---- the epilogue after a GEMM that could not take it: in place on y [M, n] (row stride ldy) --------------------------
// kEpiSigmoid: k_sigmoid's; kEpiAdd: k_add's (y + e); kEpiQ32: act_quant group 32 of y into q [M][ldy] / qs [M][ldy/32]
__global__ void k_epi(int op, bf16* y, int ldy, int M, int n, const bf16* e, int lde, uint8_t* q, uint8_t* qs) {
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;\n" :::);
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)M * n) return;  // whole warps (n % 32 == 0)
  const int m = int(i / n), c = int(i % n);
  bf16* p = y + (size_t)m * ldy + c;
  if (op == kEpiSigmoid) {
    *p = sigmoid_bf16(*p);
  } else if (op == kEpiAdd) {
    *p = __float2bfloat16_rn(__bfloat162float(*p) + __bfloat162float(e[(size_t)m * lde + c]));
  } else {
    const float h = __bfloat162float(*p);
    const uint32_t ex = act_scale_e8(warp_max(fabsf(h)), 1e-10f);
    q[(size_t)m * ldy + c] = uint8_t(quant_e4m3(h, ex));
    if ((threadIdx.x & 31) == 0) qs[(size_t)m * (ldy / 32) + c / 32] = uint8_t(ex);
  }
}

template <class... P, class... A>
void launch_pdl(void (*k)(P...), dim3 grid, unsigned block, cudaStream_t s, A... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid, cfg.blockDim = dim3(block), cfg.stream = s;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at, cfg.numAttrs = e55::patch::on(e55::patch::f8_pdl) ? 1 : 0;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, k, args...));
}

// x.pro of x.x → x.q / x.s (group 128, row stride x.ld), and the side RMSNorm if aux
void prep(const F8Act& x, int M, int k, const F8Aux* aux, cudaStream_t s) {
  PrepArgs a{x.x, x.ldx, x.pw, x.pld, M, k, x.q, x.s, x.ld, aux ? *aux : F8Aux{}, cdiv(M, 2)};
  const dim3 grid(unsigned(a.nrow + (aux ? cdiv(M, 4) : 0)), unsigned(cdiv(k / 128, kGC)));
  switch (x.pro) {
    case kProRms: return launch_pdl(k_prep<kProRms>, grid, 128, s, a);
    case kProMul: return launch_pdl(k_prep<kProMul>, grid, 128, s, a);
    case kProSitu: return launch_pdl(k_prep<kProSitu>, grid, 128, s, a);
    default: return launch_pdl(k_prep<kProNone>, grid, 128, s, a);
  }
}
void epi(const F8Out& o, int M, cudaStream_t s) {
  const size_t n = (size_t)M * o.L->n;
  launch_pdl(k_epi, dim3(unsigned((n + 255) / 256)), 256, s, o.epi, o.y, o.ldy, M, o.L->n, o.e, o.lde, o.q, o.qs);
}

}  // namespace p060
