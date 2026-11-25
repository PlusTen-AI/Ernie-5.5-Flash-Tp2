// dsa.cu — ERNIE MLA-DSA attention kernels (contract: dsa.h).
#include "kernels/attn/dsa.h"

#include <algorithm>

#include "core/log.h"
#include "kernels/attn/attn_common.cuh"
#include "kernels/gemm/mma.cuh"
#include "model/patches.h"

namespace e55 {
namespace {

constexpr int kLH = kHeads / 2;           // 32 local heads
constexpr int kQK = kKvLora + kRope;      // 576
constexpr int kMR = kMlaRowBytes;         // 656
constexpr int kIR = kIdxRowBytes;         // 68
constexpr int kChunk = 256, kKT = 32;
constexpr int kQS = kQK + 8;              // padded smem row (bf16), 1168 B
constexpr int kMaxKeys = kIdxTop + kWindow;  // 2176

__device__ __forceinline__ float pow2_ceil(float v) {  // 2^ceil(log2(v)), v > 0 normal
  const uint32_t b = __float_as_uint(v);
  return __uint_as_float((((b >> 23) & 0xFF) + ((b & 0x7FFFFF) != 0)) << 23);
}

// ---- DSA RoPE (MLA q_pe / k_pe and the indexer): vLLM get_rope's default RotaryEmbedding with the model's bf16 dtype.
// Its cos_sin_cache is built on the GPU in fp32 — inv_i = 1 / 8e6^(2i/64) (torch.pow → powf, reciprocal → IEEE
// division), angle = fl(pos · inv_i), cos / sin (cosf / sinf) — then stored as bf16; the rotation is fp32 on the bf16
// cos / sin (separate products), the output rounded to bf16 by the caller. Same values computed here per use (the
// engine's inv8 table, computed on the host, is not read). vLLM's compiled op (nvcc default contraction) runs
// FMUL + FFMA (x·c − y·s as fma(x, c, −y·s)); with bf16 x, y and bf16 c, s every product is exact in fp32, so both
// forms give the same bits — measured against that op: 0 differences in 4.1e7 MLA / NeoX outputs.
__device__ __forceinline__ void dsa_cs(int pos, int i, float& c, float& s) {
  const float inv = __fdiv_rn(1.f, powf(8000000.f, __fmul_rn(float(2 * i), 0.015625f)));
  const float a = __fmul_rn(float(pos), inv);
  c = __bfloat162float(__float2bfloat16_rn(cosf(a)));
  s = __bfloat162float(__float2bfloat16_rn(sinf(a)));
}
// interleaved pairs (2j, 2j+1) of 8 consecutive rope dims starting at r0 (0, 8, … 56)
__device__ __forceinline__ void dsa_rope8(float v[8], int r0, int pos) {
#pragma unroll
  for (int j = 0; j < 4; j++) {
    float c, s;
    dsa_cs(pos, r0 / 2 + j, c, s);
    const float e = v[2 * j], o = v[2 * j + 1];
    v[2 * j] = __fsub_rn(__fmul_rn(e, c), __fmul_rn(o, s));
    v[2 * j + 1] = __fadd_rn(__fmul_rn(o, c), __fmul_rn(e, s));
  }
}
// NeoX pair (i, i + 32) of the indexer's leading 64 dims
__device__ __forceinline__ void dsa_rope_neox(float& a, float& b, int i, int pos) {
  float c, s;
  dsa_cs(pos, i, c, s);
  const float x = a, y = b;
  a = __fsub_rn(__fmul_rn(x, c), __fmul_rn(y, s));
  b = __fadd_rn(__fmul_rn(y, c), __fmul_rn(x, s));
}

// ---- MLA cache rows: 64 threads, thread i: kv_c dims 8i … (scale group = 16 threads); threads 0–7: k_pe dims 8i …
__global__ void k_mla_rows(const bf16* __restrict__ kvn, int ldn, const bf16* __restrict__ kpe, int ldpe,
                           const RowMeta* __restrict__ meta, const float* __restrict__ inv, DsaCache c) {
  const int r = blockIdx.x, i = threadIdx.x;
  const RowMeta mt = meta[r];
  uint8_t* row = c.mla + ((size_t)mt.slot * c.ctx + mt.pos) * kMR;
  float v[8], amax = 0.f;
  load_bf16x8(kvn + (size_t)r * ldn + 8 * i, v);
#pragma unroll
  for (int j = 0; j < 8; j++) amax = fmaxf(amax, fabsf(v[j]));
#pragma unroll
  for (int off = 8; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
  const float sc = pow2_ceil(fmaxf(__fdiv_rn(amax, 448.f), 1e-4f));
  const float isc = __fdiv_rn(1.f, sc);
  uint32_t lo = 0, hi = 0;
#pragma unroll
  for (int j = 0; j < 4; j++) {
    lo |= uint32_t(__nv_cvt_float_to_fp8(fminf(fmaxf(v[j] * isc, -448.f), 448.f), __NV_SATFINITE, __NV_E4M3)) << (8 * j);
    hi |= uint32_t(__nv_cvt_float_to_fp8(fminf(fmaxf(v[4 + j] * isc, -448.f), 448.f), __NV_SATFINITE, __NV_E4M3)) << (8 * j);
  }
  *reinterpret_cast<uint2*>(row + 8 * i) = make_uint2(lo, hi);
  if ((i & 15) == 0) reinterpret_cast<float*>(row + kKvLora)[i >> 4] = sc;
  if (i < 8) {
    float e[8];
    load_bf16x8(kpe + (size_t)r * ldpe + 8 * i, e);
    dsa_rope8(e, 8 * i, mt.pos);
    store_bf16x8(reinterpret_cast<bf16*>(row + kKvLora + 16) + 8 * i, e);
  }
}

// q_pe: grid (M·32), 8 threads × 8 dims
__global__ void k_q_pe(const bf16* __restrict__ q, const RowMeta* __restrict__ meta, const float* __restrict__ inv,
                       bf16* q576) {
  const int b = blockIdx.x, r = b / kLH, i = threadIdx.x;
  float v[8];
  load_bf16x8(q + (size_t)b * 256 + kMlaNope + 8 * i, v);
  dsa_rope8(v, 8 * i, meta[r].pos);
  store_bf16x8(q576 + (size_t)b * kQK + kKvLora + 8 * i, v);
}

// ---- batched bf16 GEMM: grid (H, N/64, ceil(M/MT)), 4 warps; warp w: cols 16w … 16w+15 of the 64
template <int MT>
__global__ void __launch_bounds__(128) k_bmm(const bf16* __restrict__ x, long xm, long xh, const bf16* __restrict__ w,
                                             int N, int K, int M, bf16* y, long ym, long yh) {
  __shared__ __align__(16) bf16 sa[MT][72];
  __shared__ __align__(16) bf16 sb[64][72];
  const int h = blockIdx.x, n0 = blockIdx.y * 64, m0 = blockIdx.z * MT;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  float acc[MT / 16][2][4];
#pragma unroll
  for (int a = 0; a < MT / 16; a++)
#pragma unroll
    for (int b = 0; b < 2; b++) acc[a][b][0] = acc[a][b][1] = acc[a][b][2] = acc[a][b][3] = 0.f;
  const bf16* wh = w + (size_t)h * N * K;
  for (int k0 = 0; k0 < K; k0 += 64) {
    __syncthreads();
    for (int i = tid; i < MT * 8; i += 128) {
      const int r = i >> 3, c8 = i & 7, m = m0 + r;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (m < M) v = *reinterpret_cast<const uint4*>(x + m * xm + h * xh + k0 + c8 * 8);
      *reinterpret_cast<uint4*>(&sa[r][c8 * 8]) = v;
    }
    for (int i = tid; i < 64 * 8; i += 128) {
      const int r = i >> 3, c8 = i & 7;
      *reinterpret_cast<uint4*>(&sb[r][c8 * 8]) =
          *reinterpret_cast<const uint4*>(wh + (size_t)(n0 + r) * K + k0 + c8 * 8);
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
      uint32_t b[2][2];
#pragma unroll
      for (int nb = 0; nb < 2; nb++)
        ldsm_x2(b[nb][0], b[nb][1], &sb[16 * warp + 8 * nb + (lane & 7)][16 * kk + ((lane >> 3) & 1) * 8]);
#pragma unroll
      for (int a = 0; a < MT / 16; a++) {
        uint32_t a0, a1, a2, a3;
        ldsm_x4(a0, a1, a2, a3, &sa[16 * a + (lane & 15)][16 * kk + (lane >> 4) * 8]);
#pragma unroll
        for (int nb = 0; nb < 2; nb++) mma_bf16(acc[a][nb], a0, a1, a2, a3, b[nb][0], b[nb][1]);
      }
    }
  }
  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int a = 0; a < MT / 16; a++)
#pragma unroll
    for (int nb = 0; nb < 2; nb++) {
      const int n = n0 + 16 * warp + 8 * nb + 2 * t;
      const int ma = m0 + 16 * a + g, mb = ma + 8;
      if (ma < M) *reinterpret_cast<uint32_t*>(y + ma * ym + h * yh + n) = pack_bf16x2(acc[a][nb][0], acc[a][nb][1]);
      if (mb < M) *reinterpret_cast<uint32_t*>(y + mb * ym + h * yh + n) = pack_bf16x2(acc[a][nb][2], acc[a][nb][3]);
    }
}

// ---- indexer -----------------------------------------------------------------------------------------------------
// e2m1 code of v (already divided by the scale): round to nearest even on {0, .5, 1, 1.5, 2, 3, 4, 6}, saturating
__device__ __forceinline__ uint32_t e2m1_code(float v) {
  const float a = fabsf(v);
  const uint32_t c = (a > 0.25f) + (a >= 0.75f) + (a > 1.25f) + (a >= 1.75f) + (a > 2.5f) + (a >= 3.5f) + (a > 5.f);
  return c | (__float_as_uint(v) >> 31 << 3);
}
// vLLM downcast_to_mxfp scale of a 32-group: unbiased exponent floor(log2(round(amax))) − 2, as a UE8M0 byte
__device__ __forceinline__ uint32_t mxfp4_scale_byte(float amax) {
  uint32_t b = (__float_as_uint(amax) + 0x200000u) & 0x7F800000u;
  int e = b ? int(b >> 23) - 127 : -126;
  e = min(max(e - 2, -127), 127);
  return uint32_t(e + 127);
}

// 128 threads = one 128-vector (thread = element): in place Walsh–Hadamard (natural order) / sqrt(128), → bf16
__device__ __forceinline__ float hadamard128(float x, float* sh) {
  const int i = threadIdx.x & 127;
  sh[i] = x;
  __syncthreads();
#pragma unroll
  for (int s = 1; s < 128; s <<= 1) {
    const float a = sh[i], b = sh[i ^ s];
    __syncthreads();
    sh[i] = (i & s) ? __fsub_rn(b, a) : __fadd_rn(a, b);
    __syncthreads();
  }
  return __bfloat162float(__float2bfloat16_rn(sh[i] * 0.08838834764831845f));
}

// MXFP4 of a 128-vector held one element per thread: codes packed two per byte, 4 scales
__device__ __forceinline__ void mxfp4_store(float x, uint8_t* codes, uint8_t* scales) {
  const int i = threadIdx.x & 127, lane = threadIdx.x & 31;
  const float amax = warp_max(fabsf(x));
  const uint32_t sb = mxfp4_scale_byte(amax);
  const float inv = __uint_as_float((254u - sb) << 23);  // 1 / 2^(sb−127)
  const uint32_t code = e2m1_code(x * inv);
  const uint32_t other = __shfl_xor_sync(0xffffffffu, code, 1);
  if (!(i & 1)) codes[i >> 1] = uint8_t(code | (other << 4));
  if (lane == 0) scales[i >> 5] = uint8_t(sb);
}

// grid (M, 1 + 64): block 0 = the K row and the weights, blocks 1 … 64 = query heads; 128 threads
__global__ void __launch_bounds__(128) k_idx_prep(const bf16* __restrict__ kw, const bf16* __restrict__ qi,
                                                  const bf16* __restrict__ knw, const bf16* __restrict__ knb,
                                                  const RowMeta* __restrict__ meta, const float* __restrict__ inv,
                                                  DsaCache c, uint8_t* qf4, uint8_t* qs, float* wts) {
  const int r = blockIdx.x, j = blockIdx.y, i = threadIdx.x;
  const RowMeta mt = meta[r];
  __shared__ float sh[128], red[8];
  float x;
  if (j == 0) {
    x = __bfloat162float(kw[(size_t)r * 192 + i]);
    float s = warp_sum(x);
    if ((i & 31) == 0) red[i >> 5] = s;
    __syncthreads();
    const float mean = __fdiv_rn(((red[0] + red[1]) + red[2]) + red[3], 128.f);
    const float d = x - mean;
    s = warp_sum(d * d);
    __syncthreads();
    if ((i & 31) == 0) red[4 + (i >> 5)] = s;
    __syncthreads();
    const float var = __fdiv_rn(((red[4] + red[5]) + red[6]) + red[7], 128.f);
    x = __bfloat162float(__float2bfloat16_rn(d * rsqrtf(var + 1e-6f) * __bfloat162float(knw[i]) + __bfloat162float(knb[i])));
    if (i < 64) wts[(size_t)r * 64 + i] = __bfloat162float(kw[(size_t)r * 192 + 128 + i]) * 0.011048543456039806f;
  } else {
    x = __bfloat162float(qi[((size_t)r * 64 + (j - 1)) * 128 + i]);
  }
  // NeoX RoPE on dims 0 … 63: pairs (i, i + 32)
  sh[i] = x;
  __syncthreads();
  if (i < 32) {
    float a = sh[i], b = sh[i + 32];
    dsa_rope_neox(a, b, i, mt.pos);
    sh[i] = __bfloat162float(__float2bfloat16_rn(a)), sh[i + 32] = __bfloat162float(__float2bfloat16_rn(b));
  }
  __syncthreads();
  x = sh[i];
  __syncthreads();
  x = hadamard128(x, sh);
  if (j == 0) {
    uint8_t* row = c.idx + ((size_t)mt.slot * c.ctx + mt.pos) * kIR;
    mxfp4_store(x, row, row + 64);
  } else {
    mxfp4_store(x, qf4 + ((size_t)r * 64 + (j - 1)) * 64, qs + ((size_t)r * 64 + (j - 1)) * 4);
  }
}

// logits of rows [row0, row0 + n) over candidate key chunks of 64: grid (n, chunks), 4 warps = 4 × 16 heads
__global__ void __launch_bounds__(128) k_idx_logits(const uint8_t* __restrict__ qf4, const uint8_t* __restrict__ qs,
                                                    const float* __restrict__ wts, const RowMeta* __restrict__ meta,
                                                    DsaCache c, int row0, int ldl, float* lg) {
  const int r = row0 + blockIdx.x;
  const RowMeta mt = meta[r];
  const int ncand = mt.pos - kWindow + 1;
  const int t0 = blockIdx.y * 64;
  if (ncand <= kIdxTop || t0 >= ncand) return;
  __shared__ __align__(16) uint8_t sq[64][128 + 16];  // expanded e2m1 bytes
  __shared__ __align__(16) uint8_t sk[64][128 + 16];
  __shared__ uint8_t sqs[64][4], sks[64][4];
  __shared__ float part[4][64];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  for (int i = tid; i < 64 * 32; i += 128) {  // 64 rows × 32 words of 4 codes
    const int h = i >> 5, wd = i & 31;
    const uint16_t qv = *reinterpret_cast<const uint16_t*>(qf4 + ((size_t)r * 64 + h) * 64 + 2 * wd);
    *reinterpret_cast<uint32_t*>(&sq[h][4 * wd]) = e2m1x4_expand(qv);
    const int tk = t0 + h;
    uint32_t kv = 0;
    if (tk < ncand) kv = e2m1x4_expand(*reinterpret_cast<const uint16_t*>(c.idx + ((size_t)mt.slot * c.ctx + tk) * kIR + 2 * wd));
    *reinterpret_cast<uint32_t*>(&sk[h][4 * wd]) = kv;
  }
  for (int i = tid; i < 64 * 4; i += 128) {
    const int h = i >> 2, j = i & 3, tk = t0 + h;
    sqs[h][j] = qs[((size_t)r * 64 + h) * 4 + j];
    sks[h][j] = tk < ncand ? c.idx[((size_t)mt.slot * c.ctx + tk) * kIR + 64 + j] : 127;
  }
  __syncthreads();
  float acc[8][4];
#pragma unroll
  for (int nt = 0; nt < 8; nt++) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
  const int h0 = 16 * warp;
#pragma unroll
  for (int kk = 0; kk < 4; kk++) {
    const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&sq[h0 + g][32 * kk + 4 * t]);
    const uint32_t a1 = *reinterpret_cast<const uint32_t*>(&sq[h0 + g + 8][32 * kk + 4 * t]);
    const uint32_t a2 = *reinterpret_cast<const uint32_t*>(&sq[h0 + g][32 * kk + 16 + 4 * t]);
    const uint32_t a3 = *reinterpret_cast<const uint32_t*>(&sq[h0 + g + 8][32 * kk + 16 + 4 * t]);
    const uint32_t sfa = sqs[h0 + g + 8 * (lane & 1)][kk];
#pragma unroll
    for (int nt = 0; nt < 8; nt++) {
      const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&sk[8 * nt + g][32 * kk + 4 * t]);
      const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&sk[8 * nt + g][32 * kk + 16 + 4 * t]);
      mma_e2m1_e2m1(acc[nt], a0, a1, a2, a3, b0, b1, sfa, sks[8 * nt + g][kk]);
    }
  }
  const float w0 = wts[(size_t)r * 64 + h0 + g], w1 = wts[(size_t)r * 64 + h0 + g + 8];
#pragma unroll
  for (int nt = 0; nt < 8; nt++) {
    float s0 = fmaxf(acc[nt][0], 0.f) * w0 + fmaxf(acc[nt][2], 0.f) * w1;  // key 2t
    float s1 = fmaxf(acc[nt][1], 0.f) * w0 + fmaxf(acc[nt][3], 0.f) * w1;  // key 2t+1
#pragma unroll
    for (int off = 4; off < 32; off <<= 1) s0 += __shfl_xor_sync(0xffffffffu, s0, off), s1 += __shfl_xor_sync(0xffffffffu, s1, off);
    if (g == 0) part[warp][8 * nt + 2 * t] = s0, part[warp][8 * nt + 2 * t + 1] = s1;
  }
  __syncthreads();
  if (tid < 64 && t0 + tid < ncand)
    lg[(size_t)blockIdx.x * ldl + t0 + tid] = ((part[0][tid] + part[1][tid]) + part[2][tid]) + part[3][tid];
}

__device__ __forceinline__ uint32_t okey(float f) {
  const uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// one block (512 threads) per row: the key list in position order
__global__ void __launch_bounds__(512) k_topk(const RowMeta* __restrict__ meta, int row0, int ldl,
                                              const float* __restrict__ lg, int* list, int* cnt) {
  const int r = row0 + blockIdx.x, tid = threadIdx.x;
  const int p = meta[r].pos, ncand = p - kWindow + 1;
  int* out = list + (size_t)r * kMaxKeys;
  if (ncand <= kIdxTop) {
    for (int i = tid; i <= p; i += 512) out[i] = i;
    if (tid == 0) cnt[r] = p + 1;
    return;
  }
  const float* l = lg + (size_t)blockIdx.x * ldl;
  __shared__ int hist[256];
  __shared__ uint32_t s_prefix, s_mask;
  __shared__ int s_need;
  if (tid == 0) s_prefix = 0, s_mask = 0, s_need = kIdxTop;
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int i = tid; i < 256; i += 512) hist[i] = 0;
    __syncthreads();
    const uint32_t pre = s_prefix, msk = s_mask;
    for (int i = tid; i < ncand; i += 512) {
      const uint32_t u = okey(l[i]);
      if ((u & msk) == pre) atomicAdd(&hist[(u >> shift) & 0xFF], 1);
    }
    __syncthreads();
    if (tid == 0) {
      int need = s_need, b = 255;
      for (; b > 0; b--) {
        if (hist[b] >= need) break;
        need -= hist[b];
      }
      s_need = need;
      s_prefix = pre | (uint32_t(b) << shift);
      s_mask = msk | (0xFFu << shift);
    }
    __syncthreads();
  }
  const uint32_t thr = s_prefix;
  const int need_eq = s_need;  // how many keys equal to thr are taken (lowest positions first)
  // compaction in position order: > thr always, == thr while the running count of equals < need_eq
  __shared__ int wsum[16], weq[16], s_base, s_eqbase;
  if (tid == 0) s_base = 0, s_eqbase = 0;
  __syncthreads();
  const int lane = tid & 31, warp = tid >> 5;
  for (int c0 = 0; c0 < ncand; c0 += 512) {
    const int i = c0 + tid;
    const uint32_t u = i < ncand ? okey(l[i]) : 0;
    const bool gt = i < ncand && u > thr, eq = i < ncand && u == thr;
    const unsigned be = __ballot_sync(0xffffffffu, eq);
    if (lane == 0) weq[warp] = __popc(be);
    __syncthreads();
    int eqpre = s_eqbase;
    for (int k = 0; k < warp; k++) eqpre += weq[k];
    eqpre += __popc(be & ((1u << lane) - 1));
    const bool take = gt || (eq && eqpre < need_eq);
    const unsigned bt = __ballot_sync(0xffffffffu, take);
    if (lane == 0) wsum[warp] = __popc(bt);
    __syncthreads();
    int pos = s_base;
    for (int k = 0; k < warp; k++) pos += wsum[k];
    pos += __popc(bt & ((1u << lane) - 1));
    if (take) out[pos] = i;
    __syncthreads();
    if (tid == 0) {
      int a = 0, e = 0;
      for (int k = 0; k < 16; k++) a += wsum[k], e += weq[k];
      s_base += a, s_eqbase += e;
    }
    __syncthreads();
  }
  for (int i = tid; i < kWindow; i += 512) out[kIdxTop + i] = p - kWindow + 1 + i;
  if (tid == 0) cnt[r] = kIdxTop + kWindow;
}

// P140 dsa_sel_split: k_topk's (P046's, P091's) list entries kIdxTop … and count of row r — one CTA of kWindow threads
static_assert(kWindow == 128, "k_select_tail: one thread per window entry");
__global__ void __launch_bounds__(128) k_select_tail(const RowMeta* __restrict__ meta, int* list, int* cnt) {
  const int r = blockIdx.x, tid = threadIdx.x;
  const int p = meta[r].pos, ncand = p - kWindow + 1;
  int* out = list + (size_t)r * kMaxKeys;
  if (ncand <= kIdxTop) {
    for (int i = kIdxTop + tid; i <= p; i += kWindow) out[i] = i;
    if (tid == 0) cnt[r] = p + 1;
  } else {
    out[kIdxTop + tid] = p - kWindow + 1 + tid;
    if (tid == 0) cnt[r] = kIdxTop + kWindow;
  }
}

// ---- sparse MLA attention ----------------------------------------------------------------------------------------
struct MlaSmem {
  bf16 q[kLH][kQS];
  bf16 k[kKT][kQS];
  float s[kLH][kKT + 1];
  bf16 p[kLH][kKT + 8];
  float m[kLH], l[kLH], fac[kLH];
};

__global__ void __launch_bounds__(256) k_mla_attn(const bf16* __restrict__ q, const int* __restrict__ list,
                                                  const int* __restrict__ cnt, const RowMeta* __restrict__ meta,
                                                  DsaCache c, int row0, int nch, float* ws) {
  extern __shared__ __align__(16) uint8_t smraw[];
  MlaSmem& S = *reinterpret_cast<MlaSmem*>(smraw);
  const int r = row0 + blockIdx.x, ch = blockIdx.y;
  const RowMeta mt = meta[r];
  const int nk = cnt[r], kbeg = ch * kChunk;
  if (kbeg >= nk) return;
  const int kend = min(nk, kbeg + kChunk);
  const int* lst = list + (size_t)r * kMaxKeys;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  for (int i = tid; i < kLH * kQK / 8; i += 256) {
    const int h = i / (kQK / 8), c8 = i % (kQK / 8);
    *reinterpret_cast<uint4*>(&S.q[h][c8 * 8]) = *reinterpret_cast<const uint4*>(q + ((size_t)r * kLH + h) * kQK + c8 * 8);
  }
  if (tid < kLH) S.m[tid] = -INFINITY, S.l[tid] = 0.f;
  float o[2][8][4];
#pragma unroll
  for (int a = 0; a < 2; a++)
#pragma unroll
    for (int b = 0; b < 8; b++) o[a][b][0] = o[a][b][1] = o[a][b][2] = o[a][b][3] = 0.f;
  const float scale = 0.0625f;  // 256^-0.5
  for (int k0 = kbeg; k0 < kend; k0 += kKT) {
    __syncthreads();
    {  // thread (key tid/8, part tid%8): kv_c dims 64·part …, k_pe dims 8·part …
      const int kk = tid >> 3, part = tid & 7, key = k0 + kk;
      bf16* dst = S.k[kk];
      if (key < kend) {
        const uint8_t* row = c.mla + ((size_t)mt.slot * c.ctx + lst[key]) * kMR;
        const float sc = reinterpret_cast<const float*>(row + kKvLora)[part >> 1];
#pragma unroll
        for (int j = 0; j < 64; j += 8) {
          const uint2 u = *reinterpret_cast<const uint2*>(row + 64 * part + j);
          float v[8];
#pragma unroll
          for (int e = 0; e < 4; e++) {
            __nv_fp8_e4m3 a, b;
            a.__x = (u.x >> (8 * e)) & 0xFF, b.__x = (u.y >> (8 * e)) & 0xFF;
            v[e] = float(a) * sc, v[4 + e] = float(b) * sc;
          }
          store_bf16x8(dst + 64 * part + j, v);
        }
        *reinterpret_cast<uint4*>(dst + kKvLora + 8 * part) =
            *reinterpret_cast<const uint4*>(row + kKvLora + 16 + 16 * part);
      } else {
        for (int j = 0; j < 64; j += 8) *reinterpret_cast<uint4*>(dst + 64 * part + j) = make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(dst + kKvLora + 8 * part) = make_uint4(0, 0, 0, 0);
      }
    }
    __syncthreads();
    {
      const int mt16 = warp >> 2, nt8 = warp & 3;
      float sacc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
      for (int kk = 0; kk < kQK / 16; kk++) {
        uint32_t a0, a1, a2, a3, b0, b1;
        ldsm_x4(a0, a1, a2, a3, &S.q[16 * mt16 + (lane & 15)][16 * kk + (lane >> 4) * 8]);
        ldsm_x2(b0, b1, &S.k[8 * nt8 + (lane & 7)][16 * kk + ((lane >> 3) & 1) * 8]);
        mma_bf16(sacc, a0, a1, a2, a3, b0, b1);
      }
      const int g = lane >> 2, t = lane & 3;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int h = 16 * mt16 + g + (i >= 2 ? 8 : 0), kk = 8 * nt8 + 2 * t + (i & 1);
        S.s[h][kk] = k0 + kk < kend ? sacc[i] * scale : -INFINITY;
      }
    }
    __syncthreads();
    if (tid < kLH) {
      const int h = tid;
      float mx = S.m[h];
      for (int j = 0; j < kKT; j++) mx = fmaxf(mx, S.s[h][j]);
      const float f = expf(S.m[h] - mx);
      float sum = 0.f;
      for (int j = 0; j < kKT; j++) {
        const float e = expf(S.s[h][j] - mx);
        sum += e;
        S.p[h][j] = __float2bfloat16_rn(e);
      }
      S.l[h] = S.l[h] * f + sum;
      S.m[h] = mx;
      S.fac[h] = f;
    }
    __syncthreads();
    {
      const int g = lane >> 2;
#pragma unroll
      for (int a = 0; a < 2; a++) {
        const float f0 = S.fac[16 * a + g], f1 = S.fac[16 * a + g + 8];
#pragma unroll
        for (int b = 0; b < 8; b++) o[a][b][0] *= f0, o[a][b][1] *= f0, o[a][b][2] *= f1, o[a][b][3] *= f1;
      }
#pragma unroll
      for (int ks = 0; ks < kKT / 16; ks++)
#pragma unroll
        for (int a = 0; a < 2; a++) {
          uint32_t a0, a1, a2, a3;
          ldsm_x4(a0, a1, a2, a3, &S.p[16 * a + (lane & 15)][16 * ks + (lane >> 4) * 8]);
#pragma unroll
          for (int b = 0; b < 8; b++) {
            uint32_t b0, b1;
            ldsm_x2_t(b0, b1, &S.k[16 * ks + (lane & 15)][64 * warp + 8 * b]);
            mma_bf16(o[a][b], a0, a1, a2, a3, b0, b1);
          }
        }
    }
  }
  __syncthreads();
  float* w = ws + ((size_t)blockIdx.x * nch + ch) * (2 * kLH + kLH * kKvLora);
  if (tid < kLH) w[tid] = S.m[tid], w[kLH + tid] = S.l[tid];
  float* wo = w + 2 * kLH;
  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int a = 0; a < 2; a++)
#pragma unroll
    for (int b = 0; b < 8; b++) {
      const int h0 = 16 * a + g, d = 64 * warp + 8 * b + 2 * t;
      wo[h0 * kKvLora + d] = o[a][b][0], wo[h0 * kKvLora + d + 1] = o[a][b][1];
      wo[(h0 + 8) * kKvLora + d] = o[a][b][2], wo[(h0 + 8) * kKvLora + d + 1] = o[a][b][3];
    }
}

__global__ void __launch_bounds__(256) k_mla_merge(const int* __restrict__ cnt, const float* __restrict__ sink,
                                                   int row0, int nch, const float* __restrict__ ws, bf16* o) {
  const int r = row0 + blockIdx.x, tid = threadIdx.x, h = tid >> 3, part = tid & 7;
  const int n = (cnt[r] + kChunk - 1) / kChunk;
  const size_t stride = 2 * kLH + kLH * kKvLora;
  const float* w = ws + (size_t)blockIdx.x * nch * stride;
  float M = sink[h];
  for (int c = 0; c < n; c++) M = fmaxf(M, w[c * stride + h]);
  float den = 0.f;
  for (int c = 0; c < n; c++) den += expf(w[c * stride + h] - M) * w[c * stride + kLH + h];
  den += expf(sink[h] - M);
  for (int d = part * 64; d < part * 64 + 64; d += 2) {
    float a = 0.f, b = 0.f;
    for (int c = 0; c < n; c++) {
      const float f = expf(w[c * stride + h] - M);
      const float* x = w + c * stride + 2 * kLH + h * kKvLora + d;
      a += f * x[0], b += f * x[1];
    }
    *reinterpret_cast<uint32_t*>(o + ((size_t)r * kLH + h) * kKvLora + d) = pack_bf16x2(__fdiv_rn(a, den), __fdiv_rn(b, den));
  }
}

// o' = bf16(o + V z), y = bf16(o' · gate): grid (M, 32), 256 threads = dims
__global__ void k_vha_gate(const bf16* __restrict__ o, const float* __restrict__ z, const float* __restrict__ V,
                           const bf16* __restrict__ gate, bf16* y) {
  const int m = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  float a = 0.f;
#pragma unroll
  for (int r = 0; r < kVhaRank; r++) a += V[h * kVhaRank + r] * z[((size_t)m * kVhaRank + r) * kMlaV + d];
  const size_t i = ((size_t)m * kLH + h) * kMlaV + d;
  const float v = __bfloat162float(__float2bfloat16_rn(__bfloat162float(o[i]) + a));
  y[i] = __float2bfloat16_rn(v * __bfloat162float(gate[i]));
}

#include "kernels/attn/p045_attn.cuh"  // P045 dsa_attn_fast
#include "kernels/attn/p046_select.cuh"  // P046 dsa_select_fast
#include "kernels/attn/p047_vha.cuh"     // P047 dsa_vha_fast
#include "kernels/attn/p048_bmm.cuh"     // P048 dsa_bmm_dec
#include "kernels/attn/p071_dsa_ws.cuh"   // P071 dsa_attn_ws
#include "kernels/attn/p073_bmm_f8.cuh"   // P073 dsa_bmm_f8
#include "kernels/attn/p091_select_pf.cuh"  // P091 dsa_select_pf
#include "kernels/attn/p092_attn_pf.cuh"    // P092 dsa_attn_pf
#include "kernels/attn/p093_bmm_pf.cuh"     // P093 dsa_bmm_pf
#include "kernels/attn/p110_attn_ep.cuh"    // P110 dsa_attn_ep
#include "kernels/attn/p153_select_long.cuh"  // P153 dsa_select_long

}  // namespace

void dsa_mla_rows(const bf16* kvn, int ldn, const bf16* kpe, int ldpe, const RowMeta* meta, const float* inv8, int M,
                  DsaCache c, cudaStream_t s) {
  if (M <= 0) return;
  k_mla_rows<<<M, 64, 0, s>>>(kvn, ldn, kpe, ldpe, meta, inv8, c);
  CUDA_CHECK(cudaGetLastError());
}

void dsa_q_pe(const bf16* q, const RowMeta* meta, const float* inv8, int M, bf16* q576, cudaStream_t s) {
  if (M <= 0) return;
  k_q_pe<<<M * kLH, 8, 0, s>>>(q, meta, inv8, q576);
  CUDA_CHECK(cudaGetLastError());
}

void dsa_bmm(const bf16* x, long xm, long xh, const bf16* w, int H, int N, int K, int M, bf16* y, long ym, long yh,
             cudaStream_t s) {
  if (M <= 0) return;
  if (N % 64 || K % 64) qlog::fail("dsa", "bmm %d×%d", N, K);
  if (patch::on(patch::dsa_bmm_dec) && p048::run(x, xm, xh, w, H, N, K, M, y, ym, yh, s)) return;
  if (patch::on(patch::dsa_bmm_pf) && p093::run(x, xm, xh, w, H, N, K, M, y, ym, yh, s)) return;
  if (M <= 16) k_bmm<16><<<dim3(H, N / 64, 1), 128, 0, s>>>(x, xm, xh, w, N, K, M, y, ym, yh);
  else k_bmm<64><<<dim3(H, N / 64, cdiv(M, 64)), 128, 0, s>>>(x, xm, xh, w, N, K, M, y, ym, yh);
  CUDA_CHECK(cudaGetLastError());
}

bool dsa_bmm_f8(const bf16* x, long xm, long xh, const uint8_t* w8, const float* sc, int mode, int H, int N, int K,
                int M, bf16* y, long ym, long yh, cudaStream_t s) {
  if (M <= 0 || !patch::on(patch::dsa_bmm_f8)) return false;
  return p073::run(x, xm, xh, w8, sc, mode, H, N, K, M, y, ym, yh, s);
}

void dsa_idx_prep(const bf16* kw, const bf16* qi, const bf16* knw, const bf16* knb, const RowMeta* meta,
                  const float* inv8, int M, DsaCache c, uint8_t* qf4, uint8_t* qs, float* wts, cudaStream_t s,
                  bool q) {
  if (M <= 0) return;
  k_idx_prep<<<dim3(M, q ? 65 : 1), 128, 0, s>>>(kw, qi, knw, knb, meta, inv8, c, qf4, qs, wts);  // (y = 0: K, weights)
  CUDA_CHECK(cudaGetLastError());
}

void dsa_select(const uint8_t* qf4, const uint8_t* qs, const float* wts, const RowMeta* meta, DsaCache c, int M,
                int max_pos, int* list, int* cnt, float* ws, size_t ws_bytes, cudaStream_t s) {
  if (M <= 0) return;
  if (patch::on(patch::dsa_select_long) && p153::run(qf4, qs, wts, meta, c, M, max_pos, list, cnt, ws, ws_bytes, s))
    return;
  if (patch::on(patch::dsa_select_pf) && p091::run(qf4, qs, wts, meta, c, M, max_pos, list, cnt, ws, ws_bytes, s))
    return;
  if (patch::on(patch::dsa_select_fast) && p046::run(qf4, qs, wts, meta, c, M, max_pos, list, cnt, ws, ws_bytes, s))
    return;
  const int ldl = std::max(64, max_pos - kWindow + 1 + 63) / 64 * 64;
  const int pass = int(std::min<size_t>(M, ws_bytes / ((size_t)ldl * 4)));
  if (pass < 1) qlog::fail("dsa", "indexer workspace too small");
  for (int r0 = 0; r0 < M; r0 += pass) {
    const int n = std::min(pass, M - r0);
    if (max_pos - kWindow + 1 > kIdxTop) {
      k_idx_logits<<<dim3(n, ldl / 64), 128, 0, s>>>(qf4, qs, wts, meta, c, r0, ldl, ws);
      CUDA_CHECK(cudaGetLastError());
    }
    k_topk<<<n, 512, 0, s>>>(meta, r0, ldl, ws, list, cnt);
    CUDA_CHECK(cudaGetLastError());
  }
}

void dsa_select_tail(const RowMeta* meta, int M, int* list, int* cnt, cudaStream_t s) {
  if (M <= 0) return;
  k_select_tail<<<M, kWindow, 0, s>>>(meta, list, cnt);
  CUDA_CHECK(cudaGetLastError());
}

void dsa_attn(const bf16* q576, const int* list, const int* cnt, const RowMeta* meta, DsaCache c, const float* sink,
              int M, int max_keys, bf16* olat, float* ws, size_t ws_bytes, cudaStream_t s) {
  if (M <= 0) return;
  if (patch::on(patch::dsa_attn_ep)) {
    static int nsm = 0;
    if (!nsm) CUDA_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
    if (p110::wanted(M, max_keys, nsm))
      return p110::run(q576, list, cnt, meta, c, sink, M, max_keys, olat, ws, ws_bytes, s);
  }
  if (patch::on(patch::dsa_attn_pf)) {
    static int nsm = 0;
    if (!nsm) CUDA_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
    if (p092::wanted(M, max_keys, nsm))
      return p092::run(q576, list, cnt, meta, c, sink, M, max_keys, olat, ws, ws_bytes, s);
  }
  if (patch::on(patch::dsa_attn_ws)) {
    static int nsm = 0;
    if (!nsm) CUDA_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
    if (p071::wanted(M, max_keys, nsm))
      return p071::run(q576, list, cnt, meta, c, sink, M, max_keys, olat, ws, ws_bytes, s);
  }
  if (patch::on(patch::dsa_attn_fast))
    return p045::run(q576, list, cnt, meta, c, sink, M, max_keys, olat, ws, ws_bytes, s);
  const int nch = std::max(1, (max_keys + kChunk - 1) / kChunk);
  const size_t per_row = (size_t)nch * (2 * kLH + kLH * kKvLora) * 4;
  const int pass = int(std::min<size_t>(M, ws_bytes / per_row));
  if (pass < 1) qlog::fail("dsa", "attention workspace too small");
  static bool attr = false;
  if (!attr) {
    CUDA_CHECK(cudaFuncSetAttribute(k_mla_attn, cudaFuncAttributeMaxDynamicSharedMemorySize, int(sizeof(MlaSmem))));
    attr = true;
  }
  for (int r0 = 0; r0 < M; r0 += pass) {
    const int n = std::min(pass, M - r0);
    k_mla_attn<<<dim3(n, nch), 256, sizeof(MlaSmem), s>>>(q576, list, cnt, meta, c, r0, nch, ws);
    CUDA_CHECK(cudaGetLastError());
    k_mla_merge<<<n, 256, 0, s>>>(cnt, sink, r0, nch, ws, olat);
    CUDA_CHECK(cudaGetLastError());
  }
}

void dsa_vha_gate(const bf16* o, const float* z, const float* V, const bf16* gate, int M, bf16* y, cudaStream_t s) {
  if (M <= 0) return;
  if (M > 16 && patch::on(patch::dsa_vha_fast)) return p047::run(o, z, V, gate, M, y, s);
  k_vha_gate<<<dim3(M, kLH), kMlaV, 0, s>>>(o, z, V, gate, y);
  CUDA_CHECK(cudaGetLastError());
}


}  // namespace e55
