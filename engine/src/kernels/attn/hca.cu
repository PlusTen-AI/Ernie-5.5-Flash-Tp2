// hca.cu — ERNIE HCA attention kernels (contract: hca.h).
#include "kernels/attn/hca.h"

#include <algorithm>

#include "core/log.h"
#include "kernels/attn/attn_common.cuh"
#include "kernels/gemm/mma.cuh"
#include "model/patches.h"

namespace e55 {
namespace {

constexpr int kRowB = kKvRowBytes;  // 584
constexpr int kLocalHeads = kHeads / 2;
constexpr int kChunk = 256;         // keys per attention CTA (fixed: part of every row's arithmetic)
constexpr int kKT = 32;             // keys per tile
constexpr int kQS = kHeadDim + 8;   // padded smem row (bf16): 1040 B, conflict-free ldmatrix

// ---- KV rows -----------------------------------------------------------------------------------------------------
// 64 threads per row, thread i: dims 8i … 8i+7; quant groups of 64 dims = 8 threads
__global__ void k_kv_rows(const bf16* __restrict__ kv, int ld, const RowMeta* __restrict__ meta,
                          const float* __restrict__ inv, uint8_t* rows) {
  const int r = blockIdx.x, i = threadIdx.x, d0 = 8 * i;
  const int p = meta[r].pos;
  float v[8];
  load_bf16x8(kv + (size_t)r * ld + d0, v);
  if (d0 >= kNope) rope_pairs8(v, d0 - kNope, p, inv, false);
  uint8_t* row = rows + (size_t)r * kRowB;
  store_kv_row8(row, d0, v);
}

// ---- q prep ------------------------------------------------------------------------------------------------------
__global__ void k_q_prep(bf16* q, const RowMeta* __restrict__ meta, const float* __restrict__ inv) {
  const int r = blockIdx.x / kLocalHeads, i = threadIdx.x, d0 = 8 * i;
  bf16* h = q + (size_t)blockIdx.x * kHeadDim;
  float v[8], ss = 0.f;
  load_bf16x8(h + d0, v);
#pragma unroll
  for (int j = 0; j < 8; j++) ss += v[j] * v[j];
  __shared__ float red[2];
  ss = warp_sum(ss);
  if ((i & 31) == 0) red[i >> 5] = ss;
  __syncthreads();
  const float rs = rsqrtf(__fdiv_rn(red[0] + red[1], float(kHeadDim)) + kEps);
#pragma unroll
  for (int j = 0; j < 8; j++) v[j] *= rs;
  if (d0 >= kNope) rope_pairs8(v, d0 - kNope, meta[r].pos, inv, false);
  store_bf16x8(h + d0, v);
}

// ---- compressor --------------------------------------------------------------------------------------------------
// one block of 512 threads (thread = dim) per row that closes a group
__global__ void __launch_bounds__(512) k_compress(const float* __restrict__ ckv, const float* __restrict__ ape,
                                                  const bf16* __restrict__ nw, const RowMeta* __restrict__ meta,
                                                  const float* __restrict__ inv, HcaCache c) {
  const RowMeta mt = meta[blockIdx.x];
  const int p = mt.pos;
  if ((p + 1) % kRatio) return;
  const int d = threadIdx.x, g = p / kRatio, t0 = g * kRatio;
  const float* st = c.state + (size_t)mt.slot * kRatio * 1024;
  auto score = [&](int t) {
    return t >= mt.p0 ? ckv[(size_t)(mt.r0 + t - mt.p0) * 1024 + 512 + d] + ape[(t % kRatio) * 512 + d]
                      : st[(t % kRatio) * 1024 + 512 + d];
  };
  auto kvv = [&](int t) {
    return t >= mt.p0 ? ckv[(size_t)(mt.r0 + t - mt.p0) * 1024 + d] : st[(t % kRatio) * 1024 + d];
  };
  float mx = -INFINITY;
  for (int t = t0; t <= p; t++) mx = fmaxf(mx, score(t));
  float sum = 0.f;
  for (int t = t0; t <= p; t++) sum += expf(score(t) - mx);
  float cv = 0.f;
  for (int t = t0; t <= p; t++) cv += __fdiv_rn(expf(score(t) - mx), sum) * kvv(t);
  __shared__ float red[16], cn[512];
  float ss = warp_sum(cv * cv);
  if ((d & 31) == 0) red[d >> 5] = ss;
  __syncthreads();
  if (d < 32) {
    float x = d < 16 ? red[d] : 0.f;
    x = warp_sum(x);
    if (d == 0) red[0] = x;
  }
  __syncthreads();
  const float rs = rsqrtf(__fdiv_rn(red[0], 512.f) + kEps);
  cn[d] = cv * rs * __bfloat162float(nw[d]);
  __syncthreads();
  // 64 threads write the row (8 dims each), like k_kv_rows
  if (d < 64) {
    const int d0 = 8 * d;
    float v[8];
#pragma unroll
    for (int j = 0; j < 8; j++) v[j] = cn[d0 + j];
    if (d0 >= kNope) rope_pairs8(v, d0 - kNope, t0, inv, false);
    store_kv_row8(c.comp + ((size_t)mt.slot * c.max_comp + g) * kRowB, d0, v);
  }
}

// the state takes the rows of each sequence's open group (after k_compress has read the old ones)
__global__ void k_state(const float* __restrict__ ckv, const float* __restrict__ ape, const RowMeta* __restrict__ meta,
                        HcaCache c) {
  const RowMeta mt = meta[blockIdx.x];
  if (mt.pos < (mt.pend + 1) / kRatio * kRatio) return;
  float* st = c.state + ((size_t)mt.slot * kRatio + mt.pos % kRatio) * 1024;
  const float* src = ckv + (size_t)blockIdx.x * 1024;
  for (int d = threadIdx.x; d < 1024; d += blockDim.x)
    st[d] = d < 512 ? src[d] : src[d] + ape[(mt.pos % kRatio) * 512 + d - 512];
}

__global__ void k_ring(const uint8_t* __restrict__ rows, const RowMeta* __restrict__ meta, HcaCache c) {
  const RowMeta mt = meta[blockIdx.x];
  if (mt.pos <= mt.pend - kWindow) return;
  const uint4* src = reinterpret_cast<const uint4*>(rows + (size_t)blockIdx.x * kRowB);
  uint8_t* dst = c.ring + ((size_t)mt.slot * kWindow + mt.pos % kWindow) * kRowB;
  for (int i = threadIdx.x; i < kRowB / 8; i += blockDim.x)
    reinterpret_cast<uint2*>(dst)[i] = reinterpret_cast<const uint2*>(src)[i];
}

// ---- attention ---------------------------------------------------------------------------------------------------
struct AttnSmem {
  bf16 q[kLocalHeads][kQS];
  bf16 k[kKT][kQS];
  float s[kLocalHeads][kKT + 1];
  bf16 p[kLocalHeads][kKT + 8];
  float m[kLocalHeads], l[kLocalHeads], fac[kLocalHeads];
};

// grid (rows of the pass, chunks); 256 threads = 8 warps
__global__ void __launch_bounds__(256) k_attn(const bf16* __restrict__ q, const uint8_t* __restrict__ rows,
                                              const RowMeta* __restrict__ meta, HcaCache c, int row0, int nch,
                                              float* ws) {
  extern __shared__ __align__(16) uint8_t smraw[];
  AttnSmem& S = *reinterpret_cast<AttnSmem*>(smraw);
  const int r = row0 + blockIdx.x, ch = blockIdx.y;
  const RowMeta mt = meta[r];
  const int p = mt.pos, ncomp = (p + 1) / kRatio, nwin = min(p + 1, kWindow), nk = ncomp + nwin;
  const int kbeg = ch * kChunk;
  if (kbeg >= nk) return;
  const int kend = min(nk, kbeg + kChunk);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  // Q: 32 heads × 512
  for (int i = tid; i < kLocalHeads * kHeadDim / 8; i += 256) {
    const int h = i / (kHeadDim / 8), c8 = i % (kHeadDim / 8);
    *reinterpret_cast<uint4*>(&S.q[h][c8 * 8]) =
        *reinterpret_cast<const uint4*>(q + ((size_t)r * kLocalHeads + h) * kHeadDim + c8 * 8);
  }
  if (tid < kLocalHeads) S.m[tid] = -INFINITY, S.l[tid] = 0.f;
  float o[2][8][4];
#pragma unroll
  for (int a = 0; a < 2; a++)
#pragma unroll
    for (int b = 0; b < 8; b++) o[a][b][0] = o[a][b][1] = o[a][b][2] = o[a][b][3] = 0.f;
  const float scale = rsqrtf(float(kHeadDim));
  for (int k0 = kbeg; k0 < kend; k0 += kKT) {
    __syncthreads();
    {  // K tile: thread (key tid/8, part tid%8) dequantizes 64 dims
      const int kk = tid >> 3, part = tid & 7, key = k0 + kk;
      bf16* dst = &S.k[kk][part * 64];
      if (key < kend) {
        const uint8_t* row;
        if (key < ncomp) {
          row = c.comp + ((size_t)mt.slot * c.max_comp + key) * kRowB;
        } else {
          const int t = p - nwin + 1 + (key - ncomp);
          row = t >= mt.p0 ? rows + (size_t)(mt.r0 + t - mt.p0) * kRowB
                           : c.ring + ((size_t)mt.slot * kWindow + t % kWindow) * kRowB;
        }
        dequant_kv_part(row, part, dst);
      } else {
        for (int j = 0; j < 64; j += 8) *reinterpret_cast<uint4*>(dst + j) = make_uint4(0, 0, 0, 0);
      }
    }
    __syncthreads();
    {  // S = Q·Kᵀ: warp → heads 16·(warp/4) …, keys 8·(warp%4) …
      const int mt16 = warp >> 2, nt8 = warp & 3;
      float sacc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 8
      for (int kk = 0; kk < kHeadDim / 16; kk++) {
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
    if (tid < kLocalHeads) {  // online softmax, one thread per head
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
    {  // O = O·fac + P·V: warp → dims 64·warp …
      const int g = lane >> 2;
#pragma unroll
      for (int a = 0; a < 2; a++) {
        const float f0 = S.fac[16 * a + g], f1 = S.fac[16 * a + g + 8];
#pragma unroll
        for (int b = 0; b < 8; b++) o[a][b][0] *= f0, o[a][b][1] *= f0, o[a][b][2] *= f1, o[a][b][3] *= f1;
      }
#pragma unroll
      for (int ks = 0; ks < kKT / 16; ks++) {
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
  }
  __syncthreads();
  // partial: [m 32][l 32][O 32 × 512] fp32
  float* w = ws + ((size_t)blockIdx.x * nch + ch) * (2 * kLocalHeads + kLocalHeads * kHeadDim);
  if (tid < kLocalHeads) w[tid] = S.m[tid], w[kLocalHeads + tid] = S.l[tid];
  float* wo = w + 2 * kLocalHeads;
  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int a = 0; a < 2; a++)
#pragma unroll
    for (int b = 0; b < 8; b++) {
      const int h0 = 16 * a + g, d = 64 * warp + 8 * b + 2 * t;
      wo[h0 * kHeadDim + d] = o[a][b][0], wo[h0 * kHeadDim + d + 1] = o[a][b][1];
      wo[(h0 + 8) * kHeadDim + d] = o[a][b][2], wo[(h0 + 8) * kHeadDim + d + 1] = o[a][b][3];
    }
}

// merge the chunks of a row with the sink, in chunk order; 256 threads, thread: head tid/8, dims (tid%8)·64 …
__global__ void __launch_bounds__(256) k_merge(const RowMeta* __restrict__ meta, const float* __restrict__ sink,
                                               int row0, int nch, const float* __restrict__ ws, bf16* o) {
  const int r = row0 + blockIdx.x, tid = threadIdx.x, h = tid >> 3, part = tid & 7;
  const int p = meta[r].pos, nk = (p + 1) / kRatio + min(p + 1, kWindow), n = (nk + kChunk - 1) / kChunk;
  const size_t stride = 2 * kLocalHeads + kLocalHeads * kHeadDim;
  const float* w = ws + (size_t)blockIdx.x * nch * stride;
  float M = sink[h];
  for (int c = 0; c < n; c++) M = fmaxf(M, w[c * stride + h]);
  float den = 0.f;
  for (int c = 0; c < n; c++) den += expf(w[c * stride + h] - M) * w[c * stride + kLocalHeads + h];
  den += expf(sink[h] - M);
  for (int d = part * 64; d < part * 64 + 64; d += 2) {
    float a = 0.f, b = 0.f;
    for (int c = 0; c < n; c++) {
      const float f = expf(w[c * stride + h] - M);
      const float* x = w + c * stride + 2 * kLocalHeads + h * kHeadDim + d;
      a += f * x[0], b += f * x[1];
    }
    *reinterpret_cast<uint32_t*>(o + ((size_t)r * kLocalHeads + h) * kHeadDim + d) =
        pack_bf16x2(__fdiv_rn(a, den), __fdiv_rn(b, den));
  }
}

// ---- VHA ---------------------------------------------------------------------------------------------------------
// zp[m][r][d] = Σ_h U[h][r] · o[m][h][d]; grid (M, D/128), 128 threads
__global__ void k_vha_partial(const bf16* __restrict__ o, int D, const float* __restrict__ U, float* zp) {
  const int m = blockIdx.x, d = blockIdx.y * 128 + threadIdx.x;
  float x[kLocalHeads];
#pragma unroll
  for (int h = 0; h < kLocalHeads; h++) x[h] = __bfloat162float(o[((size_t)m * kLocalHeads + h) * D + d]);
#pragma unroll
  for (int r = 0; r < kVhaRank; r++) {
    float a = 0.f;
#pragma unroll
    for (int h = 0; h < kLocalHeads; h++) a += U[h * kVhaRank + r] * x[h];
    zp[((size_t)m * kVhaRank + r) * D + d] = a;
  }
}

// grid (M, 32 heads), 64 threads × 8 dims
__global__ void k_hca_vha_out(const bf16* __restrict__ o, const float* __restrict__ z, const float* __restrict__ V,
                              const RowMeta* __restrict__ meta, const float* __restrict__ inv, int M, uint8_t* q,
                              uint8_t* sc) {
  const int m = blockIdx.x, h = blockIdx.y, d0 = 8 * threadIdx.x;
  float v[8];
  load_bf16x8(o + ((size_t)m * kLocalHeads + h) * kHeadDim + d0, v);
#pragma unroll
  for (int j = 0; j < 8; j++) {
    float a = 0.f;
#pragma unroll
    for (int r = 0; r < kVhaRank; r++) a += V[h * kVhaRank + r] * z[((size_t)m * kVhaRank + r) * kHeadDim + d0 + j];
    v[j] = __bfloat162float(__float2bfloat16_rn(v[j] + a));
  }
  if (d0 >= kNope) rope_pairs8(v, d0 - kNope, meta[m].pos, inv, true);
  // FP8 per 128 dims = 16 threads
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

}  // namespace
}  // namespace e55

#include "kernels/attn/p040_hca_vha_out.cuh"
#include "kernels/attn/p041_hca_attn.cuh"
#include "kernels/attn/p042_hca_fuse.cuh"
#include "kernels/attn/p043_hca_attn_dec.cuh"
#include "kernels/attn/p070_hca_ws.cuh"
#include "kernels/attn/p088_hca_attn_pf.cuh"
#include "kernels/attn/p089_hca_pre_pf.cuh"
#include "kernels/attn/p150_hca_attn_long.cuh"
#include "kernels/attn/p172_hca_attn_long2.cuh"
#include "kernels/attn/p180_hca_attn_long3.cuh"

namespace e55 {

void hca_q_prep(bf16* q, const RowMeta* meta, const float* inv_freq, int M, cudaStream_t s) {
  if (M <= 0) return;
  k_q_prep<<<M * kLocalHeads, 64, 0, s>>>(q, meta, inv_freq);
  CUDA_CHECK(cudaGetLastError());
}

void hca_kv_rows(const bf16* kv, int ldkv, const RowMeta* meta, const float* inv_freq, int M, uint8_t* rows,
                 cudaStream_t s) {
  if (M <= 0) return;
  k_kv_rows<<<M, 64, 0, s>>>(kv, ldkv, meta, inv_freq, rows);
  CUDA_CHECK(cudaGetLastError());
}

void hca_compress(const float* ckv, const float* ape, const bf16* normw, const RowMeta* meta, const float* inv_freq,
                  int M, HcaCache c, cudaStream_t s) {
  if (M <= 0) return;
  k_compress<<<M, 512, 0, s>>>(ckv, ape, normw, meta, inv_freq, c);
  CUDA_CHECK(cudaGetLastError());
  k_state<<<M, 256, 0, s>>>(ckv, ape, meta, c);
  CUDA_CHECK(cudaGetLastError());
}

void hca_attn(const bf16* q, const uint8_t* rows, const RowMeta* meta, HcaCache c, const float* sink, int M,
              int max_keys, bf16* o, float* ws, size_t ws_bytes, cudaStream_t s) {
  if (M <= 0) return;
  if (M <= p043::kMaxM && patch::on(patch::hca_attn_dec))
    return p043::launch<false, false>(q, rows, meta, c, sink, nullptr, nullptr, M, max_keys, o, nullptr, ws, ws_bytes, s);
  if (M > 8 && patch::on(patch::hca_attn_tc) && patch::on(patch::hca_attn_ws))  // P070 (P041's 512-dim launches)
    return p070::launch<false, false>(q, rows, meta, c, sink, nullptr, nullptr, M, max_keys, o, nullptr, ws, ws_bytes, s);
  if (patch::on(patch::hca_attn_tc))
    return p041::attn<false, false>(q, rows, meta, c, sink, nullptr, nullptr, M, max_keys, o, nullptr, ws, ws_bytes, s);
  const int nch = std::max(1, (max_keys + kChunk - 1) / kChunk);
  const size_t per_row = (size_t)nch * (2 * kLocalHeads + kLocalHeads * kHeadDim) * 4;
  const int pass = int(std::min<size_t>(M, ws_bytes / per_row));
  if (pass < 1) qlog::fail("hca", "attention workspace %zu B < one row's %zu B", ws_bytes, per_row);
  static bool attr = false;
  if (!attr) {
    CUDA_CHECK(cudaFuncSetAttribute(k_attn, cudaFuncAttributeMaxDynamicSharedMemorySize, int(sizeof(AttnSmem))));
    attr = true;
  }
  for (int r0 = 0; r0 < M; r0 += pass) {
    const int n = std::min(pass, M - r0);
    k_attn<<<dim3(n, nch), 256, sizeof(AttnSmem), s>>>(q, rows, meta, c, r0, nch, ws);
    CUDA_CHECK(cudaGetLastError());
    k_merge<<<n, 256, 0, s>>>(meta, sink, r0, nch, ws, o);
    CUDA_CHECK(cudaGetLastError());
  }
}

void hca_ring_store(const uint8_t* rows, const RowMeta* meta, int M, HcaCache c, cudaStream_t s) {
  if (M <= 0) return;
  k_ring<<<M, 64, 0, s>>>(rows, meta, c);
  CUDA_CHECK(cudaGetLastError());
}

void vha_partial(const bf16* o, int D, const float* U, int M, float* zp, cudaStream_t s) {
  if (M <= 0) return;
  k_vha_partial<<<dim3(M, D / 128), 128, 0, s>>>(o, D, U, zp);
  CUDA_CHECK(cudaGetLastError());
}

void hca_vha_out(const bf16* o, const float* z, const float* V, const RowMeta* meta, const float* inv_freq, int M,
                 uint8_t* q, uint8_t* sc, cudaStream_t s) {
  if (M <= 0) return;
  if (patch::on(patch::hca_vha_row))
    return p040::launch<false>(o, z, V, meta, inv_freq, M, q, sc, nullptr, nullptr, nullptr, HcaCache{}, s);
  k_hca_vha_out<<<dim3(M, kLocalHeads), 64, 0, s>>>(o, z, V, meta, inv_freq, M, q, sc);
  CUDA_CHECK(cudaGetLastError());
}

// ---- P042 hca_fuse -------------------------------------------------------------------------------------------------
void hca_pre(const bf16* kv, int ldkv, const float* ckv, const float* ape, const bf16* normw, const RowMeta* meta,
             const float* inv_freq, int M, uint8_t* rows, HcaCache c, float* ws, size_t ws_bytes, cudaStream_t s) {
  if (M <= 0) return;
  if (M <= p042::kPreDecMaxM && size_t(M) * 64 <= ws_bytes && patch::on(patch::hca_compress_dec)) {
    static bool attr = false;
    if (!attr) {
      CUDA_CHECK(cudaFuncSetAttribute(p042::k_pre_dec, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      int(sizeof(p042::CompSmem))));
      attr = true;
    }
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 8, at[0].val.clusterDim.y = 1, at[0].val.clusterDim.z = 1;
    cudaLaunchConfig_t cfg{dim3(8, M), dim3(512), sizeof(p042::CompSmem), s, at, 1};
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, p042::k_pre_dec, kv, ldkv, ckv, ape, normw, meta, inv_freq, rows, c, ws));
    return;
  }
  if (M >= p089::kMinM && size_t(M) * 64 <= ws_bytes && patch::on(patch::hca_pre_pf))  // P089
    return p089::launch(kv, ldkv, ckv, ape, normw, meta, inv_freq, M, rows, c, ws, s);
  const int nkv = (M + 7) / 8;
  p042::k_pre<<<nkv + M, 512, 0, s>>>(kv, ldkv, ckv, ape, normw, meta, inv_freq, M, nkv, rows, c);
  CUDA_CHECK(cudaGetLastError());
}

// P150 / P172 dispatch: max_keys of ≥ g_long_minch / g_long2_minch chunks, ≤ g_long_G chunks per work item (0: by the
// chunk count); g_long2_cap > 0: P172 on at most that many CTAs (tests only: the overlap gate's SM split)
static int g_long_G = 0, g_long_minch = p150::kMinCh, g_long2_minch = p172::kMinCh, g_long2_cap = 0;
void hca_attn_long_tune(int G, int min_ch) {
  if (G > 0) g_long_G = G;
  if (min_ch > 0) g_long_minch = min_ch;
}
void hca_attn_long2_tune(int min_ch, int cap) {  // tests (declared by the test bindings)
  if (min_ch > 0) g_long2_minch = min_ch;
  g_long2_cap = cap;
}

void hca_attn_fused(const bf16* q, const uint8_t* rows, const RowMeta* meta, HcaCache c, const float* sink,
                    const float* inv_freq, const float* U, int M, int max_keys, bf16* o, float* zp, float* ws,
                    size_t ws_bytes, cudaStream_t s) {
  if (M <= 0) return;
  if (M <= p043::kMaxM && patch::on(patch::hca_attn_dec))
    return p043::launch<true, true>(q, rows, meta, c, sink, inv_freq, U, M, max_keys, o, zp, ws, ws_bytes, s);
  const int nchk = (max_keys + kChunk - 1) / kChunk;
  if (M >= p088::kMinM && nchk >= p180::kMinCh && nchk <= p180::kMaxCh && patch::on(patch::hca_attn_long3))
    return p180::launch(q, rows, meta, c, sink, inv_freq, U, M, max_keys, o, zp, ws, ws_bytes, g_long_G, s);  // P180
  if (M >= p088::kMinM && nchk >= g_long2_minch && nchk <= p172::kMaxCh && patch::on(patch::hca_attn_long2))
    return p172::launch(q, rows, meta, c, sink, inv_freq, U, M, max_keys, o, zp, ws, ws_bytes, g_long_G, g_long2_cap,
                        s);  // P172
  if (M >= p088::kMinM && nchk >= g_long_minch && nchk <= p150::kMaxCh && patch::on(patch::hca_attn_long))
    return p150::launch(q, rows, meta, c, sink, inv_freq, U, M, max_keys, o, zp, ws, ws_bytes, g_long_G, s);  // P150
  if (M >= p088::kMinM && patch::on(patch::hca_attn_pf))  // P088 (prefill)
    return p088::launch(q, rows, meta, c, sink, inv_freq, U, M, max_keys, o, zp, ws, ws_bytes, s);
  if (M > 8 && patch::on(patch::hca_attn_ws))  // P070 (P041's 512-dim launches)
    return p070::launch<true, true>(q, rows, meta, c, sink, inv_freq, U, M, max_keys, o, zp, ws, ws_bytes, s);
  p041::attn<true, true>(q, rows, meta, c, sink, inv_freq, U, M, max_keys, o, zp, ws, ws_bytes, s);
}

void hca_post(const bf16* o, const float* z, const float* V, const RowMeta* meta, const float* inv_freq,
              const uint8_t* rows, const float* ckv, const float* ape, int M, HcaCache c, uint8_t* q, uint8_t* sc,
              cudaStream_t s) {
  if (M <= 0) return;
  p040::launch<true>(o, z, V, meta, inv_freq, M, q, sc, rows, ckv, ape, c, s);
}
void hca_vha_own(const bf16* om, const bf16* op, const float* zp, const float* Up, const float* Vm, const float* Vp,
                 const RowMeta* meta, const float* inv_freq, int M, uint8_t* qm, uint8_t* sm, uint8_t* qp, uint8_t* sp,
                 cudaStream_t s) {  // P100
  if (M <= 0) return;
  p042::k_vha_own<<<M, 64, 0, s>>>(om, op, zp, Up, Vm, Vp, meta, inv_freq, M, qm, sm, qp, sp);
  CUDA_CHECK(cudaGetLastError());
}
void hca_post_state(const RowMeta* meta, const uint8_t* rows, const float* ckv, const float* ape, int M, HcaCache c,
                    cudaStream_t s) {  // P100
  if (M <= 0) return;
  p042::k_post_state<<<M, 64, 0, s>>>(meta, rows, ckv, ape, c);
  CUDA_CHECK(cudaGetLastError());
}


}  // namespace e55
