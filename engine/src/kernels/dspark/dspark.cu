// dspark.cu — DSpark draft kernels and the verify rollback (contract: dspark.h).
#include "kernels/dspark/dspark.h"

#include <cfloat>
#include <climits>

#include "core/log.h"
#include "kernels/attn/attn_common.cuh"
#include "kernels/gemm/mma.cuh"

namespace e55 {
namespace {

constexpr int kRowB = kKvRowBytes;  // 584
constexpr int kLH = kHeads / 2;     // this rank's heads
constexpr int kKT = 32;             // keys per tile
constexpr int kQS = kHeadDim + 8;   // padded smem row (bf16)

// ---- aux features ------------------------------------------------------------------------------------------------
__global__ void k_aux(const bf16* __restrict__ R, const int* __restrict__ rows, int which, bf16* aux) {
  const int k = blockIdx.x, h0 = 16 * threadIdx.x;
  const bf16* r = R + (size_t)rows[k] * kHcDim;
#pragma unroll
  for (int part = 0; part < 2; part++) {
    const int h = h0 + 8 * part;
    float a[8], v[8];
    load_bf16x8(r + h, a);
#pragma unroll
    for (int j = 1; j < kStreams; j++) {
      load_bf16x8(r + j * kHidden + h, v);
#pragma unroll
      for (int i = 0; i < 8; i++) a[i] = __fadd_rn(a[i], v[i]);
    }
#pragma unroll
    for (int i = 0; i < 8; i++) a[i] = __fmul_rn(a[i], 0.25f);
    store_bf16x8(aux + (size_t)k * kDspAux + which * kHidden + h, a);
  }
}

// ---- context rows ------------------------------------------------------------------------------------------------
__global__ void k_ctx_rows(const bf16* __restrict__ kv, int ld, const int* __restrict__ pos,
                           const int* __restrict__ slot, const float* __restrict__ inv, uint8_t* ring, int slots) {
  const int k = blockIdx.x, st = blockIdx.y, d0 = 8 * threadIdx.x, p = pos[k];
  float v[8];
  load_bf16x8(kv + (size_t)k * ld + st * kHeadDim + d0, v);
  if (d0 >= kNope) rope_pairs8(v, d0 - kNope, p, inv, false);
  store_kv_row8(ring + (((size_t)st * slots + slot[k]) * kDspRing + p % kDspRing) * kRowB, d0, v);
}

// ---- attention ---------------------------------------------------------------------------------------------------
struct AttnSmem {
  bf16 q[kLH][kQS];
  bf16 k[kKT][kQS];
  float s[kLH][kKT + 1];
  bf16 p[kLH][kKT + 8];
  float m[kLH], l[kLH], fac[kLH];
};

// one CTA (256 threads = 8 warps) per block row; keys: context rows (ring) then the block's rows (non-causal)
__global__ void __launch_bounds__(256) k_attn(const bf16* __restrict__ q, const uint8_t* __restrict__ rows,
                                              const RowMeta* __restrict__ meta, const uint8_t* __restrict__ ring,
                                              const float* __restrict__ sink, bf16* out) {
  extern __shared__ __align__(16) uint8_t smraw[];
  AttnSmem& S = *reinterpret_cast<AttnSmem*>(smraw);
  const int r = blockIdx.x;
  const RowMeta mt = meta[r];
  const int nctx = min(mt.p0, kDspCtx), nblk = mt.pend - mt.p0 + 1, nk = nctx + nblk;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  for (int i = tid; i < kLH * kHeadDim / 8; i += 256) {
    const int h = i / (kHeadDim / 8), c8 = i % (kHeadDim / 8);
    *reinterpret_cast<uint4*>(&S.q[h][c8 * 8]) =
        *reinterpret_cast<const uint4*>(q + ((size_t)r * kLH + h) * kHeadDim + c8 * 8);
  }
  if (tid < kLH) S.m[tid] = -INFINITY, S.l[tid] = 0.f;
  float o[2][8][4];
#pragma unroll
  for (int a = 0; a < 2; a++)
#pragma unroll
    for (int b = 0; b < 8; b++) o[a][b][0] = o[a][b][1] = o[a][b][2] = o[a][b][3] = 0.f;
  const float scale = rsqrtf(float(kHeadDim));
  for (int k0 = 0; k0 < nk; k0 += kKT) {
    __syncthreads();
    {  // K tile: thread (key tid/8, part tid%8) dequantizes 64 dims
      const int kk = tid >> 3, part = tid & 7, key = k0 + kk;
      bf16* dst = &S.k[kk][part * 64];
      if (key < nk) {
        const uint8_t* row;
        if (key < nctx) {
          const int t = mt.p0 - nctx + key;
          row = ring + ((size_t)mt.slot * kDspRing + t % kDspRing) * kRowB;
        } else {
          row = rows + (size_t)(mt.r0 + key - nctx) * kRowB;
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
        S.s[h][kk] = k0 + kk < nk ? sacc[i] * scale : -INFINITY;
      }
    }
    __syncthreads();
    if (tid < kLH) {  // online softmax, one thread per head
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
  if (tid < kLH) {  // the sink: M = max(m, sink), den = e^(m−M)·l + e^(sink−M); fac = e^(m−M) / den
    const float sk = sink[tid], M = fmaxf(S.m[tid], sk), f = expf(S.m[tid] - M);
    S.fac[tid] = f, S.l[tid] = f * S.l[tid] + expf(sk - M);
  }
  __syncthreads();
  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int a = 0; a < 2; a++)
#pragma unroll
    for (int b = 0; b < 8; b++) {
      const int h0 = 16 * a + g, d = 64 * warp + 8 * b + 2 * t;
      bf16* o0 = out + ((size_t)r * kLH + h0) * kHeadDim + d;
      bf16* o1 = out + ((size_t)r * kLH + h0 + 8) * kHeadDim + d;
      *reinterpret_cast<uint32_t*>(o0) = pack_bf16x2(__fdiv_rn(S.fac[h0] * o[a][b][0], S.l[h0]),
                                                     __fdiv_rn(S.fac[h0] * o[a][b][1], S.l[h0]));
      *reinterpret_cast<uint32_t*>(o1) = pack_bf16x2(__fdiv_rn(S.fac[h0 + 8] * o[a][b][2], S.l[h0 + 8]),
                                                     __fdiv_rn(S.fac[h0 + 8] * o[a][b][3], S.l[h0 + 8]));
    }
}

// ---- Markov head -------------------------------------------------------------------------------------------------
constexpr int kMkRows = 128;  // vocabulary rows per CTA (8 warps × 4 rows × 4 passes)

__device__ __forceinline__ bool better(float v1, int i1, float v2, int i2) {  // torch.argmax: NaN max, first index
  const bool n1 = v1 != v1, n2 = v2 != v2;
  if (n1 != n2) return n1;
  if (!n1 && v1 != v2) return v1 > v2;
  return i1 < i2;
}

__device__ __forceinline__ float mmax(float x, const float* rt) {  // misc.cu k_multimax, same order
  const float a = fmaxf(rt[0] - x, 0.f), b = fmaxf(x - rt[1], 0.f), cc = fmaxf(rt[2] - x, 0.f), d = fmaxf(x - rt[3], 0.f);
  float y = __fadd_rn(x, __fmul_rn(rt[4], a));
  y = __fadd_rn(y, __fmul_rn(rt[5], b));
  y = __fadd_rn(y, __fmul_rn(rt[6], __fmul_rn(cc, cc)));
  return __fadd_rn(y, __fmul_rn(rt[7], __fmul_rn(d, d)));
}

// 256 threads; lane group j = lane / 8 (4 rows per warp pass), lane i = lane % 8 holds elements 32i … 32i+31 of the
// row; items k ≡ i (mod 8) are finished (bias → logit → argmax) by lane i of each group. The launch is CTAs g0 … of
// the sweep's nb (vocabulary block vb = g, or nb − 1 − g with back: P161 / P163); every value is per element, so
// neither the order nor the split changes a bit (part is indexed by vb)
template <bool F8>
__global__ void __launch_bounds__(256) k_markov(const float* __restrict__ base, const int* __restrict__ brow,
                                                const int* __restrict__ seq, const int* __restrict__ prev,
                                                const bf16* __restrict__ w1t, const bf16* __restrict__ w2,
                                                const uint8_t* __restrict__ w2q, const uint8_t* __restrict__ w2s,
                                                int Vh, int off, const float* __restrict__ mm, int K, float* out,
                                                const int* __restrict__ orow, float2* part, int g0, int nb, int back) {
  __shared__ float e[kDspMaxN][8 * 33];  // lane i's 32 values at 33·i: the 8 lanes of a row group on distinct banks
  __shared__ float2 red[8][kDspMaxN];
  __shared__ float rt[8];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gi = lane >> 3, li = lane & 7;
  for (int i = tid; i < K * kDspRank; i += 256) {
    const int k = i / kDspRank, c = i % kDspRank;
    e[k][(c >> 5) * 33 + (c & 31)] = __bfloat162float(w1t[(size_t)prev[seq[k]] * kDspRank + c]);
  }
  if (tid < 8) rt[tid] = mm[tid];
  __syncthreads();
  float bv[2] = {-INFINITY, -INFINITY};
  int bi[2] = {INT_MAX, INT_MAX};
  const int g = g0 + int(blockIdx.x), vb = back ? nb - 1 - g : g, v0 = vb * kMkRows;
  for (int pass = 0; pass < kMkRows / 32; pass++) {
    const int v = v0 + pass * 32 + warp * 4 + gi;
    float w[32];
    if (F8 && v < Vh) {
      const uint4* src = reinterpret_cast<const uint4*>(w2q + (size_t)v * kDspRank + 32 * li);
      const float sc = ue8m0_f32(w2s[(size_t)v * (kDspRank / 32) + li]);
#pragma unroll
      for (int q2 = 0; q2 < 2; q2++) {
        const uint4 u = src[q2];
        const uint32_t x[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int j = 0; j < 16; j++) {
          __nv_fp8_e4m3 e;
          e.__x = (x[j >> 2] >> (8 * (j & 3))) & 0xFF;
          w[16 * q2 + j] = float(e) * sc;
        }
      }
    } else if (v < Vh) {
      const uint4* src = reinterpret_cast<const uint4*>(w2 + (size_t)v * kDspRank + 32 * li);
#pragma unroll
      for (int q4 = 0; q4 < 4; q4++) {
        const uint4 u = src[q4];
        const uint32_t x[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int j = 0; j < 4; j++) w[8 * q4 + 2 * j] = bf16_lo(x[j]), w[8 * q4 + 2 * j + 1] = bf16_hi(x[j]);
      }
    } else {
#pragma unroll
      for (int j = 0; j < 32; j++) w[j] = 0.f;
    }
#pragma unroll
    for (int k = 0; k < kDspMaxN; k++) {
      if (k >= K) break;
      float acc = 0.f;
#pragma unroll
      for (int j = 0; j < 32; j++) acc = fmaf(w[j], e[k][33 * li + j], acc);
      acc += __shfl_xor_sync(0xffffffffu, acc, 1);
      acc += __shfl_xor_sync(0xffffffffu, acc, 2);
      acc += __shfl_xor_sync(0xffffffffu, acc, 4);
      if ((k & 7) == li && v < Vh) {
        const float y = mmax(__fadd_rn(base[(size_t)brow[k] * Vh + v], acc), rt);
        if (out) out[(size_t)orow[k] * Vh + v] = y;
        if (better(y, off + v, bv[k >> 3], bi[k >> 3])) bv[k >> 3] = y, bi[k >> 3] = off + v;
      }
    }
  }
  // the 4 row groups of a warp hold the same items: lanes li, li+8, li+16, li+24
#pragma unroll
  for (int h = 0; h < 2; h++)
#pragma unroll
    for (int o = 8; o < 32; o <<= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv[h], o);
      const int oi = __shfl_xor_sync(0xffffffffu, bi[h], o);
      if (better(ov, oi, bv[h], bi[h])) bv[h] = ov, bi[h] = oi;
    }
  if (gi == 0)
#pragma unroll
    for (int h = 0; h < 2; h++)
      if (8 * h + li < K) red[warp][8 * h + li] = make_float2(bv[h], __int_as_float(bi[h]));
  __syncthreads();
  if (tid < K) {
    float v = red[0][tid].x;
    int i = __float_as_int(red[0][tid].y);
    for (int w = 1; w < 8; w++) {
      const float ov = red[w][tid].x;
      const int oi = __float_as_int(red[w][tid].y);
      if (better(ov, oi, v, i)) v = ov, i = oi;
    }
    part[(size_t)tid * nb + vb] = make_float2(v, __int_as_float(i));
  }
}

// P083: one CTA (64 threads) per item, a thread per candidate
__global__ void __launch_bounds__(64) k_markov_topk(const SparseCand* __restrict__ cands, const int* __restrict__ brow,
                                                    const int* __restrict__ seq, const int* __restrict__ prev,
                                                    const bf16* __restrict__ w1t, const bf16* __restrict__ w2, int off,
                                                    const float* __restrict__ mm, int K, int rank, float* exch,
                                                    SparseCand* oc) {
  __shared__ float e[kDspRank];
  __shared__ float sv[2];
  __shared__ int si[2];
  const int k = blockIdx.x, j = threadIdx.x;
  const bf16* er = w1t + (size_t)prev[seq[k]] * kDspRank;
  for (int c = j; c < kDspRank; c += 64) e[c] = __bfloat162float(er[c]);
  __syncthreads();
  const SparseCand cd = cands[(size_t)brow[k] * kSparseRec + j];
  const uint4* w = reinterpret_cast<const uint4*>(w2 + (size_t)(cd.id - off) * kDspRank);
  float acc = 0.f;
#pragma unroll 4
  for (int q = 0; q < kDspRank / 8; q++) {
    const uint4 u = w[q];
    const uint32_t x[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int i = 0; i < 4; i++) {
      acc = fmaf(bf16_lo(x[i]), e[8 * q + 2 * i], acc);
      acc = fmaf(bf16_hi(x[i]), e[8 * q + 2 * i + 1], acc);
    }
  }
  float rt[8];
#pragma unroll
  for (int i = 0; i < 8; i++) rt[i] = mm[i];
  float v = mmax(__fadd_rn(cd.x, acc), rt);
  int id = cd.id;
  if (oc) oc[(size_t)k * kSparseK + j] = SparseCand{id, v};
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, v, o);
    const int oi = __shfl_xor_sync(0xffffffffu, id, o);
    if (better(ov, oi, v, id)) v = ov, id = oi;
  }
  if ((j & 31) == 0) sv[j >> 5] = v, si[j >> 5] = id;
  __syncthreads();
  if (j == 0 && exch) {
    if (better(sv[1], si[1], v, id)) v = sv[1], id = si[1];
    float* me = exch + ((size_t)rank * K + k) * 2;
    float* other = exch + ((size_t)(1 - rank) * K + k) * 2;
    me[0] = v, me[1] = float(id);
    other[0] = 0.f, other[1] = 0.f;
  }
}

__global__ void k_cand_fill(float* q, size_t n) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) q[i] = -INFINITY;
}
__global__ void k_cand_put(const SparseCand* __restrict__ oc2, int K, float* q) {
  const int k = blockIdx.x, r = threadIdx.x >> 6, j = threadIdx.x & 63;
  const SparseCand c = oc2[((size_t)r * K + k) * kSparseK + j];
  q[(size_t)k * kVocab + c.id] = c.x;
}

__global__ void k_argmax(const float2* __restrict__ part, int nb, int rank, float* exch, int K) {
  const int k = blockIdx.x, tid = threadIdx.x;
  float v = -INFINITY;
  int i = INT_MAX;
  for (int b = tid; b < nb; b += blockDim.x) {
    const float2 p = part[(size_t)k * nb + b];
    if (better(p.x, __float_as_int(p.y), v, i)) v = p.x, i = __float_as_int(p.y);
  }
  __shared__ float sv[32];
  __shared__ int si[32];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, v, o);
    const int oi = __shfl_xor_sync(0xffffffffu, i, o);
    if (better(ov, oi, v, i)) v = ov, i = oi;
  }
  if ((tid & 31) == 0) sv[tid >> 5] = v, si[tid >> 5] = i;
  __syncthreads();
  if (tid == 0) {
    for (int w = 1; w < int(blockDim.x) / 32; w++)
      if (better(sv[w], si[w], v, i)) v = sv[w], i = si[w];
    // the id travels as an exact float (< 2^24); the other rank's entries are zeros (summed by the all-reduce)
    float* me = exch + ((size_t)rank * K + k) * 2;
    float* other = exch + ((size_t)(1 - rank) * K + k) * 2;
    me[0] = v, me[1] = float(i == INT_MAX ? -1 : i);
    other[0] = 0.f, other[1] = 0.f;
  }
}

__global__ void k_pick(const float* __restrict__ exch, int K, const int* __restrict__ seq, int step, int* prev,
                       int* drafts) {
  const int k = threadIdx.x;
  if (k >= K) return;
  const float v0 = exch[2 * k], v1 = exch[(K + k) * 2];
  const int i0 = int(exch[2 * k + 1]), i1 = int(exch[(K + k) * 2 + 1]);
  const int tok = i0 < 0 ? i1 : i1 < 0 ? i0 : better(v1, i1, v0, i0) ? i1 : i0;
  prev[seq[k]] = tok;
  drafts[seq[k] * kDspMaxN + step] = tok;
}

__global__ void k_put(const int* __restrict__ tok, int K, const int* __restrict__ seq, int step, int* prev,
                      int* drafts) {
  const int k = threadIdx.x;
  if (k >= K) return;
  prev[seq[k]] = tok[k];
  drafts[seq[k] * kDspMaxN + step] = tok[k];
}

// ---- confidence head, expert count -------------------------------------------------------------------------------
__global__ void __launch_bounds__(128) k_conf(const bf16* __restrict__ h, const int* __restrict__ rseq,
                                              const int* __restrict__ ridx, const int* __restrict__ last,
                                              const int* __restrict__ drafts, const bf16* __restrict__ w1t,
                                              const float* __restrict__ w, float* conf) {
  const int r = blockIdx.x, tid = threadIdx.x, b = rseq[r], i = ridx[r];
  const int prev = i == 0 ? last[b] : drafts[b * kDspMaxN + i - 1];
  float acc = 0.f;
  for (int d = tid; d < kHidden; d += 128) acc = fmaf(w[d], __bfloat162float(h[(size_t)r * kHidden + d]), acc);
  for (int d = tid; d < kDspRank; d += 128) acc = fmaf(w[kHidden + d], __bfloat162float(w1t[(size_t)prev * kDspRank + d]), acc);
  __shared__ float red[4];
  acc = warp_sum(acc);
  if ((tid & 31) == 0) red[tid >> 5] = acc;
  __syncthreads();
  if (tid == 0) conf[r] = (red[0] + red[1]) + (red[2] + red[3]);
}

__global__ void __launch_bounds__(512) k_experts(const int* __restrict__ ids, int n, int* cnt) {
  __shared__ int f[kExperts];
  __shared__ int red[16];
  const int tid = threadIdx.x;
  f[tid] = 0;
  __syncthreads();
  for (int i = tid; i < n; i += 512) f[ids[i] & (kExperts - 1)] = 1;
  __syncthreads();
  int c = f[tid];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) c += __shfl_xor_sync(0xffffffffu, c, o);
  if ((tid & 31) == 0) red[tid >> 5] = c;
  __syncthreads();
  if (tid == 0) {
    int t = 0;
    for (int w = 0; w < 16; w++) t += red[w];
    *cnt += t;
  }
}

// ---- rollback ----------------------------------------------------------------------------------------------------
__global__ void k_backup(const RowMeta* __restrict__ meta, const int* __restrict__ cross, DspRoll R) {
  const int l = blockIdx.x, r = blockIdx.y, tid = threadIdx.x;
  const RowMeta mt = meta[r];
  const uint2* src = reinterpret_cast<const uint2*>(R.ring[l] + ((size_t)mt.slot * kWindow + mt.pos % kWindow) * kRowB);
  uint2* dst = reinterpret_cast<uint2*>(R.bk_ring + ((size_t)l * R.maxr + r) * kRowB);
  for (int i = tid; i < kRowB / 8; i += blockDim.x) dst[i] = src[i];
  const int c = cross[r];
  if (c >= 0 && mt.pos > c) {
    const float4* ss = reinterpret_cast<const float4*>(R.state[l] + ((size_t)mt.slot * kRatio + mt.pos % kRatio) * 1024);
    float4* sd = reinterpret_cast<float4*>(R.bk_state + ((size_t)l * R.maxr + r) * 1024);
    for (int i = tid; i < 256; i += blockDim.x) sd[i] = ss[i];
  }
}

__global__ void k_restore(const RowMeta* __restrict__ meta, const int2* __restrict__ info, DspRoll R) {
  const int l = blockIdx.x, r = blockIdx.y, tid = threadIdx.x;
  const RowMeta mt = meta[r];
  const int c = info[r].x, kend = info[r].y;
  if (mt.pos >= kend) {
    const uint2* src = reinterpret_cast<const uint2*>(R.bk_ring + ((size_t)l * R.maxr + r) * kRowB);
    uint2* dst = reinterpret_cast<uint2*>(R.ring[l] + ((size_t)mt.slot * kWindow + mt.pos % kWindow) * kRowB);
    for (int i = tid; i < kRowB / 8; i += blockDim.x) dst[i] = src[i];
  }
  if (c < 0 || c < kend) return;  // no group closed in the block, or its closing row was kept
  float* st = R.state[l] + ((size_t)mt.slot * kRatio + mt.pos % kRatio) * 1024;
  if (mt.pos > c) {
    const float* src = R.bk_state + ((size_t)l * R.maxr + r) * 1024;
    for (int i = tid; i < 1024; i += blockDim.x) st[i] = src[i];
  } else if (mt.pos < kend) {
    const float* src = R.stash + ((size_t)l * R.maxr + r) * 1024;
    const float* ape = R.ape[l] + (mt.pos % kRatio) * 512;
    for (int d = tid; d < 1024; d += blockDim.x) st[d] = d < 512 ? src[d] : src[d] + ape[d - 512];
  }
}

}  // namespace

#include "kernels/dspark/p129_draft_attn.cuh"
#include "kernels/dspark/p130_draft_chain.cuh"

void dsp_aux(const bf16* R, const int* rows, int K, int which, bf16* aux, cudaStream_t s) {
  if (K <= 0) return;
  k_aux<<<K, kHidden / 16, 0, s>>>(R, rows, which, aux);
  CUDA_CHECK(cudaGetLastError());
}

void dsp_ctx_rows(const bf16* kv, int ldkv, const int* pos, const int* slot, int K, const float* inv, uint8_t* ring,
                  int slots, cudaStream_t s) {
  if (K <= 0) return;
  k_ctx_rows<<<dim3(K, kDspStages), 64, 0, s>>>(kv, ldkv, pos, slot, inv, ring, slots);
  CUDA_CHECK(cudaGetLastError());
}

void dsp_attn(const bf16* q, const uint8_t* rows, const RowMeta* meta, const uint8_t* ring, const float* sink, int M,
              bf16* o, cudaStream_t s) {
  if (M <= 0) return;
  static bool attr = false;
  if (!attr) {
    CUDA_CHECK(cudaFuncSetAttribute(k_attn, cudaFuncAttributeMaxDynamicSharedMemorySize, int(sizeof(AttnSmem))));
    attr = true;
  }
  k_attn<<<M, 256, sizeof(AttnSmem), s>>>(q, rows, meta, ring, sink, o);
  CUDA_CHECK(cudaGetLastError());
}

void dsp_attn_fused(const bf16* q, const bf16* kv, int ldkv, const RowMeta* meta, const uint8_t* ring, const float* sink,
                    const float* inv, const float* U, int M, bf16* o, float* zp, cudaStream_t s) {
  if (M <= 0) return;
  p129::launch(q, kv, ldkv, meta, ring, sink, inv, U, M, o, zp, s);
}

int dsp_markov_blocks(int Vh) { return cdiv(Vh, kMkRows); }
int dsp_markov_rows() { return kMkRows; }

void dsp_markov(const float* base, const int* brow, const int* seq, const int* prev, const bf16* w1t, const bf16* w2,
                const uint8_t* w2q, const uint8_t* w2s, int Vh, int off, const float* mm, int K, float* out,
                const int* orow, float2* part, cudaStream_t s, int g0, int ng, bool back) {
  if (K <= 0) return;
  if (K > kDspMaxN) qlog::fail("dspark", "markov: %d items > %d", K, kDspMaxN);
  const int nb = dsp_markov_blocks(Vh);
  if (ng < 0) ng = nb - g0;
  if (g0 < 0 || ng <= 0 || g0 + ng > nb) qlog::fail("dspark", "markov: CTAs %d + %d of %d", g0, ng, nb);
  (w2q ? k_markov<true> : k_markov<false>)<<<ng, 256, 0, s>>>(base, brow, seq, prev, w1t, w2, w2q, w2s, Vh, off, mm, K,
                                                              out, orow, part, g0, nb, back ? 1 : 0);
  CUDA_CHECK(cudaGetLastError());
}

void dsp_markov_topk(const SparseCand* cands, const int* brow, const int* seq, const int* prev, const bf16* w1t,
                     const bf16* w2, int off, const float* mm, int K, int rank, float* exch, SparseCand* oc,
                     cudaStream_t s) {
  if (K <= 0) return;
  k_markov_topk<<<K, 64, 0, s>>>(cands, brow, seq, prev, w1t, w2, off, mm, K, rank, exch, oc);
  CUDA_CHECK(cudaGetLastError());
}

void dsp_cand_rows(const SparseCand* oc2, int K, float* q, cudaStream_t s) {
  if (K <= 0) return;
  const size_t n = size_t(K) * kVocab;
  k_cand_fill<<<unsigned((n + 255) / 256), 256, 0, s>>>(q, n);
  k_cand_put<<<K, 128, 0, s>>>(oc2, K, q);
  CUDA_CHECK(cudaGetLastError());
}

void dsp_argmax(const float2* part, int nb, int K, int rank, float* exch, cudaStream_t s) {
  if (K <= 0) return;
  k_argmax<<<K, 256, 0, s>>>(part, nb, rank, exch, K);
  CUDA_CHECK(cudaGetLastError());
}

void dsp_pick(const float* exch, int K, const int* seq, int step, int* prev, int* drafts, cudaStream_t s) {
  if (K <= 0) return;
  k_pick<<<1, 32, 0, s>>>(exch, K, seq, step, prev, drafts);
  CUDA_CHECK(cudaGetLastError());
}

void dsp_chain(const float* base, const int* brow, const int* seq, int K, const int* pseq, int pK, const float* pex,
               const int* last, int* drafts, int step, const bf16* w1t, const uint8_t* w2q, const uint8_t* w2s, int Vh,
               int off, const float* mm, float2* part, unsigned* cnt, int rank, float* ex, cudaStream_t s) {
  if (K <= 0) return;
  if (K > kDspMaxN || pK > kDspMaxN) qlog::fail("dspark", "chain: %d / %d items > %d", K, pK, kDspMaxN);
  const p130::Args a{base, brow, seq, K, pseq, pK, pex, last, drafts, step, w1t, w2q, w2s, Vh, off, mm, part, cnt, rank,
                     ex};
  p130::launch(a, dsp_markov_blocks(Vh), s);
}

void dsp_put(const int* tok, int K, const int* seq, int step, int* prev, int* drafts, cudaStream_t s) {
  if (K <= 0) return;
  k_put<<<1, 32, 0, s>>>(tok, K, seq, step, prev, drafts);
  CUDA_CHECK(cudaGetLastError());
}

void dsp_conf(const bf16* h, const int* rseq, const int* ridx, const int* last, const int* drafts, const bf16* w1t,
              const float* w, int M, float* conf, cudaStream_t s) {
  if (M <= 0) return;
  k_conf<<<M, 128, 0, s>>>(h, rseq, ridx, last, drafts, w1t, w, conf);
  CUDA_CHECK(cudaGetLastError());
}

void dsp_experts(const int* ids, int n, int* cnt, cudaStream_t s) {
  if (n <= 0) return;
  k_experts<<<1, 512, 0, s>>>(ids, n, cnt);
  CUDA_CHECK(cudaGetLastError());
}

void dsp_backup(const RowMeta* meta, const int* cross, int M, DspRoll r, cudaStream_t s) {
  if (M <= 0 || r.nl <= 0) return;
  k_backup<<<dim3(r.nl, M), 128, 0, s>>>(meta, cross, r);
  CUDA_CHECK(cudaGetLastError());
}

void dsp_restore(const RowMeta* meta, const int2* info, int M, DspRoll r, cudaStream_t s) {
  if (M <= 0 || r.nl <= 0) return;
  k_restore<<<dim3(r.nl, M), 128, 0, s>>>(meta, info, r);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace e55
