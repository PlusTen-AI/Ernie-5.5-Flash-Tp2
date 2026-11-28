// mhc.cu — mHC pre / post / head (contract and the exact arithmetic: mhc.h). Base kernels: one CTA of 256 threads per
// row; P050 mhc_dec (p050_mhc_dec.cuh; P107 mhc_rows_pdl there), P051 mhc_big (p051_mhc_big.cuh), P072 mhc_fuse
// (p072_mhc_fuse.cuh), P106 mhc_prefill (p106_mhc_prefill.cuh, post + pre only) and P164 mhc_verify
// (p164_mhc_verify.cuh, M ≤ 32; P192 mhc_verify64 p192_mhc_verify64.cuh, 33 … 64) compute the same values faster.
#include "kernels/mhc/mhc.h"

#include <cmath>
#include <cstring>
#include <type_traits>
#include <vector>

#include "core/log.h"
#include "kernels/gemm/mma.cuh"
#include "model/patches.h"

namespace e55 {

namespace {
constexpr int kChunks = 64, kCq = kHcDim / kChunks, kCh = kHidden / kChunks, kOut = 25;  // 256 q / 64 h per chunk
constexpr int kDecM = 16;                                                                // P050 row limit (ws size)

std::vector<float> f32(const Weights& W, const std::string& n) {
  auto b = W.slice(n);
  std::vector<float> v(b.size() / 4);
  std::memcpy(v.data(), b.data(), b.size());
  return v;
}
float* upload(const std::vector<float>& v) {
  float* d;
  CUDA_CHECK(cudaMalloc(&d, v.size() * 4));
  CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
  return d;
}
void init_kernels();
}  // namespace

MhcW mhc_upload(const Weights& W, const std::string& prefix, const std::string& norm) {
  init_kernels();
  MhcW m;
  auto fn = f32(W, prefix + "mapping_proj.weight");  // stored [16384, 24] (in = j·4096 + h, out)
  if (fn.size() != size_t(kHcDim) * kHcMix) qlog::fail("mhc", "%smapping_proj shape", prefix.c_str());
  std::vector<float> q(fn.size());
  for (int h = 0; h < kHidden; h++)
    for (int j = 0; j < kStreams; j++)
      for (int n = 0; n < kHcMix; n++) q[(size_t(h) * 4 + j) * kHcMix + n] = fn[(size_t(j) * kHidden + h) * kHcMix + n];
  m.fnq = upload(q);
  if (prefix.rfind("model.layers.0.self_attention", 0) == 0) {  // layer 0's first pre: Σ_j fn, j in order
    std::vector<float> b(size_t(kHidden) * kHcMix);
    for (int h = 0; h < kHidden; h++)
      for (int n = 0; n < kHcMix; n++) {
        float s = fn[size_t(h) * kHcMix + n];
        for (int j = 1; j < kStreams; j++) s += fn[(size_t(j) * kHidden + h) * kHcMix + n];
        b[size_t(h) * kHcMix + n] = s;
      }
    m.fnb = upload(b);
  }
  std::vector<float> sb(27);
  auto base = f32(W, prefix + "bias");
  std::memcpy(sb.data(), base.data(), 24 * 4);
  sb[24] = f32(W, prefix + "alpha_pre")[0];
  sb[25] = f32(W, prefix + "alpha_post")[0];
  sb[26] = f32(W, prefix + "alpha_res")[0];
  m.sb = upload(sb);
  auto nw = W.slice(norm);
  CUDA_CHECK(cudaMalloc(&m.normw, nw.size()));
  CUDA_CHECK(cudaMemcpy(m.normw, nw.data(), nw.size(), cudaMemcpyHostToDevice));
  const size_t wsb = size_t(kChunks) * kDecM * kOut * 4;
  CUDA_CHECK(cudaMalloc(&m.ws, wsb + 16));
  CUDA_CHECK(cudaMemset(m.ws, 0, wsb + 16));
  m.cnt = reinterpret_cast<unsigned*>(reinterpret_cast<uint8_t*>(m.ws) + wsb);
  return m;
}

void mhc_free(MhcW& m) {
  cudaFree(m.fnq), cudaFree(m.fnb), cudaFree(m.sb), cudaFree(m.normw), cudaFree(m.ws);
  m = MhcW{};
}

HcHead hc_head_upload(const Weights& W, const std::string& prefix) {
  HcHead h;
  auto fn = f32(W, prefix + "hc_head_fn");  // [16384, 4]
  std::vector<float> t(size_t(4) * kHcDim);
  for (int o = 0; o < 4; o++)
    for (int i = 0; i < kHcDim; i++) t[(size_t)o * kHcDim + i] = fn[(size_t)i * 4 + o];
  h.fn = upload(t);
  std::vector<float> sb(5);
  auto base = f32(W, prefix + "hc_head_base");
  std::memcpy(sb.data(), base.data(), 16);
  sb[4] = f32(W, prefix + "hc_head_scale")[0];
  h.sb = upload(sb);
  return h;
}

namespace {
__device__ __forceinline__ float sigm(float x) { return __fdiv_rn(1.0f, 1.0f + expf(-x)); }

template <int NT>
__device__ float block_sum(float v, float* red) {
  v = warp_sum(v);
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  __syncthreads();
  if (l == 0) red[w] = v;
  __syncthreads();
  if (w == 0) {
    v = l < NT / 32 ? red[l] : 0.f;
    v = warp_sum(v);
    if (l == 0) red[0] = v;
  }
  __syncthreads();
  return red[0];
}

// the post of one element (mhc.h): post_j · out first, then the comb terms i = 0..3 as FMAs (vLLM's order)
__device__ __forceinline__ float post1(const float (&c)[16], const float (&p)[4], int j, float x0, float x1, float x2,
                                       float x3, float o) {
  float v = __fmul_rn(p[j], o);
  v = __fmaf_rn(c[j], x0, v);
  v = __fmaf_rn(c[4 + j], x1, v);
  v = __fmaf_rn(c[8 + j], x2, v);
  return __fmaf_rn(c[12 + j], x3, v);
}

__device__ __forceinline__ void unpack8(const uint4 u, float* f) {
  const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
  for (int i = 0; i < 4; i++) f[2 * i] = bf16_lo(w[i]), f[2 * i + 1] = bf16_hi(w[i]);
}

// Shared-memory slot of x_q within one row (q < 256, P050 / P051): the 8 float4 of each 32-q segment rotated by the
// segment index, so the 8 threads that write a row's 8 segments (STS.128) hit distinct banks.
__device__ __forceinline__ int xsw(int q) { return (q & ~31) | ((((q >> 2) + (q >> 5)) & 7) << 2) | (q & 3); }

// the reduction of block_sum<256> (warp butterfly, then warp 0 over the 8 warp sums); NAMED: barrier 2 over threads
// 0–255 only (P050 runs a 9th warp beside them)
template <bool NAMED>
__device__ __forceinline__ void bsync() {
  if constexpr (NAMED) asm volatile("bar.sync 2, 256;\n" ::: "memory");
  else __syncthreads();
}
template <bool NAMED>
__device__ __forceinline__ float block_sum256(float v, float* red) {
  v = warp_sum(v);
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  bsync<NAMED>();
  if (l == 0) red[w] = v;
  bsync<NAMED>();
  if (w == 0) {
    v = l < 8 ? red[l] : 0.f;
    v = warp_sum(v);
    if (l == 0) red[0] = v;
  }
  bsync<NAMED>();
  return red[0];
}

// thread tid's 16 norm weights (dims 16·tid …)
__device__ __forceinline__ void load_nw(const bf16* __restrict__ nw, float (&w)[16]) {
  unpack8(*reinterpret_cast<const uint4*>(nw + threadIdx.x * 16), w);
  unpack8(*reinterpret_cast<const uint4*>(nw + threadIdx.x * 16 + 8), w + 8);
}

// inp = RMSNorm_w(bf16(Σ_j pre_j x_j)) for thread tid's 16 dims (x: the stored bf16 residual, k_pre layout:
// x[j][i] = stream j, dim 16·tid + i; w: load_nw). Threads 0–255.
template <bool NAMED>
__device__ __forceinline__ void collapse(const float (&x)[4][16], const float (&pre)[4], const float (&w)[16], float* red,
                                         bf16* inp_row) {
  const int tid = threadIdx.x;
  float ol[16], s2 = 0.f;
#pragma unroll
  for (int i = 0; i < 16; i++) {
    ol[i] = ((pre[0] * x[0][i] + pre[1] * x[1][i]) + pre[2] * x[2][i]) + pre[3] * x[3][i];
    s2 += ol[i] * ol[i];
  }
  s2 = block_sum256<NAMED>(s2, red);
  const float rs = rsqrtf(__fdiv_rn(s2, float(kHidden)) + kEps);
  uint32_t o[8];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const float a = __bfloat162float(__float2bfloat16_rn(ol[2 * i])) * rs * w[2 * i];
    const float b = __bfloat162float(__float2bfloat16_rn(ol[2 * i + 1])) * rs * w[2 * i + 1];
    o[i] = pack_bf16x2(a, b);
  }
  uint4* d = reinterpret_cast<uint4*>(inp_row + tid * 16);
  d[0] = make_uint4(o[0], o[1], o[2], o[3]);
  d[1] = make_uint4(o[4], o[5], o[6], o[7]);
}

// Base: one CTA (256 threads) per row. MODE 0: pre on R · 1: post + pre (R ← R') · 2: layer 0's broadcast pre.
template <int MODE>
__global__ void __launch_bounds__(256) k_mix(const float* __restrict__ w, const float* __restrict__ sb,
                                             const bf16* __restrict__ nw, bf16* R, const bf16* __restrict__ out,
                                             float* post, float* comb, bf16* inp) {
  constexpr int K = MODE == 2 ? kCh : kCq;  // terms per chunk
  extern __shared__ float xs[];              // x_q: [16384] (MODE 2: e [4096])
  __shared__ float P[kChunks][kOut], vmix[kOut], red[8], pre_s[4];
  const int row = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  bf16* r = R + (size_t)row * kHcDim;
  float x[4][16];  // the stored residual (bf16 values) for the collapse
  if constexpr (MODE == 1) {
    float c[16], p[4];
#pragma unroll
    for (int i = 0; i < 16; i++) c[i] = comb[(size_t)row * 16 + i];
#pragma unroll
    for (int i = 0; i < 4; i++) p[i] = post[(size_t)row * 4 + i];
#pragma unroll
    for (int c8 = 0; c8 < 2; c8++) {
      const int h0 = tid * 16 + c8 * 8;
      float xi[4][8], o[8];
#pragma unroll
      for (int j = 0; j < 4; j++) unpack8(*reinterpret_cast<const uint4*>(r + j * kHidden + h0), xi[j]);
      unpack8(*reinterpret_cast<const uint4*>(out + (size_t)row * kHidden + h0), o);
#pragma unroll
      for (int j = 0; j < 4; j++) {
        uint32_t wv[4];
#pragma unroll
        for (int k = 0; k < 8; k += 2) {
          const float v0 = post1(c, p, j, xi[0][k], xi[1][k], xi[2][k], xi[3][k], o[k]);
          const float v1 = post1(c, p, j, xi[0][k + 1], xi[1][k + 1], xi[2][k + 1], xi[3][k + 1], o[k + 1]);
          xs[4 * (h0 + k) + j] = v0, xs[4 * (h0 + k + 1) + j] = v1;
          wv[k / 2] = pack_bf16x2(v0, v1);
          x[j][c8 * 8 + k] = bf16_lo(wv[k / 2]), x[j][c8 * 8 + k + 1] = bf16_hi(wv[k / 2]);
        }
        *reinterpret_cast<uint4*>(r + j * kHidden + h0) = make_uint4(wv[0], wv[1], wv[2], wv[3]);  // own dims only
      }
    }
  } else {
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
      for (int c8 = 0; c8 < 2; c8++) unpack8(*reinterpret_cast<const uint4*>(r + j * kHidden + tid * 16 + c8 * 8), &x[j][c8 * 8]);
#pragma unroll
    for (int i = 0; i < 16; i++) {
      const int h = tid * 16 + i;
      if constexpr (MODE == 0) {
#pragma unroll
        for (int j = 0; j < 4; j++) xs[4 * h + j] = x[j][i];
      } else {
        xs[h] = x[0][i];
      }
    }
  }
  __syncthreads();
  for (int cc = warp; cc < kChunks; cc += 8)  // chunk partials: lane n < 24 the mixer rows, lane 24 Σx²
    if (lane < kOut) {
      float acc = 0.f;
      for (int q = cc * K; q < (cc + 1) * K; q++) {
        const float xv = xs[q];
        const float wv = lane < kHcMix ? w[(size_t)q * kHcMix + lane] : xv;
        acc = __fmaf_rn(wv, xv, acc);
      }
      P[cc][lane] = acc;
    }
  __syncthreads();
  if (tid < kOut) {
    float y = P[0][tid];
    for (int cc = 1; cc < kChunks; cc++) y = __fadd_rn(y, P[cc][tid]);
    vmix[tid] = y;
  }
  __syncthreads();
  if (tid == 0) {
    const float f = __fdiv_rn(1.0f, sqrtf(__fdiv_rn(vmix[24], float(MODE == 2 ? kHidden : kHcDim))) + kEps);
    float m[24];
    for (int i = 0; i < 24; i++) m[i] = vmix[i] * f;
    float pr[4], po[4], cm[16];
    for (int i = 0; i < 4; i++) pr[i] = sigm(m[i] * sb[24] + sb[i]) + kEps;
    for (int i = 0; i < 4; i++) po[i] = sigm(m[4 + i] * sb[25] + sb[4 + i]) * 2.0f;
    for (int i = 0; i < 4; i++) {
      float l[4], mx = -INFINITY;
      for (int j = 0; j < 4; j++) l[j] = m[8 + 4 * i + j] * sb[26] + sb[8 + 4 * i + j], mx = fmaxf(mx, l[j]);
      float s = 0.f;
      for (int j = 0; j < 4; j++) l[j] = expf(l[j] - mx), s += l[j];
      for (int j = 0; j < 4; j++) cm[4 * i + j] = __fdiv_rn(l[j], s) + kEps;
    }
    auto colnorm = [&] {
      for (int j = 0; j < 4; j++) {
        const float s = ((cm[j] + cm[4 + j]) + cm[8 + j]) + cm[12 + j] + kEps;
        for (int i = 0; i < 4; i++) cm[4 * i + j] = __fdiv_rn(cm[4 * i + j], s);
      }
    };
    auto rownorm = [&] {
      for (int i = 0; i < 4; i++) {
        const float s = ((cm[4 * i] + cm[4 * i + 1]) + cm[4 * i + 2]) + cm[4 * i + 3] + kEps;
        for (int j = 0; j < 4; j++) cm[4 * i + j] = __fdiv_rn(cm[4 * i + j], s);
      }
    };
    colnorm();
    for (int it = 0; it < 19; it++) rownorm(), colnorm();
    for (int i = 0; i < 4; i++) pre_s[i] = pr[i], post[(size_t)row * 4 + i] = po[i];
    for (int i = 0; i < 16; i++) comb[(size_t)row * 16 + i] = cm[i];
  }
  __syncthreads();
  float nwv[16];
  load_nw(nw, nwv);
  const float pr[4] = {pre_s[0], pre_s[1], pre_s[2], pre_s[3]};
  collapse<false>(x, pr, nwv, red, inp + (size_t)row * kHidden);
}

// last post alone: one block of 256 threads per row; thread: 16 dims of every stream (reads before writes: Ro may be R)
__global__ void __launch_bounds__(256) k_post(const bf16* __restrict__ out, const bf16* R, const float* __restrict__ post,
                                              const float* __restrict__ comb, bf16* Ro) {
  const int row = blockIdx.x, tid = threadIdx.x;
  float c[16], p[4];
#pragma unroll
  for (int i = 0; i < 16; i++) c[i] = comb[(size_t)row * 16 + i];
#pragma unroll
  for (int i = 0; i < 4; i++) p[i] = post[(size_t)row * 4 + i];
#pragma unroll
  for (int c8 = 0; c8 < 2; c8++) {
    const int h0 = tid * 16 + c8 * 8;
    float xi[4][8], o[8];
#pragma unroll
    for (int j = 0; j < 4; j++) unpack8(*reinterpret_cast<const uint4*>(R + (size_t)row * kHcDim + j * kHidden + h0), xi[j]);
    unpack8(*reinterpret_cast<const uint4*>(out + (size_t)row * kHidden + h0), o);
#pragma unroll
    for (int j = 0; j < 4; j++) {
      uint32_t wv[4];
#pragma unroll
      for (int k = 0; k < 8; k += 2)
        wv[k / 2] = pack_bf16x2(post1(c, p, j, xi[0][k], xi[1][k], xi[2][k], xi[3][k], o[k]),
                                post1(c, p, j, xi[0][k + 1], xi[1][k + 1], xi[2][k + 1], xi[3][k + 1], o[k + 1]));
      *reinterpret_cast<uint4*>(Ro + (size_t)row * kHcDim + j * kHidden + h0) = make_uint4(wv[0], wv[1], wv[2], wv[3]);
    }
  }
}

__global__ void __launch_bounds__(256) k_head(const float* __restrict__ fn, const float* __restrict__ sb,
                                              const bf16* __restrict__ R, const int* __restrict__ rows, bf16* out) {
  const int row = rows[blockIdx.x], tid = threadIdx.x;
  __shared__ float red[8], pre_s[4];
  const bf16* r = R + (size_t)row * kHcDim;
  float ss = 0.f, mx[4] = {0.f, 0.f, 0.f, 0.f};
  for (int i = tid; i < kHcDim; i += 256) {
    const float v = __bfloat162float(r[i]);
    ss += v * v;
#pragma unroll
    for (int o = 0; o < 4; o++) mx[o] += v * fn[(size_t)o * kHcDim + i];
  }
  ss = block_sum<256>(ss, red);
  for (int o = 0; o < 4; o++) {
    const float v = block_sum<256>(mx[o], red);
    if (tid == 0) pre_s[o] = v;
    __syncthreads();
  }
  if (tid == 0) {
    const float f = rsqrtf(__fdiv_rn(ss, float(kHcDim)) + kEps);
    for (int o = 0; o < 4; o++) pre_s[o] = sigm(pre_s[o] * f * sb[4] + sb[o]) + kEps;
  }
  __syncthreads();
  for (int h = tid; h < kHidden; h += 256) {
    float v = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++) v += pre_s[j] * __bfloat162float(r[j * kHidden + h]);
    out[(size_t)blockIdx.x * kHidden + h] = __float2bfloat16_rn(v);
  }
}

template <int MODE>
void base(const MhcW& m, const float* w, bf16* R, const bf16* out, int M, float* post, float* comb, bf16* inp,
          cudaStream_t s) {
  const size_t smem = (MODE == 2 ? kHidden : kHcDim) * 4;
  k_mix<MODE><<<M, 256, smem, s>>>(w, m.sb, m.normw, R, out, post, comb, inp);
  CUDA_CHECK(cudaGetLastError());
}

#include "kernels/mhc/p050_mhc_dec.cuh"  // P050 mhc_dec (M ≤ 16)
#include "kernels/mhc/p051_mhc_big.cuh"  // P051 mhc_big (M > 16)
#include "kernels/mhc/p072_mhc_fuse.cuh"  // P072 mhc_fuse (M > 16, one launch)
#include "kernels/mhc/p106_mhc_prefill.cuh"  // P106 mhc_prefill (post + pre, M >= 256)
#include "kernels/mhc/p164_mhc_verify.cuh"  // P164 mhc_verify (M <= 32)
#include "kernels/mhc/p192_mhc_verify64.cuh"  // P192 mhc_verify64 (33 <= M <= 64, with P164)

void init_kernels() {  // shared-memory opt-ins, once, at upload (not inside a later graph capture)
  static const bool done = [] {
    CUDA_CHECK(cudaFuncSetAttribute(k_mix<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, kHcDim * 4));
    CUDA_CHECK(cudaFuncSetAttribute(k_mix<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, kHcDim * 4));
    CUDA_CHECK(cudaFuncSetAttribute(k_mix<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, kHidden * 4));
    p051::init();
    p072::init();
    return true;
  }();
  (void)done;
}

template <int MODE>
void run(const MhcW& m, bf16* R, const bf16* out, int M, float* mix, float* post, float* comb, bf16* inp,
         cudaStream_t s) {
  if (M <= 0) return;
  const float* w = MODE == 2 ? m.fnb : m.fnq;
  if (!w) qlog::fail("mhc", "broadcast pre on a mixer without fn_b (only layer 0's attention mixer has one)");
  if (M <= p164::kMaxM && patch::on(patch::mhc_verify)) return p164::run<MODE>(m, w, R, out, M, post, comb, inp, s);  // P164
  if (M <= p192::kMaxM && patch::on(patch::mhc_verify) && patch::on(patch::mhc_verify64))
    return p192::run<MODE>(m, w, R, out, M, post, comb, inp, s);  // P192
  if (M <= kDecM && patch::on(patch::mhc_dec)) return p050::run<MODE>(m, w, R, out, M, post, comb, inp, s);  // P050
  if constexpr (MODE == 1)
    if (M >= p106::kMinM && patch::on(patch::mhc_prefill)) return p106::run(m, w, R, out, M, post, comb, inp, s);  // P106
  if (M >= p072::kMinM && patch::on(patch::mhc_fuse)) return p072::run<MODE>(m, w, R, out, M, post, comb, inp, s);  // P072
  if (M > kDecM && patch::on(patch::mhc_big)) return p051::run<MODE>(m, w, R, out, M, post, comb, inp, s);  // P051
  base<MODE>(m, w, R, out, M, post, comb, inp, s);
}
}  // namespace

void mhc_pre(const MhcW& m, const bf16* R, int M, float* mix, float* post, float* comb, bf16* inp, cudaStream_t s) {
  run<0>(m, const_cast<bf16*>(R), nullptr, M, mix, post, comb, inp, s);
}

void mhc_post_pre(const MhcW& m, const bf16* out, bf16* R, int M, float* mix, float* post, float* comb, bf16* inp,
                  cudaStream_t s) {
  run<1>(m, R, out, M, mix, post, comb, inp, s);
}

void mhc_pre_embed(const MhcW& m, const bf16* R, int M, float* mix, float* post, float* comb, bf16* inp, cudaStream_t s) {
  run<2>(m, const_cast<bf16*>(R), nullptr, M, mix, post, comb, inp, s);
}

void mhc_post(const bf16* out, const bf16* R, const float* post, const float* comb, int M, bf16* Rout, cudaStream_t s) {
  if (M <= 0) return;
  k_post<<<M, 256, 0, s>>>(out, R, post, comb, Rout);
  CUDA_CHECK(cudaGetLastError());
}

void mhc_p164_rows(int r) { p164::g_rows = r; }

void hc_head(const HcHead& h, const bf16* R, const int* rows, int n, bf16* out, cudaStream_t s) {
  if (n <= 0) return;
  k_head<<<n, 256, 0, s>>>(h.fn, h.sb, R, rows, out);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace e55
