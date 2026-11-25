// misc.cu — contract: misc.h.
#include "kernels/misc/misc.h"

#include "core/log.h"
#include "kernels/gemm/mma.cuh"

namespace e55 {
namespace {

__global__ void k_embed4(const bf16* __restrict__ tab, const int* __restrict__ ids, bf16* R) {
  const int row = blockIdx.x;
  const uint4* src = reinterpret_cast<const uint4*>(tab + (size_t)ids[row] * kHidden);
  uint4* dst = reinterpret_cast<uint4*>(R + (size_t)row * kHcDim);
  for (int i = threadIdx.x; i < kHidden / 8; i += blockDim.x) {
    const uint4 v = src[i];
#pragma unroll
    for (int j = 0; j < 4; j++) dst[j * (kHidden / 8) + i] = v;
  }
}

template <typename T>
__device__ __forceinline__ float ld(const T* p);
template <>
__device__ __forceinline__ float ld<bf16>(const bf16* p) { return __bfloat162float(*p); }

// one block (256 threads) per row
template <typename T>
__global__ void k_rms(const T* __restrict__ x, int ldx, const bf16* __restrict__ w, int n, bf16* y, int ldy) {
  const T* r = x + (size_t)blockIdx.x * ldx;
  __shared__ float red[8];
  float ss = 0.f;
  for (int i = threadIdx.x; i < n; i += 256) {
    const float v = ld(r + i);
    ss += v * v;
  }
  ss = warp_sum(ss);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < 8 ? red[threadIdx.x] : 0.f;
    v = warp_sum(v);
    if (threadIdx.x == 0) red[0] = v;
  }
  __syncthreads();
  const float rs = rsqrtf(__fdiv_rn(red[0], float(n)) + kEps);
  bf16* o = y + (size_t)blockIdx.x * ldy;
  for (int i = threadIdx.x; i < n; i += 256) o[i] = __float2bfloat16_rn(ld(r + i) * rs * __bfloat162float(w[i]));
}

__global__ void k_sigmoid(bf16* x, size_t n) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = __float2bfloat16_rn(__fdiv_rn(1.0f, 1.0f + expf(-__bfloat162float(x[i]))));
}

__global__ void k_mul(const bf16* a, int lda, const bf16* b, int ldb, int M, int n, bf16* y, int ldy) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)M * n) return;
  const int m = int(i / n), c = int(i % n);
  y[(size_t)m * ldy + c] = __float2bfloat16_rn(__bfloat162float(a[(size_t)m * lda + c]) * __bfloat162float(b[(size_t)m * ldb + c]));
}

__global__ void k_situ(const bf16* gu, int M, int n, bf16* h) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)M * n) return;
  const int m = int(i / n), c = int(i % n);
  const float g = __bfloat162float(gu[(size_t)m * 2 * n + c]), u = __bfloat162float(gu[(size_t)m * 2 * n + n + c]);
  // vLLM's situ_and_mul (csrc/activation_kernels.cu situ_activation, beta 4, linear_beta 25), same operations in the
  // same order: sigmoid(g) as (1 + tanh(g/2)) / 2 folded into 0.5·beta, up clipped as 25·tanh(u · fl(1/25))
  const float gate = (0.5f * 4.0f) * tanhf(g * 0.25f) * (1.0f + tanhf(g * 0.5f));
  const float up = 25.0f * tanhf(u * (1.0f / 25.0f));
  h[i] = __float2bfloat16_rn(gate * up);
}

__global__ void k_multimax(float* lg, int V, int ld, const float* __restrict__ rt, size_t n) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int m = int(i / V), c = int(i % V);
  float* p = lg + (size_t)m * ld + c;
  const float x = *p;
  const float a = fmaxf(rt[0] - x, 0.f), b = fmaxf(x - rt[1], 0.f), cc = fmaxf(rt[2] - x, 0.f), d = fmaxf(x - rt[3], 0.f);
  // the reference's order, no fused multiply-add (torch / its Triton kernel with fp fusion off)
  float y = __fadd_rn(x, __fmul_rn(rt[4], a));
  y = __fadd_rn(y, __fmul_rn(rt[5], b));
  y = __fadd_rn(y, __fmul_rn(rt[6], __fmul_rn(cc, cc)));
  y = __fadd_rn(y, __fmul_rn(rt[7], __fmul_rn(d, d)));
  *p = y;
}

__global__ void k_add(const bf16* a, const bf16* b, int ldb, int M, int n, bf16* y) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)M * n) return;
  const int m = int(i / n), c = int(i % n);
  y[i] = __float2bfloat16_rn(__bfloat162float(a[i]) + __bfloat162float(b[(size_t)m * ldb + c]));
}

__global__ void k_gather(const uint8_t* src, int rb, const int* rows, int M, uint8_t* dst) {
  const int j = blockIdx.x, r = rows[j];
  if (r >= M) return;
  for (int i = threadIdx.x * 4; i < rb; i += blockDim.x * 4)
    *reinterpret_cast<uint32_t*>(dst + (size_t)j * rb + i) = *reinterpret_cast<const uint32_t*>(src + (size_t)r * rb + i);
}

__global__ void k_halves(const float* src, int R, int Vh, float* dst) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)2 * R * Vh) return;
  const int h = int(i / ((size_t)R * Vh)), r = int((i / Vh) % R), c = int(i % Vh);
  dst[(size_t)r * 2 * Vh + (size_t)h * Vh + c] = src[i];
}

// P094: one block per row, 16-byte units
__global__ void k_pick(const uint8_t* __restrict__ src, long sld, int w, const int* __restrict__ own, uint8_t* dst,
                       long dld) {
  const int i = blockIdx.x;
  const uint4* s = reinterpret_cast<const uint4*>(src + (size_t)i * sld + (size_t)own[i] * w);
  uint4* d = reinterpret_cast<uint4*>(dst + (size_t)i * dld);
  for (int j = threadIdx.x; j < w / 16; j += blockDim.x) d[j] = s[j];
}

// P095: as k_embed4, from this rank's half
__global__ void k_embed_half(const bf16* __restrict__ tab, const int* __restrict__ ids, int v0, int vh, bf16* E, long ld) {
  const int row = blockIdx.x, id = ids[row] - v0;
  if (id < 0 || id >= vh) return;
  const uint4* src = reinterpret_cast<const uint4*>(tab + (size_t)id * kHidden);
  uint4* dst = reinterpret_cast<uint4*>(E + (size_t)row * ld);
  for (int i = threadIdx.x; i < kHidden / 8; i += blockDim.x) dst[i] = src[i];
}
// each thread reads its 16-byte units of stream 2 + h and writes the same units of all four streams (no other thread
// touches them: in place is safe)
__global__ void k_embed4_pick(const int* __restrict__ ids, int vh, bf16* R) {
  const int row = blockIdx.x, h = min(max(ids[row] / vh, 0), 1);
  uint4* r = reinterpret_cast<uint4*>(R + (size_t)row * kHcDim);
  constexpr int n = kHidden / 8;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const uint4 v = r[(2 + h) * n + i];
#pragma unroll
    for (int j = 0; j < 4; j++) r[j * n + i] = v;
  }
}

// P100: grid (M, D / 128), 128 threads; the chains as hca.cu k_vha_partial (32 heads, ranks outer, heads inner)
__global__ void k_vha_z(const bf16* __restrict__ o, const float* __restrict__ U, const bf16* __restrict__ o2,
                        const float* __restrict__ U2, const float* zo, int D, float* z) {
  constexpr int kLH = kHeads / 2;
  const int m = blockIdx.x, d = blockIdx.y * 128 + threadIdx.x;
  float x[kLH], x2[kLH];
#pragma unroll
  for (int h = 0; h < kLH; h++) x[h] = __bfloat162float(o[((size_t)m * kLH + h) * D + d]);
  if (o2) {
#pragma unroll
    for (int h = 0; h < kLH; h++) x2[h] = __bfloat162float(o2[((size_t)m * kLH + h) * D + d]);
  }
#pragma unroll
  for (int r = 0; r < kVhaRank; r++) {
    float a = 0.f;
#pragma unroll
    for (int h = 0; h < kLH; h++) a += U[h * kVhaRank + r] * x[h];
    const size_t i = ((size_t)m * kVhaRank + r) * D + d;
    float b;
    if (o2) {
      b = 0.f;
#pragma unroll
      for (int h = 0; h < kLH; h++) b += U2[h * kVhaRank + r] * x2[h];
    } else {
      b = zo[i];
    }
    z[i] = __fadd_rn(b, a);
  }
}

unsigned nb(size_t n) { return unsigned((n + 255) / 256); }
}  // namespace

void embed4(const bf16* table, const int* ids, int M, bf16* R, cudaStream_t s) {
  if (M <= 0) return;
  k_embed4<<<M, 256, 0, s>>>(table, ids, R);
  CUDA_CHECK(cudaGetLastError());
}
void rmsnorm(const bf16* x, int ldx, const bf16* w, int M, int n, bf16* y, int ldy, cudaStream_t s) {
  if (M <= 0) return;
  k_rms<bf16><<<M, 256, 0, s>>>(x, ldx, w, n, y, ldy);
  CUDA_CHECK(cudaGetLastError());
}
void sigmoid_inplace(bf16* x, size_t n, cudaStream_t s) {
  if (!n) return;
  k_sigmoid<<<nb(n), 256, 0, s>>>(x, n);
  CUDA_CHECK(cudaGetLastError());
}
void mul_bf16(const bf16* a, int lda, const bf16* b, int ldb, int M, int n, bf16* y, int ldy, cudaStream_t s) {
  if (M <= 0) return;
  k_mul<<<nb((size_t)M * n), 256, 0, s>>>(a, lda, b, ldb, M, n, y, ldy);
  CUDA_CHECK(cudaGetLastError());
}
void situ_bf16(const bf16* gu, int M, int n, bf16* h, cudaStream_t s) {
  if (M <= 0) return;
  k_situ<<<nb((size_t)M * n), 256, 0, s>>>(gu, M, n, h);
  CUDA_CHECK(cudaGetLastError());
}
void multimax(float* lg, int M, int V, int ld, const float* rt, cudaStream_t s) {
  if (M <= 0) return;
  k_multimax<<<nb((size_t)M * V), 256, 0, s>>>(lg, V, ld, rt, (size_t)M * V);
  CUDA_CHECK(cudaGetLastError());
}
void add_bf16(const bf16* a, const bf16* b, int ldb, int M, int n, bf16* y, cudaStream_t s) {
  if (M <= 0) return;
  k_add<<<nb((size_t)M * n), 256, 0, s>>>(a, b, ldb, M, n, y);
  CUDA_CHECK(cudaGetLastError());
}
void gather_rows(const void* src, int row_bytes, const int* rows, int n, int M, void* dst, cudaStream_t s) {
  if (n <= 0) return;
  k_gather<<<n, 256, 0, s>>>(static_cast<const uint8_t*>(src), row_bytes, rows, M, static_cast<uint8_t*>(dst));
  CUDA_CHECK(cudaGetLastError());
}
void pick_rows(const void* src, long sld, int w, const int* own, int n, void* dst, long dld, cudaStream_t s) {
  if (n <= 0) return;
  if ((w | sld | dld) & 15) qlog::fail("misc", "pick_rows: widths / strides must be multiples of 16");
  k_pick<<<n, 256, 0, s>>>(static_cast<const uint8_t*>(src), sld, w, own, static_cast<uint8_t*>(dst), dld);
  CUDA_CHECK(cudaGetLastError());
}
void embed_half(const bf16* half, const int* ids, int M, int v0, int vh, bf16* E, long ld, cudaStream_t s) {
  if (M <= 0) return;
  k_embed_half<<<M, 256, 0, s>>>(half, ids, v0, vh, E, ld);
  CUDA_CHECK(cudaGetLastError());
}
void embed4_pick(const int* ids, int M, int vh, bf16* R, cudaStream_t s) {
  if (M <= 0) return;
  k_embed4_pick<<<M, 256, 0, s>>>(ids, vh, R);
  CUDA_CHECK(cudaGetLastError());
}
void vha_z(const bf16* o, const float* U, const bf16* o2, const float* U2, const float* zo, int D, int M, float* z,
           cudaStream_t s) {
  if (M <= 0) return;
  if (D % 128 || (!o2 && !zo)) qlog::fail("misc", "vha_z: D %d", D);
  k_vha_z<<<dim3(M, D / 128), 128, 0, s>>>(o, U, o2, U2, zo, D, z);
  CUDA_CHECK(cudaGetLastError());
}
void halves_to_rows(const float* src, int R, int Vh, float* dst, cudaStream_t s) {
  if (R <= 0) return;
  k_halves<<<nb((size_t)2 * R * Vh), 256, 0, s>>>(src, R, Vh, dst);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace e55
