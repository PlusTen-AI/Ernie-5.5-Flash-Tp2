// comm_k.cu — test-only kernel for engine/tests/test_comm.py: a weight-stream stand-in (reads `bytes` once, like a
// decode GEMM, launched as a programmatic dependent with griddepcontrol.wait + launch_dependents at its start, as
// P030's decode GEMM) (its result is never stored: the buffers hold ones).
#include <cuda_runtime.h>

#include <cstdint>

namespace {
__global__ void __launch_bounds__(512) k_stream(const uint4* __restrict__ src, size_t n, uint4* dst) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  uint4 acc = make_uint4(0, 0, 0, 0);
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    const uint4 v = __ldcs(src + i);
    acc.x ^= v.x, acc.y ^= v.y, acc.z ^= v.z, acc.w ^= v.w;
  }
  // the buffers hold ones: acc is a XOR of ones, never this pattern — the loads stay, dst is not written
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x9E3779B9u) dst[0] = acc;
}
// GPU ↔ CPU round trip through pinned host memory (rdma_ar's doorbell / flag path without the NIC): thread 0 writes i
// to a (release, system scope), the CPU answers i in b; n rounds, total globaltimer ns → out[0]
__global__ void k_ping(unsigned long long* a, const unsigned long long* b, int n, unsigned long long* out) {
  if (threadIdx.x) return;
  unsigned long long t0, t1, v;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  for (unsigned long long i = 1; i <= (unsigned long long)n; i++) {
    asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(a), "l"(i) : "memory");
    do {
      asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(b) : "memory");
    } while (v != i);
  }
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
  out[0] = t1 - t0;
}
}  // namespace

void comm_test_ping(unsigned long long* a, const unsigned long long* b, int n, unsigned long long* out, cudaStream_t s) {
  k_ping<<<1, 32, 0, s>>>(a, b, n, out);
}

// reads src[bytes] (multiple of 16); dst is the next collective's buffer (never written in practice)
void comm_test_stream(const void* src, size_t bytes, void* dst, cudaStream_t s) {
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t cfg{dim3(96), dim3(512), 0, s, at, 1};
  cudaLaunchKernelEx(&cfg, k_stream, reinterpret_cast<const uint4*>(src), bytes / 16, reinterpret_cast<uint4*>(dst));
}
