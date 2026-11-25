// Test binding for P176's feasibility check (tests/zcmem.py): GPU kernels on cuMemCreate HOST_NUMA memory (what the
// NIC can register on GB10) against cudaMalloc memory — CUDA events around each kernel.
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <vector>

void* zc_alloc(int kind, size_t bytes, unsigned long long* handle, size_t* mapped);
void zc_free(void* p, unsigned long long h, size_t mapped);
void zc_run(int op, void* p, size_t bytes, int reps, unsigned* sink, cudaStream_t s);

namespace {
// per kind (0 cudaMalloc, 1 HOST_NUMA): ms of each op (min of `iters`) on a buffer of `mib` MiB
std::vector<double> bench(int kind, int mib, int l2_mib, int iters) {
  const size_t bytes = size_t(mib) << 20;
  unsigned long long h = 0;
  size_t mapped = 0;
  void* p = zc_alloc(kind, bytes, &h, &mapped);
  TORCH_CHECK(p, "allocation failed");
  unsigned* sink = nullptr;
  cudaMalloc(&sink, 16);
  cudaStream_t s = c10::cuda::getCurrentCUDAStream().stream();
  cudaEvent_t a, b;
  cudaEventCreate(&a), cudaEventCreate(&b);
  zc_run(1, p, bytes, 1, sink, s);  // first touch
  cudaStreamSynchronize(s);
  std::vector<double> out;
  auto t = [&](int op, size_t by, int reps) {
    float best = 1e30f;
    for (int i = 0; i < iters; i++) {
      if (op == 3) zc_run(0, p, by, 1, sink, s);  // warm the L2 set
      cudaEventRecord(a, s);
      zc_run(op, p, by, reps, sink, s);
      cudaEventRecord(b, s);
      cudaEventSynchronize(b);
      float ms = 0;
      cudaEventElapsedTime(&ms, a, b);
      best = std::min(best, ms);
    }
    out.push_back(best);
  };
  t(0, bytes, 1);
  t(1, bytes, 1);
  t(2, bytes, 1);
  t(3, size_t(l2_mib) << 20, 32);
  // the set read (4) after a kernel read it · (5) after a kernel wrote it (a producer's output: should hit L2)
  for (int pre : {0, 1}) {
    float best = 1e30f;
    for (int i = 0; i < iters; i++) {
      zc_run(0, static_cast<char*>(p) + (bytes / 2), bytes / 2, 1, sink, s);  // evict
      zc_run(pre, p, size_t(l2_mib) << 20, 1, sink, s);
      cudaEventRecord(a, s);
      zc_run(0, p, size_t(l2_mib) << 20, 1, sink, s);
      cudaEventRecord(b, s);
      cudaEventSynchronize(b);
      float ms = 0;
      cudaEventElapsedTime(&ms, a, b);
      best = std::min(best, ms);
    }
    out.push_back(best);
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "kernel error");
  cudaEventDestroy(a), cudaEventDestroy(b);
  cudaFree(sink);
  zc_free(p, h, mapped);
  return out;
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("bench", &bench); }
