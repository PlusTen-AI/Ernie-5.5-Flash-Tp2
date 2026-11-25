// Test binding for the collectives (core/comm.cc, kernels/comm/rdma_ar.cu): P001 rdma_ar, P075 ar_lanes (async
// tickets), P076 rdma_gather (row-piece all-gathers), P077 ar_pdl. Calls run on torch's current stream; timing loops
// with CUDA events (back-to-back calls, optionally each behind a weight-stream stand-in kernel, tests/comm_k.cu).
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include "core/comm.h"
#include "core/log.h"
#include "model/patches.h"

void comm_test_stream(const void* src, size_t bytes, void* dst, cudaStream_t s);  // tests/comm_k.cu
void comm_test_ping(unsigned long long* a, const unsigned long long* b, int n, unsigned long long* out, cudaStream_t s);

namespace {
std::unique_ptr<e55::Comm> C;
cudaEvent_t E0, E1;
cudaStream_t cur() { return c10::cuda::getCurrentCUDAStream().stream(); }

void init(int rank, const std::string& ip, int port) {
  setenv("E55_RDMA_INIT", "1", 1);  // RDMA queue pairs even while P001 is switched off (per-call switching)
  C = std::make_unique<e55::Comm>(rank, ip, port);
  CUDA_CHECK(cudaEventCreate(&E0));
  CUDA_CHECK(cudaEventCreate(&E1));
}
void close_all() {
  CUDA_CHECK(cudaDeviceSynchronize());
  C.reset();
}
void set_patch(const std::string& n, bool on) { e55::patch::set(n, on); }
bool patch_on(const std::string& n) { return e55::patch::on(n); }
void barrier() { C->barrier(cur()); }

void ar_raw(void* p, int64_t n, bool f32, cudaStream_t s) {
  if (f32) C->all_reduce_f32(static_cast<float*>(p), size_t(n), s);
  else C->all_reduce_bf16(static_cast<e55::bf16*>(p), size_t(n), s);
}
bool is_f32(const torch::Tensor& t) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous());
  TORCH_CHECK(t.scalar_type() == torch::kFloat32 || t.scalar_type() == torch::kBFloat16);
  return t.scalar_type() == torch::kFloat32;
}
void ar(torch::Tensor t) { ar_raw(t.data_ptr(), t.numel(), is_f32(t), cur()); }
int ar_async(torch::Tensor t) {
  return is_f32(t) ? C->all_reduce_f32_async(t.data_ptr<float>(), t.numel(), cur())
                   : C->all_reduce_bf16_async(reinterpret_cast<e55::bf16*>(t.data_ptr()), t.numel(), cur());
}
void join(int t) { C->join(t, cur()); }

// pieces: src [M, w] (row stride from the tensor), dst (row stride from the tensor), rank r's part at off + r·stride
std::vector<e55::AgPiece> pieces(const std::vector<torch::Tensor>& src, const std::vector<torch::Tensor>& dst,
                                 const std::vector<int64_t>& off, const std::vector<int64_t>& stride) {
  std::vector<e55::AgPiece> v;
  for (size_t j = 0; j < src.size(); j++) {
    const int64_t es = src[j].element_size(), ed = dst[j].element_size();
    v.push_back({src[j].data_ptr(), dst[j].data_ptr(), src[j].stride(0) * es, dst[j].stride(0) * ed, off[j], stride[j],
                 int(src[j].size(1) * es)});
  }
  return v;
}
int ag(std::vector<torch::Tensor> src, std::vector<torch::Tensor> dst, std::vector<int64_t> off,
       std::vector<int64_t> stride, int M, bool async) {
  auto v = pieces(src, dst, off, stride);
  if (async) return C->all_gather_rows_async(v.data(), int(v.size()), M, cur());
  C->all_gather_rows(v.data(), int(v.size()), M, cur());
  return -1;
}
void ag_flat(torch::Tensor send, torch::Tensor recv) {  // recv = [rank0 | rank1]
  if (is_f32(send)) C->all_gather_f32(send.data_ptr<float>(), recv.data_ptr<float>(), send.numel(), cur());
  else C->all_gather_bf16(reinterpret_cast<e55::bf16*>(send.data_ptr()), reinterpret_cast<e55::bf16*>(recv.data_ptr()),
                          send.numel(), cur());
}

// µs per iteration (median, best) of `iters` back-to-back iterations, each: [the stream stand-in over pre_bytes of a
// rotation of 8 cold buffers] then op. reps after a barrier each (first rep = warm-up).
template <typename F>
std::vector<double> timed(int iters, int reps, int64_t pre_bytes, void* pre_dst, F op) {
  std::vector<torch::Tensor> rot;
  if (pre_bytes > 0)
    for (int i = 0; i < 8; i++)
      rot.push_back(torch::ones({pre_bytes}, torch::dtype(torch::kUInt8).device(torch::kCUDA)));
  CUDA_CHECK(cudaDeviceSynchronize());
  const cudaStream_t s = cur();
  std::vector<double> ts;
  for (int rep = 0; rep < reps + 1; rep++) {
    C->barrier(s);
    CUDA_CHECK(cudaEventRecord(E0, s));
    for (int i = 0; i < iters; i++) {
      if (pre_bytes > 0) comm_test_stream(rot[i % 8].data_ptr(), size_t(pre_bytes), pre_dst, s);
      op(s);
    }
    CUDA_CHECK(cudaEventRecord(E1, s));
    CUDA_CHECK(cudaEventSynchronize(E1));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, E0, E1));
    if (rep) ts.push_back(ms * 1000.0 / iters);
  }
  std::sort(ts.begin(), ts.end());
  return {ts[ts.size() / 2], ts.front()};
}
// all-reduce of n zeros (zeros stay zeros); f32 or bf16
std::vector<double> time_ar(int64_t n, bool f32, int iters, int reps, int64_t pre_bytes) {
  auto t = torch::zeros({n}, torch::dtype(f32 ? torch::kFloat32 : torch::kBFloat16).device(torch::kCUDA));
  void* p = t.data_ptr();
  return timed(iters, reps, pre_bytes, p, [&](cudaStream_t s) { ar_raw(p, n, f32, s); });
}
// all-gather of M rows of pieces of widths[j] bytes (pieces into separate dst buffers)
std::vector<double> time_ag(std::vector<int64_t> widths, int M, int iters, int reps, int64_t pre_bytes) {
  std::vector<torch::Tensor> src, dst;
  std::vector<e55::AgPiece> v;
  const int np = int(widths.size());
  for (int j = 0; j < np; j++) {
    const int64_t width = widths[j];
    src.push_back(torch::zeros({M, width}, torch::dtype(torch::kUInt8).device(torch::kCUDA)));
    dst.push_back(torch::zeros({M, 2 * width}, torch::dtype(torch::kUInt8).device(torch::kCUDA)));
    v.push_back({src[j].data_ptr(), dst[j].data_ptr(), width, 2 * width, 0, width, int(width)});
  }
  return timed(iters, reps, pre_bytes, src[0].data_ptr(),
               [&](cudaStream_t s) { C->all_gather_rows(v.data(), np, M, s); });
}
// the phase stamps (RdmaAR::probe) over `iters` back-to-back calls after a barrier: probe_read()'s vector
template <typename F>
std::vector<double> probed(int iters, F op) {
  const cudaStream_t s = cur();
  C->barrier(s);
  C->rdma_ar()->probe(true);
  for (int i = 0; i < iters; i++) op(s);
  CUDA_CHECK(cudaStreamSynchronize(s));
  auto r = C->rdma_ar()->probe_read();
  C->rdma_ar()->probe(false);
  return r;
}
std::vector<double> probe_ar(int64_t n, bool f32, int iters) {
  auto t = torch::zeros({n}, torch::dtype(f32 ? torch::kFloat32 : torch::kBFloat16).device(torch::kCUDA));
  void* p = t.data_ptr();
  return probed(iters, [&](cudaStream_t s) { ar_raw(p, n, f32, s); });
}
std::vector<double> probe_ag(std::vector<int64_t> widths, int M, int iters) {
  std::vector<torch::Tensor> src, dst;
  std::vector<e55::AgPiece> v;
  for (size_t j = 0; j < widths.size(); j++) {
    src.push_back(torch::zeros({M, widths[j]}, torch::dtype(torch::kUInt8).device(torch::kCUDA)));
    dst.push_back(torch::zeros({M, 2 * widths[j]}, torch::dtype(torch::kUInt8).device(torch::kCUDA)));
    v.push_back({src[j].data_ptr(), dst[j].data_ptr(), widths[j], 2 * widths[j], 0, widths[j], int(widths[j])});
  }
  return probed(iters, [&](cudaStream_t s) { C->all_gather_rows(v.data(), int(v.size()), M, s); });
}
// large calls: tuning (RdmaAR::tune) and ms per all-reduce of `bytes` (median of reps, after a barrier each)
void tune(int chunk_kb, int copy_blocks, int add_blocks, int ahead, int xfer_only) {
  C->rdma_ar()->tune(chunk_kb, copy_blocks, add_blocks, ahead, xfer_only);
}
double time_big(int64_t bytes, bool f32, int reps) {
  auto t = torch::zeros({bytes / (f32 ? 4 : 2)}, torch::dtype(f32 ? torch::kFloat32 : torch::kBFloat16).device(torch::kCUDA));
  const cudaStream_t s = cur();
  std::vector<double> ts;
  for (int r = 0; r < reps + 1; r++) {
    C->barrier(s);
    CUDA_CHECK(cudaEventRecord(E0, s));
    ar_raw(t.data_ptr(), t.numel(), f32, s);
    CUDA_CHECK(cudaEventRecord(E1, s));
    CUDA_CHECK(cudaEventSynchronize(E1));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, E0, E1));
    if (r) ts.push_back(ms);
  }
  std::sort(ts.begin(), ts.end());
  return ts[ts.size() / 2];
}
// transport probe (RdmaAR::probe): per-call phase times of the small paths and the proxy's post cost
void probe(bool on) { C->rdma_ar()->probe(on); }
std::vector<double> probe_read() { return C->rdma_ar()->probe_read(); }
void tune_small(int ll_max, int ag_ll_max, int fused_chunks, int ll_split, int ll_block, int fused_block) {
  C->rdma_ar()->tune_small(ll_max, ag_ll_max, fused_chunks, ll_split, ll_block, fused_block);
}
// GPU ↔ CPU round trip (ns) through pinned host memory: a CPU thread on the biggest core other than the proxy's
// answers each GPU write (n rounds)
double ping(int n) {
  int cpu = -1, best = -1;
  const long nc = sysconf(_SC_NPROCESSORS_CONF);
  for (int i = 0; i < nc; i++) {
    int cap = 0;
    char path[96];
    snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpu_capacity", i);
    if (FILE* f = std::fopen(path, "r")) {
      if (std::fscanf(f, "%d", &cap) != 1) cap = 0;
      std::fclose(f);
    }
    if (i != C->rdma_ar()->proxy_cpu() && cap >= best) best = cap, cpu = i;
  }
  unsigned long long* h = nullptr;
  CUDA_CHECK(cudaHostAlloc(&h, 256, cudaHostAllocMapped));
  std::memset(h, 0, 256);
  unsigned long long* d = nullptr;
  CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d), h, 0));
  unsigned long long* out = nullptr;
  CUDA_CHECK(cudaMalloc(&out, 8));
  std::thread th([&] {
    if (cpu >= 0) {
      cpu_set_t set;
      CPU_ZERO(&set);
      CPU_SET(cpu, &set);
      pthread_setaffinity_np(pthread_self(), sizeof set, &set);
    }
    for (unsigned long long i = 1; i <= (unsigned long long)n; i++) {
      while (__atomic_load_n(h, __ATOMIC_ACQUIRE) != i) asm volatile("yield" ::: "memory");
      __atomic_store_n(h + 8, i, __ATOMIC_RELEASE);
    }
  });
  comm_test_ping(d, d + 8, n, out, cur());
  CUDA_CHECK(cudaStreamSynchronize(cur()));
  th.join();
  unsigned long long ns = 0;
  CUDA_CHECK(cudaMemcpy(&ns, out, 8, cudaMemcpyDeviceToHost));
  cudaFree(out);
  cudaFreeHost(h);
  return double(ns) / n;
}
// the stand-in alone (pre_bytes) — the baseline the two above add to
std::vector<double> time_pre(int64_t pre_bytes, int iters, int reps) {
  auto t = torch::zeros({16}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
  return timed(iters, reps, pre_bytes, t.data_ptr(), [](cudaStream_t) {});
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", &init);
  m.def("close", &close_all);
  m.def("set_patch", &set_patch);
  m.def("patch_on", &patch_on);
  m.def("barrier", &barrier);
  m.def("ar", &ar);
  m.def("ar_async", &ar_async);
  m.def("join", &join);
  m.def("ag", &ag);
  m.def("ag_flat", &ag_flat);
  m.def("time_ar", &time_ar);
  m.def("time_ag", &time_ag);
  m.def("time_pre", &time_pre);
  m.def("tune", &tune);
  m.def("time_big", &time_big);
  m.def("probe", &probe);
  m.def("probe_read", &probe_read);
  m.def("tune_small", &tune_small);
  m.def("ping", &ping);
  m.def("probe_ar", &probe_ar);
  m.def("probe_ag", &probe_ag);
}
