// Test binding for P114 lin_pair / P115 lin_small (kernels/linear p114_lin_pair.cuh): the checkpoint's router / HCA
// compressor (12-bit copy only, as the engine uploads them), their P078 column halves (views, as model.cc lin_rows),
// the DSA indexer k ‖ weights_proj (bf16) and lm_head (12-bit only); outputs for bitwise comparison, and cold per-call
// timing (L2 flushed by reading a large buffer before every call; CUDA events around the call alone).
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <memory>

#include "core/log.h"
#include "kernels/linear/linear.h"
#include "model/patches.h"

void lindec_spin(long ns, cudaStream_t s);  // lindec_spin.cu

namespace {
using e55::bf16;
std::unique_ptr<e55::Weights> g_w;
std::vector<e55::Lin> g_lin;
std::vector<bool> g_view;

void open(const std::string& model_dir, const std::string& table, int rank) {
  g_w = std::make_unique<e55::Weights>(model_dir, table, rank);
}
// mode: 0 bf16 only · 1 bf16 + 12-bit · 2 12-bit only
int lin(const std::vector<std::string>& names, int mode) {
  g_lin.push_back(e55::lin_upload(*g_w, names, mode));
  g_view.push_back(false);
  return int(g_lin.size()) - 1;
}
// rows [r0, r0 + n) of handle h as a Lin of its own (model.cc lin_rows, P078)
int rows(int h, int r0, int n) {
  const e55::Lin L = g_lin.at(h);
  e55::Lin v = L;
  v.n = v.npad = n;
  if (L.w) v.w = static_cast<uint8_t*>(L.w) + size_t(r0 / 16) * (L.k / 32) * 1024;
  if (L.pk) v.pk = L.pk + size_t(r0 / 16) * (L.k / 32) * 768, v.pk_base = L.pk_base + r0, v.pk_xo = L.pk_xo + r0 / 16;
  g_lin.push_back(v);
  g_view.push_back(true);
  return int(g_lin.size()) - 1;
}
std::vector<int64_t> lin_info(int h) {
  const auto& L = g_lin.at(h);
  return {L.n, L.k, L.split, L.w ? 1 : 0, L.pk ? 1 : 0, L.pk_nx};
}
void set_patch(const std::string& name, bool on) { e55::patch::set(name, on); }
void warps(int w) { e55::lin_p114_warps(w); }

const bf16* P(const torch::Tensor& t) { return reinterpret_cast<const bf16*>(t.data_ptr()); }
cudaStream_t cur() { return c10::cuda::getCurrentCUDAStream().stream(); }

torch::Tensor fwd(int h, torch::Tensor x) {
  const auto& L = g_lin.at(h);
  auto y = torch::empty({x.size(0), L.n}, x.options().dtype(torch::kFloat32));
  e55::lin_forward_f32(L, P(x), y.data_ptr<float>(), int(x.size(0)), cur());
  return y;
}
torch::Tensor fwd_bf16(int h, torch::Tensor x) {
  const auto& L = g_lin.at(h);
  auto y = torch::empty({x.size(0), L.n}, x.options());
  e55::lin_forward(L, P(x), reinterpret_cast<bf16*>(y.data_ptr()), int(x.size(0)), cur());
  return y;
}
std::vector<torch::Tensor> pair(int a, int b, torch::Tensor x) {
  const auto &A = g_lin.at(a), &B = g_lin.at(b);
  auto ya = torch::empty({x.size(0), A.n}, x.options().dtype(torch::kFloat32));
  auto yb = torch::empty({x.size(0), B.n}, x.options().dtype(torch::kFloat32));
  e55::lin_forward_f32_pair(A, B, P(x), ya.data_ptr<float>(), yb.data_ptr<float>(), int(x.size(0)), cur());
  return {ya, yb};
}

// µs per call, cold: for each round, for each item: read `flush` (evicts L2), read x (it is hot in the engine: the
// previous kernel wrote it), event, the call, event. kind 0: lin_forward_f32 (hs: handles) · 1: lin_forward (bf16
// out) · 2: lin_forward_f32_pair (hs: pairs a0, b0, a1, b1, …)
// spin_ns > 0: the call is launched behind lindec_spin(spin_ns) (events around both; the spin's length is subtracted),
// as in the engine, where a decode linear is a programmatic dependent of the kernel before it
std::vector<double> timed(int kind, const std::vector<int>& hs, torch::Tensor x, torch::Tensor flush, int rounds,
                          long spin_ns) {
  auto s = cur();
  const int M = int(x.size(0)), items = kind == 2 ? int(hs.size()) / 2 : int(hs.size());
  int nmax = 0;
  for (int h : hs) nmax = std::max(nmax, g_lin.at(h).n);
  auto y = torch::empty({2, M, nmax}, x.options().dtype(torch::kFloat32));
  float* y0 = y.data_ptr<float>();
  float* y1 = y0 + size_t(M) * nmax;
  auto call = [&](int i) {
    if (kind == 0) e55::lin_forward_f32(g_lin.at(hs[i]), P(x), y0, M, s);
    else if (kind == 1) e55::lin_forward(g_lin.at(hs[i]), P(x), reinterpret_cast<bf16*>(y0), M, s);
    else e55::lin_forward_f32_pair(g_lin.at(hs[2 * i]), g_lin.at(hs[2 * i + 1]), P(x), y0, y1, M, s);
  };
  for (int i = 0; i < items; i++) call(i);  // warm-up (first launches)
  const int n = rounds * items;
  std::vector<cudaEvent_t> ev(2 * n);
  for (auto& e : ev) C10_CUDA_CHECK(cudaEventCreate(&e));
  std::vector<torch::Tensor> keep;
  for (int r = 0; r < rounds; r++)
    for (int i = 0; i < items; i++) {
      keep.push_back(flush.sum());
      keep.push_back(x.sum());
      const int j = r * items + i;
      C10_CUDA_CHECK(cudaEventRecord(ev[2 * j], s));
      if (spin_ns > 0) lindec_spin(spin_ns, s);
      call(i);
      C10_CUDA_CHECK(cudaEventRecord(ev[2 * j + 1], s));
    }
  C10_CUDA_CHECK(cudaStreamSynchronize(s));
  std::vector<double> us(n);
  for (int j = 0; j < n; j++) {
    float f;
    C10_CUDA_CHECK(cudaEventElapsedTime(&f, ev[2 * j], ev[2 * j + 1]));
    us[j] = f * 1000.0 - spin_ns / 1000.0;
  }
  for (auto& e : ev) cudaEventDestroy(e);
  return us;
}
// µs per call of each round: an event, then for each item [lindec_spin(spin_ns)] + the call, an event; minus the spins,
// divided by the items (one interval over the round: no per-call event granularity). Cold weights by rotation (the
// caller passes more weight bytes than L2); x stays hot. kind as timed().
std::vector<double> timed_batch(int kind, const std::vector<int>& hs, torch::Tensor x, int rounds, long spin_ns) {
  auto s = cur();
  const int M = int(x.size(0)), items = kind == 2 ? int(hs.size()) / 2 : int(hs.size());
  int nmax = 0;
  for (int h : hs) nmax = std::max(nmax, g_lin.at(h).n);
  auto y = torch::empty({2, M, nmax}, x.options().dtype(torch::kFloat32));
  float* y0 = y.data_ptr<float>();
  float* y1 = y0 + size_t(M) * nmax;
  auto call = [&](int i) {
    if (kind == 0) e55::lin_forward_f32(g_lin.at(hs[i]), P(x), y0, M, s);
    else if (kind == 1) e55::lin_forward(g_lin.at(hs[i]), P(x), reinterpret_cast<bf16*>(y0), M, s);
    else e55::lin_forward_f32_pair(g_lin.at(hs[2 * i]), g_lin.at(hs[2 * i + 1]), P(x), y0, y1, M, s);
  };
  for (int i = 0; i < items; i++) call(i);  // warm-up
  cudaEvent_t e0, e1;
  C10_CUDA_CHECK(cudaEventCreate(&e0));
  C10_CUDA_CHECK(cudaEventCreate(&e1));
  std::vector<double> us;
  for (int r = 0; r < rounds; r++) {
    C10_CUDA_CHECK(cudaEventRecord(e0, s));
    for (int i = 0; i < items; i++) {
      if (spin_ns > 0) lindec_spin(spin_ns, s);
      call(i);
    }
    C10_CUDA_CHECK(cudaEventRecord(e1, s));
    C10_CUDA_CHECK(cudaEventSynchronize(e1));
    float f;
    C10_CUDA_CHECK(cudaEventElapsedTime(&f, e0, e1));
    us.push_back((f * 1000.0 - double(items) * spin_ns / 1000.0) / items);
  }
  cudaEventDestroy(e0), cudaEventDestroy(e1);
  return us;
}
void free_all() {
  for (size_t i = 0; i < g_lin.size(); i++)
    if (!g_view[i]) e55::lin_free(g_lin[i]);
  g_lin.clear(), g_view.clear();
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("open", &open);
  m.def("lin", &lin);
  m.def("rows", &rows);
  m.def("lin_info", &lin_info);
  m.def("set_patch", &set_patch);
  m.def("warps", &warps);
  m.def("fwd", &fwd);
  m.def("fwd_bf16", &fwd_bf16);
  m.def("pair", &pair);
  m.def("timed", &timed);
  m.def("timed_batch", &timed_batch);
  m.def("free_all", &free_all);
}
