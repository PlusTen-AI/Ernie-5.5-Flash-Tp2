// Test binding for P097 lm12_only / P098 lin12_prefill (kernels/linear: p098_lin12_prefill.cuh): bf16 linears of the
// checkpoint uploaded with both copies (base: the bf16 prefill GEMM at M > 64) or with P013's 12-bit copy only (every
// M reads it), outputs returned for bitwise comparison, and CUDA-event timing of back-to-back calls (tests/test_lin12.py).
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <memory>

#include "core/log.h"
#include "kernels/linear/linear.h"
#include "model/patches.h"

namespace {
using e55::bf16;
std::unique_ptr<e55::Weights> g_w;
std::vector<e55::Lin> g_lin;

void open(const std::string& model_dir, const std::string& table, int rank) {
  g_w = std::make_unique<e55::Weights>(model_dir, table, rank);
}
// mode: 0 bf16 only · 1 bf16 + 12-bit · 2 12-bit only
int lin(const std::vector<std::string>& names, int mode) {
  g_lin.push_back(e55::lin_upload(*g_w, names, mode));
  return int(g_lin.size()) - 1;
}
std::vector<int64_t> lin_info(int h) {
  const auto& L = g_lin.at(h);
  return {L.n, L.k, L.split, L.w ? 1 : 0, L.pk ? 1 : 0, L.pk_nx};
}
void set_patch(const std::string& name, bool on) { e55::patch::set(name, on); }
void tile(int t) { e55::lin_p098_tile(t); }

const bf16* P(const torch::Tensor& t) { return reinterpret_cast<const bf16*>(t.data_ptr()); }

torch::Tensor fwd(int h, torch::Tensor x) {
  const auto& L = g_lin.at(h);
  const int M = int(x.size(0));
  auto y = torch::empty({M, L.n}, x.options().dtype(torch::kFloat32));
  e55::lin_forward_f32(L, P(x), y.data_ptr<float>(), M, c10::cuda::getCurrentCUDAStream().stream());
  return y;
}
// ms per call (CUDA events): runs × (one call per handle, back to back — distinct layers' weights)
std::vector<double> bench(const std::vector<int>& hs, torch::Tensor x, int runs) {
  auto s = c10::cuda::getCurrentCUDAStream().stream();
  const int M = int(x.size(0));
  int nmax = 0;
  for (int h : hs) nmax = std::max(nmax, g_lin.at(h).n);
  auto y = torch::empty({M, nmax}, x.options().dtype(torch::kFloat32));
  auto go = [&] {
    for (int h : hs) e55::lin_forward_f32(g_lin.at(h), P(x), y.data_ptr<float>(), M, s);
  };
  go();
  std::vector<cudaEvent_t> ev(runs + 1);
  for (auto& e : ev) C10_CUDA_CHECK(cudaEventCreate(&e));
  C10_CUDA_CHECK(cudaStreamSynchronize(s));
  C10_CUDA_CHECK(cudaEventRecord(ev[0], s));
  for (int i = 0; i < runs; i++) {
    go();
    C10_CUDA_CHECK(cudaEventRecord(ev[i + 1], s));
  }
  C10_CUDA_CHECK(cudaEventSynchronize(ev[runs]));
  std::vector<double> ms(runs);
  for (int i = 0; i < runs; i++) {
    float f;
    C10_CUDA_CHECK(cudaEventElapsedTime(&f, ev[i], ev[i + 1]));
    ms[i] = f / hs.size();
  }
  for (auto& e : ev) cudaEventDestroy(e);
  return ms;
}
void free_all() {
  for (auto& L : g_lin) e55::lin_free(L);
  g_lin.clear();
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("open", &open);
  m.def("lin", &lin);
  m.def("lin_info", &lin_info);
  m.def("set_patch", &set_patch);
  m.def("tile", &tile);
  m.def("fwd", &fwd);
  m.def("bench", &bench);
  m.def("free_all", &free_all);
}
