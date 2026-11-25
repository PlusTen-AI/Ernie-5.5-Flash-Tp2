// Test binding for the MoE family (engine/src/kernels/moe/moe.cu): one layer's experts (either rank), routing,
// expert pass, timing.
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <memory>

#include <algorithm>

#include "core/log.h"
#include "kernels/moe/moe.h"

namespace e55 {
void test_act_quant32(const bf16* x, int M, uint8_t* q, uint8_t* s, cudaStream_t st);  // moe_aq.cu
void moe_test_plan(MoeScratch* S, const int* ids, int M, bool fast, cudaStream_t s);  // moe.cu
}
#include "model/patches.h"

namespace {
std::string g_dir, g_table;
std::unique_ptr<e55::Weights> g_w[2];
std::vector<e55::MoeExperts*> g_e;
e55::MoeScratch* g_s = nullptr;

void open(const std::string& model_dir, const std::string& table) { g_dir = model_dir, g_table = table; }
int upload(int rank, int layer) {
  if (!g_w[rank]) g_w[rank] = std::make_unique<e55::Weights>(g_dir, g_table, rank);
  g_e.push_back(e55::moe_upload(*g_w[rank], layer));
  if (!g_s) g_s = e55::moe_scratch(8192);
  return int(g_e.size()) - 1;
}
std::vector<torch::Tensor> route(torch::Tensor lg, torch::Tensor bias, torch::Tensor scale) {
  const int M = int(lg.size(0));
  auto ids = torch::empty({M, 10}, lg.options().dtype(torch::kInt32));
  auto w = torch::empty({M, 10}, lg.options().dtype(torch::kFloat32));
  e55::moe_route(lg.data_ptr<float>(), bias.data_ptr<float>(), scale.data_ptr<float>(), M, ids.data_ptr<int>(),
                 w.data_ptr<float>(), c10::cuda::getCurrentCUDAStream().stream());
  return {ids, w};
}
torch::Tensor experts(int h, torch::Tensor q, torch::Tensor s, torch::Tensor ids, torch::Tensor w, bool bf) {
  const int M = int(q.size(0));
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  auto part = torch::empty({M, 2048}, q.options().dtype(bf ? torch::kBFloat16 : torch::kFloat32));
  if (bf)
    e55::moe_experts_bf16(g_e.at(h), g_s, q.data_ptr<uint8_t>(), s.data_ptr<uint8_t>(), ids.data_ptr<int>(),
                          w.data_ptr<float>(), M, reinterpret_cast<e55::bf16*>(part.data_ptr()), 2048, st);
  else
    e55::moe_experts(g_e.at(h), g_s, q.data_ptr<uint8_t>(), s.data_ptr<uint8_t>(), ids.data_ptr<int>(),
                     w.data_ptr<float>(), M, part.data_ptr<float>(), 2048, st);
  return part;
}
double bench(int h, torch::Tensor q, torch::Tensor s, torch::Tensor ids, torch::Tensor w, int iters) {
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  const int M = int(q.size(0));
  auto part = torch::empty({M, 2048}, q.options().dtype(torch::kFloat32));
  auto go = [&] {
    e55::moe_experts(g_e.at(h), g_s, q.data_ptr<uint8_t>(), s.data_ptr<uint8_t>(), ids.data_ptr<int>(),
                     w.data_ptr<float>(), M, part.data_ptr<float>(), 2048, st);
  };
  go();
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0), cudaEventCreate(&e1);
  cudaEventRecord(e0, st);
  for (int i = 0; i < iters; i++) go();
  cudaEventRecord(e1, st);
  cudaEventSynchronize(e1);
  float ms;
  cudaEventElapsedTime(&ms, e0, e1);
  return ms / iters;
}
void set_patch(const std::string& name, bool on) { e55::patch::set(name, on); }

// the latent as MXFP8 per 32 (the engine's act_quant, as model.cc runs it before the experts)
std::vector<torch::Tensor> act_q32(torch::Tensor lat) {
  const int M = int(lat.size(0));
  auto q = torch::empty({M, 2048}, lat.options().dtype(torch::kUInt8));
  auto s = torch::empty({M, 64}, lat.options().dtype(torch::kUInt8));
  e55::test_act_quant32(reinterpret_cast<const e55::bf16*>(lat.data_ptr()), M, q.data_ptr<uint8_t>(),
                        s.data_ptr<uint8_t>(), c10::cuda::getCurrentCUDAStream().stream());
  return {q, s};
}
// P066: route + latent quantization + experts in one call → ids, w, part (bf16 or fp32)
std::vector<torch::Tensor> fused(int h, torch::Tensor lg, torch::Tensor bias, torch::Tensor scale, torch::Tensor lat,
                                 bool bf, bool preq) {
  const int M = int(lg.size(0));
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  auto ids = torch::empty({M, 10}, lg.options().dtype(torch::kInt32));
  auto w = torch::empty({M, 10}, lg.options().dtype(torch::kFloat32));
  auto part = torch::empty({M, 2048}, lg.options().dtype(bf ? torch::kBFloat16 : torch::kFloat32));
  const auto* lp = reinterpret_cast<const e55::bf16*>(lat.data_ptr());
  auto q = torch::empty({M, 2048}, lg.options().dtype(torch::kUInt8));
  auto sc = torch::empty({M, 64}, lg.options().dtype(torch::kUInt8));
  if (preq) e55::test_act_quant32(lp, M, q.data_ptr<uint8_t>(), sc.data_ptr<uint8_t>(), st);
  const e55::bf16* l = preq ? nullptr : lp;
  const uint8_t *lq = preq ? q.data_ptr<uint8_t>() : nullptr, *ls = preq ? sc.data_ptr<uint8_t>() : nullptr;
  if (bf)
    e55::moe_fused(g_e.at(h), g_s, lg.data_ptr<float>(), bias.data_ptr<float>(), scale.data_ptr<float>(), l, 2048, lq,
                   ls, M, ids.data_ptr<int>(), w.data_ptr<float>(), reinterpret_cast<e55::bf16*>(part.data_ptr()), 2048,
                   st);
  else
    e55::moe_fused_f32(g_e.at(h), g_s, lg.data_ptr<float>(), bias.data_ptr<float>(), scale.data_ptr<float>(), l, 2048,
                       lq, ls, M, ids.data_ptr<int>(), w.data_ptr<float>(), part.data_ptr<float>(), 2048, st);
  return {ids, w, part};
}

// Cold timing: L2 flushed before each call by reading 256 MB (clean lines, as in a step where other layers' weights
// pass through L2); call i uses ids[i % N] / w[i % N] ([N, M, 10]). Returns the median ms of the calls (CUDA events
// around each call only).
torch::Tensor g_flush;
double median_ms(std::vector<cudaEvent_t>& ev, int n) {
  std::vector<float> t(n);
  for (int i = 0; i < n; i++) cudaEventElapsedTime(&t[i], ev[2 * i], ev[2 * i + 1]);
  for (auto& e : ev) cudaEventDestroy(e);
  std::sort(t.begin(), t.end());
  return t[n / 2];
}
template <class F>
double time_cold(int iters, bool cold, F go) {
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  if (cold && !g_flush.defined())
    g_flush = torch::ones({64 << 20}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  go(0);
  std::vector<cudaEvent_t> ev(2 * iters);
  for (auto& e : ev) cudaEventCreate(&e);
  for (int i = 0; i < iters; i++) {
    if (cold) g_flush.sum();
    cudaEventRecord(ev[2 * i], st);
    go(i);
    cudaEventRecord(ev[2 * i + 1], st);
  }
  cudaStreamSynchronize(st);
  return median_ms(ev, iters);
}
double bench_cold(int h, torch::Tensor q, torch::Tensor s, torch::Tensor ids, torch::Tensor w, int iters, bool cold) {
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  const int N = int(ids.size(0)), M = int(q.size(0));
  auto part = torch::empty({M, 2048}, q.options().dtype(torch::kFloat32));
  return time_cold(iters, cold, [&](int i) {
    e55::moe_experts(g_e.at(h), g_s, q.data_ptr<uint8_t>(), s.data_ptr<uint8_t>(), ids[i % N].data_ptr<int>(),
                     w[i % N].data_ptr<float>(), M, part.data_ptr<float>(), 2048, st);
  });
}
// ms of one cold read pass over n MB (torch sum; the practical DRAM read rate)
double bench_read(int mb, int iters) {
  auto x = torch::ones({int64_t(mb) << 18}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  return time_cold(iters, true, [&](int) { x.sum(); });
}
// calibration: a 4-byte memset (one tiny kernel) timed the same way
double bench_null(int iters) {
  auto x = torch::empty({1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  return time_cold(iters, false, [&](int) { cudaMemsetAsync(x.data_ptr(), 0, 4, st); });
}
// decode MoE as model.cc runs it, cold: route + act_quant + experts (bf16) — or with P066 the fused launch;
// call i uses logits lg[i % N] ([N, M, 1024]) and latent lat [M, 2048] bf16
double bench_dec(int h, torch::Tensor lg, torch::Tensor bias, torch::Tensor scale, torch::Tensor lat, int iters,
                 bool preq) {
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  const int N = int(lg.size(0)), M = int(lg.size(1));
  auto ids = torch::empty({M, 10}, lg.options().dtype(torch::kInt32));
  auto w = torch::empty({M, 10}, lg.options().dtype(torch::kFloat32));
  auto q = torch::empty({M, 2048}, lg.options().dtype(torch::kUInt8));
  auto sc = torch::empty({M, 64}, lg.options().dtype(torch::kUInt8));
  auto part = torch::empty({M, 6144}, lg.options().dtype(torch::kBFloat16));
  const auto* lp = reinterpret_cast<const e55::bf16*>(lat.data_ptr());
  auto* pp = reinterpret_cast<e55::bf16*>(part.data_ptr());
  if (preq) e55::test_act_quant32(lp, M, q.data_ptr<uint8_t>(), sc.data_ptr<uint8_t>(), st);
  return time_cold(iters, true, [&](int i) {
    const float* l = lg[i % N].data_ptr<float>();
    if (e55::moe_fused_ok(M)) {
      e55::moe_fused(g_e.at(h), g_s, l, bias.data_ptr<float>(), scale.data_ptr<float>(), preq ? nullptr : lp, 2048,
                     preq ? q.data_ptr<uint8_t>() : nullptr, preq ? sc.data_ptr<uint8_t>() : nullptr, M,
                     ids.data_ptr<int>(), w.data_ptr<float>(), pp, 6144, st);
      return;
    }
    e55::moe_route(l, bias.data_ptr<float>(), scale.data_ptr<float>(), M, ids.data_ptr<int>(), w.data_ptr<float>(), st);
    if (!preq) e55::test_act_quant32(lp, M, q.data_ptr<uint8_t>(), sc.data_ptr<uint8_t>(), st);
    e55::moe_experts_bf16(g_e.at(h), g_s, q.data_ptr<uint8_t>(), sc.data_ptr<uint8_t>(), ids.data_ptr<int>(),
                          w.data_ptr<float>(), M, pp, 6144, st);
  });
}
// the routing plan alone (base four launches or P065's one), cold, ms
double bench_plan(torch::Tensor ids, bool fast, int iters) {
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  return time_cold(iters, true, [&](int) { e55::moe_test_plan(g_s, ids.data_ptr<int>(), int(ids.size(0)), fast, st); });
}
double bench_route(torch::Tensor lg, torch::Tensor bias, torch::Tensor scale, int iters) {
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  const int M = int(lg.size(0));
  auto ids = torch::empty({M, 10}, lg.options().dtype(torch::kInt32));
  auto w = torch::empty({M, 10}, lg.options().dtype(torch::kFloat32));
  return time_cold(iters, false, [&](int) {
    e55::moe_route(lg.data_ptr<float>(), bias.data_ptr<float>(), scale.data_ptr<float>(), M, ids.data_ptr<int>(),
                   w.data_ptr<float>(), st);
  });
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("open", &open);
  m.def("upload", &upload);
  m.def("route", &route);
  m.def("experts", &experts, pybind11::arg("h"), pybind11::arg("q"), pybind11::arg("s"), pybind11::arg("ids"),
        pybind11::arg("w"), pybind11::arg("bf") = false);
  m.def("bench", &bench);
  m.def("bench_cold", &bench_cold);
  m.def("bench_route", &bench_route);
  m.def("bench_read", &bench_read);
  m.def("bench_null", &bench_null);
  m.def("set_patch", &set_patch);
  m.def("act_q32", &act_q32);
  m.def("fused", &fused);
  m.def("bench_dec", &bench_dec);
  m.def("bench_plan", &bench_plan);
}
