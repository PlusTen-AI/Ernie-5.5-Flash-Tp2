// Test binding for the overlap gate (engine/tests/test_overlap.py): one HCA layer's attention (kernels/attn/hca.cu,
// hca_attn_fused at the default dispatch) and one MoE layer's experts (kernels/moe/moe.cu, moe_experts_bf16 at the
// default prefill path) on caller-owned tensors, each launched on the caller's current stream.
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cmath>
#include <memory>

#include "core/weights.h"
#include "kernels/attn/hca.h"
#include "kernels/moe/moe.h"
#include "model/patches.h"

namespace e55 {
void hca_attn_long2_tune(int min_ch, int cap);                                         // hca.cu (tests)
void test_act_quant32(const bf16* x, int M, uint8_t* q, uint8_t* s, cudaStream_t st);  // tests/bind/moe_aq.cu
}  // namespace e55

namespace {
using e55::bf16;
cudaStream_t st() { return c10::cuda::getCurrentCUDAStream().stream(); }
bf16* B(torch::Tensor t) { return reinterpret_cast<bf16*>(t.data_ptr()); }
const e55::RowMeta* MT(torch::Tensor m) {
  TORCH_CHECK(m.dtype() == torch::kInt32 && m.size(1) == 5);
  return reinterpret_cast<const e55::RowMeta*>(m.data_ptr<int>());
}
e55::HcaCache C(torch::Tensor ring, torch::Tensor comp, torch::Tensor state) {
  e55::HcaCache c;
  c.ring = ring.data_ptr<uint8_t>(), c.comp = comp.data_ptr<uint8_t>(), c.state = state.data_ptr<float>();
  c.max_comp = int(comp.size(1));
  return c;
}

// model.cc inv_yarn(160000, 16, 65536, 32, 1) (as tests/bind/hca.cpp)
torch::Tensor inv_freq() {
  const float theta = 160000.f, factor = 16.f;
  const double orig = 65536, bfast = 32, bslow = 1;
  auto corr = [&](double r) { return 64.0 * std::log(orig / (r * 2 * M_PI)) / (2 * std::log(double(theta))); };
  double low = std::max(std::floor(corr(bfast)), 0.0), high = std::min(std::ceil(corr(bslow)), 63.0);
  if (low == high) high += 0.001;
  auto v = torch::empty({32}, torch::kFloat32);
  for (int i = 0; i < 32; i++) {
    const float pf = powf(theta, float(2 * i) / 64.0f);
    const float extra = 1.0f / pf, inter = 1.0f / (factor * pf);
    const float ramp = std::min(1.0f, std::max(0.0f, (float(i) - float(low)) / float(high - low)));
    const float mask = 1.0f - ramp;
    v[i] = inter * (1.0f - mask) + extra * mask;
  }
  return v.cuda();
}

// ---- HCA
void kv_rows(torch::Tensor kv, torch::Tensor meta, torch::Tensor inv, torch::Tensor rows) {
  e55::hca_kv_rows(B(kv), int(kv.size(1)), MT(meta), inv.data_ptr<float>(), int(meta.size(0)), rows.data_ptr<uint8_t>(),
                   st());
}
void pre(torch::Tensor kv, torch::Tensor ckv, torch::Tensor ape, torch::Tensor nw, torch::Tensor meta, torch::Tensor inv,
         torch::Tensor rows, torch::Tensor ring, torch::Tensor comp, torch::Tensor state, torch::Tensor ws) {
  e55::hca_pre(B(kv), int(kv.size(1)), ckv.data_ptr<float>(), ape.data_ptr<float>(), B(nw), MT(meta),
               inv.data_ptr<float>(), int(meta.size(0)), rows.data_ptr<uint8_t>(), C(ring, comp, state),
               ws.data_ptr<float>(), size_t(ws.numel()) * 4, st());
}
void attn_fused(torch::Tensor q, torch::Tensor rows, torch::Tensor meta, torch::Tensor ring, torch::Tensor comp,
                torch::Tensor state, torch::Tensor sink, torch::Tensor inv, torch::Tensor U, int max_keys,
                torch::Tensor o, torch::Tensor zp, torch::Tensor ws) {
  e55::hca_attn_fused(B(q), rows.data_ptr<uint8_t>(), MT(meta), C(ring, comp, state), sink.data_ptr<float>(),
                      inv.data_ptr<float>(), U.data_ptr<float>(), int(meta.size(0)), max_keys, B(o),
                      zp.data_ptr<float>(), ws.data_ptr<float>(), size_t(ws.numel()) * 4, st());
}
void hca_cap(int cap) { e55::hca_attn_long2_tune(0, cap); }  // P172 on at most cap CTAs (0: every SM)

// ---- MoE (rank 0's experts of one layer)
std::unique_ptr<e55::Weights> g_w;
e55::MoeExperts* g_e = nullptr;
e55::MoeScratch* g_s = nullptr;
void moe_open(const std::string& model_dir, const std::string& table, int layer) {
  g_w = std::make_unique<e55::Weights>(model_dir, table, 0);
  g_e = e55::moe_upload(*g_w, layer);
  g_s = e55::moe_scratch(8192);
}
std::vector<torch::Tensor> route(torch::Tensor lg, torch::Tensor bias, torch::Tensor scale) {
  const int M = int(lg.size(0));
  auto ids = torch::empty({M, 10}, lg.options().dtype(torch::kInt32));
  auto w = torch::empty({M, 10}, lg.options().dtype(torch::kFloat32));
  e55::moe_route(lg.data_ptr<float>(), bias.data_ptr<float>(), scale.data_ptr<float>(), M, ids.data_ptr<int>(),
                 w.data_ptr<float>(), st());
  return {ids, w};
}
std::vector<torch::Tensor> act_q32(torch::Tensor lat) {
  const int M = int(lat.size(0));
  auto q = torch::empty({M, 2048}, lat.options().dtype(torch::kUInt8));
  auto s = torch::empty({M, 64}, lat.options().dtype(torch::kUInt8));
  e55::test_act_quant32(B(lat), M, q.data_ptr<uint8_t>(), s.data_ptr<uint8_t>(), st());
  return {q, s};
}
void experts(torch::Tensor q, torch::Tensor s, torch::Tensor ids, torch::Tensor w, torch::Tensor part) {
  e55::moe_experts_bf16(g_e, g_s, q.data_ptr<uint8_t>(), s.data_ptr<uint8_t>(), ids.data_ptr<int>(),
                        w.data_ptr<float>(), int(q.size(0)), B(part), int(part.size(1)), st());
}
void moe_close() {
  if (g_e) e55::moe_free(g_e), g_e = nullptr;
  if (g_s) e55::moe_scratch_free(g_s), g_s = nullptr;
  g_w.reset();
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("inv_freq", &inv_freq);
  m.def("kv_rows", &kv_rows);
  m.def("pre", &pre);
  m.def("attn_fused", &attn_fused);
  m.def("hca_cap", &hca_cap);
  m.def("moe_open", &moe_open);
  m.def("route", &route);
  m.def("act_q32", &act_q32);
  m.def("experts", &experts);
  m.def("moe_close", &moe_close);
}
