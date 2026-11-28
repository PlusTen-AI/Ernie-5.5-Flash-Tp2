// Test binding for the prefill-rule HCA attention (P210 hca_attn_1p, engine/tests/test_hca_pf.py): the fused chain's
// stages on caller-owned tensors (KV rows, hca_pre, hca_attn_fused, hca_post), patch switches, the step kind
// (core/step_kind.h: P210 is taken only when a prefill step is running) and an L2 flush for timing.
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cmath>

#include "core/step_kind.h"
#include "kernels/attn/hca.h"
#include "model/patches.h"

namespace {
using e55::bf16;
cudaStream_t st() { return c10::cuda::getCurrentCUDAStream().stream(); }
bf16* B(torch::Tensor t) { return reinterpret_cast<bf16*>(t.data_ptr()); }
const e55::RowMeta* MT(torch::Tensor m) {
  TORCH_CHECK(m.dtype() == torch::kInt32 && m.size(1) == 5 && m.is_contiguous());
  return reinterpret_cast<const e55::RowMeta*>(m.data_ptr<int>());
}
e55::HcaCache C(torch::Tensor ring, torch::Tensor comp, torch::Tensor state) {
  e55::HcaCache c;
  c.ring = ring.data_ptr<uint8_t>(), c.comp = comp.data_ptr<uint8_t>(), c.state = state.data_ptr<float>();
  c.max_comp = int(comp.size(1));
  return c;
}

// model.cc inv_yarn(160000, 16, 65536, 32, 1), the same float code (as tests/bind/hca.cpp)
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

void patch(const std::string& n, bool on) { e55::patch::set(n, on); }
void set_prefill(bool on) { e55::set_step_prefill(on); }  // the step kind the dispatchers read (this module's copy)
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
void post(torch::Tensor o, torch::Tensor z, torch::Tensor V, torch::Tensor meta, torch::Tensor inv, torch::Tensor rows,
          torch::Tensor ckv, torch::Tensor ape, torch::Tensor ring, torch::Tensor comp, torch::Tensor state,
          torch::Tensor q8, torch::Tensor sc) {
  e55::hca_post(B(o), z.data_ptr<float>(), V.data_ptr<float>(), MT(meta), inv.data_ptr<float>(),
                rows.data_ptr<uint8_t>(), ckv.data_ptr<float>(), ape.data_ptr<float>(), int(meta.size(0)),
                C(ring, comp, state), q8.data_ptr<uint8_t>(), sc.data_ptr<uint8_t>(), st());
}

torch::Tensor g_flush;
void flush() {
  if (!g_flush.defined()) g_flush = torch::zeros({int64_t(256) << 20}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
  g_flush.add_(1);
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("inv_freq", &inv_freq);
  m.def("patch", &patch);
  m.def("set_prefill", &set_prefill);
  m.def("kv_rows", &kv_rows);
  m.def("pre", &pre);
  m.def("attn_fused", &attn_fused);
  m.def("post", &post);
  m.def("flush", &flush);
}
