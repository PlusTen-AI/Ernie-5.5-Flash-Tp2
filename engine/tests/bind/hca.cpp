// Test binding for the HCA attention family (engine/src/kernels/attn/hca.cu): every stage of attn_hca between the
// projections (q prep, KV rows, compressor, attention, ring, VHA) on caller-owned tensors, the decode-fused chain, patch
// switches, and GPU helpers for timing (L2 flush, spin).
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cmath>

#include "kernels/attn/hca.h"
#include "model/patches.h"

namespace e55 {
void hca_attn_long2_tune(int min_ch, int cap);  // hca.cu (tests): P172's dispatch bound and CTA cap
}
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

// model.cc inv_yarn(160000, 16, 65536, 32, 1), the same float code
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
void long_tune(int G, int min_ch) { e55::hca_attn_long_tune(G, min_ch); }
void long2_tune(int min_ch, int cap) { e55::hca_attn_long2_tune(min_ch, cap); }
void q_prep(torch::Tensor q, torch::Tensor meta, torch::Tensor inv) {
  e55::hca_q_prep(B(q), MT(meta), inv.data_ptr<float>(), int(meta.size(0)), st());
}
void kv_rows(torch::Tensor kv, torch::Tensor meta, torch::Tensor inv, torch::Tensor rows) {
  e55::hca_kv_rows(B(kv), int(kv.size(1)), MT(meta), inv.data_ptr<float>(), int(meta.size(0)), rows.data_ptr<uint8_t>(),
                   st());
}
void compress(torch::Tensor ckv, torch::Tensor ape, torch::Tensor nw, torch::Tensor meta, torch::Tensor inv,
              torch::Tensor ring, torch::Tensor comp, torch::Tensor state) {
  e55::hca_compress(ckv.data_ptr<float>(), ape.data_ptr<float>(), B(nw), MT(meta), inv.data_ptr<float>(),
                    int(meta.size(0)), C(ring, comp, state), st());
}
void attn(torch::Tensor q, torch::Tensor rows, torch::Tensor meta, torch::Tensor ring, torch::Tensor comp,
          torch::Tensor state, torch::Tensor sink, int max_keys, torch::Tensor o, torch::Tensor ws) {
  e55::hca_attn(B(q), rows.data_ptr<uint8_t>(), MT(meta), C(ring, comp, state), sink.data_ptr<float>(),
                int(meta.size(0)), max_keys, B(o), ws.data_ptr<float>(), size_t(ws.numel()) * 4, st());
}
void ring_store(torch::Tensor rows, torch::Tensor meta, torch::Tensor ring, torch::Tensor comp, torch::Tensor state) {
  e55::hca_ring_store(rows.data_ptr<uint8_t>(), MT(meta), int(meta.size(0)), C(ring, comp, state), st());
}
void vha_partial(torch::Tensor o, int D, torch::Tensor U, torch::Tensor zp) {
  e55::vha_partial(B(o), D, U.data_ptr<float>(), int(o.size(0)), zp.data_ptr<float>(), st());
}
void vha_out(torch::Tensor o, torch::Tensor z, torch::Tensor V, torch::Tensor meta, torch::Tensor inv, torch::Tensor q8,
             torch::Tensor sc) {
  e55::hca_vha_out(B(o), z.data_ptr<float>(), V.data_ptr<float>(), MT(meta), inv.data_ptr<float>(), int(meta.size(0)),
                   q8.data_ptr<uint8_t>(), sc.data_ptr<uint8_t>(), st());
}

// ---- P042 fused chain
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

// ---- timing helper: an L2 flush between cold runs (torch.cuda._sleep lets the host queue the timed work ahead)
torch::Tensor g_flush;
void flush() {
  if (!g_flush.defined()) g_flush = torch::zeros({int64_t(256) << 20}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
  g_flush.add_(1);
}
}  // namespace


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("inv_freq", &inv_freq);
  m.def("patch", &patch);
  m.def("long_tune", &long_tune);
  m.def("long2_tune", &long2_tune);
  m.def("q_prep", &q_prep);
  m.def("kv_rows", &kv_rows);
  m.def("compress", &compress);
  m.def("attn", &attn);
  m.def("ring_store", &ring_store);
  m.def("vha_partial", &vha_partial);
  m.def("vha_out", &vha_out);
  m.def("pre", &pre);
  m.def("attn_fused", &attn_fused);
  m.def("post", &post);
  m.def("flush", &flush);
}
