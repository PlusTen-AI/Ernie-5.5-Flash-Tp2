// Test binding for the DSA attention family (engine/src/kernels/attn/dsa.cu): every stage of attn_dsa between the
// projections (MLA cache rows, q_pe, the W_UK / W_UV batched GEMMs, indexer prep, top-k selection, sparse MLA attention,
// VHA + gate) on caller-owned tensors, patch switches, and an L2 flush for cold timing.
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cmath>

#include "kernels/attn/dsa.h"
#include "model/patches.h"

void dsa_mma_peak(int ctas, int threads, int chains, int iters, float* ms, long long* cycles, double* mmas);  // dsa_peak.cu

namespace {
using e55::bf16;
cudaStream_t st() { return c10::cuda::getCurrentCUDAStream().stream(); }
bf16* B(torch::Tensor t) { return reinterpret_cast<bf16*>(t.data_ptr()); }
const e55::RowMeta* MT(torch::Tensor m) {
  TORCH_CHECK(m.dtype() == torch::kInt32 && m.size(1) == 5);
  return reinterpret_cast<const e55::RowMeta*>(m.data_ptr<int>());
}
e55::DsaCache C(torch::Tensor mla, torch::Tensor idx) {  // [slots, ctx, 656] / [slots, ctx, 68] uint8
  TORCH_CHECK(mla.size(1) == idx.size(1) && mla.size(2) == e55::kMlaRowBytes && idx.size(2) == e55::kIdxRowBytes);
  e55::DsaCache c;
  c.mla = mla.data_ptr<uint8_t>(), c.idx = idx.data_ptr<uint8_t>(), c.ctx = int(mla.size(1));
  return c;
}

// model.cc inv_plain(8e6), the same float code
torch::Tensor inv8() {
  auto v = torch::empty({32}, torch::kFloat32);
  for (int i = 0; i < 32; i++) v[i] = 1.0f / powf(8000000.f, float(2 * i) / 64.0f);
  return v.cuda();
}

void patch(const std::string& n, bool on) { e55::patch::set(n, on); }

void mla_rows(torch::Tensor kvn, torch::Tensor kpe, torch::Tensor meta, torch::Tensor inv, torch::Tensor mla,
              torch::Tensor idx) {
  e55::dsa_mla_rows(B(kvn), int(kvn.stride(0)), B(kpe), int(kpe.stride(0)), MT(meta), inv.data_ptr<float>(),
                    int(meta.size(0)), C(mla, idx), st());
}
void q_pe(torch::Tensor q, torch::Tensor meta, torch::Tensor inv, torch::Tensor q576) {
  e55::dsa_q_pe(B(q), MT(meta), inv.data_ptr<float>(), int(meta.size(0)), B(q576), st());
}
// Y[m][h][n] = Σ_k X[m][h][k] W[h][n][k]; x / y given with their element offsets and strides
void bmm(torch::Tensor x, int64_t xoff, int64_t xm, int64_t xh, torch::Tensor w, int H, int N, int K, int M,
         torch::Tensor y, int64_t yoff, int64_t ym, int64_t yh) {
  e55::dsa_bmm(B(x) + xoff, xm, xh, B(w), H, N, K, M, B(y) + yoff, ym, yh, st());
}
bool bmm_f8(torch::Tensor x, int64_t xoff, int64_t xm, int64_t xh, torch::Tensor w8, torch::Tensor sc, int mode, int H,
            int N, int K, int M, torch::Tensor y, int64_t yoff, int64_t ym, int64_t yh) {
  return e55::dsa_bmm_f8(B(x) + xoff, xm, xh, w8.data_ptr<uint8_t>(), sc.data_ptr<float>(), mode, H, N, K, M,
                         B(y) + yoff, ym, yh, st());
}
void idx_prep(torch::Tensor kw, torch::Tensor qi, torch::Tensor knw, torch::Tensor knb, torch::Tensor meta,
              torch::Tensor inv, torch::Tensor mla, torch::Tensor idx, torch::Tensor qf4, torch::Tensor qs,
              torch::Tensor wts) {
  e55::dsa_idx_prep(B(kw), B(qi), B(knw), B(knb), MT(meta), inv.data_ptr<float>(), int(meta.size(0)), C(mla, idx),
                    qf4.data_ptr<uint8_t>(), qs.data_ptr<uint8_t>(), wts.data_ptr<float>(), st());
}
// P141 dsa_q_own: the K rows (and weights) only
void idx_prep_k(torch::Tensor kw, torch::Tensor knw, torch::Tensor knb, torch::Tensor meta, torch::Tensor inv,
                torch::Tensor mla, torch::Tensor idx, torch::Tensor wts) {
  e55::dsa_idx_prep(B(kw), nullptr, B(knw), B(knb), MT(meta), inv.data_ptr<float>(), int(meta.size(0)), C(mla, idx),
                    nullptr, nullptr, wts.data_ptr<float>(), st(), false);
}
// P140 dsa_sel_split: list entries 2048 … and counts from the positions
void sel_tail(torch::Tensor meta, torch::Tensor list, torch::Tensor cnt) {
  e55::dsa_select_tail(MT(meta), int(meta.size(0)), list.data_ptr<int>(), cnt.data_ptr<int>(), st());
}
void sel(torch::Tensor qf4, torch::Tensor qs, torch::Tensor wts, torch::Tensor meta, torch::Tensor mla,
            torch::Tensor idx, int max_pos, torch::Tensor list, torch::Tensor cnt, torch::Tensor ws) {
  e55::dsa_select(qf4.data_ptr<uint8_t>(), qs.data_ptr<uint8_t>(), wts.data_ptr<float>(), MT(meta), C(mla, idx),
                  int(meta.size(0)), max_pos, list.data_ptr<int>(), cnt.data_ptr<int>(), ws.data_ptr<float>(),
                  size_t(ws.numel()) * 4, st());
}
void attn(torch::Tensor q576, torch::Tensor list, torch::Tensor cnt, torch::Tensor meta, torch::Tensor mla,
          torch::Tensor idx, torch::Tensor sink, int max_keys, torch::Tensor olat, torch::Tensor ws) {
  e55::dsa_attn(B(q576), list.data_ptr<int>(), cnt.data_ptr<int>(), MT(meta), C(mla, idx), sink.data_ptr<float>(),
                int(meta.size(0)), max_keys, B(olat), ws.data_ptr<float>(), size_t(ws.numel()) * 4, st());
}
void vha_gate(torch::Tensor o, torch::Tensor z, torch::Tensor V, torch::Tensor gate, torch::Tensor y) {
  e55::dsa_vha_gate(B(o), z.data_ptr<float>(), V.data_ptr<float>(), B(gate), int(o.size(0)), B(y), st());
}

// the k64 FP4 MMA's peak (dsa_peak.cu): (ms, cycles of CTA 0, MMAs)
std::tuple<double, int64_t, double> mma_peak(int ctas, int threads, int chains, int iters) {
  float ms = 0.f;
  long long cyc = 0;
  double n = 0;
  dsa_mma_peak(ctas, threads, chains, iters, &ms, &cyc, &n);
  return {double(ms), int64_t(cyc), n};
}

// ---- timing helper: an L2 flush between cold runs
torch::Tensor g_flush;
void flush() {
  if (!g_flush.defined()) g_flush = torch::zeros({int64_t(256) << 20}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
  g_flush.add_(1);
}
}  // namespace


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("inv8", &inv8);
  m.def("patch", &patch);
  m.def("mla_rows", &mla_rows);
  m.def("q_pe", &q_pe);
  m.def("bmm", &bmm);
  m.def("bmm_f8", &bmm_f8);
  m.def("idx_prep", &idx_prep);
  m.def("idx_prep_k", &idx_prep_k);
  m.def("select", &sel);
  m.def("select_cfg", &e55::dsa_select_cfg);
  m.def("sel_tail", &sel_tail);
  m.def("attn", &attn);
  m.def("vha_gate", &vha_gate);
  m.def("flush", &flush);
  m.def("mma_peak", &mma_peak);
}
