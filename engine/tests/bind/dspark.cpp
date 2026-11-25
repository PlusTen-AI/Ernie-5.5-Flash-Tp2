// Test binding for the DSpark draft's own attention (engine/src/kernels/dspark/dspark.cu): the base chain of a draft
// stage's attention (hca_q_prep + hca_kv_rows + dsp_attn + vha_partial) and P129's fused launch on caller-owned tensors.
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <map>

#include "kernels/attn/hca.h"
#include "kernels/dspark/dspark.h"
#include "model/patches.h"

namespace {
using e55::bf16;
cudaStream_t st() { return c10::cuda::getCurrentCUDAStream().stream(); }
bf16* B(torch::Tensor t) { return reinterpret_cast<bf16*>(t.data_ptr()); }
const e55::RowMeta* MT(torch::Tensor m) {
  TORCH_CHECK(m.dtype() == torch::kInt32 && m.size(1) == 5);
  return reinterpret_cast<const e55::RowMeta*>(m.data_ptr<int>());
}
// context rows of the 3 stages: kv [K, 1536] bf16 at positions pos [K] of slots slot [K] → ring [3][slots][256][584]
void ctx_rows(torch::Tensor kv, torch::Tensor pos, torch::Tensor slot, torch::Tensor inv, torch::Tensor ring, int slots) {
  e55::dsp_ctx_rows(B(kv), int(kv.size(1)), pos.data_ptr<int>(), slot.data_ptr<int>(), int(kv.size(0)),
                    inv.data_ptr<float>(), ring.data_ptr<uint8_t>(), slots, st());
}
// the base chain: q (raw, prepped in place), kv [M, 512] → rows, o, zp
void base(torch::Tensor q, torch::Tensor kv, torch::Tensor meta, torch::Tensor ring, torch::Tensor sink,
          torch::Tensor inv, torch::Tensor U, torch::Tensor rows, torch::Tensor o, torch::Tensor zp) {
  const int M = int(meta.size(0));
  e55::hca_q_prep(B(q), MT(meta), inv.data_ptr<float>(), M, st());
  e55::hca_kv_rows(B(kv), int(kv.size(1)), MT(meta), inv.data_ptr<float>(), M, rows.data_ptr<uint8_t>(), st());
  e55::dsp_attn(B(q), rows.data_ptr<uint8_t>(), MT(meta), ring.data_ptr<uint8_t>(), sink.data_ptr<float>(), M, B(o), st());
  e55::vha_partial(B(o), e55::kHeadDim, U.data_ptr<float>(), M, zp.data_ptr<float>(), st());
}
// one stage of the base chain by name (timing)
void base_one(int which, torch::Tensor q, torch::Tensor kv, torch::Tensor meta, torch::Tensor ring, torch::Tensor sink,
              torch::Tensor inv, torch::Tensor U, torch::Tensor rows, torch::Tensor o, torch::Tensor zp) {
  const int M = int(meta.size(0));
  if (which == 0) e55::hca_q_prep(B(q), MT(meta), inv.data_ptr<float>(), M, st());
  if (which == 1)
    e55::hca_kv_rows(B(kv), int(kv.size(1)), MT(meta), inv.data_ptr<float>(), M, rows.data_ptr<uint8_t>(), st());
  if (which == 2)
    e55::dsp_attn(B(q), rows.data_ptr<uint8_t>(), MT(meta), ring.data_ptr<uint8_t>(), sink.data_ptr<float>(), M, B(o),
                  st());
  if (which == 3) e55::vha_partial(B(o), e55::kHeadDim, U.data_ptr<float>(), M, zp.data_ptr<float>(), st());
}
// P129: q raw (not written), kv [M, 512] → o, zp
void fused(torch::Tensor q, torch::Tensor kv, torch::Tensor meta, torch::Tensor ring, torch::Tensor sink,
           torch::Tensor inv, torch::Tensor U, torch::Tensor o, torch::Tensor zp) {
  e55::dsp_attn_fused(B(q), B(kv), int(kv.size(1)), MT(meta), ring.data_ptr<uint8_t>(), sink.data_ptr<float>(),
                      inv.data_ptr<float>(), U.data_ptr<float>(), int(meta.size(0)), B(o), zp.data_ptr<float>(), st());
}
// the greedy Markov chain of one sequence over N steps (no exchange: one GPU — the other half's candidates stay zero):
// base = dsp_markov + dsp_argmax + dsp_pick a step; chain = P130's dsp_chain a step + dsp_pick after the last.
// base [N, Vh] fp32, w1t [V, 256] bf16, w2q [Vh, 256] e4m3, w2s [Vh, 8], mm [8]; ws: int32 scratch [≥ 4096]; returns
// the drafts [N] (in drafts [16 × 16] row 0)
void chain(bool fused, torch::Tensor base, torch::Tensor w1t, torch::Tensor w2q, torch::Tensor w2s, torch::Tensor mm,
           int last, int N, torch::Tensor ws, torch::Tensor part, torch::Tensor drafts) {
  const int Vh = int(base.size(1));
  int* w = ws.data_ptr<int>();
  int *seq = w, *brow = w + 16, *prev = w + 32, *lastd = w + 48;
  unsigned* cnt = reinterpret_cast<unsigned*>(w + 64);
  float* ex = reinterpret_cast<float*>(w + 128);  // [16][64]
  std::vector<int> h(64, 0);
  for (int i = 0; i < 16; i++) h[16 + i] = i;
  h[32] = last, h[48] = last;
  cudaMemcpyAsync(w, h.data(), 64 * 4, cudaMemcpyHostToDevice, st());
  float2* pt = reinterpret_cast<float2*>(part.data_ptr<float>());
  int* dr = drafts.data_ptr<int>();
  for (int i = 0; i < N; i++) {
    if (fused) {
      e55::dsp_chain(base.data_ptr<float>(), brow + i, seq, 1, i ? seq : nullptr, i ? 1 : 0, i ? ex + (i - 1) * 64 : nullptr,
                     lastd, dr, i, B(w1t), w2q.data_ptr<uint8_t>(), w2s.data_ptr<uint8_t>(), Vh, 0,
                     mm.data_ptr<float>(), pt, cnt + i, 0, ex + i * 64, st());
      if (i + 1 == N) e55::dsp_pick(ex + i * 64, 1, seq, i, prev, dr, st());
    } else {
      e55::dsp_markov(base.data_ptr<float>(), brow + i, seq, prev, B(w1t), nullptr, w2q.data_ptr<uint8_t>(),
                      w2s.data_ptr<uint8_t>(), Vh, 0, mm.data_ptr<float>(), 1, nullptr, seq, pt, st());
      e55::dsp_argmax(pt, e55::dsp_markov_blocks(Vh), 1, 0, ex, st());
      e55::dsp_pick(ex, 1, seq, i, prev, dr, st());
    }
  }
}
// one sampled Markov sweep launch (P161 / P163: CTAs g0 … g0 + ng − 1, back): items' base rows brow, sequences seq,
// previous tokens prev (int32 [16] etc.); out [K, Vh] fp32 (rows orow), part [K][nb] (float2 as fp32 pairs)
void markov(torch::Tensor base, torch::Tensor brow, torch::Tensor seq, torch::Tensor prev, torch::Tensor w1t,
            torch::Tensor w2q, torch::Tensor w2s, torch::Tensor mm, int K, torch::Tensor out, torch::Tensor orow,
            torch::Tensor part, int g0, int ng, bool back) {
  e55::dsp_markov(base.data_ptr<float>(), brow.data_ptr<int>(), seq.data_ptr<int>(), prev.data_ptr<int>(), B(w1t),
                  nullptr, w2q.data_ptr<uint8_t>(), w2s.data_ptr<uint8_t>(), int(base.size(1)), 0, mm.data_ptr<float>(),
                  K, out.data_ptr<float>(), orow.data_ptr<int>(), reinterpret_cast<float2*>(part.data_ptr<float>()),
                  st(), g0, ng, back);
}
int markov_blocks(int Vh) { return e55::dsp_markov_blocks(Vh); }
void patch(const std::string& n, bool on) { e55::patch::set(n, on); }
std::map<std::string, long> props() {
  std::map<std::string, long> m;
  int v = 0;
  for (auto [k, a] : {std::pair<const char*, cudaDeviceAttr>{"l2", cudaDevAttrL2CacheSize},
                      {"persist_max", cudaDevAttrMaxPersistingL2CacheSize},
                      {"window_max", cudaDevAttrMaxAccessPolicyWindowSize}, {"clock_khz", cudaDevAttrClockRate},
                      {"sms", cudaDevAttrMultiProcessorCount}, {"smem_sm", cudaDevAttrMaxSharedMemoryPerMultiprocessor}}) {
    cudaDeviceGetAttribute(&v, a, 0);
    m[k] = v;
  }
  return m;
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ctx_rows", &ctx_rows);
  m.def("base", &base);
  m.def("base_one", &base_one);
  m.def("fused", &fused);
  m.def("chain", &chain);
  m.def("markov", &markov);
  m.def("markov_blocks", &markov_blocks);
  m.def("patch", &patch);
  m.def("props", &props);
}
