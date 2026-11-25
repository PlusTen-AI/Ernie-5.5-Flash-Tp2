// Test binding for the decode-path fusions (tests/test_fuse.py): P060 f8_fuse_ops (kernels/gemm p030: norms / gates /
// SiTU / sum inside the FP8 GEMM launches) against the separate kernels (kernels/misc), on real checkpoint weights —
// the decode step's sequences of model.cc (HCA / DSA attention projections, MoE shared expert + fc2, dense FFN) run
// base or fused, outputs returned for bitwise comparison, and timed back to back over several layers (cold weights);
// P061 lin_pdl / P062 lin12 (kernels/linear): the router / compressor linears; P063 sparse_wire (kernels/sample P019):
// the step's sampling from both vocab halves' candidates vs from the gathered full rows.
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <algorithm>
#include <memory>

#include "core/log.h"
#include "kernels/gemm/fp8.h"
#include "kernels/linear/linear.h"
#include "kernels/misc/misc.h"
#include "kernels/sample/sample.h"
#include "model/patches.h"

namespace {
using e55::bf16;
std::string g_dir, g_table;
std::unique_ptr<e55::Weights> g_w[2];
std::vector<e55::F8Lin> g_f8;
std::vector<e55::Lin> g_lin;

e55::Weights& W(int rank) {
  TORCH_CHECK(rank == 0 || rank == 1);
  if (!g_w[rank]) g_w[rank] = std::make_unique<e55::Weights>(g_dir, g_table, rank);
  return *g_w[rank];
}
void open(const std::string& model_dir, const std::string& table) { g_dir = model_dir, g_table = table; }
int f8(int rank, const std::vector<std::string>& names, int groups) {
  g_f8.push_back(e55::f8_upload(W(rank), names, groups));
  return int(g_f8.size()) - 1;
}
int lin(int rank, const std::vector<std::string>& names, bool pack12) {
  g_lin.push_back(e55::lin_upload(W(rank), names, pack12));
  return int(g_lin.size()) - 1;
}
std::vector<int64_t> lin_info(int h) {
  const auto& L = g_lin.at(h);
  return {L.n, L.k, L.split, L.pk ? 1 : 0, L.pk_nx};
}
// a bf16 vector (norm weight) of this rank
torch::Tensor vec(int rank, const std::string& name) {
  auto b = W(rank).slice(name);
  auto t = torch::empty({int64_t(b.size() / 2)}, torch::dtype(torch::kBFloat16));
  std::memcpy(t.data_ptr(), b.data(), b.size());
  return t.cuda();
}
void set_patch(const std::string& name, bool on) { e55::patch::set(name, on); }
bool fuse_ok(int M) { return e55::f8_fuse_ok(M); }

std::vector<torch::Tensor> quant(torch::Tensor x, int group) {
  x = x.contiguous();
  const int M = int(x.size(0)), k = int(x.size(1));
  auto q = torch::empty({M, k}, x.options().dtype(torch::kUInt8));
  auto s = torch::empty({M, k / group}, x.options().dtype(torch::kUInt8));
  e55::act_quant(reinterpret_cast<const bf16*>(x.data_ptr()), k, M, k, group, {q.data_ptr<uint8_t>(),
                 s.data_ptr<uint8_t>(), k}, c10::cuda::getCurrentCUDAStream().stream());
  return {q, s};
}

// ---- the decode step's buffers (M ≤ 8) and sequences ---------------------------------------------------------------
constexpr int kRows = 4096;  // rows the sequences take at most
struct Buf {
  torch::Tensor t1, t2, t2n, t3, t4, t4w, q, o, aout, fout, pb, qf4, qs, xq, xs, ws, lg;
};
Buf g_b;
const Buf& bufs() {
  if (!g_b.t1.defined()) {
    auto bf = torch::dtype(torch::kBFloat16).device(torch::kCUDA), u8 = torch::dtype(torch::kUInt8).device(torch::kCUDA);
    const int R = kRows;
    g_b.t1 = torch::zeros({R, 18432}, bf), g_b.t2 = torch::zeros({R, 8192}, bf), g_b.t2n = torch::zeros({R, 8192}, bf);
    g_b.t3 = torch::zeros({R, 8192}, bf), g_b.t4 = torch::zeros({R, 8192}, bf), g_b.t4w = torch::zeros({R, 4096}, bf);
    g_b.q = torch::zeros({R, 16384}, bf), g_b.o = torch::zeros({R, 16384}, bf), g_b.aout = torch::zeros({R, 4096}, bf);
    g_b.fout = torch::zeros({R, 4096}, bf), g_b.pb = torch::zeros({R, 6144}, bf);
    g_b.qf4 = torch::zeros({R, 2048}, u8), g_b.qs = torch::zeros({R, 64}, u8);
    g_b.xq = torch::zeros({R, 16384}, u8), g_b.xs = torch::zeros({R, 128}, u8);
    g_b.ws = torch::zeros({int64_t(64) << 20}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    g_b.lg = torch::zeros({R, 1024}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  }
  return g_b;
}
template <typename T = bf16>
T* P(const torch::Tensor& t) { return reinterpret_cast<T*>(t.data_ptr()); }

// P060 activation op on x for a GEMM over k inputs (quantized into the act buffers when M > 8)
e55::F8Act pro(int op, const bf16* x, int ldx, int k, const bf16* pw = nullptr, int pld = 0) {
  const Buf& b = bufs();
  return e55::f8_pro(op, x, ldx, pw, pld, P<uint8_t>(b.xq), P<uint8_t>(b.xs), k);
}
// as model.cc's Engine::act
e55::F8Act act(const bf16* x, int ldx, int M, int k, cudaStream_t s) {
  const Buf& b = bufs();
  e55::F8Act a{P<uint8_t>(b.xq), P<uint8_t>(b.xs), k};
  if (e55::f8_defer_act(M)) {
    a.x = x, a.ldx = ldx;
    return a;
  }
  e55::act_quant(x, ldx, M, k, 128, a, s);
  return a;
}

// HCA: qkv → q / kv norms → q_up + gate (+ σ) … wo_a → z ⊙ gate → o_proj. h: qkv, qup, gate, woa, oproj
void seq_hca(const std::vector<int>& h, const torch::Tensor& qn, const torch::Tensor& kvn, const torch::Tensor& inp,
             const torch::Tensor& oq, const torch::Tensor& os, int M, bool fused, cudaStream_t s) {
  const Buf& b = bufs();
  float* ws = b.ws.data_ptr<float>();
  const auto &qkv = g_f8.at(h[0]), &qup = g_f8.at(h[1]), &gate = g_f8.at(h[2]), &woa = g_f8.at(h[3]),
             &oproj = g_f8.at(h[4]);
  e55::f8_forward(qkv, act(P(inp), 4096, M, 4096, s), M, P(b.t1), 1536, ws, s);
  if (fused) {
    e55::F8Aux kv{P(b.t1) + 1024, 1536, P(kvn), 512, P(b.t4), 512};
    e55::f8_forward_multi({{&qup, P(b.q), 16384}, {&gate, P(b.t3), 4096, e55::kEpiSigmoid}},
                          pro(e55::kProRms, P(b.t1), 1536, 1024, P(qn)), M, ws, s, &kv);
  } else {
    e55::rmsnorm(P(b.t1), 1536, P(qn), M, 1024, P(b.t2), 1024, s);
    e55::rmsnorm(P(b.t1) + 1024, 1536, P(kvn), M, 512, P(b.t4), 512, s);
    e55::f8_forward_multi({{&qup, P(b.q), 16384}, {&gate, P(b.t3), 4096}}, act(P(b.t2), 1024, M, 1024, s), M, ws, s);
    e55::sigmoid_inplace(P(b.t3), size_t(M) * 4096, s);
  }
  e55::F8Act ga{P<uint8_t>(oq), P<uint8_t>(os), 4096};
  e55::f8_forward(woa, ga, M, P(b.t4w), 4096, ws, s, long(M) * 4096, long(M) * 32);
  if (fused) {
    e55::f8_forward_out({&oproj, P(b.aout), 4096}, pro(e55::kProMul, P(b.t4w), 4096, 4096, P(b.t3), 4096), M, ws, s);
  } else {
    e55::mul_bf16(P(b.t4w), 4096, P(b.t3), 4096, M, 4096, P(b.t2), 4096, s);
    e55::f8_forward(oproj, act(P(b.t2), 4096, M, 4096, s), M, P(b.aout), 4096, ws, s);
  }
}
// DSA: qkv → q_a / kv_a norms → q_b + gate (+ σ) + indexer q. h: qkv, qb, gate, iq
void seq_dsa(const std::vector<int>& h, const torch::Tensor& qan, const torch::Tensor& kvan, const torch::Tensor& inp,
             int M, bool fused, cudaStream_t s) {
  const Buf& b = bufs();
  float* ws = b.ws.data_ptr<float>();
  const auto &qkv = g_f8.at(h[0]), &qb = g_f8.at(h[1]), &gate = g_f8.at(h[2]), &iq = g_f8.at(h[3]);
  e55::f8_forward(qkv, act(P(inp), 4096, M, 4096, s), M, P(b.t1), 1600, ws, s);
  if (fused) {
    e55::F8Aux kv{P(b.t1) + 1024, 1600, P(kvan), 512, P(b.t4), 512};
    e55::f8_forward_multi({{&qb, P(b.q), 8192}, {&gate, P(b.t3), 8192, e55::kEpiSigmoid}, {&iq, P(b.o), 8192}},
                          pro(e55::kProRms, P(b.t1), 1600, 1024, P(qan)), M, ws, s, &kv);
  } else {
    e55::rmsnorm(P(b.t1), 1600, P(qan), M, 1024, P(b.t2), 1024, s);
    e55::rmsnorm(P(b.t1) + 1024, 1600, P(kvan), M, 512, P(b.t4), 512, s);
    e55::f8_forward_multi({{&qb, P(b.q), 8192}, {&gate, P(b.t3), 8192}, {&iq, P(b.o), 8192}},
                          act(P(b.t2), 1024, M, 1024, s), M, ws, s);
    e55::sigmoid_inplace(P(b.t3), size_t(M) * 8192, s);
  }
}
// MoE without the experts: fc1 + shared gate/up (→ the latent's MXFP8 copy) · SiTU → shared down · latent_norm → fc2
// (+ shared). pb's latent half [:, :2048] is the caller's (stands for the all-reduced experts). h: fc1, gu, down, fc2
void seq_moe(const std::vector<int>& h, const torch::Tensor& lnorm, const torch::Tensor& inp, int M, bool fused,
             cudaStream_t s) {
  const Buf& b = bufs();
  float* ws = b.ws.data_ptr<float>();
  const auto &fc1 = g_f8.at(h[0]), &gu = g_f8.at(h[1]), &down = g_f8.at(h[2]), &fc2 = g_f8.at(h[3]);
  e55::F8Act xa = act(P(inp), 4096, M, 4096, s);
  bf16* pb = P(b.pb);
  if (fused) {
    e55::f8_forward_multi({{&fc1, P(b.t2), 2048, e55::kEpiQ32, nullptr, 0, P<uint8_t>(b.qf4), P<uint8_t>(b.qs)},
                           {&gu, P(b.t1), 2048}}, xa, M, ws, s);
    e55::f8_forward_out({&down, pb + 2048, 6144}, pro(e55::kProSitu, P(b.t1), 2048, 1024), M, ws, s);
    e55::f8_forward_out({&fc2, P(b.fout), 4096, e55::kEpiAdd, pb + 2048, 6144},
                        pro(e55::kProRms, pb, 6144, 2048, P(lnorm)), M, ws, s);
  } else {
    e55::f8_forward_multi({{&fc1, P(b.t2), 2048}, {&gu, P(b.t1), 2048}}, xa, M, ws, s);
    e55::act_quant(P(b.t2), 2048, M, 2048, 32, e55::F8Act{P<uint8_t>(b.qf4), P<uint8_t>(b.qs), 2048}, s);
    e55::situ_bf16(P(b.t1), M, 1024, P(b.t3), s);
    e55::f8_forward(down, act(P(b.t3), 1024, M, 1024, s), M, pb + 2048, 6144, ws, s);
    e55::rmsnorm(pb, 6144, P(lnorm), M, 2048, P(b.t2n), 2048, s);
    e55::f8_forward(fc2, act(P(b.t2n), 2048, M, 2048, s), M, P(b.t4), 4096, ws, s);
    e55::add_bf16(P(b.t4), pb + 2048, 6144, M, 4096, P(b.fout), s);
  }
}
// dense FFN (layer 0): gate/up → SiTU → down. h: gu, down
void seq_dense(const std::vector<int>& h, const torch::Tensor& inp, int M, bool fused, cudaStream_t s) {
  const Buf& b = bufs();
  float* ws = b.ws.data_ptr<float>();
  const auto &gu = g_f8.at(h[0]), &down = g_f8.at(h[1]);
  static torch::Tensor big;  // [8, 14336]
  if (!big.defined()) big = torch::zeros({8, 14336}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
  TORCH_CHECK(M <= 8, "dense: M ≤ 8");
  e55::f8_forward(gu, act(P(inp), 4096, M, 4096, s), M, P(big), 14336, ws, s);
  if (fused) {
    e55::f8_forward_out({&down, P(b.fout), 4096}, pro(e55::kProSitu, P(big), 14336, 7168), M, ws, s);
  } else {
    e55::situ_bf16(P(big), M, 7168, P(b.q), s);
    e55::f8_forward(down, act(P(b.q), 7168, M, 7168, s), M, P(b.fout), 4096, ws, s);
  }
}

// one fused op's own cost: the GEMM with the op (on) vs the same GEMM on a plain deferred activation (off) — the
// misc kernel it replaces not counted. HCA h: qkv, qup, gate, woa, oproj; MoE h: fc1, gu, down, fc2
void seq_op(const std::string& op, const std::vector<int>& h, const torch::Tensor& w, int M, bool on, cudaStream_t s) {
  const Buf& b = bufs();
  float* ws = b.ws.data_ptr<float>();
  bf16* pb = P(b.pb);
  if (op == "rmsq") {  // q_up + gate: q norm prologue + kv norm aux + σ epilogue
    const auto &qup = g_f8.at(h[1]), &gate = g_f8.at(h[2]);
    if (on) {
      e55::F8Aux kv{P(b.t1) + 1024, 1536, P(w), 512, P(b.t4), 512};
      e55::f8_forward_multi({{&qup, P(b.q), 16384}, {&gate, P(b.t3), 4096, e55::kEpiSigmoid}},
                            pro(e55::kProRms, P(b.t1), 1536, 1024, P(w)), M, ws, s, &kv);
    } else {
      e55::f8_forward_multi({{&qup, P(b.q), 16384}, {&gate, P(b.t3), 4096}}, act(P(b.t2), 1024, M, 1024, s), M, ws, s);
    }
  } else if (op == "mul") {
    const auto& oproj = g_f8.at(h[4]);
    e55::f8_forward_out({&oproj, P(b.aout), 4096},
                        on ? pro(e55::kProMul, P(b.t4w), 4096, 4096, P(b.t3), 4096) : act(P(b.t2), 4096, M, 4096, s), M, ws, s);
  } else if (op == "q32") {
    const auto &fc1 = g_f8.at(h[0]), &gu = g_f8.at(h[1]);
    e55::F8Out o1{&fc1, P(b.t2), 2048};
    if (on) o1.epi = e55::kEpiQ32, o1.q = P<uint8_t>(b.qf4), o1.qs = P<uint8_t>(b.qs);
    e55::f8_forward_multi({o1, {&gu, P(b.t1), 2048}}, act(P(b.t4w), 4096, M, 4096, s), M, ws, s);
  } else if (op == "situ") {
    e55::f8_forward_out({&g_f8.at(h[2]), pb + 2048, 6144},
                        on ? pro(e55::kProSitu, P(b.t1), 2048, 1024) : act(P(b.t3), 1024, M, 1024, s), M, ws, s);
  } else if (op == "rms") {  // fc2: latent norm prologue + add epilogue
    if (on)
      e55::f8_forward_out({&g_f8.at(h[3]), P(b.fout), 4096, e55::kEpiAdd, pb + 2048, 6144},
                          pro(e55::kProRms, pb, 6144, 2048, P(w)), M, ws, s);
    else
      e55::f8_forward(g_f8.at(h[3]), act(P(b.t2n), 2048, M, 2048, s), M, P(b.t4), 4096, ws, s);
  }
}
// one sequence (kind: hca / dsa / moe / dense) for each layer group, then the buffers that hold its results
void run_kind(const std::string& kind, const std::vector<std::vector<int>>& hs, const std::vector<torch::Tensor>& v,
              const torch::Tensor& inp, const torch::Tensor& oq, const torch::Tensor& os, int M, bool fused,
              cudaStream_t s) {
  for (size_t i = 0; i < hs.size(); i++) {
    if (kind == "hca") seq_hca(hs[i], v.at(2 * i), v.at(2 * i + 1), inp, oq, os, M, fused, s);
    else if (kind == "dsa") seq_dsa(hs[i], v.at(2 * i), v.at(2 * i + 1), inp, M, fused, s);
    else if (kind == "moe") seq_moe(hs[i], v.at(i), inp, M, fused, s);
    else if (kind == "dense") seq_dense(hs[i], inp, M, fused, s);
    else if (kind.rfind("op:", 0) == 0) seq_op(kind.substr(3), hs[i], v.at(i), M, fused, s);
    else TORCH_CHECK(false, "kind");
  }
}
std::vector<torch::Tensor> run(const std::string& kind, const std::vector<int>& h, const std::vector<torch::Tensor>& v,
                               torch::Tensor inp, torch::Tensor oq, torch::Tensor os, torch::Tensor pb_lat, int M,
                               bool fused) {
  auto s = c10::cuda::getCurrentCUDAStream().stream();
  const Buf& b = bufs();
  for (auto* t : {&g_b.t1, &g_b.t2, &g_b.t2n, &g_b.t3, &g_b.t4, &g_b.t4w, &g_b.q, &g_b.o, &g_b.aout, &g_b.fout,
                  &g_b.qf4, &g_b.qs})
    t->zero_();
  g_b.pb.zero_();
  if (pb_lat.numel()) g_b.pb.narrow(1, 0, 2048).narrow(0, 0, M).copy_(pb_lat);
  run_kind(kind, {h}, v, inp, oq, os, M, fused, s);
  auto r = [&](const torch::Tensor& t, int n) { return t.narrow(0, 0, M).narrow(1, 0, n).clone(); };
  if (kind == "hca") return {r(b.t1, 1536), r(b.q, 16384), r(b.t3, 4096), r(b.t4, 512), r(b.t4w, 4096), r(b.aout, 4096)};
  if (kind == "dsa") return {r(b.t1, 1600), r(b.q, 8192), r(b.t3, 8192), r(b.o, 8192), r(b.t4, 512)};
  if (kind == "moe") return {r(b.t2, 2048), r(b.qf4, 2048), r(b.qs, 64), r(b.t1, 2048), r(b.pb, 6144), r(b.fout, 4096)};
  return {r(b.fout, 4096)};
}
// ms per run (median taken by the caller): each run = the sequence over every layer group, back to back
std::vector<double> bench(const std::string& kind, const std::vector<std::vector<int>>& hs,
                          const std::vector<torch::Tensor>& v, torch::Tensor inp, torch::Tensor oq, torch::Tensor os,
                          int M, bool fused, int runs) {
  auto s = c10::cuda::getCurrentCUDAStream().stream();
  run_kind(kind, hs, v, inp, oq, os, M, fused, s);  // warm-up
  std::vector<cudaEvent_t> ev(runs + 1);
  for (auto& e : ev) C10_CUDA_CHECK(cudaEventCreate(&e));
  C10_CUDA_CHECK(cudaStreamSynchronize(s));
  C10_CUDA_CHECK(cudaEventRecord(ev[0], s));
  for (int i = 0; i < runs; i++) {
    run_kind(kind, hs, v, inp, oq, os, M, fused, s);
    C10_CUDA_CHECK(cudaEventRecord(ev[i + 1], s));
  }
  C10_CUDA_CHECK(cudaEventSynchronize(ev[runs]));
  std::vector<double> ms(runs);
  for (int i = 0; i < runs; i++) {
    float f;
    C10_CUDA_CHECK(cudaEventElapsedTime(&f, ev[i], ev[i + 1]));
    ms[i] = f;
  }
  for (auto& e : ev) cudaEventDestroy(e);
  return ms;
}

// ---- linears (router / compressor) ---------------------------------------------------------------------------------
torch::Tensor lin_fwd(int h, torch::Tensor x) {
  const auto& L = g_lin.at(h);
  const int M = int(x.size(0));
  auto y = torch::empty({M, L.n}, x.options().dtype(torch::kFloat32));
  e55::lin_forward_f32(L, P(x), y.data_ptr<float>(), M, c10::cuda::getCurrentCUDAStream().stream());
  return y;
}
// ms per run: each run = for every handle, [an FP8 GEMM h8 (the kernel before, e.g. mhc / q_up), then] the linear —
// back to back over all handles (distinct weights: cold)
std::vector<double> bench_lin(const std::vector<int>& hs, const std::vector<int>& before, torch::Tensor x, int runs) {
  auto s = c10::cuda::getCurrentCUDAStream().stream();
  const Buf& b = bufs();
  const int M = int(x.size(0));
  int nmax = 0;
  for (int h : hs) nmax = std::max(nmax, g_lin.at(h).n);
  auto y = torch::empty({M, nmax}, x.options().dtype(torch::kFloat32));
  auto go = [&] {
    for (size_t i = 0; i < hs.size(); i++) {
      if (i < before.size() && before[i] >= 0)
        e55::f8_forward(g_f8.at(before[i]), act(P(x), 4096, M, g_f8.at(before[i]).k, s), M, P(b.q), 16384,
                        b.ws.data_ptr<float>(), s);
      e55::lin_forward_f32(g_lin.at(hs[i]), P(x), y.data_ptr<float>(), M, s);
    }
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
    ms[i] = f;
  }
  for (auto& e : ev) cudaEventDestroy(e);
  return ms;
}
// ---- sampling: full rows vs both halves' candidates (P019 kernels, as the engine's two ranks would run them) --------
struct SampP {
  torch::Tensor temp, top_k, top_p, min_p, seed, ctr;
};
// lg [R, V] fp32 (V = 2·Vh): (token, prob) of sample_rows on the full rows, then of sample_rows_sparse on the records
// of the halves [0, Vh) and [Vh, V) (sample_cands each, as rank 0 / rank 1), [2][R][kSparseRec]
std::vector<torch::Tensor> samp_cmp(torch::Tensor lg, torch::Tensor temp, torch::Tensor top_k, torch::Tensor top_p,
                                    torch::Tensor min_p, torch::Tensor seed, torch::Tensor ctr) {
  auto s = c10::cuda::getCurrentCUDAStream().stream();
  const int R = int(lg.size(0)), V = int(lg.size(1)), Vh = V / 2;
  auto i32 = lg.options().dtype(torch::kInt32);
  auto tf = torch::empty({R}, i32), ts = torch::empty({R}, i32), pf = torch::empty({R}, lg.options()),
       ps = torch::empty({R}, lg.options());
  e55::sample_rows(lg.data_ptr<float>(), R, V, temp.data_ptr<float>(), top_k.data_ptr<int>(), top_p.data_ptr<float>(),
                   min_p.data_ptr<float>(), reinterpret_cast<uint64_t*>(seed.data_ptr()),
                   reinterpret_cast<uint64_t*>(ctr.data_ptr()), tf.data_ptr<int>(), pf.data_ptr<float>(), s);
  auto h0 = lg.narrow(1, 0, Vh).contiguous(), h1 = lg.narrow(1, Vh, Vh).contiguous();
  auto c = torch::empty({2, R, e55::kSparseRec, 2}, i32);
  auto* cp = reinterpret_cast<e55::SparseCand*>(c.data_ptr());
  e55::sample_cands(h0.data_ptr<float>(), R, Vh, 0, cp, s);
  e55::sample_cands(h1.data_ptr<float>(), R, Vh, Vh, cp + size_t(R) * e55::kSparseRec, s);
  e55::sample_rows_sparse(cp, R, R, V, temp.data_ptr<float>(), top_k.data_ptr<int>(), top_p.data_ptr<float>(),
                          min_p.data_ptr<float>(), reinterpret_cast<uint64_t*>(seed.data_ptr()),
                          reinterpret_cast<uint64_t*>(ctr.data_ptr()), ts.data_ptr<int>(), ps.data_ptr<float>(), s);
  return {tf, pf, ts, ps};
}
// ms per call of this rank's work after multimax: full = halves_to_rows of the gathered [2][R][Vh] + sample_rows;
// sparse = sample_cands of its half + sample_rows_sparse of the gathered records (the all-gathers are not included)
std::vector<double> samp_bench(torch::Tensor lg, torch::Tensor temp, torch::Tensor top_k, torch::Tensor top_p,
                               torch::Tensor min_p, torch::Tensor seed, torch::Tensor ctr, bool sparse, int runs) {
  auto s = c10::cuda::getCurrentCUDAStream().stream();
  const int R = int(lg.size(0)), V = int(lg.size(1)), Vh = V / 2;
  auto i32 = lg.options().dtype(torch::kInt32);
  auto tok = torch::empty({R}, i32), prob = torch::empty({R}, lg.options()), full = torch::empty_like(lg);
  auto halves = torch::cat({lg.narrow(1, 0, Vh).contiguous().view({-1}), lg.narrow(1, Vh, Vh).contiguous().view({-1})});
  auto c = torch::empty({2, R, e55::kSparseRec, 2}, i32);
  auto* cp = reinterpret_cast<e55::SparseCand*>(c.data_ptr());
  auto go = [&] {
    if (sparse) {
      e55::sample_cands(halves.data_ptr<float>(), R, Vh, 0, cp, s);
      e55::sample_rows_sparse(cp, R, R, V, temp.data_ptr<float>(), top_k.data_ptr<int>(), top_p.data_ptr<float>(),
                              min_p.data_ptr<float>(), reinterpret_cast<uint64_t*>(seed.data_ptr()),
                              reinterpret_cast<uint64_t*>(ctr.data_ptr()), tok.data_ptr<int>(), prob.data_ptr<float>(), s);
    } else {
      e55::halves_to_rows(halves.data_ptr<float>(), R, Vh, full.data_ptr<float>(), s);
      e55::sample_rows(full.data_ptr<float>(), R, V, temp.data_ptr<float>(), top_k.data_ptr<int>(),
                       top_p.data_ptr<float>(), min_p.data_ptr<float>(), reinterpret_cast<uint64_t*>(seed.data_ptr()),
                       reinterpret_cast<uint64_t*>(ctr.data_ptr()), tok.data_ptr<int>(), prob.data_ptr<float>(), s);
    }
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
    ms[i] = f;
  }
  for (auto& e : ev) cudaEventDestroy(e);
  return ms;
}
bool sparse_ok(float temp, int top_k, float top_p) { return e55::sparse_ok(temp, top_k, top_p, e55::kVocab); }
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("open", &open);
  m.def("f8", &f8);
  m.def("lin", &lin);
  m.def("lin_info", &lin_info);
  m.def("vec", &vec);
  m.def("set_patch", &set_patch);
  m.def("fuse_ok", &fuse_ok);
  m.def("quant", &quant);
  m.def("run", &run);
  m.def("bench", &bench);
  m.def("lin_fwd", &lin_fwd);
  m.def("bench_lin", &bench_lin);
  m.def("samp_cmp", &samp_cmp);
  m.def("samp_bench", &samp_bench);
  m.def("sparse_ok", &sparse_ok);
  m.def("test_pfn", &e55::f8_test_pfn);
}
