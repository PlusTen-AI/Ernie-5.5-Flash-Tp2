// Test binding for the FP8 GEMM family (engine/src/kernels/gemm/fp8.cu): upload from the checkpoint (either rank),
// activation quantization, forward (bf16 / fp32 out, quantized or — P032 — bf16 activation), patch switches, timing
// per call with CUDA events (cold: rotating over weight copies that together exceed L2 several times).
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstring>
#include <memory>

#include "core/log.h"
#include "kernels/gemm/fp8.h"
#include "model/patches.h"

namespace {
std::string g_dir, g_table;
std::unique_ptr<e55::Weights> g_w[2];
std::vector<e55::F8Lin> g_lin;

void open(const std::string& model_dir, const std::string& table) { g_dir = model_dir, g_table = table; }

int upload(int rank, const std::vector<std::string>& names, int groups) {
  TORCH_CHECK(rank == 0 || rank == 1);
  if (!g_w[rank]) g_w[rank] = std::make_unique<e55::Weights>(g_dir, g_table, rank);
  g_lin.push_back(e55::f8_upload(*g_w[rank], names, groups));
  return int(g_lin.size()) - 1;
}

void release(int h) { e55::f8_free(g_lin.at(h)); }

std::vector<int64_t> info(int h) {
  const auto& L = g_lin.at(h);
  return {L.n, L.k, L.npad, L.split, L.groups};
}

void set_patch(const std::string& name, bool on) { e55::patch::set(name, on); }
bool patch_on(const std::string& name) { return e55::patch::on(name); }
bool defer_act(int M) { return e55::f8_defer_act(M); }

// x bf16 [M, k] → (q uint8 [M, k], s uint8 [M, k/group])
std::vector<torch::Tensor> quant(torch::Tensor x, int group) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kBFloat16 && x.dim() == 2);
  x = x.contiguous();
  const int M = int(x.size(0)), k = int(x.size(1));
  auto q = torch::empty({M, k}, x.options().dtype(torch::kUInt8));
  auto s = torch::empty({M, k / group}, x.options().dtype(torch::kUInt8));
  e55::act_quant(reinterpret_cast<const e55::bf16*>(x.data_ptr()), k, M, k, group, {q.data_ptr<uint8_t>(),
                 s.data_ptr<uint8_t>(), k}, c10::cuda::getCurrentCUDAStream().stream());
  return {q, s};
}

void run(const e55::F8Lin& L, const e55::F8Act& a, int M, bool f32, void* y, float* w, cudaStream_t st, long gx,
         long gs) {
  if (f32) e55::f8_forward_f32(L, a, M, static_cast<float*>(y), L.n, w, st, gx, gs);
  else e55::f8_forward(L, a, M, static_cast<e55::bf16*>(y), L.n, w, st, gx, gs);
}

// q [M or G·M, ld], s [·, ld/128] → y [M, n]; or (P032) x bf16 [M, k] given as `x` with q / s as scratch
torch::Tensor forward(int h, torch::Tensor q, torch::Tensor s, int M, bool f32, int64_t gx, int64_t gs,
                      c10::optional<torch::Tensor> x) {
  const auto& L = g_lin.at(h);
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  auto y = torch::empty({M, L.n}, q.options().dtype(f32 ? torch::kFloat32 : torch::kBFloat16));
  torch::Tensor ws;
  if (L.split > 1) ws = torch::empty({int64_t(e55::f8_ws(L, M) / 4)}, q.options().dtype(torch::kFloat32));
  e55::F8Act a{q.data_ptr<uint8_t>(), s.data_ptr<uint8_t>(), int(q.size(1))};
  if (x) a.x = reinterpret_cast<const e55::bf16*>(x->data_ptr()), a.ldx = int(x->size(1));
  run(L, a, M, f32, y.data_ptr(), L.split > 1 ? ws.data_ptr<float>() : nullptr, st, gx, gs);
  return y;
}

// per-call ms (CUDA events around each call — or around `chain` back-to-back calls, divided — queued behind a GPU
// sleep issued by the caller).
// cold: rotate over weight copies (≥ 160 MB in total, L2 is 24 MB) so every call reads its weights from DRAM.
// x given: the call is act_quant + GEMM (deferred = false) or the GEMM on the bf16 activation (deferred = true).
std::vector<double> bench(int h, torch::Tensor q, torch::Tensor s, int M, int runs, bool cold, int64_t gx, int64_t gs,
                          c10::optional<torch::Tensor> x, bool deferred, int chain) {
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  const auto& L0 = g_lin.at(h);
  std::vector<e55::F8Lin> cp{L0};
  const size_t wb = size_t(L0.npad) * L0.k, sb = size_t(L0.npad / 16) * (L0.k / 128);
  if (cold) {
    const int nc = int(std::min<size_t>(64, std::max<size_t>(2, (160u << 20) / wb + 1)));
    for (int i = 1; i < nc; i++) {
      e55::F8Lin c = L0;
      C10_CUDA_CHECK(cudaMalloc(&c.w, wb));
      C10_CUDA_CHECK(cudaMalloc(&c.s, sb));
      C10_CUDA_CHECK(cudaMemcpy(c.w, L0.w, wb, cudaMemcpyDeviceToDevice));
      C10_CUDA_CHECK(cudaMemcpy(c.s, L0.s, sb, cudaMemcpyDeviceToDevice));
      cp.push_back(c);
    }
  }
  auto y = torch::empty({M, L0.n}, q.options().dtype(torch::kBFloat16));
  torch::Tensor ws;
  if (L0.split > 1) ws = torch::empty({int64_t(e55::f8_ws(L0, M) / 4)}, q.options().dtype(torch::kFloat32));
  float* w = L0.split > 1 ? ws.data_ptr<float>() : nullptr;
  e55::F8Act a{q.data_ptr<uint8_t>(), s.data_ptr<uint8_t>(), int(q.size(1))};
  auto call = [&](int i) {
    const auto& L = cp[i % cp.size()];
    e55::F8Act aa = a;
    if (x && deferred) aa.x = reinterpret_cast<const e55::bf16*>(x->data_ptr()), aa.ldx = int(x->size(1));
    else if (x) e55::act_quant(reinterpret_cast<const e55::bf16*>(x->data_ptr()), int(x->size(1)), M, L.k, 128, a, st);
    run(L, aa, M, false, y.data_ptr(), w, st, gx, gs);
  };
  for (int i = 0; i < int(cp.size()); i++) call(i);  // warm-up (also every copy once)
  std::vector<cudaEvent_t> ev(runs + 1);
  for (auto& e : ev) C10_CUDA_CHECK(cudaEventCreate(&e));
  C10_CUDA_CHECK(cudaStreamSynchronize(st));
  C10_CUDA_CHECK(cudaEventRecord(ev[0], st));
  for (int i = 0; i < runs; i++) {
    for (int j = 0; j < chain; j++) call(i * chain + j + 1);
    C10_CUDA_CHECK(cudaEventRecord(ev[i + 1], st));
  }
  C10_CUDA_CHECK(cudaEventSynchronize(ev[runs]));
  std::vector<double> ms(runs);
  for (int i = 0; i < runs; i++) {
    float f;
    C10_CUDA_CHECK(cudaEventElapsedTime(&f, ev[i], ev[i + 1]));
    ms[i] = f / chain;
  }
  for (auto& e : ev) cudaEventDestroy(e);
  for (size_t i = 1; i < cp.size(); i++) cudaFree(cp[i].w), cudaFree(cp[i].s);
  return ms;
}

// P034: weights hs on one activation → [y_i]; q / s quantized (or scratch with x given, P032)
std::vector<torch::Tensor> forward_multi(std::vector<int> hs, torch::Tensor q, torch::Tensor s, int M,
                                         c10::optional<torch::Tensor> x) {
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  std::vector<torch::Tensor> ys;
  size_t wsb = 0;
  for (int h : hs) {
    ys.push_back(torch::empty({M, g_lin.at(h).n}, q.options().dtype(torch::kBFloat16)));
    wsb = std::max(wsb, e55::f8_ws(g_lin.at(h), M));
  }
  auto ws = torch::empty({int64_t(wsb / 4 + 1)}, q.options().dtype(torch::kFloat32));
  e55::F8Act a{q.data_ptr<uint8_t>(), s.data_ptr<uint8_t>(), int(q.size(1))};
  if (x) a.x = reinterpret_cast<const e55::bf16*>(x->data_ptr()), a.ldx = int(x->size(1));
  auto o = [&](size_t i) { return e55::F8Out{&g_lin.at(hs[i]), reinterpret_cast<e55::bf16*>(ys[i].data_ptr()),
                                             int(ys[i].size(1))}; };
  if (hs.size() == 2) e55::f8_forward_multi({o(0), o(1)}, a, M, ws.data_ptr<float>(), st);
  else if (hs.size() == 3) e55::f8_forward_multi({o(0), o(1), o(2)}, a, M, ws.data_ptr<float>(), st);
  else TORCH_CHECK(false, "2 or 3 weights");
  return ys;
}

// per-call ms of f8_forward_multi over cold weight copies (x given: deferred activation, P032 — or with quant_first,
// one act_quant of x into q / s first, as the engine does without P032)
std::vector<double> bench_multi(std::vector<int> hs, torch::Tensor q, torch::Tensor s, int M, int runs,
                                c10::optional<torch::Tensor> x, int chain, bool quant_first) {
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  size_t tot = 0, wsb = 0;
  for (int h : hs) tot += size_t(g_lin.at(h).npad) * g_lin.at(h).k, wsb = std::max(wsb, e55::f8_ws(g_lin.at(h), M));
  const int nc = int(std::min<size_t>(64, std::max<size_t>(2, (160u << 20) / tot + 1)));
  std::vector<std::vector<e55::F8Lin>> cp(nc);
  std::vector<torch::Tensor> ys;
  for (int h : hs) ys.push_back(torch::empty({M, g_lin.at(h).n}, q.options().dtype(torch::kBFloat16)));
  for (int c = 0; c < nc; c++)
    for (int h : hs) {
      e55::F8Lin L = g_lin.at(h);
      if (c) {
        const size_t wb = size_t(L.npad) * L.k, sb = size_t(L.npad / 16) * (L.k / 128);
        C10_CUDA_CHECK(cudaMalloc(&L.w, wb));
        C10_CUDA_CHECK(cudaMalloc(&L.s, sb));
        C10_CUDA_CHECK(cudaMemcpy(L.w, g_lin.at(h).w, wb, cudaMemcpyDeviceToDevice));
        C10_CUDA_CHECK(cudaMemcpy(L.s, g_lin.at(h).s, sb, cudaMemcpyDeviceToDevice));
      }
      cp[c].push_back(L);
    }
  auto ws = torch::empty({int64_t(wsb / 4 + 1)}, q.options().dtype(torch::kFloat32));
  e55::F8Act a{q.data_ptr<uint8_t>(), s.data_ptr<uint8_t>(), int(q.size(1))};
  if (x && !quant_first) a.x = reinterpret_cast<const e55::bf16*>(x->data_ptr()), a.ldx = int(x->size(1));
  auto call = [&](int i) {
    auto& L = cp[i % nc];
    if (x && quant_first)
      e55::act_quant(reinterpret_cast<const e55::bf16*>(x->data_ptr()), int(x->size(1)), M, L[0].k, 128, a, st);
    auto o = [&](size_t j) { return e55::F8Out{&L[j], reinterpret_cast<e55::bf16*>(ys[j].data_ptr()),
                                               int(ys[j].size(1))}; };
    if (L.size() == 2) e55::f8_forward_multi({o(0), o(1)}, a, M, ws.data_ptr<float>(), st);
    else e55::f8_forward_multi({o(0), o(1), o(2)}, a, M, ws.data_ptr<float>(), st);
  };
  for (int i = 0; i < nc; i++) call(i);
  std::vector<cudaEvent_t> ev(runs + 1);
  for (auto& e : ev) C10_CUDA_CHECK(cudaEventCreate(&e));
  C10_CUDA_CHECK(cudaStreamSynchronize(st));
  C10_CUDA_CHECK(cudaEventRecord(ev[0], st));
  for (int i = 0; i < runs; i++) {
    for (int j = 0; j < chain; j++) call(i * chain + j + 1);
    C10_CUDA_CHECK(cudaEventRecord(ev[i + 1], st));
  }
  C10_CUDA_CHECK(cudaEventSynchronize(ev[runs]));
  std::vector<double> ms(runs);
  for (int i = 0; i < runs; i++) {
    float f;
    C10_CUDA_CHECK(cudaEventElapsedTime(&f, ev[i], ev[i + 1]));
    ms[i] = f / chain;
  }
  for (auto& e : ev) cudaEventDestroy(e);
  for (int c = 1; c < nc; c++)
    for (auto& L : cp[c]) cudaFree(L.w), cudaFree(L.s);
  return ms;
}

// act_quant alone, per-call ms
std::vector<double> bench_act(torch::Tensor x, int group, int runs) {
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  const int M = int(x.size(0)), k = int(x.size(1));
  auto q = torch::empty({M, k}, x.options().dtype(torch::kUInt8));
  auto s = torch::empty({M, k / group}, x.options().dtype(torch::kUInt8));
  e55::F8Act a{q.data_ptr<uint8_t>(), s.data_ptr<uint8_t>(), k};
  auto xp = reinterpret_cast<const e55::bf16*>(x.data_ptr());
  e55::act_quant(xp, k, M, k, group, a, st);
  std::vector<cudaEvent_t> ev(runs + 1);
  for (auto& e : ev) C10_CUDA_CHECK(cudaEventCreate(&e));
  C10_CUDA_CHECK(cudaEventRecord(ev[0], st));
  for (int i = 0; i < runs; i++) {
    e55::act_quant(xp, k, M, k, group, a, st);
    C10_CUDA_CHECK(cudaEventRecord(ev[i + 1], st));
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
// ---- P103 f8_tile / P104 f8_tile_epi: the prefill call sites as model.cc runs them (P060 on) --------------------------
// a bf16 vector (norm weight) of this rank
torch::Tensor vec(int rank, const std::string& name) {
  if (!g_w[rank]) g_w[rank] = std::make_unique<e55::Weights>(g_dir, g_table, rank);
  auto b = g_w[rank]->slice(name);
  auto t = torch::empty({int64_t(b.size() / 2)}, torch::dtype(torch::kBFloat16));
  std::memcpy(t.data_ptr(), b.data(), b.size());
  return t.cuda();
}
// a device copy of a weight (cold timing: rotate over copies)
int clone_w(int h) {
  e55::F8Lin L = g_lin.at(h);
  const size_t wb = size_t(L.npad) * L.k, sb = size_t(L.npad / 16) * (L.k / 128);
  C10_CUDA_CHECK(cudaMalloc(&L.w, wb));
  C10_CUDA_CHECK(cudaMalloc(&L.s, sb));
  C10_CUDA_CHECK(cudaMemcpy(L.w, g_lin.at(h).w, wb, cudaMemcpyDeviceToDevice));
  C10_CUDA_CHECK(cudaMemcpy(L.s, g_lin.at(h).s, sb, cudaMemcpyDeviceToDevice));
  g_lin.push_back(L);
  return int(g_lin.size()) - 1;
}
template <typename T = e55::bf16>
T* P(const torch::Tensor& t) { return reinterpret_cast<T*>(t.data_ptr()); }
torch::Tensor g_ws;
// one call site (model.cc line in the profile): h its weights, T its tensors (inputs, outputs, scratch), M rows
void site(const std::string& k, const std::vector<int>& h, const std::vector<torch::Tensor>& T, int M, cudaStream_t s) {
  using namespace e55;
  if (!g_ws.defined()) g_ws = torch::zeros({int64_t(64) << 20}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  float* ws = g_ws.data_ptr<float>();
  auto L = [&](int i) -> const F8Lin& { return g_lin.at(h.at(i)); };
  auto U = [&](int i) { return P<uint8_t>(T.at(i)); };
  if (k == "hca.qkv" || k == "dsa.qkv" || k == "moe.gu" || k == "dense.gu" || k == "dense.down" || k == "dsa.o") {
    // 926 / 932 attn_front, 1014 ffn_tp shared gate/up, 994 dense, 699 DSA o_proj: [xq, xs, y]
    f8_forward(L(0), F8Act{U(0), U(1), int(T[0].size(1))}, M, P(T[2]), int(T[2].size(1)), ws, s);
  } else if (k == "hca.qg") {  // 527: q norm → q_up + gate (σ), kv norm aside. T: t1, qn, kvn, xq, xs, q, t3, t4
    const F8Aux kv{P(T[0]) + 1024, 1536, P(T[2]), 512, P(T[7]), 512};
    f8_forward_multi({{&L(0), P(T[5]), 16384}, {&L(1), P(T[6]), 4096, kEpiSigmoid}},
                     f8_pro(kProRms, P(T[0]), 1536, P(T[1]), 0, U(3), U(4), 1024), M, ws, s, &kv);
  } else if (k == "hca.wo") {  // 601 + 604: wo_a → z ⊙ gate → o_proj. T: oq, os, t3, t4, xq, xs, aout
    const F8Act ga{U(0), U(1), 4096};
    if (f8_epi_fused(L(0), M, kEpiMulQ128, ws)) {  // P104: o_proj's activation from wo_a's store
      f8_forward_out({&L(0), nullptr, 4096, kEpiMulQ128, P(T[2]), 4096, U(4), U(5)}, ga, M, ws, s, long(M) * 4096,
                     long(M) * 32);
      f8_forward(L(1), F8Act{U(4), U(5), 4096}, M, P(T[6]), 4096, ws, s);
    } else {
      f8_forward(L(0), ga, M, P(T[3]), 4096, ws, s, long(M) * 4096, long(M) * 32);
      f8_forward_out({&L(1), P(T[6]), 4096}, f8_pro(kProMul, P(T[3]), 4096, P(T[2]), 4096, U(4), U(5), 4096), M, ws, s);
    }
  } else if (k == "dsa.qgi") {  // 648: q_a norm → q_b + gate (σ) + indexer q, kv_a norm aside. T: t1, qan, kvan, xq, xs, q, t3, qi, t4
    const F8Aux kv{P(T[0]) + 1024, 1600, P(T[2]), 512, P(T[8]), 512};
    f8_forward_multi({{&L(0), P(T[5]), 8192}, {&L(1), P(T[6]), 8192, kEpiSigmoid}, {&L(2), P(T[7]), 8192}},
                     f8_pro(kProRms, P(T[0]), 1600, P(T[1]), 0, U(3), U(4), 1024), M, ws, s, &kv);
  } else if (k == "hca.qup") {  // 747 (P100: the gate on the owner): q norm → q_up, kv norm aside. T: t1, qn, kvn, xq, xs, q, t4
    const F8Aux kv{P(T[0]) + 1024, 1536, P(T[2]), 512, P(T[6]), 512};
    f8_forward_multi({{&L(0), P(T[5]), 16384}}, f8_pro(kProRms, P(T[0]), 1536, P(T[1]), 0, U(3), U(4), 1024), M, ws, s,
                     &kv);
  } else if (k == "dsa.qbi") {  // 886: q_a norm → q_b + iq, kv_a norm aside. T: t1, qan, kvan, xq, xs, q, qi, t4
    const F8Aux kv{P(T[0]) + 1024, 1600, P(T[2]), 512, P(T[7]), 512};
    f8_forward_multi({{&L(0), P(T[5]), 8192}, {&L(1), P(T[6]), 8192}},
                     f8_pro(kProRms, P(T[0]), 1600, P(T[1]), 0, U(3), U(4), 1024), M, ws, s, &kv);
  } else if (k == "hca.gate" || k == "dsa.gate") {  // 1323 attn_gate: q norm → this rank's + the peer's gate (σ). T: t1, qn, xq, xs, g
    const int w = k == "hca.gate" ? 1536 : 1600, n = k == "hca.gate" ? 4096 : 8192;
    f8_forward_multi({{&L(0), P(T[4]), n, kEpiSigmoid}, {&L(1), P(T[4]) + size_t(M) * n, n, kEpiSigmoid}},
                     f8_pro(kProRms, P(T[0]), w, P(T[1]), 0, U(2), U(3), 1024), M, ws, s);
  } else if (k == "hca.woa") {  // 1346 attn_own: wo_a of this rank's and of the peer's heads. T: qm, sm, qp, sp, t4m, t4p
    f8_forward(L(0), F8Act{U(0), U(1), 4096}, M, P(T[4]), 4096, ws, s, long(M) * 4096, long(M) * 32);
    f8_forward(L(1), F8Act{U(2), U(3), 4096}, M, P(T[5]), 4096, ws, s, long(M) * 4096, long(M) * 32);
  } else if (k == "hca.oproj") {  // 1354: z ⊙ gate → o_proj K halves, the 2nd + the 1st's partial. T: t4m, t4p, gm, gp, xq, xs, pm, aout
    f8_forward_out({&L(0), P(T[6]), 4096}, f8_pro(kProMul, P(T[0]), 4096, P(T[2]), 4096, U(4), U(5), 4096), M, ws, s);
    f8_forward_out({&L(1), P(T[7]), 4096, kEpiAdd, P(T[6]), 4096},
                   f8_pro(kProMul, P(T[1]), 4096, P(T[3]), 4096, U(4), U(5), 4096), M, ws, s);
  } else if (k == "dsa.oproj") {  // 1372: act → o_proj K halves, the 2nd + the 1st's partial. T: ym, yp, xq, xs, pm, aout
    const F8Act xa{U(2), U(3), 8192};
    act_quant(P(T[0]), 8192, M, 8192, 128, xa, s);
    f8_forward(L(0), xa, M, P(T[4]), 4096, ws, s);
    act_quant(P(T[1]), 8192, M, 8192, 128, xa, s);
    f8_forward_out({&L(1), P(T[5]), 4096, kEpiAdd, P(T[4]), 4096}, xa, M, ws, s);
  } else if (k == "moe.down.g") {  // the GEMM alone (activation given): 1269's. T: t1, xq, xs, pk
    f8_forward(L(0), F8Act{U(1), U(2), 1024}, M, P(T[3]), 4096, ws, s);
  } else if (k == "moe.fc2.g") {  // 1116's GEMM + its epilogue alone. T: pl, lnorm, ps, xq, xs, fout
    f8_forward_out({&L(0), P(T[5]), 4096, kEpiAdd, P(T[2]), 4096}, F8Act{U(3), U(4), 2048}, M, ws, s);
  } else if (k == "hca.gate.g") {  // 1323's GEMMs + epilogues alone. T: t1, qn, xq, xs, g
    f8_forward_multi({{&L(0), P(T[4]), 4096, kEpiSigmoid}, {&L(1), P(T[4]) + size_t(M) * 4096, 4096, kEpiSigmoid}},
                     F8Act{U(2), U(3), 1024}, M, ws, s);
  } else if (k == "hca.oproj.g") {  // 1354's GEMMs alone (activation given). T: t4m, t4p, gm, gp, xq, xs, pm, aout
    f8_forward(L(0), F8Act{U(4), U(5), 4096}, M, P(T[6]), 4096, ws, s);
    f8_forward_out({&L(1), P(T[7]), 4096, kEpiAdd, P(T[6]), 4096}, F8Act{U(4), U(5), 4096}, M, ws, s);
  } else if (k == "moe.gd") {  // 1261 + 1269 ffn_tp shared: gate/up → SiTU → down. T: xq, xs, t1, xq2, xs2, pk
    const F8Act xa{U(0), U(1), 4096};
    if (f8_epi_fused(L(0), M, kEpiSituQ128, ws)) {  // P134: SiTU + the down's FP8 activation in gate/up's store
      f8_forward_out({&L(0), nullptr, 1024, kEpiSituQ128, nullptr, 0, U(3), U(4)}, xa, M, ws, s);
      f8_forward(L(1), F8Act{U(3), U(4), 1024}, M, P(T[5]), 4096, ws, s);
    } else {
      f8_forward(L(0), xa, M, P(T[2]), 2048, ws, s);
      f8_forward_out({&L(1), P(T[5]), 4096}, f8_pro(kProSitu, P(T[2]), 2048, nullptr, 0, U(3), U(4), 1024), M, ws, s);
    }
  } else if (k == "moe.fc1") {  // 964 ffn_front: fc1 + the latent's MXFP8 copy. T: xq, xs, t2, lq, ls
    f8_forward_out({&L(0), P(T[2]), 2048, kEpiQ32, nullptr, 0, U(3), U(4)}, F8Act{U(0), U(1), 4096}, M, ws, s);
  } else if (k == "moe.front") {  // 1218 ffn_front at decode: fc1 + the latent's MXFP8 copy and the shared gate/up, one call. T: xq, xs, t2, lq, ls, t1
    f8_forward_multi({{&L(0), P(T[2]), 2048, kEpiQ32, nullptr, 0, U(3), U(4)}, {&L(1), P(T[5]), 2048}},
                     F8Act{U(0), U(1), 4096}, M, ws, s);
  } else if (k == "moe.down") {  // 1022: SiTU → shared down. T: t1, xq, xs, pk
    f8_forward_out({&L(0), P(T[3]), 4096}, f8_pro(kProSitu, P(T[0]), 2048, nullptr, 0, U(1), U(2), 1024), M, ws, s);
  } else if (k == "moe.fc2") {  // 869 ffn_b: latent_norm → fc2 + shared. T: pl, lnorm, ps, xq, xs, fout
    f8_forward_out({&L(0), P(T[5]), 4096, kEpiAdd, P(T[2]), 4096},
                   f8_pro(kProRms, P(T[0]), 2048, P(T[1]), 0, U(3), U(4), 2048), M, ws, s);
  } else {
    TORCH_CHECK(false, "site ", k);
  }
}
void site_run(const std::string& k, const std::vector<int>& h, const std::vector<torch::Tensor>& T, int M) {
  site(k, h, T, M, c10::cuda::getCurrentCUDAStream().stream());
}
// ms per call (CUDA events around each), weights rotated over the copies hs[i % n] (cold)
std::vector<double> site_bench(const std::string& k, const std::vector<std::vector<int>>& hs,
                               const std::vector<torch::Tensor>& T, int M, int runs) {
  auto s = c10::cuda::getCurrentCUDAStream().stream();
  for (size_t i = 0; i < hs.size(); i++) site(k, hs[i], T, M, s);  // warm-up
  std::vector<cudaEvent_t> ev(runs + 1);
  for (auto& e : ev) C10_CUDA_CHECK(cudaEventCreate(&e));
  C10_CUDA_CHECK(cudaStreamSynchronize(s));
  C10_CUDA_CHECK(cudaEventRecord(ev[0], s));
  for (int i = 0; i < runs; i++) {
    site(k, hs[(i + 1) % hs.size()], T, M, s);
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

int test_seq(int c) { return e55::f8_test_seq(c); }
int test_mid(int mt) { return e55::f8_test_mid(mt); }
std::string path(int h, int M, int epi) { return e55::f8_test_path(g_lin.at(h), M, epi); }
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("test_seq", &test_seq);
  m.def("test_mid", &test_mid);
  m.def("path", &path);
  m.def("vec", &vec);
  m.def("clone", &clone_w);
  m.def("site_run", &site_run);
  m.def("site_bench", &site_bench);
  m.def("open", &open);
  m.def("upload", &upload);
  m.def("release", &release);
  m.def("info", &info);
  m.def("set_patch", &set_patch);
  m.def("patch_on", &patch_on);
  m.def("defer_act", &defer_act);
  m.def("quant", &quant);
  m.def("forward", &forward, py::arg("h"), py::arg("q"), py::arg("s"), py::arg("M"), py::arg("f32"), py::arg("gx"),
        py::arg("gs"), py::arg("x") = py::none());
  m.def("bench", &bench, py::arg("h"), py::arg("q"), py::arg("s"), py::arg("M"), py::arg("runs"), py::arg("cold"),
        py::arg("gx") = 0, py::arg("gs") = 0, py::arg("x") = py::none(), py::arg("deferred") = false,
        py::arg("chain") = 1);
  m.def("bench_act", &bench_act);
  m.def("forward_multi", &forward_multi, py::arg("hs"), py::arg("q"), py::arg("s"), py::arg("M"),
        py::arg("x") = py::none());
  m.def("bench_multi", &bench_multi, py::arg("hs"), py::arg("q"), py::arg("s"), py::arg("M"), py::arg("runs"),
        py::arg("x") = py::none(), py::arg("chain") = 1, py::arg("quant_first") = false);
}
