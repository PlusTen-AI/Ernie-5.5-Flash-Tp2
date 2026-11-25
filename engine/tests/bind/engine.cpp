// Binding of the whole engine (engine/src): init, step (logits copied out), reset, gate taps — and the server's calls:
// run (a step whose last row of each sequence is sampled: penalties, temperature, min_p, top-k, top-p, the token — on
// the GPU, kernels/sample), penalty state per slot. Built by the server and the gate (tests/ext.py, family "engine":
// every source under src/). Every rank makes the same calls (the server mirrors rank 0's to rank 1, positional
// arguments only); the tokens rank 0 returns are the ones the server uses.
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <algorithm>
#include <map>
#include <memory>
#include <tuple>
#include <vector>

#include "core/log.h"
#include "kernels/dspark/dspark.h"
#include "kernels/sample/sample.h"
#include "model/model.h"
#include "model/patches.h"

namespace {
std::unique_ptr<e55::Engine> g_e;

// ---- sampling state --------------------------------------------------------------------------------------------
struct Samp {
  int cap = 0;  // sequences per call
  float *temp = nullptr, *top_p = nullptr, *min_p = nullptr, *prob = nullptr;
  int *top_k = nullptr, *tok = nullptr, *start = nullptr, *ids = nullptr;
  uint64_t *seed = nullptr, *ctr = nullptr;
  e55::PenSeq* seqs = nullptr;
  int ids_cap = 0;
};
Samp g_s;
std::vector<e55::PenSeen> g_pen;  // per slot; cnt == nullptr until the slot's first pen_begin

template <typename T>
T* dalloc(size_t n) {
  T* p = nullptr;
  CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
  return p;
}

void samp_init(int cap) {
  g_s.cap = cap;
  g_s.temp = dalloc<float>(cap), g_s.top_p = dalloc<float>(cap), g_s.min_p = dalloc<float>(cap);
  g_s.prob = dalloc<float>(cap);
  g_s.top_k = dalloc<int>(cap), g_s.tok = dalloc<int>(cap), g_s.start = dalloc<int>(cap + 1);
  g_s.seed = dalloc<uint64_t>(cap), g_s.ctr = dalloc<uint64_t>(cap);
  g_s.seqs = dalloc<e55::PenSeq>(cap);
  std::vector<int> st(cap + 1);
  for (int i = 0; i <= cap; i++) st[i] = i;
  CUDA_CHECK(cudaMemcpy(g_s.start, st.data(), st.size() * 4, cudaMemcpyHostToDevice));
}
void samp_free() {
  for (void* p : {(void*)g_s.temp, (void*)g_s.top_p, (void*)g_s.min_p, (void*)g_s.prob, (void*)g_s.top_k,
                  (void*)g_s.tok, (void*)g_s.start, (void*)g_s.ids, (void*)g_s.seed, (void*)g_s.ctr, (void*)g_s.seqs})
    cudaFree(p);
  for (auto& p : g_pen) cudaFree(p.cnt), cudaFree(p.ids), cudaFree(p.n);
  g_pen.clear();
  g_s = Samp{};
}
// device copy of host ids (grows)
const int* dev_ids(const std::vector<int>& v, cudaStream_t s) {
  if (int(v.size()) > g_s.ids_cap) {
    cudaFree(g_s.ids);
    g_s.ids_cap = std::max<int>(v.size(), 4096);
    g_s.ids = dalloc<int>(g_s.ids_cap);
  }
  CUDA_CHECK(cudaMemcpyAsync(g_s.ids, v.data(), v.size() * 4, cudaMemcpyHostToDevice, s));
  return g_s.ids;
}

// ---- calls -----------------------------------------------------------------------------------------------------
// P125 warm_up: a prefill chunk at this position too (DSA rows with ≥ 65,536 candidates take P046's select, whose
// workspace grows with the rows per pass)
constexpr int kWarmLong = 65536;
// pool_warm (P125 kv_pool, the server): the pool starts with only the warm-up's room (kv_pool_init sizes it after the
// warm-up); false: every slot's ctx positions, as the per-slot caches were (tests)
void init(const std::string& model, const std::string& table, int rank, const std::string& head_ip, int slots, int ctx,
          int max_rows, int logit_rows, int port, bool pool_warm) {
  e55::EngineCfg c;
  c.model_dir = model, c.table = table, c.rank = rank, c.head_ip = head_ip, c.slots = slots, c.ctx = ctx;
  c.max_rows = max_rows, c.logit_rows = std::max(logit_rows, slots), c.port = port;
  if (pool_warm) {  // warm_up(): slot 0's rows (the others' fewer); with ctx past kWarmLong, its long chunk
    c.pool_tokens = c.max_rows + 2 * c.logit_rows + 128;
    if (ctx > kWarmLong + c.max_rows) c.pool_tokens = std::max(c.pool_tokens, (kWarmLong + c.max_rows + 64) / slots + 1);
  }
  g_e = std::make_unique<e55::Engine>(c);
  samp_init(c.logit_rows);
  g_pen.assign(slots, e55::PenSeen{nullptr, nullptr, nullptr});
}
void engine_close() {
  if (g_e) cudaStreamSynchronize(g_e->stream());
  samp_free();
  g_e.reset();
}

torch::Tensor step(const std::vector<int>& ids, const std::vector<int>& pos, const std::vector<int>& slot,
                   const std::vector<int>& seq_start, bool all_logits, const std::vector<int>& rows) {
  const float* lg = g_e->step(ids, pos, slot, seq_start, all_logits, rows);
  const int R = g_e->logit_rows_last();
  auto out = torch::empty({R, e55::kVocab}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  cudaMemcpyAsync(out.data_ptr<float>(), lg, size_t(R) * e55::kVocab * 4, cudaMemcpyDeviceToDevice, g_e->stream());
  cudaStreamSynchronize(g_e->stream());
  return out;
}

// A step that only fills caches (a prompt chunk before its last): the logits of its last rows are not used.
void fill(const std::vector<int>& ids, const std::vector<int>& pos, const std::vector<int>& slot,
          const std::vector<int>& seq_start) {
  g_e->step(ids, pos, slot, seq_start, false);
}

// A step, then one token per sequence from its last row. Per sequence b: temp (0 = greedy), top_k (0 = off), top_p,
// min_p, seed and ctr (Philox counter: the caller passes the new token's position, so a token never depends on
// batching), penalties (rep, freq, pres; (1, 0, 0) = off — a penalised sequence needs pen_begin first, its sampled
// token is counted here). Returns the tokens (host).
std::vector<int> run(const std::vector<int>& ids, const std::vector<int>& pos, const std::vector<int>& slot,
                     const std::vector<int>& seq_start, const std::vector<double>& temp, const std::vector<int>& top_k,
                     const std::vector<double>& top_p, const std::vector<double>& min_p,
                     const std::vector<int64_t>& seed, const std::vector<int64_t>& ctr, const std::vector<double>& rep,
                     const std::vector<double>& freq, const std::vector<double>& pres) {
  const int B = int(slot.size());
  if (B > g_s.cap) qlog::fail("run", "%d sequences > %d", B, g_s.cap);
  cudaStream_t s = g_e->stream();
  // P063 sparse_wire: when every sequence's settings allow it (sparse_ok), sample from both vocab halves' top-64
  // records instead of all-gathered full rows — the same tokens; the halves are penalised in place first
  bool sp = e55::patch::on(e55::patch::sparse_wire);
  for (int b = 0; b < B && sp; b++) sp = e55::sparse_ok(float(temp[b]), top_k[b], float(top_p[b]), e55::kVocab);
  float* lg = const_cast<float*>(g_e->step(ids, pos, slot, seq_start, false, {}, !sp));
  std::vector<float> t(B), tp(B), mp(B);
  std::vector<uint64_t> sd(B), ct(B);
  std::vector<e55::PenSeq> pq(B);
  bool pen = false;
  for (int b = 0; b < B; b++) {
    t[b] = float(temp[b]), tp[b] = float(top_p[b]), mp[b] = float(min_p[b]);
    sd[b] = uint64_t(seed[b]), ct[b] = uint64_t(ctr[b]);
    const bool on = rep[b] != 1.0 || freq[b] != 0.0 || pres[b] != 0.0;
    if (on && !g_pen.at(slot[b]).cnt) qlog::fail("run", "slot %d: penalties without pen_begin", slot[b]);
    pq[b] = e55::PenSeq{on ? g_pen[slot[b]] : e55::PenSeen{nullptr, nullptr, nullptr}, float(rep[b]), float(freq[b]),
                        float(pres[b])};
    pen |= on;
  }
  CUDA_CHECK(cudaMemcpyAsync(g_s.temp, t.data(), B * 4, cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaMemcpyAsync(g_s.top_p, tp.data(), B * 4, cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaMemcpyAsync(g_s.min_p, mp.data(), B * 4, cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaMemcpyAsync(g_s.top_k, top_k.data(), B * 4, cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaMemcpyAsync(g_s.seed, sd.data(), B * 8, cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaMemcpyAsync(g_s.ctr, ct.data(), B * 8, cudaMemcpyHostToDevice, s));
  if (pen) {
    CUDA_CHECK(cudaMemcpyAsync(g_s.seqs, pq.data(), B * sizeof(e55::PenSeq), cudaMemcpyHostToDevice, s));
    if (sp)  // this rank's half (ids rank·Vh …): the same values as on the full rows
      e55::penalize_rows(lg, e55::kVocabHalf, g_e->rank() * e55::kVocabHalf, e55::kVocabHalf, e55::kVocab, B, g_s.start,
                         nullptr, nullptr, B, g_s.seqs, s);
    else
      e55::penalize_rows(lg, e55::kVocab, 0, e55::kVocab, e55::kVocab, B, g_s.start, nullptr, nullptr, B, g_s.seqs, s);
  }
  std::vector<int> out(B);
  if (sp) {
    e55::sample_rows_sparse(g_e->sparse_cands(), B, B, e55::kVocab, g_s.temp, g_s.top_k, g_s.top_p, g_s.min_p,
                            g_s.seed, g_s.ctr, g_s.tok, g_s.prob, s);
    CUDA_CHECK(cudaMemcpyAsync(out.data(), g_s.tok, B * 4, cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    // kSparseDense: a NaN / inf in a half — the full rows (both ranks hold the same records: the same decision)
    if (std::find(out.begin(), out.end(), e55::kSparseDense) == out.end()) {
      if (pen) e55::penalty_add(g_s.seqs, B, g_s.tok, g_s.start, nullptr, nullptr, e55::kVocab, s);  // before the next step
      return out;
    }
    lg = const_cast<float*>(g_e->gather_logits());
  }
  e55::sample_rows(lg, B, e55::kVocab, g_s.temp, g_s.top_k, g_s.top_p, g_s.min_p, g_s.seed, g_s.ctr, g_s.tok,
                   g_s.prob, s);
  if (pen) e55::penalty_add(g_s.seqs, B, g_s.tok, g_s.start, nullptr, nullptr, e55::kVocab, s);
  CUDA_CHECK(cudaMemcpyAsync(out.data(), g_s.tok, B * 4, cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  return out;
}

// Penalty state of a slot: the prompt's tokens marked, no generated tokens yet (the slot's request is penalised).
void pen_begin(int slot, const std::vector<int>& prompt) {
  auto& p = g_pen.at(slot);
  if (!p.cnt) p = e55::PenSeen{dalloc<uint32_t>(e55::kVocab), dalloc<int>(e55::kVocab), dalloc<int>(1)};
  cudaStream_t s = g_e->stream();
  e55::penalty_reset(p, e55::kVocab, s);
  if (!prompt.empty()) e55::penalty_prompt(p, dev_ids(prompt, s), int(prompt.size()), e55::kVocab, s);
  CUDA_CHECK(cudaStreamSynchronize(s));
}
// Generated tokens the server put in itself (a forced </think>, tool_choice's forced call start) count as generated.
void pen_add(int slot, const std::vector<int>& toks) {
  auto& p = g_pen.at(slot);
  if (!p.cnt || toks.empty()) return;
  cudaStream_t s = g_e->stream();
  e55::PenSeq q{p, 1.f, 0.f, 0.f};
  e55::PenSeq* dq = dalloc<e55::PenSeq>(1);
  int* st = dalloc<int>(2);
  const int h[2] = {0, int(toks.size())};
  CUDA_CHECK(cudaMemcpyAsync(dq, &q, sizeof q, cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaMemcpyAsync(st, h, 8, cudaMemcpyHostToDevice, s));
  e55::penalty_add(dq, 1, dev_ids(toks, s), st, nullptr, nullptr, e55::kVocab, s);
  CUDA_CHECK(cudaStreamSynchronize(s));
  cudaFree(dq), cudaFree(st);
}

// ---- DSpark speculative decoding (model/dspark.cc; model.h SpecSeq) ----------------------------------------------
// The scheduler's calls: dspark_init once after init; per request: (pen_begin if penalised), the prompt through
// fill / step, spec_begin (first token from the last step's logits row, then drafts), then spec_round over the slots
// decoding together (returns each slot's emitted tokens); spec_depth / spec_gate / spec_cap / spec_last between
// rounds; reset as before. Every rank makes the same calls (the drafts and accept decisions are the same bits on both).
std::map<int, e55::SpecSeq> g_sq;  // per slot

void dspark_init() { g_e->dspark_init(); }
std::vector<e55::SpecSeq*> seqs_of(const std::vector<int>& slots) {
  std::vector<e55::SpecSeq*> v;
  for (int s : slots) {
    auto it = g_sq.find(s);
    if (it == g_sq.end()) qlog::fail("spec", "slot %d has no speculative sequence (spec_begin)", s);
    v.push_back(&it->second);
  }
  return v;
}
// lrow: the row of this sequence in the last step's logits; gate > 0: confidence-gated verify length (SpecSeq)
int spec_begin(int slot, int depth, double temp, int64_t seed, int lrow, int top_k, double top_p, double min_p,
               double rep, double freq, double pres, double gate, bool draft) {
  e55::SpecSeq q;
  q.slot = slot, q.depth = depth, q.temp = float(temp), q.seed = uint64_t(seed), q.top_k = top_k;
  q.top_p = float(top_p), q.min_p = float(min_p), q.rep = float(rep), q.freq = float(freq), q.pres = float(pres);
  q.gate = float(gate);
  if (rep != 1.0 || freq != 0.0 || pres != 0.0) {
    if (!g_pen.at(slot).cnt) qlog::fail("spec", "slot %d: penalties without pen_begin", slot);
    q.pen = g_pen[slot];
  }
  g_sq[slot] = q;
  e55::SpecSeq& r = g_sq[slot];
  const int tok = g_e->spec_first(r, lrow);
  if (draft) g_e->spec_draft({&r});
  return tok;
}
void spec_draft(const std::vector<int>& slots) { g_e->spec_draft(seqs_of(slots)); }
// caps (optional, one per slot): the most tokens each sequence may emit this round (as spec_cap; kept for later rounds)
std::vector<std::vector<int>> spec_round(const std::vector<int>& slots, bool draft_next, const std::vector<int>& caps) {
  auto v = seqs_of(slots);
  if (!caps.empty()) {
    if (caps.size() != slots.size()) qlog::fail("spec", "spec_round: %zu caps for %zu slots", caps.size(), slots.size());
    for (size_t i = 0; i < v.size(); i++) v[i]->cap = caps[i];
  }
  return g_e->spec_round(v, draft_next);
}
// P168 copy rounds (server/copy.py): slot's round verifying `copy` in place of its drafts, at most cap tokens (0: no
// cap; kept for later rounds as spec_round's caps) → the emitted tokens. follow ≥ 0: the token after `copy` in the
// source — when every copied token and then it come out, no DSpark drafts are made (spec_drafts empty: the caller
// copies again or calls spec_draft before a DSpark round)
std::vector<int> spec_copy(int slot, const std::vector<int>& copy, int cap, int follow, bool draft_next) {
  auto v = seqs_of({slot});
  v[0]->cap = cap;
  return g_e->spec_copy(*v[0], copy, follow, draft_next);
}
void spec_commit(const std::vector<int>& keep) { g_e->spec_commit(keep); }
torch::Tensor spec_logits() {
  int R = 0;
  const float* lg = g_e->spec_logits(&R);
  auto out = torch::empty({R, e55::kVocab}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  cudaMemcpyAsync(out.data_ptr<float>(), lg, size_t(R) * e55::kVocab * 4, cudaMemcpyDeviceToDevice, g_e->stream());
  cudaStreamSynchronize(g_e->stream());
  return out;
}
std::map<std::string, double> spec_stats(bool reset) {
  const e55::SpecStats s = g_e->spec_stats(reset);
  return {{"ms_verify", s.ms_verify}, {"ms_ctx", s.ms_ctx}, {"ms_accept", s.ms_accept}, {"ms_commit", s.ms_commit},
          {"ms_draft", s.ms_draft}, {"rounds", double(s.rounds)}, {"seqs", double(s.seqs)}, {"rows", double(s.rows)},
          {"verified", double(s.verified)}, {"accepted", double(s.accepted)}, {"emitted", double(s.emitted)}};
}
void dspark_ctx(int slot, torch::Tensor aux, const std::vector<int>& pos) {
  TORCH_CHECK(aux.is_cuda() && aux.dtype() == torch::kBFloat16 && aux.is_contiguous() && aux.dim() == 2 &&
              aux.size(0) == int64_t(pos.size()) && aux.size(1) == e55::kDspAux, "aux: cuda bf16 [K, 12288]");
  cudaDeviceSynchronize();
  g_e->dspark_ctx(slot, reinterpret_cast<const e55::bf16*>(aux.data_ptr()), pos);
}
std::pair<std::vector<double>, long> expert_take() {
  long n = 0;
  auto v = g_e->expert_take(&n);
  return {v, n};
}

void sync_mode(int m) { e55::set_sync_mode(m); }
void sync_device() {  // the queued GPU work done (server.prefill_marks)
  const cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) qlog::fail("engine", "sync: %s", cudaGetErrorString(e));
}
// (stage, model.cc line, ms, count) since the last call (sync mode 2)
std::vector<std::tuple<std::string, int, double, long>> prof() {
  std::vector<std::tuple<std::string, int, double, long>> out;
  for (auto& p : e55::prof_take()) out.emplace_back(p.what, p.line, p.ms, p.n);
  return out;
}
void reset(int slot) {
  g_sq.erase(slot);
  g_e->reset(slot);
}
int history(int slot) { return g_e->history(slot); }
// ---- P125 kv_pool -----------------------------------------------------------------------------------------------
// warm_up: the server's paths once each before the pool takes the room left, so whatever they allocate lazily (kernel
// modules, NCCL / RDMA connections, the kernels' own workspaces, penalty state, id buffers) comes out of that room now,
// not out of the reserve later: (ctx past 65,536 + max_rows: a prefill chunk at position 65,536 on zero caches —
// P046's long-context select), a 100-row step, a prefill chunk (max_rows rows), decode through run() greedy with
// penalties (P063's sparse sampling) and sampled (full rows), a logit_rows-row all-logits step, every slot decoding
// together, and with DSpark a greedy and a sampled round over every slot. Every slot is reset at the end. Same calls on
// every rank (the server mirrors kv_pool_init).
void warm_up() {
  const e55::EngineCfg& c = g_e->cfg();
  const int S = c.slots, LR = c.logit_rows, R = std::max(1, std::min(c.max_rows, c.ctx - 2 * LR - 64));
  auto ids = [](int n, int o) {
    std::vector<int> v(n);
    for (int i = 0; i < n; i++) v[i] = 1000 + int((long(o + i) * 7919) % 90000);
    return v;
  };
  auto seq = [](int a, int n) {
    std::vector<int> v(n);
    for (int i = 0; i < n; i++) v[i] = a + i;
    return v;
  };
  for (int s = 0; s < S; s++) reset(s), pen_begin(s, {});
  dev_ids(std::vector<int>(std::max(c.ctx, 4096)), g_e->stream());  // pen_begin's prompt copies (≤ ctx ids)
  if (c.ctx > kWarmLong + c.max_rows) {  // first, on caches still all zero (finite inputs; the output is dropped)
    g_e->set_history(0, kWarmLong);
    fill(ids(c.max_rows, 0), seq(kWarmLong, c.max_rows), {0}, {0, c.max_rows});
    reset(0);
  }
  const int m = std::min(100, R);
  fill(ids(m, 0), seq(0, m), {0}, {0, m});
  reset(0);
  fill(ids(R, 0), seq(0, R), {0}, {0, R});
  int p = R;
  int t = run({ids(1, p)[0]}, {p}, {0}, {0, 1}, {0.0}, {0}, {1.0}, {0.0}, {1}, {p + 1}, {1.1}, {0.0}, {0.0})[0];
  p++;
  t = run({t}, {p}, {0}, {0, 1}, {1.0}, {0}, {1.0}, {0.0}, {2}, {p + 1}, {1.0}, {0.0}, {0.0})[0];
  p++;
  const int v = std::min(LR, R);
  g_e->step(ids(v, p), seq(p, v), {0}, {0, v}, true);
  p += v;
  std::vector<int> tk(S), ps(S), sl(S), st(S + 1);
  for (int s = 0; s < S; s++) tk[s] = 1000 + s, ps[s] = s ? 0 : p, sl[s] = s, st[s + 1] = s + 1;
  g_e->step(tk, ps, sl, st, false);  // (gathered rows: spec_begin samples its first token from them)
  if (g_e->dspark() && LR >= 2 * S) {  // (a round verifies 1 + depth rows a sequence)
    const int depth = std::min(15, LR / S - 1);
    for (int s = 0; s < S; s++) spec_begin(s, depth, 0.0, s + 1, s, 0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, true);
    std::vector<int> all(S);
    for (int s = 0; s < S; s++) all[s] = s;
    spec_round(all, true, {});
    for (int s = 0; s < S; s++) g_sq.at(s).temp = 0.8f;
    spec_round(all, false, {});
  }
  for (int s = 0; s < S; s++) reset(s);
  CUDA_CHECK(cudaStreamSynchronize(g_e->stream()));
}
// both ranks: (warm: warm_up() first) then the pool to the room left (Engine::kv_pool_init) → its chunks (0: P125 off)
size_t kv_pool_init(double reserve_gib, double max_gib, bool warm) {
  if (!g_e->kv_pool()) return 0;
  if (warm) warm_up();
  return g_e->kv_pool_init(reserve_gib, max_gib);
}

void set_taps(const std::vector<int>& rows) { g_e->set_taps(rows); }
// (resid [Ln, n, 4, 4096], attn [Ln, n, 4096], ffn [Ln, n, 4096], route [Ln, n, 10]) copies of the listed layers
// (empty: all 43)
std::vector<torch::Tensor> taps(int n, const std::vector<int>& layers) {
  cudaStreamSynchronize(g_e->stream());
  std::vector<int> ls = layers;
  if (ls.empty())
    for (int i = 0; i < e55::kLayers; i++) ls.push_back(i);
  const long Ln = long(ls.size());
  auto bf = torch::dtype(torch::kBFloat16).device(torch::kCUDA);
  auto r = torch::empty({Ln, n, 4, e55::kHidden}, bf), a = torch::empty({Ln, n, e55::kHidden}, bf);
  auto f = torch::empty({Ln, n, e55::kHidden}, bf);
  auto rt = torch::empty({Ln, n, 10}, torch::dtype(torch::kInt32).device(torch::kCUDA));
  auto f32 = torch::dtype(torch::kFloat32).device(torch::kCUDA);
  auto m = torch::empty({Ln, n, 4, e55::kHidden}, bf), po = torch::empty({Ln, n, 4}, f32), cb = torch::empty({Ln, n, 16}, f32);
  const size_t R = size_t(n) * 4 * e55::kHidden, H = size_t(n) * e55::kHidden, K = size_t(n) * 10;
  for (long j = 0; j < Ln; j++) {
    const size_t L = size_t(ls[j]);
    cudaMemcpy((e55::bf16*)r.data_ptr() + j * R, g_e->tap_resid() + L * R, R * 2, cudaMemcpyDeviceToDevice);
    cudaMemcpy((e55::bf16*)a.data_ptr() + j * H, g_e->tap_attn() + L * H, H * 2, cudaMemcpyDeviceToDevice);
    cudaMemcpy((e55::bf16*)f.data_ptr() + j * H, g_e->tap_ffn() + L * H, H * 2, cudaMemcpyDeviceToDevice);
    cudaMemcpy(rt.data_ptr<int>() + j * K, g_e->tap_route() + L * K, K * 4, cudaMemcpyDeviceToDevice);
    cudaMemcpy((e55::bf16*)m.data_ptr() + j * R, g_e->tap_mid() + L * R, R * 2, cudaMemcpyDeviceToDevice);
    cudaMemcpy(po.data_ptr<float>() + j * n * 4, g_e->tap_post() + L * n * 4, size_t(n) * 16, cudaMemcpyDeviceToDevice);
    cudaMemcpy(cb.data_ptr<float>() + j * n * 16, g_e->tap_comb() + L * n * 16, size_t(n) * 64, cudaMemcpyDeviceToDevice);
  }
  // resid, attn, ffn, route, then the mid-layer state: residual after the attention post, the FFN pre's post / comb
  return {r, a, f, rt, m, po, cb};
}
}  // namespace

// Engine work runs without the GIL (arguments are converted before, results after), so the server's HTTP threads keep
// reading bodies and streaming while a step, a prompt chunk or the load is in flight. Nothing here calls into Python.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using nogil = py::call_guard<py::gil_scoped_release>;
  m.def("init", &init, py::arg("model"), py::arg("table"), py::arg("rank"), py::arg("head_ip"), py::arg("slots"),
        py::arg("ctx"), py::arg("max_rows"), py::arg("logit_rows"), py::arg("port"), py::arg("pool_warm") = false, nogil());
  m.def("close", &engine_close, nogil());
  m.def("step", &step, py::arg("ids"), py::arg("pos"), py::arg("slot"), py::arg("seq_start"), py::arg("all_logits"),
        py::arg("rows") = std::vector<int>{}, nogil());
  m.def("fill", &fill, nogil());
  m.def("run", &run, nogil());
  m.def("pen_begin", &pen_begin, nogil());
  m.def("pen_add", &pen_add, nogil());
  m.def("sync_mode", &sync_mode);
  m.def("sync", &sync_device);
  m.def("prof", &prof);
  m.def("reset", &reset, nogil());
  // P146 prefix_ckpt (model.h): entries per slot, save / restore / copy / drop change state (both ranks: the server
  // mirrors them); a restored or copied slot's speculative sequence starts again (spec_begin), as after reset
  m.def("ckpt_on", []() { return g_e->ckpt_on(); });
  m.def("ckpt_init", [](int n) { g_e->ckpt_init(n); }, nogil());
  m.def("ckpt_cap", []() { return g_e->ckpt_cap(); });
  m.def("ckpt_save", [](int s, int k) { return g_e->ckpt_save(s, k); }, nogil());
  m.def("ckpt_restore", [](int s, int k) { g_sq.erase(s), g_e->ckpt_restore(s, k); }, nogil());
  m.def("ckpt_copy", [](int d, int s, int k) { g_sq.erase(d); return g_e->ckpt_copy(d, s, k); }, nogil());
  m.def("ckpt_drop", [](int s, int k) { g_e->ckpt_drop(s, k); }, nogil());
  m.def("ckpt_pos", [](int s, int k) { return g_e->ckpt_pos(s, k); });
  m.def("ckpt_bytes", [](long pos) { return g_e->ckpt_bytes(pos); });
  m.def("ckpt_need", [](long pos) { return g_e->ckpt_need(pos); });
  m.def("ckpt_held", [](int s) { return g_e->ckpt_held(s); });
  m.def("history", &history);
  m.def("set_taps", &set_taps);
  m.def("dspark_init", &dspark_init, nogil());
  m.def("spec_begin", &spec_begin, py::arg("slot"), py::arg("depth"), py::arg("temp"), py::arg("seed"), py::arg("lrow"),
        py::arg("top_k") = 0, py::arg("top_p") = 1.0, py::arg("min_p") = 0.0, py::arg("rep") = 1.0,
        py::arg("freq") = 0.0, py::arg("pres") = 0.0, py::arg("gate") = 0.0, py::arg("draft") = true, nogil());
  m.def("spec_draft", &spec_draft, nogil());
  m.def("spec_round", &spec_round, py::arg("slots"), py::arg("draft_next") = true, py::arg("caps") = std::vector<int>{},
        nogil());
  // P143: the round cost table [B][rmax + 1] (ms; empty: off) · each sequence's verified drafts in the last round
  m.def("spec_round_cost", [](const std::vector<float>& ms, int rmax) { g_e->spec_round_cost(ms, rmax); },
        py::arg("ms"), py::arg("rmax"));
  m.def("spec_verified", []() { return g_e->spec_verified(); });
  m.def("spec_copy", &spec_copy, py::arg("slot"), py::arg("copy"), py::arg("cap") = 0, py::arg("follow") = -1,
        py::arg("draft_next") = true, nogil());
  m.def("spec_commit", &spec_commit, nogil());
  m.def("spec_logits", &spec_logits, nogil());
  m.def("spec_stats", &spec_stats, py::arg("reset") = true);
  m.def("spec_drafts", [](int s) { return g_sq.at(s).drafts; });
  m.def("spec_conf", [](int s) { return g_sq.at(s).conf; });
  m.def("spec_pos", [](int s) { return g_sq.at(s).pos; });
  m.def("spec_last", [](int s, int tok) { g_sq.at(s).last = tok; }, py::arg("slot"), py::arg("token"));
  m.def("spec_depth", [](int s, int d) { g_sq.at(s).depth = d; }, py::arg("slot"), py::arg("depth"));
  m.def("spec_gate", [](int s, double g) { g_sq.at(s).gate = float(g); }, py::arg("slot"), py::arg("gate"));
  m.def("spec_cost", [](const std::vector<float>& ms) { g_e->spec_cost(ms); });
  m.def("spec_cap", [](int s, int c) { g_sq.at(s).cap = c; }, py::arg("slot"), py::arg("cap"));
  m.def("dspark_ctx", &dspark_ctx);
  m.def("patch", [](const std::string& n, bool on) { e55::patch::set(n, on); });  // tests: A / B in one process
  m.def("patch_on", [](const std::string& n) { return e55::patch::on(n); });
  m.def("set_history", [](int s, int n) { g_e->set_history(s, n); });
  // test: greedy drafts at anchor (last, pos) of slot (its context rows written), depth rows → (drafts, conf)
  m.def("draft_at", [](int s, int last, int pos, int depth) {
    e55::SpecSeq q;
    q.slot = s, q.last = last, q.pos = pos, q.depth = depth;
    g_e->spec_draft({&q});
    return std::make_pair(q.drafts, q.conf);
  }, nogil());
  // P125 kv_pool: the pool (both ranks, the server mirrors it) and its accounting (rank 0's queries, not mirrored)
  m.def("kv_pool_init", &kv_pool_init, py::arg("reserve_gib"), py::arg("max_gib") = 0.0, py::arg("warm") = true, nogil());
  m.def("kv_on", []() { return g_e->kv_pool(); });
  m.def("kv_need", [](long tokens) { return g_e->kv_need(tokens); });
  m.def("kv_held", [](int slot) { return g_e->kv_held(slot); });
  m.def("kv_free", []() { return g_e->kv_free(); });
  m.def("kv_total", []() { return g_e->kv_total(); });
  m.def("mem_free", []() { return g_e->mem_free_gib(); }, nogil());  // [rank0, rank1] MemFree GiB (collective)
  m.def("expert_count", [](bool on) { g_e->expert_count(on); });
  m.def("expert_take", &expert_take);
  m.def("taps", &taps, py::arg("n"), py::arg("layers") = std::vector<int>{});
}
