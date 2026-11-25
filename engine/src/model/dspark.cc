// dspark.cc — DSpark speculative decoding on this rank (contract: model.h "DSpark"; the draft's math: docs/model.md).
//
// The draft (checkpoint model.mtp.*, vLLM-ERNIE's Ernie5_5DSparkForCausalLM; tools/ref/dspark_ref.py is its plain
// PyTorch form): 3 stages, each an ERNIE MoE decoder layer whose attention is HCA without compressor over a sliding
// window, on the target's embedding / lm_head / multimax, plus a Markov head and a confidence head. TP = 2 like the
// target (tools/tensor_table.py): heads and expert / shared intermediates split, everything feeding both halves
// replicated, markov_w2 split by vocabulary rows like lm_head. Native formats: FP8 block linears through kernels/gemm,
// MXFP4 experts through kernels/moe (W4A8), mHC through kernels/mhc, bf16 through kernels/linear.
//
// Context: every target step writes, for each sequence's last ≤ 128 rows, main_x = RMSNorm(feature_projection(
// [aux40 ‖ aux41 ‖ aux42])) (aux_L: the stream mean of the bf16 residual after layer L) and from it each stage's
// context KV row (RMSNorm_kv(linear_kv_proj(main_x)), RoPE at the row's position) into a per-slot ring of 256.
// Draft at anchor P (the last emitted token, not yet through the target): rows [last, 1024 × (N−1)] at P … P+N−1,
// attention over the context rows P−128 … P−1 and the N block rows; the head (mhc_contract, norm, lm_head) gives the
// base logits; then left to right: logits_i = multimax(base_i + markov_w2 · markov_w1[prev]), draft_i = argmax
// (greedy) or a sample_rows draw (temp > 0, kept as the q row of that draft); confidence_i = w·[h_i, markov_w1[prev]].
//
// Verify: one target step over [last, d_1 … d_k] of every sequence (all logits). The step writes the HCA window ring
// and compressor state for all rows; positional caches (DSA latent / indexer keys, compressed entries, the draft's
// 256-ring) need nothing, since a rejected row's entry is rewritten before anything reads it. The ring rows (by
// position % 128) a rejected row overwrote are still read by later queries: they are saved before the step and put
// back by spec_commit. A block that contains a compressor group's closing position c: the step keeps only the rows
// after c in the state (slots 0 …, overwriting the old group's first rows) — if c is rejected, the saved state rows are
// put back and the kept rows before c get their state rows from the step's stashed projections. Afterwards every cache
// and the history equal what plain decoding of the kept rows leaves (deterministic, batch-invariant kernels).
#include <malloc.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <thread>

#include "core/log.h"
#include "kernels/dspark/dspark.h"
#include "kernels/misc/misc.h"
#include "kernels/sample/sample.h"
#include "model/layer.h"
#include "model/model.h"
#include "model/patches.h"

namespace e55 {
int pack12_rc();  // model.cc: the routers' copies (P062 / P098)
namespace {

template <typename T>
T* dmalloc(size_t n) {
  T* p = nullptr;
  CUDA_CHECK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
  CUDA_CHECK(cudaMemset(p, 0, std::max<size_t>(n, 1) * sizeof(T)));
  return p;
}
// cudaFree each and null it (unset pointers are null: a no-op)
template <typename... T>
void dfree(T*&... p) {
  ((cudaFree(const_cast<void*>(static_cast<const void*>(p))), p = nullptr), ...);
}
float bf2f(uint16_t b) {
  uint32_t u = uint32_t(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
std::vector<float> as_f32(const Weights& W, const std::string& n) {
  const auto& ti = W.info(n);
  auto b = W.slice(n);
  std::vector<float> v;
  if (ti.dtype == "F32") {
    v.resize(b.size() / 4);
    std::memcpy(v.data(), b.data(), b.size());
  } else if (ti.dtype == "BF16") {
    v.resize(b.size() / 2);
    for (size_t i = 0; i < v.size(); i++) v[i] = bf2f(uint16_t(b[2 * i] | (b[2 * i + 1] << 8)));
  } else {
    qlog::fail("dspark", "%s: dtype %s is not a float", n.c_str(), ti.dtype.c_str());
  }
  return v;
}
float* up_f32(const std::vector<float>& v) {
  float* p = dmalloc<float>(v.size());
  CUDA_CHECK(cudaMemcpy(p, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
  return p;
}
template <typename T>
T* up(const std::vector<T>& v) {
  T* p = dmalloc<T>(v.size());
  CUDA_CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return p;
}
bf16* up_bf16(const Weights& W, const std::string& n) {
  auto b = W.slice(n);
  bf16* p = dmalloc<bf16>(b.size() / 2);
  CUDA_CHECK(cudaMemcpy(p, b.data(), b.size(), cudaMemcpyHostToDevice));
  return p;
}
float* up_vha(const Weights& W, const std::string& n, int rank) {  // this rank's 32 heads × 16 (as model.cc)
  auto v = as_f32(W, n);
  if (v.size() != size_t(kHeads) * kVhaRank) qlog::fail("dspark", "%s shape", n.c_str());
  return up_f32(std::vector<float>(v.begin() + rank * 32 * kVhaRank, v.begin() + (rank + 1) * 32 * kVhaRank));
}
// vLLM-ERNIE's draft RoPE: build_deepseek_v4_rope(compress_ratio = 1) — deepseek_yarn with factor 1 at rope_theta
// 8e6 (plain RoPE), the formula of model.cc's inv_yarn with factor 1 (fp32, the reference's op order)
std::vector<float> inv_draft() {
  const float theta = 8000000.f, factor = 1.f;
  const double orig = 1048576, bfast = 32, bslow = 1;
  std::vector<float> v(32);
  auto corr = [&](double r) { return 64.0 * std::log(orig / (r * 2 * M_PI)) / (2 * std::log(double(theta))); };
  double low = std::max(std::floor(corr(bfast)), 0.0), high = std::min(std::ceil(corr(bslow)), 63.0);
  if (low == high) high += 0.001;
  for (int i = 0; i < 32; i++) {
    const float pf = powf(theta, float(2 * i) / 64.0f);
    const float extra = 1.0f / pf, inter = 1.0f / (factor * pf);
    const float ramp = std::min(1.0f, std::max(0.0f, (float(i) - float(low)) / float(high - low)));
    const float mask = 1.0f - ramp;
    v[i] = inter * (1.0f - mask) + extra * mask;
  }
  return v;
}
float sigm(float x) { return 1.f / (1.f + std::exp(-x)); }
// E55_DSP_PROF=n (tests; read per draft pass / round): CUDA events at the draft pass's stage boundaries (g_dp) and at
// a round's phase boundaries (g_rp: verify, accept, commit); rank 0 logs the ms between them by stage name (summed
// over the 3 stages and the Markov steps), averaged over n passes (n = 1: every pass). An event between two launches
// costs the PDL overlap there: the split sums a little above the pass's own time.
struct DProf {
  const char* tag;
  bool on = false;
  int n = 0, every = 1, passes = 0;
  cudaEvent_t ev[192] = {};
  const char* nm[192] = {};
  std::vector<std::pair<const char*, double>> acc;
  double tot = 0.0;
};
DProf g_dp{"dsp-prof"}, g_rp{"round-prof"};
void dmark(DProf& p, cudaStream_t s, const char* nm) {
  if (!p.on || p.n >= 192) return;
  if (!p.ev[p.n]) CUDA_CHECK(cudaEventCreate(&p.ev[p.n]));
  CUDA_CHECK(cudaEventRecord(p.ev[p.n], s));
  p.nm[p.n++] = nm;
}
void dmark(cudaStream_t s, const char* nm) { dmark(g_dp, s, nm); }
void dprof_begin(DProf& p, cudaStream_t s) {
  const char* e = std::getenv("E55_DSP_PROF");
  const int n = e ? std::atoi(e) : 0;
  p.on = n > 0, p.n = 0;
  if (!p.on || n != p.every) p.every = std::max(1, n), p.passes = 0, p.acc.clear(), p.tot = 0.0;
  dmark(p, s, "start");
}
void dprof_begin(cudaStream_t s) { dprof_begin(g_dp, s); }
void dprof_end(DProf& p, int rank, int M) {  // after the pass's synchronize
  if (!p.on || rank != 0 || p.n < 2) return;
  for (int i = 1; i < p.n; i++) {
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, p.ev[i - 1], p.ev[i]));
    p.tot += ms;
    auto it = std::find_if(p.acc.begin(), p.acc.end(), [&](const auto& x) { return !std::strcmp(x.first, p.nm[i]); });
    if (it == p.acc.end()) p.acc.emplace_back(p.nm[i], ms);
    else it->second += ms;
  }
  if (++p.passes < p.every) return;
  std::string s;
  char b[96];
  for (const auto& [k, v] : p.acc) {
    std::snprintf(b, sizeof b, "%s%s %.3f", s.empty() ? "" : " · ", k, v / p.passes);
    s += b;
  }
  if (p.every == 1) qlog::info(p.tag, "M=%d · events %.3f ms: %s", M, p.tot, s.c_str());
  else qlog::info(p.tag, "M=%d · mean of %d · events %.3f ms: %s", M, p.passes, p.tot / p.passes, s.c_str());
  p.passes = 0, p.acc.clear(), p.tot = 0.0;
}
void dprof_end(int rank, int M) { dprof_end(g_dp, rank, M); }
// bf16 [n, k] (host) → e4m3 [n, k] + UE8M0 [n, k/32]: per 32, scale 2^ceil(log2(amax / 448)), RNE, saturating
void mxfp8(const uint16_t* w, size_t n, int k, std::vector<uint8_t>& q, std::vector<uint8_t>& sc) {
  q.resize(n * k), sc.resize(n * (k / 32));
  const int nt = std::max(1, load_threads());
  std::vector<std::thread> th;
  for (int t = 0; t < nt; t++)
    th.emplace_back([&, t] {
      for (size_t blk = t; blk < n * (k / 32); blk += nt) {
        const uint16_t* x = w + blk * 32;
        float amax = 0.f;
        for (int j = 0; j < 32; j++) amax = std::max(amax, std::fabs(bf2f(x[j])));
        int e = amax > 0.f ? int(std::ceil(std::log2(amax / 448.f))) : -127;
        e = std::max(-127, std::min(127, e));
        sc[blk] = uint8_t(e + 127);
        const float inv = std::ldexp(1.f, -e);
        for (int j = 0; j < 32; j++)
          q[blk * 32 + j] = uint8_t(__nv_cvt_float_to_fp8(bf2f(x[j]) * inv, __NV_SATFINITE, __NV_E4M3));
      }
    });
  for (auto& x : th) x.join();
}

}  // namespace

int SpecSeq::verify(const std::vector<float>* cost) const {
  const int n = int(drafts.size());
  if (gate < 0.f && cost && !cost->empty() && int(conf.size()) >= n) {  // the best expected tokens per ms
    float s = 1.f, e = 1.f, best = 1.f / (*cost)[0];
    int k = 0;
    for (int j = 1; j <= n && j < int(cost->size()); j++) {
      s *= sigm(conf[j - 1]);
      e += s;
      if (e / (*cost)[j] > best) best = e / (*cost)[j], k = j;
    }
    return k;
  }
  if (gate <= 0.f || int(conf.size()) < n) return n;
  float s = 1.f;
  int k = 0;
  while (k < n) {
    s *= sigm(conf[k]);
    if (s < gate) break;
    k++;
  }
  return k;
}

struct Engine::Dsp {
  // weights
  F8Lin fproj, ckv;  // feature_projection [4096, 12288]; the stages' linear_kv_proj [3 × 512, 4096] (context KV)
  bf16* fnorm = nullptr;
  std::vector<std::unique_ptr<Layer>> st;
  HcHead head;
  bf16* norm = nullptr;
  bf16* w1t = nullptr;   // markov_w1ᵀ [V, 256]
  bf16* w2 = nullptr;    // markov_w2 rows of this rank's vocabulary half [Vh, 256]
  float* cw = nullptr;   // confidence_head [4096 + 256] fp32
  Lin lm8, lm4;          // P080 / P081: MXFP8 / NVFP4 copies of this rank's lm_head half (draft logits only)
  uint8_t *w2q = nullptr, *w2s = nullptr;  // P082: markov_w2 as e4m3 [Vh, 256] + UE8M0 [Vh, 8]
  float* inv = nullptr;  // the draft's RoPE
  uint8_t* ring = nullptr;  // [3][slots][256][584]
  // context rows of a step
  int maxK = 0, K = 0;
  bf16* aux = nullptr;          // [maxK, 12288]
  uint8_t *xq = nullptr, *xs = nullptr;
  int *d_rows = nullptr, *d_pos = nullptr, *d_slot = nullptr;
  // verify rollback
  DspRoll roll{};
  int* d_cross = nullptr;
  int2* d_info = nullptr;
  bool cross_any = false, have_verify = false;
  std::vector<int> v_slot, v_p0, v_n, v_cross;
  int v_M = 0;
  // drafting
  int maxD = 0;                  // draft rows of one pass (≤ logit_rows)
  float *qh = nullptr, *q2 = nullptr, *qf = nullptr;  // this half [maxD, Vh] · gathered [2, maxD, Vh] · full [maxD, V]
  float* qs = nullptr;           // per slot: the pending drafts' q rows [slots][16][V]
  float2* part = nullptr;
  SparseCand *cands = nullptr, *oc = nullptr, *oc2 = nullptr;  // P083: [maxD][65] · [16][64] · [2][16][64]
  float* exch = nullptr;         // [2][16][2]
  float* exs = nullptr;          // P130: per Markov step [16][2][16][2]
  unsigned* mcnt = nullptr;      // P130: per Markov step, the sweep's CTA counter
  int *prev = nullptr, *dtok = nullptr, *last = nullptr;  // [16] · [16][16] · [16]
  int *it = nullptr;             // items: seq, brow, orow [3][maxD]
  int *rseq = nullptr, *ridx = nullptr;  // [maxD]
  float* conf = nullptr;         // [maxD]
  float *s_temp = nullptr, *s_topp = nullptr, *s_prob = nullptr;
  int *s_topk = nullptr, *s_tok = nullptr;
  uint64_t *s_seed = nullptr, *s_ctr = nullptr;
  // accept
  float* qbuf = nullptr;         // [maxD, V] the verified drafts' q rows in batch order
  int *a_pst = nullptr, *a_qst = nullptr, *a_dr = nullptr, *a_nacc = nullptr, *a_next = nullptr, *a_ntok = nullptr;
  float *a_temp = nullptr, *a_topp = nullptr, *a_minp = nullptr;
  int* a_topk = nullptr;
  uint64_t *a_seed = nullptr, *a_ctr = nullptr;
  PenSeq* a_pen = nullptr;
  // P160 ctx_side: a round's verify runs its context rows on cs (forked after the logits: ca), joined (cb) in the next
  // draft pass — under the accept, its host round trip, the rollback and the pass's start
  cudaStream_t cs = nullptr;
  cudaEvent_t ca = nullptr, cb = nullptr;
  bool cok = false, cpend = false;
  // stats
  SpecStats stats;
  cudaEvent_t ev[6];
  bool ecount = false;
  int* ecnt = nullptr;  // [43]
  long esteps = 0;
};

namespace {
// P160: s waits (on the GPU) for the context rows a round's verify ran on the side stream (D: Engine::Dsp)
template <class D>
void dsp_join(D& d, cudaStream_t s) {
  if (!d.cpend) return;
  CUDA_CHECK(cudaStreamWaitEvent(s, d.cb, 0));
  d.cpend = false;
}
}  // namespace

// every device buffer of the draft (the engine's destructor, after its streams are idle; P078 views are not owned)
void Engine::dsp_delete(Dsp* d) {
  if (!d) return;
  for (auto& l : d->st) {
    mhc_free(l->ha), mhc_free(l->hf);
    auto& h = l->h;
    for (F8Lin* f : {&h.qkv, &h.qup, &h.gate, &h.woa, &h.oproj}) f8_free(*f);
    lin_free(h.comp);
    dfree(h.qn, h.kvn, h.cnw, h.ape, h.sink, h.U, h.V);
    auto& f = l->f;
    for (F8Lin* x : {&f.gu, &f.down, &f.fc1, &f.fc2}) f8_free(*x);
    lin_free(f.router);
    dfree(f.bias, f.rscale, f.lnorm);
    moe_free(f.ex), f.ex = nullptr;
  }
  f8_free(d->fproj), f8_free(d->ckv);
  lin_free(d->lm8), lin_free(d->lm4);
  dfree(d->fnorm, d->head.fn, d->head.sb, d->norm, d->w1t, d->w2, d->cw, d->w2q, d->w2s, d->inv, d->ring);
  dfree(d->aux, d->xq, d->xs, d->d_rows, d->d_pos, d->d_slot);
  dfree(d->roll.ring, d->roll.state, d->roll.ape, d->roll.bk_ring, d->roll.bk_state, d->roll.stash);
  dfree(d->d_cross, d->d_info, d->qh, d->q2, d->qf, d->qs, d->part, d->cands, d->oc, d->oc2, d->exch, d->exs, d->mcnt);
  dfree(d->prev, d->dtok, d->last, d->it, d->rseq, d->ridx, d->conf);
  dfree(d->s_temp, d->s_topp, d->s_prob, d->s_topk, d->s_tok, d->s_seed, d->s_ctr, d->qbuf);
  dfree(d->a_pst, d->a_qst, d->a_dr, d->a_nacc, d->a_next, d->a_ntok, d->a_temp, d->a_topp, d->a_minp, d->a_topk);
  dfree(d->a_seed, d->a_ctr, d->a_pen, d->ecnt);
  for (auto& e : d->ev) cudaEventDestroy(e);
  if (d->cs) cudaStreamSynchronize(d->cs), cudaStreamDestroy(d->cs);
  if (d->ca) cudaEventDestroy(d->ca);
  if (d->cb) cudaEventDestroy(d->cb);
  delete d;
}

void Engine::dspark_init() {
  if (dsp_) return;
  const auto t0 = std::chrono::steady_clock::now();
  const Weights& W = *w_;
  const int r = c_.rank;
  dsp_.reset(new Dsp);
  Dsp& d = *dsp_;
  d.fproj = f8_upload(W, {"model.mtp.feature_projection"});
  d.fnorm = up_bf16(W, "model.mtp.feature_norm.weight");
  std::vector<std::string> kvn;
  for (int i = 0; i < kDspStages; i++) {
    auto l = std::make_unique<Layer>();
    const std::string p = "model.mtp." + std::to_string(i) + ".", a = p + "self_attn.", m = p + "block_sparse_moe.";
    l->ha = mhc_upload(W, p + "self_attention_hyper_connection.", p + "input_layernorm.weight");
    l->hf = mhc_upload(W, p + "mlp_hyper_connection.", p + "post_attention_layernorm.weight");
    auto& h = l->h;
    h.qkv = f8_upload(W, {a + "linear_q_down_proj", a + "linear_kv_proj"});
    h.qup = f8_upload(W, {a + "linear_q_up_proj"});
    h.gate = f8_upload(W, {a + "gate_proj"});
    h.woa = f8_upload(W, {a + "linear_o_group_proj"}, 4);
    h.oproj = f8_upload(W, {a + "o_proj"});
    h.qn = up_bf16(W, a + "q_layernorm.weight");
    h.kvn = up_bf16(W, a + "kv_layernorm.weight");
    h.sink = up_f32(as_f32(W, a + "core_attention.attn_sink"));
    h.U = up_vha(W, a + "vha_postmix_U", r);
    h.V = up_vha(W, a + "vha_postmix_V", r);
    auto& f = l->f;
    f.router = lin_upload(W, {m + "gate.weight", m + "gate.weight_1"}, pack12_rc());  // P062 / P098 (model.cc)
    f.bias = up_f32(as_f32(W, m + "e_score_correction_bias"));
    f.rscale = up_f32(as_f32(W, m + "gate.routed_scaling_factor_param"));
    f.fc1 = f8_upload(W, {m + "fc1_latent_proj"});
    f.fc2 = f8_upload(W, {m + "fc2_latent_proj"});
    f.lnorm = up_bf16(W, m + "latent_norm.weight");
    f.gu = f8_upload(W, {m + "shared_experts.w1", m + "shared_experts.w3"});
    f.down = f8_upload(W, {m + "shared_experts.w2"});
    f.ex = moe_upload(W, m + "experts.");
    f.r0h = lin_rows(f.router, r * 256, 256), f.r1h = lin_rows(f.router, 512 + r * 256, 256);  // P078 views (ffn_a)
    f.fc1h = f8_rows(f.fc1, r * 1024, 1024), f.fc2h = f8_rows(f.fc2, r * 2048, 2048);
    kvn.push_back(a + "linear_kv_proj");
    d.st.push_back(std::move(l));
  }
  d.ckv = f8_upload(W, kvn);
  d.head = hc_head_upload(W, "model.mtp.mhc_contract.");
  d.norm = up_bf16(W, "model.mtp.norm.weight");
  {  // Markov head: w1 stored [256, V] → [V, 256]; w2 [V, 256] split by vocabulary rows (this rank's half)
    std::vector<int64_t> sh;
    auto w1 = W.slice("model.mtp.markov_head.markov_w1.weight", &sh);
    if (sh.size() != 2 || sh[0] != kDspRank || sh[1] != kVocab) qlog::fail("dspark", "markov_w1 shape");
    std::vector<uint16_t> t(size_t(kVocab) * kDspRank);
    const uint16_t* src = reinterpret_cast<const uint16_t*>(w1.data());
    for (int c = 0; c < kDspRank; c++)
      for (int v = 0; v < kVocab; v++) t[size_t(v) * kDspRank + c] = src[size_t(c) * kVocab + v];
    d.w1t = reinterpret_cast<bf16*>(up(t));
    auto w2 = W.slice("model.mtp.markov_head.markov_w2.weight", &sh);
    if (sh[0] != kVocabHalf || sh[1] != kDspRank) qlog::fail("dspark", "markov_w2 slice shape");
    d.w2 = dmalloc<bf16>(w2.size() / 2);
    CUDA_CHECK(cudaMemcpy(d.w2, w2.data(), w2.size(), cudaMemcpyHostToDevice));
    std::vector<uint8_t> q, sc;  // P082
    mxfp8(reinterpret_cast<const uint16_t*>(w2.data()), kVocabHalf, kDspRank, q, sc);
    d.w2q = up(q), d.w2s = up(sc);
  }
  if (patch::on(patch::draft_lm8) || patch::on(patch::draft_lm4)) {  // P080 / P081: draft copies of this rank's
    std::vector<int64_t> sh;                                             // lm_head half (only the switched-on ones)
    auto lw = W.slice("lm_head.weight", &sh);
    if (sh[0] != kVocabHalf || sh[1] != kHidden) qlog::fail("dspark", "lm_head slice shape");
    if (patch::on(patch::draft_lm8)) {
      std::vector<uint8_t> q, sc;
      mxfp8(reinterpret_cast<const uint16_t*>(lw.data()), kVocabHalf, kHidden, q, sc);
      d.lm8 = lin_make(WMat::MXFP8, kVocabHalf, kHidden, q.data(), sc.data());
    }
    if (patch::on(patch::draft_lm4)) d.lm4 = lin_make_nvfp4(kVocabHalf, kHidden, reinterpret_cast<const bf16*>(lw.data()));
  }
  {
    auto cw = as_f32(W, "model.mtp.confidence_head.projection.weight");
    if (cw.size() != size_t(kHidden + kDspRank)) qlog::fail("dspark", "confidence_head shape");
    d.cw = up_f32(cw);
  }
  d.inv = up_f32(inv_draft());
  w_->drop_cache();
  const int S = c_.slots;
  d.ring = dmalloc<uint8_t>(size_t(kDspStages) * S * kDspRing * kKvRowBytes);
  d.maxK = std::min(c_.max_rows, kDspCtx * S);
  d.aux = dmalloc<bf16>(size_t(d.maxK) * kDspAux);
  d.xq = dmalloc<uint8_t>(size_t(d.maxK) * kDspAux);
  d.xs = dmalloc<uint8_t>(size_t(d.maxK) * kDspAux / 32);
  d.d_rows = dmalloc<int>(d.maxK), d.d_pos = dmalloc<int>(d.maxK), d.d_slot = dmalloc<int>(d.maxK);
  // rollback: HCA layers in step order
  const int maxr = c_.logit_rows;
  std::vector<uint8_t*> rings;
  std::vector<float*> states;
  std::vector<const float*> apes;
  int il = 0;
  for (auto& l : L_)
    if (!l->dsa) rings.push_back(hc_[il].ring), states.push_back(hc_[il].state), apes.push_back(l->h.ape), il++;
  d.roll.nl = il, d.roll.maxr = maxr;
  d.roll.ring = up(rings);
  d.roll.state = up(states);
  d.roll.ape = up(apes);
  d.roll.bk_ring = dmalloc<uint8_t>(size_t(il) * maxr * kKvRowBytes);
  d.roll.bk_state = dmalloc<float>(size_t(il) * maxr * 1024);
  d.roll.stash = dmalloc<float>(size_t(il) * maxr * 1024);
  d.d_cross = dmalloc<int>(maxr);
  d.d_info = dmalloc<int2>(maxr);
  // drafting
  d.maxD = maxr;
  d.qh = dmalloc<float>(size_t(kDspMaxN) * kVocabHalf);
  d.q2 = dmalloc<float>(size_t(2) * kDspMaxN * kVocabHalf);
  d.qf = dmalloc<float>(size_t(kDspMaxN) * kVocab);
  d.qs = dmalloc<float>(size_t(S) * kDspMaxN * kVocab);
  d.part = dmalloc<float2>(size_t(kDspMaxN) * dsp_markov_blocks(kVocabHalf));
  d.cands = dmalloc<SparseCand>(size_t(maxr) * kSparseRec);
  d.oc = dmalloc<SparseCand>(size_t(kDspMaxN) * kSparseK);
  d.oc2 = dmalloc<SparseCand>(size_t(2) * kDspMaxN * kSparseK);
  d.exch = dmalloc<float>(2 * kDspMaxN * 2);
  d.exs = dmalloc<float>(size_t(kDspMaxN) * 2 * kDspMaxN * 2), d.mcnt = dmalloc<unsigned>(kDspMaxN);
  d.prev = dmalloc<int>(kDspMaxN), d.dtok = dmalloc<int>(kDspMaxN * kDspMaxN), d.last = dmalloc<int>(kDspMaxN);
  d.it = dmalloc<int>(3 * size_t(maxr));
  d.rseq = dmalloc<int>(maxr), d.ridx = dmalloc<int>(maxr);
  d.conf = dmalloc<float>(maxr);
  d.s_temp = dmalloc<float>(maxr), d.s_topp = dmalloc<float>(maxr), d.s_prob = dmalloc<float>(maxr);
  d.s_topk = dmalloc<int>(maxr), d.s_tok = dmalloc<int>(maxr);
  d.s_seed = dmalloc<uint64_t>(maxr), d.s_ctr = dmalloc<uint64_t>(maxr);
  d.qbuf = dmalloc<float>(size_t(maxr) * kVocab);
  d.a_pst = dmalloc<int>(maxr + 1), d.a_qst = dmalloc<int>(maxr + 1), d.a_dr = dmalloc<int>(maxr);
  d.a_nacc = dmalloc<int>(maxr), d.a_next = dmalloc<int>(maxr), d.a_ntok = dmalloc<int>(maxr);
  d.a_temp = dmalloc<float>(maxr), d.a_topp = dmalloc<float>(maxr), d.a_minp = dmalloc<float>(maxr);
  d.a_topk = dmalloc<int>(maxr);
  d.a_seed = dmalloc<uint64_t>(maxr), d.a_ctr = dmalloc<uint64_t>(maxr);
  d.a_pen = dmalloc<PenSeq>(maxr);
  for (auto& e : d.ev) CUDA_CHECK(cudaEventCreate(&e));
  CUDA_CHECK(cudaStreamCreateWithFlags(&d.cs, cudaStreamNonBlocking));  // P160
  CUDA_CHECK(cudaEventCreateWithFlags(&d.ca, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreateWithFlags(&d.cb, cudaEventDisableTiming));
  d.ecnt = dmalloc<int>(kLayers);
  // GEMM workspace for the draft's shapes
  size_t fws = fws_bytes_;
  fws = std::max(fws, f8_ws(d.fproj, d.maxK));
  fws = std::max(fws, f8_ws(d.ckv, d.maxK));
  for (auto& l : d.st)
    for (const F8Lin* f : {&l->h.qkv, &l->h.qup, &l->h.gate, &l->h.woa, &l->h.oproj, &l->f.gu, &l->f.down, &l->f.fc1,
                           &l->f.fc2})
      fws = std::max(fws, f8_ws(*f, maxr));
  if (fws > fws_bytes_) {
    CUDA_CHECK(cudaFree(fws_));
    fws_bytes_ = fws;
    fws_ = dmalloc<float>(fws / 4 + 1);
  }
  malloc_trim(0);  // the loaders' freed host buffers back to the system
  CUDA_CHECK(cudaDeviceSynchronize());
  size_t fr, tot;
  cudaMemGetInfo(&fr, &tot);
  qlog::ok("dspark", "rank%d · draft loaded (3 stages, Markov + confidence heads) · %.0f s · %.1f GiB free", r,
           std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), fr / 1073741824.0);
}

// ---- hooks of step() -------------------------------------------------------------------------------------------------
void Engine::dsp_step_begin(const std::vector<RowMeta>& meta, const std::vector<int>& slot,
                            const std::vector<int>& seq_start) {
  Dsp& d = *dsp_;
  const int B = int(slot.size()), M = int(meta.size());
  std::vector<int> rows, pos, sl;
  for (int b = 0; b < B; b++)
    for (int r = std::max(seq_start[b], seq_start[b + 1] - kDspCtx); r < seq_start[b + 1]; r++)
      rows.push_back(r), pos.push_back(meta[r].pos), sl.push_back(slot[b]);
  d.K = int(rows.size());
  if (d.K > d.maxK) qlog::fail("dspark", "%d context rows > %d", d.K, d.maxK);
  if (d.K) {
    CUDA_CHECK(cudaMemcpyAsync(d.d_rows, rows.data(), d.K * 4, cudaMemcpyHostToDevice, s_));
    CUDA_CHECK(cudaMemcpyAsync(d.d_pos, pos.data(), d.K * 4, cudaMemcpyHostToDevice, s_));
    CUDA_CHECK(cudaMemcpyAsync(d.d_slot, sl.data(), d.K * 4, cudaMemcpyHostToDevice, s_));
  }
  if (sp_on_ && d.K) {  // P094 sp_resid: each context row's owner (its aux rows are exchanged in dsp_step_end)
    std::vector<int> own(d.K);
    for (int i = 0; i < d.K; i++) own[i] = sp_owner(rows[i]);
    CUDA_CHECK(cudaMemcpyAsync(d_own_ + c_.logit_rows, own.data(), d.K * 4, cudaMemcpyHostToDevice, s_));
  }
  if (!verify_) return;
  if (M > d.roll.maxr) qlog::fail("dspark", "verify step of %d rows > %d", M, d.roll.maxr);
  d.v_slot = slot, d.v_p0.assign(B, 0), d.v_n.assign(B, 0), d.v_cross.assign(B, -1), d.v_M = M;
  d.cross_any = false;
  std::vector<int> cross(M);
  for (int b = 0; b < B; b++) {
    const int r0 = seq_start[b], n = seq_start[b + 1] - r0, p0 = meta[r0].pos, pend = p0 + n - 1;
    if (n > kWindow) qlog::fail("dspark", "verify block of %d rows", n);
    const int c = (pend + 1) / kRatio * kRatio - 1;  // the last group-closing position ≤ pend
    d.v_p0[b] = p0, d.v_n[b] = n, d.v_cross[b] = c >= p0 ? c : -1;
    d.cross_any |= c >= p0;
    for (int i = 0; i < n; i++) cross[r0 + i] = d.v_cross[b];
  }
  CUDA_CHECK(cudaMemcpyAsync(d.d_cross, cross.data(), M * 4, cudaMemcpyHostToDevice, s_));
  dsp_backup(d_meta_, d.d_cross, M, d.roll, s_);
  d.have_verify = true;
}

void Engine::dsp_stash(int il, int o, int M) {  // rows [o, o + M) of the step (a verify: one lane, o = 0)
  Dsp& d = *dsp_;
  if (!d.cross_any) return;
  if (o + M > d.roll.maxr) qlog::fail("dspark", "stash rows %d + %d > %d", o, M, d.roll.maxr);
  CUDA_CHECK(cudaMemcpyAsync(d.roll.stash + (size_t(il) * d.roll.maxr + o) * 1024, ckv_ + size_t(o) * 1024,
                             size_t(M) * 1024 * 4, cudaMemcpyDeviceToDevice, s_));
}

void Engine::dsp_after_layer(int L, int M) {
  Dsp& d = *dsp_;
  if (L >= 40 && L <= 42 && d.K) dsp_aux(R_, d.d_rows, d.K, L - 40, d.aux, s_);
  (void)M;
}

void Engine::dsp_count(int L, int M) {
  Dsp& d = *dsp_;
  if (d.ecount && L >= 1) dsp_experts(d_moe_ids_, M * kTopK, d.ecnt + L, s_);
}

void Engine::dsp_step_end(int M) {
  Dsp& d = *dsp_;
  if (d.ecount) d.esteps++;
  if (verify_) CUDA_CHECK(cudaEventRecord(d.ev[5], s_));
  if (verify_) dmark(g_rp, s_, "v.step");
  if (d.K && sp_on_)  // P094: the aux rows (stream means after layers 40–42) of each context row from its owner
    for (int j = 0; j < 3; j++) sp_rows(d.aux + j * kHidden, kDspAux * 2, kHidden * 2, d_own_ + c_.logit_rows, d.K);
  if (d.K && d.cok && verify_ && !sp_on_) {  // P160 ctx_side: on cs (the same calls; dsp_context launches on s_)
    CUDA_CHECK(cudaEventRecord(d.ca, s_));
    CUDA_CHECK(cudaStreamWaitEvent(d.cs, d.ca, 0));
    std::swap(s_, d.cs);
    dsp_context(d.K);
    std::swap(s_, d.cs);
    CUDA_CHECK(cudaEventRecord(d.cb, d.cs));
    d.cpend = true;
  } else if (d.K) {
    dsp_context(d.K);
  }
  if (verify_) dmark(g_rp, s_, "v.ctx");
  (void)M;
}

void Engine::dsp_context(int K) {
  Dsp& d = *dsp_;
  F8Act a{d.xq, d.xs, kDspAux};
  act_quant(d.aux, kDspAux, K, kDspAux, 128, a, s_);
  f8_forward(d.fproj, a, K, t1_, kHidden, fws_, s_);
  rmsnorm(t1_, kHidden, d.fnorm, K, kHidden, t2_, kHidden, s_);  // main_x
  F8Act b = act(t2_, kHidden, K, kHidden, 128);
  f8_forward(d.ckv, b, K, t3_, 3 * kHeadDim, fws_, s_);
  for (int i = 0; i < kDspStages; i++)
    rmsnorm(t3_ + i * kHeadDim, 3 * kHeadDim, d.st[i]->h.kvn, K, kHeadDim, t4_ + i * kHeadDim, 3 * kHeadDim, s_);
  dsp_ctx_rows(t4_, 3 * kHeadDim, d.d_pos, d.d_slot, K, d.inv, d.ring, c_.slots, s_);
}

// P146 prefix_ckpt: the context rows a draft anchored at pos + 1 … reads first — positions pos − 128 … pos − 1, ring
// rows (pos + 128) % 256 … (+ 128, wrapping) — of each stage ⇄ buf [3][128][584] (stream-ordered on s_)
void Engine::dsp_ckpt(int slot, long pos, uint8_t* buf, bool save) {
  Dsp& d = *dsp_;
  const size_t row = kKvRowBytes, stage = size_t(c_.slots) * kDspRing * row;
  const int a = int((pos + kDspCtx) % kDspRing), n1 = std::min(kDspCtx, kDspRing - a);
  for (int i = 0; i < kDspStages; i++) {
    uint8_t* r = d.ring + i * stage + size_t(slot) * kDspRing * row;
    uint8_t* b = buf + size_t(i) * kDspCtx * row;
    const std::pair<size_t, size_t> piece[2] = {{size_t(a), size_t(n1)}, {0, size_t(kDspCtx - n1)}};
    size_t o = 0;
    for (const auto& [at, n] : piece) {
      if (n) CUDA_CHECK(cudaMemcpyAsync(save ? b + o * row : r + at * row, save ? r + at * row : b + o * row, n * row,
                                        cudaMemcpyDeviceToDevice, s_));
      o += n;
    }
  }
}

void Engine::dspark_ctx(int slot, const bf16* aux, const std::vector<int>& pos) {
  Dsp& d = *dsp_;
  const int K = int(pos.size());
  if (K > d.maxK) qlog::fail("dspark", "%d context rows > %d", K, d.maxK);
  std::vector<int> sl(K, slot);
  CUDA_CHECK(cudaMemcpyAsync(d.aux, aux, size_t(K) * kDspAux * 2, cudaMemcpyDeviceToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(d.d_pos, pos.data(), K * 4, cudaMemcpyHostToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(d.d_slot, sl.data(), K * 4, cudaMemcpyHostToDevice, s_));
  dsp_context(K);
  CUDA_CHECK(cudaStreamSynchronize(s_));
}

// ---- the draft forward -----------------------------------------------------------------------------------------------
void Engine::dsp_forward(const std::vector<SpecSeq*>& seqs, int M) {
  Dsp& d = *dsp_;
  const size_t ring_stage = size_t(c_.slots) * kDspRing * kKvRowBytes;
  embed(M);  // (P095: the rows exchanged)
  dmark(s_, "embed");
  for (int i = 0; i < kDspStages; i++) {
    Layer& l = *d.st[i];
    auto& h = l.h;
    if (i == 0) mhc_pre(l.ha, R_, M, mix_, post_, comb_, inp_, s_);  // on the repeated embedding itself (vLLM)
    if (i == 0) dsp_join(d, s_);  // P160: the verify's context rows (ring rows read below; t1_ … t4_, xq_ reused)
    if (i == 0) dmark(s_, "mhc");
    F8Act xa = act(inp_, kHidden, M, kHidden, 128);
    f8_forward(h.qkv, xa, M, t1_, 1536, fws_, s_);
    dmark(s_, "a.qkv");
    // P129 draft_attn: P060's fused calls as the target's attn_hca_a / _b (q_norm as q_up + gate's activation op,
    // kv_norm as its side RMSNorm, σ as gate's epilogue, z ⊙ gate as o_proj's activation op), and q prep + the block's
    // KV rows + attention + VHA partial in one launch — the same bytes as the base calls below
    const bool fa = patch::on(patch::draft_attn), fz = fa && f8_fuse_ok(M);
    if (fz) {
      const F8Aux kv{t1_ + kQLora, 1536, h.kvn, kHeadDim, t4_, kHeadDim};
      f8_forward_multi({{&h.qup, q_, 16384}, {&h.gate, t3_, 4096, kEpiSigmoid}},
                       f8_pro(kProRms, t1_, 1536, h.qn, 0, xq_, xs_, kQLora), M, fws_, s_, &kv);
    } else {
      rmsnorm(t1_, 1536, h.qn, M, kQLora, t2_, kQLora, s_);
      rmsnorm(t1_ + kQLora, 1536, h.kvn, M, kHeadDim, t4_, kHeadDim, s_);
      F8Act qa = act(t2_, kQLora, M, kQLora, 128);
      f8_forward_multi({{&h.qup, q_, 16384}, {&h.gate, t3_, 4096}}, qa, M, fws_, s_);
    }
    dmark(s_, "a.qup");
    if (fa) {
      if (!fz) sigmoid_inplace(t3_, size_t(M) * 4096, s_);
      dsp_attn_fused(q_, t4_, kHeadDim, d_meta_, d.ring + i * ring_stage, h.sink, d.inv, h.U, M, o_, zp_, s_);
      dmark(s_, "a.attn");
    } else {
      hca_q_prep(q_, d_meta_, d.inv, M, s_);
      sigmoid_inplace(t3_, size_t(M) * 4096, s_);
      hca_kv_rows(t4_, kHeadDim, d_meta_, d.inv, M, rows_, s_);
      dmark(s_, "a.prep");
      dsp_attn(q_, rows_, d_meta_, d.ring + i * ring_stage, h.sink, M, o_, s_);
      dmark(s_, "a.attn");
      vha_partial(o_, kHeadDim, h.U, M, zp_, s_);
      dmark(s_, "a.vha_p");
    }
    comm_->all_reduce_f32(zp_, size_t(M) * kVhaRank * kHeadDim, s_);
    dmark(s_, "a.ar_z");
    hca_vha_out(o_, zp_, h.V, d_meta_, d.inv, M, oq_, os_, s_);
    dmark(s_, "a.vha_o");
    F8Act ga{oq_, os_, 4096};
    f8_forward(h.woa, ga, M, t4_, 4096, fws_, s_, long(M) * 4096, long(M) * 32);
    dmark(s_, "a.woa");
    if (fz) {
      f8_forward_out({&h.oproj, aout_, kHidden}, f8_pro(kProMul, t4_, 4096, t3_, 4096, xq_, xs_, 4096), M, fws_, s_);
    } else {
      mul_bf16(t4_, 4096, t3_, 4096, M, 4096, t2_, 4096, s_);
      F8Act za = act(t2_, 4096, M, 4096, 128);
      f8_forward(h.oproj, za, M, aout_, kHidden, fws_, s_);
    }
    dmark(s_, "a.oproj");
    comm_->all_reduce_bf16(aout_, size_t(M) * kHidden, s_);
    dmark(s_, "a.ar_o");
    mhc_post_pre(l.hf, aout_, R_, M, mix_, post_, comb_, inp_, s_);
    dmark(s_, "mhc");
    const Lane all{0, M, 0, 0, 0};  // the target's FFN path on one lane (its per-M patches: P060, P066, P078 …)
    ffn_a(l, all, &all, 1, false, false, nullptr, nullptr);
    dmark(s_, "ffn_a");
    ffn_b(l, all, false);
    dmark(s_, "ffn_b");
    if (i + 1 < kDspStages) mhc_post_pre(d.st[i + 1]->ha, fout_, R_, M, mix_, post_, comb_, inp_, s_);
    else mhc_post(fout_, R_, post_, comb_, M, R_, s_);
    dmark(s_, "mhc");
  }
  std::vector<int> rows(M);
  for (int i = 0; i < M; i++) rows[i] = i;
  CUDA_CHECK(cudaMemcpyAsync(d_rows_, rows.data(), M * 4, cudaMemcpyHostToDevice, s_));
  hc_head(d.head, R_, d_rows_, M, hbuf_, s_);
  dmark(s_, "h.contract");
  rmsnorm(hbuf_, kHidden, d.norm, M, kHidden, t2_, kHidden, s_);
  dmark(s_, "h.norm");
  const bool l4 = patch::on(patch::draft_lm4), l8 = !l4 && patch::on(patch::draft_lm8);
  if ((l4 && !d.lm4.w) || (l8 && !d.lm8.w))
    qlog::fail("dspark", "P08%d %s switched on after dspark_init (its copy was not built)", l4 ? 1 : 0,
               l4 ? "draft_lm4" : "draft_lm8");
  const Lin& lm = l4 ? d.lm4 : l8 ? d.lm8 : lm_;
  lin_forward_f32(lm, t2_, lg_half_, M, s_);  // base logits of this rank's vocabulary half [M, Vh]
  dmark(s_, "h.lm");
  (void)seqs;
}

void Engine::spec_draft(const std::vector<SpecSeq*>& seqs) {
  Dsp& d = *dsp_;
  const int B = int(seqs.size());
  if (B < 1 || B > kDspMaxN) qlog::fail("dspark", "spec_draft: %d sequences", B);
  std::vector<int> ids, rs, ri, r0(B), lasts(B);
  std::vector<RowMeta> meta;
  int maxN = 0;
  bool greedy = true;
  for (int b = 0; b < B; b++) {
    const SpecSeq& q = *seqs[b];
    const int N = q.depth;
    if (N < 1 || N >= kDspMaxN) qlog::fail("dspark", "draft depth %d", N);
    if (hist_.at(q.slot) != q.pos) qlog::fail("dspark", "slot %d: anchor at %d, history %d", q.slot, q.pos, hist_[q.slot]);
    if (q.pos + N > c_.ctx) qlog::fail("dspark", "slot %d: draft past the context", q.slot);
    r0[b] = int(ids.size());
    for (int i = 0; i < N; i++) {
      ids.push_back(i ? kDspNoise : q.last);
      meta.push_back(RowMeta{q.pos + i, q.slot, r0[b], q.pos, q.pos + N - 1});
      rs.push_back(b), ri.push_back(i);
    }
    lasts[b] = q.last;
    maxN = std::max(maxN, N);
    greedy &= !(q.temp > 0.f);
  }
  const int M = int(ids.size());
  if (M > d.maxD || M > c_.logit_rows) qlog::fail("dspark", "%d draft rows > %d", M, std::min(d.maxD, c_.logit_rows));
  CUDA_CHECK(cudaEventRecord(d.ev[3], s_));
  dprof_begin(s_);
  CUDA_CHECK(cudaMemcpyAsync(d_ids_, ids.data(), M * 4, cudaMemcpyHostToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(d_meta_, meta.data(), M * sizeof(RowMeta), cudaMemcpyHostToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(d.rseq, rs.data(), M * 4, cudaMemcpyHostToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(d.ridx, ri.data(), M * 4, cudaMemcpyHostToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(d.last, lasts.data(), B * 4, cudaMemcpyHostToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(d.prev, lasts.data(), B * 4, cudaMemcpyHostToDevice, s_));
  dmark(s_, "h2d");
  dsp_forward(seqs, M);
  // the Markov chain, step i over the sequences with depth > i (items in batch order)
  std::vector<int> item, off(maxN + 1, 0);
  std::vector<float> tmp, tpp;
  std::vector<int> tpk;
  std::vector<uint64_t> sd, ct;
  for (int i = 0; i < maxN; i++) {
    off[i] = int(item.size()) / 3;
    std::vector<int> sq, br, orw;
    for (int b = 0; b < B; b++)
      if (seqs[b]->depth > i) {
        const SpecSeq& q = *seqs[b];
        sq.push_back(b), br.push_back(r0[b] + i), orw.push_back(int(sq.size()) - 1);
        tmp.push_back(q.temp), tpp.push_back(q.top_p), tpk.push_back(q.top_k), sd.push_back(q.seed);
        ct.push_back((uint64_t(1) << 63) | (uint64_t(q.pos) << 4) | uint64_t(i));  // fresh per round and draft
      }
    for (size_t k = 0; k < sq.size(); k++) item.push_back(sq[k]);
    for (size_t k = 0; k < sq.size(); k++) item.push_back(br[k]);
    for (size_t k = 0; k < sq.size(); k++) item.push_back(orw[k]);
  }
  off[maxN] = int(item.size()) / 3;
  const int NI = off[maxN];
  // layout per step: [seq K][brow K][orow K] at 3·off[i]
  CUDA_CHECK(cudaMemcpyAsync(d.it, item.data(), item.size() * 4, cudaMemcpyHostToDevice, s_));
  if (!greedy) {
    CUDA_CHECK(cudaMemcpyAsync(d.s_temp, tmp.data(), NI * 4, cudaMemcpyHostToDevice, s_));
    CUDA_CHECK(cudaMemcpyAsync(d.s_topp, tpp.data(), NI * 4, cudaMemcpyHostToDevice, s_));
    CUDA_CHECK(cudaMemcpyAsync(d.s_topk, tpk.data(), NI * 4, cudaMemcpyHostToDevice, s_));
    CUDA_CHECK(cudaMemcpyAsync(d.s_seed, sd.data(), NI * 8, cudaMemcpyHostToDevice, s_));
    CUDA_CHECK(cudaMemcpyAsync(d.s_ctr, ct.data(), NI * 8, cudaMemcpyHostToDevice, s_));
  }
  const int nb = dsp_markov_blocks(kVocabHalf), off_v = c_.rank * kVocabHalf;
  const bool topk = patch::on(patch::markov_topk);
  if (topk) sample_cands(lg_half_, M, kVocabHalf, off_v, d.cands, s_);
  dmark(s_, "h2d");
  // P130 draft_chain (greedy, P082's w2 copy): a step = one launch (the previous step's pick, the sweep, the argmax)
  // + the all-reduce of its candidates; the last step's pick after the loop — the same drafts as the loop below
  const bool chain = greedy && !topk && patch::on(patch::draft_chain) && patch::on(patch::markov8);
  // sampled (P161 / P162 / P163, below): the sweep direction, the in-place q rows (P076's row gather over RDMA), the
  // pipelined sweep; qslot[i]: the slot of step i's first item
  const bool back = patch::on(patch::markov_back);
  const bool qrows = patch::on(patch::draft_qrows) && patch::on(patch::rdma_gather) && comm_->rdma();
  const bool pipe = qrows && patch::on(patch::markov_pipe);
  std::vector<int> qslot(maxN, -1);
  for (int i = 0; i < maxN; i++)
    for (int b = 0; b < B && qslot[i] < 0; b++)
      if (seqs[b]->depth > i) qslot[i] = seqs[b]->slot;
  for (int i = 0; chain && i < maxN; i++) {
    const int K = off[i + 1] - off[i], pK = i ? off[i] - off[i - 1] : 0;
    const int* sq = d.it + 3 * off[i];
    float* ex = d.exs + size_t(i) * 4 * kDspMaxN;
    dsp_chain(lg_half_, sq + K, sq, K, i ? d.it + 3 * off[i - 1] : nullptr, pK, i ? ex - 4 * kDspMaxN : nullptr, d.last,
              d.dtok, i, d.w1t, d.w2q, d.w2s, kVocabHalf, off_v, mm_, d.part, d.mcnt + i, c_.rank, ex, s_);
    dmark(s_, "m.markov");
    comm_->all_reduce_f32(ex, size_t(4) * K, s_);
    dmark(s_, "m.ar");
    if (i + 1 == maxN) {
      dsp_pick(ex, K, sq, i, d.prev, d.dtok, s_);
      dmark(s_, "m.pick");
    }
  }
  for (int i = 0; !chain && i < maxN; i++) {
    const int K = off[i + 1] - off[i];
    const int* sq = d.it + 3 * off[i];
    const int *br = sq + K, *orw = sq + 2 * K;
    if (topk) {  // P083: the bias on each half's top-64 base logits only
      dsp_markov_topk(d.cands, br, sq, d.prev, d.w1t, d.w2, off_v, mm_, K, c_.rank, greedy ? d.exch : nullptr,
                      greedy ? nullptr : d.oc, s_);
      if (greedy) {
        comm_->all_reduce_f32(d.exch, size_t(4) * K, s_);
        dsp_pick(d.exch, K, sq, i, d.prev, d.dtok, s_);
        continue;
      }
      comm_->all_gather_f32(reinterpret_cast<const float*>(d.oc), reinterpret_cast<float*>(d.oc2),
                            size_t(K) * kSparseK * 2, s_);
      dsp_cand_rows(d.oc2, K, d.qf, s_);
    } else if (!greedy && qrows) {
      // P162 draft_qrows: this rank's q half lands in place in the full rows (P076's row gather) — in the slot's q store
      // when the step has one item (sampled there; nothing copied), else in qf as before. P161 markov_back: odd steps
      // sweep the vocabulary backwards (they start on the w2 rows the previous sweep read last, still in L2). P163
      // markov_pipe: the sweep in np launches, each part's gather on the comm stream under the next part's sweep.
      const bool m8 = patch::on(patch::markov8), bk = back && (i & 1);
      float* dst = K == 1 ? d.qs + (size_t(qslot[i]) * kDspMaxN + i) * kVocab : d.qf;
      const int np = pipe ? 2 : 1;  // (one GPU, 7-step chains: 2 parts cost +9 / +15 µs a step at 1 / 4 items, 4 parts +25 / +52)
      int tk[2];
      for (int c = 0; c < np; c++) {
        const int g0 = nb * c / np, g1 = nb * (c + 1) / np;  // sweep CTAs [g0, g1) → vocabulary blocks
        dsp_markov(lg_half_, br, sq, d.prev, d.w1t, d.w2, m8 ? d.w2q : nullptr, m8 ? d.w2s : nullptr, kVocabHalf,
                   off_v, mm_, K, d.qh, orw, d.part, s_, g0, g1 - g0, bk);
        const int v0 = (bk ? nb - g1 : g0) * dsp_markov_rows(), v1 = std::min(kVocabHalf, (bk ? nb - g0 : g1) * dsp_markov_rows());
        const AgPiece pc{d.qh + v0, dst, 4LL * kVocabHalf, 4LL * kVocab, 4LL * v0, 4LL * kVocabHalf, 4 * (v1 - v0)};
        if (np > 1) tk[c] = comm_->all_gather_rows_async(&pc, 1, K, s_);
        else comm_->all_gather_rows(&pc, 1, K, s_);
      }
      for (int c = 0; np > 1 && c < np; c++) comm_->join(tk[c], s_);
      dmark(s_, "m.markov+gather");
      const int o = off[i];
      sample_rows(dst, K, kVocab, d.s_temp + o, d.s_topk + o, d.s_topp + o, nullptr, d.s_seed + o, d.s_ctr + o, d.s_tok,
                  d.s_prob, s_);
      dmark(s_, "m.sample");
      dsp_put(d.s_tok, K, sq, i, d.prev, d.dtok, s_);
      for (int b = 0, k = 0; K > 1 && b < B; b++)
        if (seqs[b]->depth > i) {  // the q row the draft came from, kept for the accept test
          CUDA_CHECK(cudaMemcpyAsync(d.qs + (size_t(seqs[b]->slot) * kDspMaxN + i) * kVocab,
                                     d.qf + size_t(k) * kVocab, size_t(kVocab) * 4, cudaMemcpyDeviceToDevice, s_));
          k++;
        }
      dmark(s_, "m.qcopy");
      continue;
    } else {
      const bool m8 = patch::on(patch::markov8);
      dsp_markov(lg_half_, br, sq, d.prev, d.w1t, d.w2, m8 ? d.w2q : nullptr, m8 ? d.w2s : nullptr, kVocabHalf,
                 off_v, mm_, K, greedy ? nullptr : d.qh, orw, d.part, s_, 0, -1, !greedy && back && (i & 1));
      dmark(s_, "m.markov");
      if (greedy) {
        dsp_argmax(d.part, nb, K, c_.rank, d.exch, s_);
        dmark(s_, "m.argmax");
        comm_->all_reduce_f32(d.exch, size_t(4) * K, s_);
        dmark(s_, "m.ar");
        dsp_pick(d.exch, K, sq, i, d.prev, d.dtok, s_);
        dmark(s_, "m.pick");
        continue;
      }
      comm_->all_gather_f32(d.qh, d.q2, size_t(K) * kVocabHalf, s_);
      dmark(s_, "m.gather");
      halves_to_rows(d.q2, K, kVocabHalf, d.qf, s_);
      dmark(s_, "m.h2r");
    }
    {
      const int o = off[i];
      sample_rows(d.qf, K, kVocab, d.s_temp + o, d.s_topk + o, d.s_topp + o, nullptr, d.s_seed + o, d.s_ctr + o,
                  d.s_tok, d.s_prob, s_);
      dmark(s_, "m.sample");
      dsp_put(d.s_tok, K, sq, i, d.prev, d.dtok, s_);
      int k = 0;
      for (int b = 0; b < B; b++)
        if (seqs[b]->depth > i) {  // the q row the draft came from, kept for the accept test
          CUDA_CHECK(cudaMemcpyAsync(d.qs + (size_t(seqs[b]->slot) * kDspMaxN + i) * kVocab,
                                     d.qf + size_t(k) * kVocab, size_t(kVocab) * 4, cudaMemcpyDeviceToDevice, s_));
          k++;
        }
      dmark(s_, "m.qcopy");
    }
  }
  dsp_conf(hbuf_, d.rseq, d.ridx, d.last, d.dtok, d.w1t, d.cw, M, d.conf, s_);
  dmark(s_, "conf");
  std::vector<int> dt(size_t(kDspMaxN) * B);
  std::vector<float> cf(M);
  CUDA_CHECK(cudaMemcpyAsync(dt.data(), d.dtok, dt.size() * 4, cudaMemcpyDeviceToHost, s_));
  CUDA_CHECK(cudaMemcpyAsync(cf.data(), d.conf, M * 4, cudaMemcpyDeviceToHost, s_));
  dmark(s_, "d2h");
  CUDA_CHECK(cudaEventRecord(d.ev[4], s_));
  CUDA_CHECK(cudaStreamSynchronize(s_));
  dprof_end(c_.rank, M);
  for (int b = 0; b < B; b++) {
    SpecSeq& q = *seqs[b];
    q.drafts.assign(dt.begin() + b * kDspMaxN, dt.begin() + b * kDspMaxN + q.depth);
    q.conf.assign(cf.begin() + r0[b], cf.begin() + r0[b] + q.depth);
  }
}

// ---- first token, rounds -------------------------------------------------------------------------------------------
int Engine::spec_first(SpecSeq& q, int lrow) {
  Dsp& d = *dsp_;
  if (lrow < 0 || lrow >= last_rows_) qlog::fail("dspark", "spec_first: logits row %d of %d", lrow, last_rows_);
  q.pos = hist_.at(q.slot);
  float* row = lg_ + size_t(lrow) * kVocab;
  if (q.penal()) {
    PenSeq ps{q.pen, q.rep, q.freq, q.pres};
    const int st[2] = {0, 1};
    CUDA_CHECK(cudaMemcpyAsync(d.a_pen, &ps, sizeof ps, cudaMemcpyHostToDevice, s_));
    CUDA_CHECK(cudaMemcpyAsync(d.a_pst, st, 8, cudaMemcpyHostToDevice, s_));
    penalize_rows(row, kVocab, 0, kVocab, kVocab, 1, d.a_pst, nullptr, nullptr, 1, d.a_pen, s_);
  }
  const float t = q.temp, tp = q.top_p, mp = q.min_p;
  const uint64_t sd = q.seed, ct = uint64_t(q.pos);
  CUDA_CHECK(cudaMemcpyAsync(d.a_temp, &t, 4, cudaMemcpyHostToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(d.a_topp, &tp, 4, cudaMemcpyHostToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(d.a_minp, &mp, 4, cudaMemcpyHostToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(d.a_topk, &q.top_k, 4, cudaMemcpyHostToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(d.a_seed, &sd, 8, cudaMemcpyHostToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(d.a_ctr, &ct, 8, cudaMemcpyHostToDevice, s_));
  sample_rows(row, 1, kVocab, d.a_temp, d.a_topk, d.a_topp, d.a_minp, d.a_seed, d.a_ctr, d.a_next, d.s_prob, s_);
  int tok = 0;
  CUDA_CHECK(cudaMemcpyAsync(&tok, d.a_next, 4, cudaMemcpyDeviceToHost, s_));
  CUDA_CHECK(cudaStreamSynchronize(s_));
  q.last = tok;
  q.drafts.clear(), q.conf.clear();
  return tok;
}

void Engine::spec_commit(const std::vector<int>& keep) {
  Dsp& d = *dsp_;
  if (!d.have_verify) qlog::fail("dspark", "spec_commit without a verify step");
  const int B = int(d.v_slot.size());
  if (int(keep.size()) != B) qlog::fail("dspark", "spec_commit: %zu sequences, the verify had %d", keep.size(), B);
  std::vector<int2> info(d.v_M);
  int r = 0;
  bool any = false;
  for (int b = 0; b < B; b++) {
    if (keep[b] < 1 || keep[b] > d.v_n[b]) qlog::fail("dspark", "spec_commit: keep %d of %d rows", keep[b], d.v_n[b]);
    any |= keep[b] < d.v_n[b];
    for (int i = 0; i < d.v_n[b]; i++) info[r++] = make_int2(d.v_cross[b], d.v_p0[b] + keep[b]);
    hist_.at(d.v_slot[b]) = d.v_p0[b] + keep[b];
  }
  if (any) {
    CUDA_CHECK(cudaMemcpyAsync(d.d_info, info.data(), d.v_M * sizeof(int2), cudaMemcpyHostToDevice, s_));
    dsp_restore(d_meta_, d.d_info, d.v_M, d.roll, s_);
  }
  d.have_verify = false;
}

void Engine::spec_round_cost(const std::vector<float>& ms, int rmax) {
  if (ms.empty()) {
    rcost_.clear(), rcost_rmax_ = 0;
    return;
  }
  if (rmax < 1 || ms.size() % size_t(rmax + 1))
    qlog::fail("dspark", "spec_round_cost: %zu entries is not B × (rmax %d + 1)", ms.size(), rmax);
  // rank 0's numbers on both ranks (one small all-gather): the lengths shape the verify step, so a table that differs
  // in the last bit (a caller's measured ms) would part the ranks
  const size_t n = ms.size();
  std::vector<float> t(n);
  float* dv = dmalloc<float>(3 * n);
  CUDA_CHECK(cudaMemcpyAsync(dv, ms.data(), n * 4, cudaMemcpyHostToDevice, s_));
  comm_->all_gather_f32(dv, dv + n, n, s_);  // [rank 0 ‖ rank 1]
  CUDA_CHECK(cudaMemcpyAsync(t.data(), dv + n, n * 4, cudaMemcpyDeviceToHost, s_));
  CUDA_CHECK(cudaStreamSynchronize(s_));
  dfree(dv);
  const int nb = int(n) / (rmax + 1);
  for (int b = 1; b <= nb; b++)
    for (int r = b; r <= rmax; r++)
      if (!(t[size_t(b - 1) * (rmax + 1) + r] > 0.f)) qlog::fail("dspark", "spec_round_cost: ms(B %d, R %d) ≤ 0", b, r);
  rcost_ = t, rcost_rmax_ = rmax;
}

// The verify length (drafts) of each sequence of a round. P121 conf_gate off: every draft; on: SpecSeq::verify (q.gate)
// — and P143 verify_joint (a spec_round_cost table): the gate < 0 sequences' lengths chosen together, maximizing the
// round's expected emitted tokens over its ms: for every total R the most expected tokens take the largest products
// p_b,j = Π_{i ≤ j} σ(conf_b,i) (a sequence's products only fall, so its picks are a prefix: picking the largest next
// one is exact), then the R with the best tokens / ms(B, R) (the smallest on ties). The other sequences' rows and
// expected tokens count in it as they are. P122 cap_verify: no row past the cap (a round emits at most q.cap tokens;
// cap tokens need cap − 1 drafts). Host fp32 on the same confidences and table: the same lengths on both ranks.
std::vector<int> Engine::spec_lengths(const std::vector<SpecSeq*>& seqs) const {
  const int B = int(seqs.size()), W = rcost_rmax_ + 1;
  const bool gate = patch::on(patch::conf_gate), capv = patch::on(patch::cap_verify);
  const bool joint = gate && patch::on(patch::verify_joint) && rcost_rmax_ > 0 && size_t(B) * W <= rcost_.size();
  std::vector<int> k(B, 0), n(B, 0), jb;
  int R = B;
  float E = float(B);
  for (int b = 0; b < B; b++) {
    const SpecSeq& q = *seqs[b];
    n[b] = int(q.drafts.size());
    if (q.cap > 0 && capv) n[b] = std::max(0, std::min(n[b], q.cap - 1));
    if (joint && q.gate < 0.f && q.conf.size() >= q.drafts.size()) {
      jb.push_back(b);
      continue;
    }
    k[b] = std::min(gate ? q.verify(&spec_cost_) : int(q.drafts.size()), n[b]);
    R += k[b];
    float p = 1.f;
    for (int j = 0; j < k[b]; j++) p *= j < int(q.conf.size()) ? sigm(q.conf[j]) : 1.f, E += p;
  }
  if (jb.empty() || R > rcost_rmax_) return k;
  const float* ms = rcost_.data() + size_t(B - 1) * W;
  std::vector<float> p(jb.size(), 1.f);
  std::vector<int> nx(jb.size(), 0), pick;
  float best = E / ms[R];
  size_t bm = 0;
  while (R < rcost_rmax_) {
    int c = -1;
    float g = -1.f;
    for (size_t i = 0; i < jb.size(); i++)
      if (nx[i] < n[jb[i]]) {
        const float v = p[i] * sigm(seqs[jb[i]]->conf[nx[i]]);
        if (v > g) g = v, c = int(i);
      }
    if (c < 0) break;
    p[c] = g, nx[c]++, R++, E += g;
    pick.push_back(c);
    if (E / ms[R] > best) best = E / ms[R], bm = pick.size();
  }
  for (size_t i = 0; i < bm; i++) k[jb[pick[i]]]++;
  return k;
}

std::vector<std::vector<int>> Engine::spec_round(const std::vector<SpecSeq*>& seqs, bool draft_next) {
  Dsp& d = *dsp_;
  const int B = int(seqs.size());
  if (B < 1 || B > kDspMaxN) qlog::fail("dspark", "spec_round: %d sequences", B);
  std::vector<int> ids, pos, slot, ss{0}, qst{0}, dr;
  std::vector<float> tmp, tpp, mpp;
  std::vector<int> tpk;
  std::vector<uint64_t> sd, ct;
  std::vector<PenSeq> pen(B);
  bool sampled = false, penal = false;
  const std::vector<int> kv = spec_lengths(seqs);  // P121 / P122 / P143
  for (int b = 0; b < B; b++) {
    const SpecSeq& q = *seqs[b];
    const int k = kv[b];
    if (hist_.at(q.slot) != q.pos) qlog::fail("dspark", "slot %d: last at %d, history %d", q.slot, q.pos, hist_[q.slot]);
    for (int j = 0; j <= k; j++) ids.push_back(j ? q.drafts[j - 1] : q.last), pos.push_back(q.pos + j);
    for (int j = 0; j < k; j++) dr.push_back(q.drafts[j]);
    slot.push_back(q.slot);
    ss.push_back(int(ids.size()));
    qst.push_back(int(dr.size()));
    tmp.push_back(q.temp), tpp.push_back(q.top_p), mpp.push_back(q.min_p), tpk.push_back(q.top_k);
    sd.push_back(q.seed), ct.push_back(uint64_t(q.pos));
    pen[b] = PenSeq{q.penal() ? q.pen : PenSeen{nullptr, nullptr, nullptr}, q.rep, q.freq, q.pres};
    sampled |= q.temp > 0.f;
    penal |= q.penal();
  }
  const int M = int(ids.size()), ND = int(dr.size());
  if (M > c_.logit_rows) qlog::fail("dspark", "verify of %d rows > logit_rows %d", M, c_.logit_rows);
  if (pool_) {  // P125: the verify rows mapped before anything changes (KvFull: the round did not happen)
    std::vector<long> end(B);
    for (int b = 0; b < B; b++) end[b] = long(pos[ss[b + 1] - 1]) + 1;
    kv_ensure(slot, end);
  }
  CUDA_CHECK(cudaEventRecord(d.ev[0], s_));
  dprof_begin(g_rp, s_);
  if (penal) {  // the round's `last` counts as generated before its rows are penalised (a replaced last as replaced)
    std::vector<int> lasts(B), one(B + 1);
    for (int b = 0; b < B; b++) lasts[b] = seqs[b]->last;
    CUDA_CHECK(cudaMemcpyAsync(d.a_pen, pen.data(), B * sizeof(PenSeq), cudaMemcpyHostToDevice, s_));
    CUDA_CHECK(cudaMemcpyAsync(d.a_next, lasts.data(), B * 4, cudaMemcpyHostToDevice, s_));
    penalty_add(d.a_pen, B, nullptr, nullptr, nullptr, d.a_next, kVocab, s_);
  }
  verify_ = true;
  d.cok = patch::on(patch::ctx_side);  // P160: this round joins the side stream (the next draft pass, or below)
  step(ids, pos, slot, ss, true);
  verify_ = false, d.cok = false;
  CUDA_CHECK(cudaEventRecord(d.ev[1], s_));
  CUDA_CHECK(cudaMemcpyAsync(d.a_pst, ss.data(), (B + 1) * 4, cudaMemcpyHostToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(d.a_qst, qst.data(), (B + 1) * 4, cudaMemcpyHostToDevice, s_));
  if (ND) CUDA_CHECK(cudaMemcpyAsync(d.a_dr, dr.data(), ND * 4, cudaMemcpyHostToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(d.a_temp, tmp.data(), B * 4, cudaMemcpyHostToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(d.a_topp, tpp.data(), B * 4, cudaMemcpyHostToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(d.a_minp, mpp.data(), B * 4, cudaMemcpyHostToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(d.a_topk, tpk.data(), B * 4, cudaMemcpyHostToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(d.a_seed, sd.data(), B * 8, cudaMemcpyHostToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(d.a_ctr, ct.data(), B * 8, cudaMemcpyHostToDevice, s_));
  dmark(g_rp, s_, "a.h2d");
  if (penal) penalize_rows(lg_, kVocab, 0, kVocab, kVocab, M, d.a_pst, d.a_qst, d.a_dr, B, d.a_pen, s_);
  // the verified drafts' q rows, in batch order (P162 draft_qrows, one sequence: read in its slot's q store)
  const float* qrow = d.qbuf;
  if (sampled && B == 1 && patch::on(patch::draft_qrows)) qrow = d.qs + size_t(seqs[0]->slot) * kDspMaxN * kVocab;
  else if (sampled)
    for (int b = 0; b < B; b++)
      if (seqs[b]->temp > 0.f && kv[b])
        CUDA_CHECK(cudaMemcpyAsync(d.qbuf + size_t(qst[b]) * kVocab, d.qs + size_t(seqs[b]->slot) * kDspMaxN * kVocab,
                                   size_t(kv[b]) * kVocab * 4, cudaMemcpyDeviceToDevice, s_));
  dmark(g_rp, s_, "a.qcopy");
  spec_accept(lg_, d.a_pst, qrow, d.a_qst, d.a_dr, B, kVocab, d.a_temp, d.a_topk, d.a_topp, d.a_minp, nullptr,
              d.a_seed, d.a_ctr, d.a_nacc, d.a_next, s_);
  dmark(g_rp, s_, "a.accept");
  std::vector<int> nacc(B), next(B);
  CUDA_CHECK(cudaMemcpyAsync(nacc.data(), d.a_nacc, B * 4, cudaMemcpyDeviceToHost, s_));
  CUDA_CHECK(cudaMemcpyAsync(next.data(), d.a_next, B * 4, cudaMemcpyDeviceToHost, s_));
  dmark(g_rp, s_, "a.d2h");
  CUDA_CHECK(cudaEventRecord(d.ev[2], s_));
  CUDA_CHECK(cudaStreamSynchronize(s_));
  std::vector<std::vector<int>> out(B);
  std::vector<int> keep(B), ntok(B);
  for (int b = 0; b < B; b++) {
    SpecSeq& q = *seqs[b];
    if (nacc[b] < 0 || nacc[b] > kv[b]) qlog::fail("dspark", "spec_accept: n_acc %d of %d drafts", nacc[b], kv[b]);
    std::vector<int> t(q.drafts.begin(), q.drafts.begin() + nacc[b]);
    t.push_back(next[b]);
    const int kp = q.cap > 0 ? std::min<int>(q.cap, int(t.size())) : int(t.size());
    t.resize(kp);
    keep[b] = kp, ntok[b] = kp - 1;
    out[b] = t;
    d.stats.verified += kv[b], d.stats.accepted += nacc[b], d.stats.emitted += kp;
  }
  spec_commit(keep);
  if (penal) {  // the kept drafts count now; the last emitted token counts with its round
    CUDA_CHECK(cudaMemcpyAsync(d.a_ntok, ntok.data(), B * 4, cudaMemcpyHostToDevice, s_));
    penalty_add(d.a_pen, B, d.a_dr, d.a_qst, d.a_ntok, nullptr, kVocab, s_);
  }
  for (int b = 0; b < B; b++) {
    SpecSeq& q = *seqs[b];
    q.pos += keep[b];
    q.last = out[b].back();
    q.drafts.clear(), q.conf.clear();
  }
  CUDA_CHECK(cudaEventRecord(d.ev[3], s_));
  dmark(g_rp, s_, "c.commit");
  if (draft_next) spec_draft(seqs);  // (P160: its first stage joins the context rows after its mHC pre)
  else {
    dsp_join(d, s_);
    CUDA_CHECK(cudaEventRecord(d.ev[4], s_));
    CUDA_CHECK(cudaStreamSynchronize(s_));
  }
  dprof_end(g_rp, c_.rank, M);
  float ms[5];
  CUDA_CHECK(cudaEventElapsedTime(&ms[0], d.ev[0], d.ev[5]));  // verify step without the context rows
  CUDA_CHECK(cudaEventElapsedTime(&ms[1], d.ev[5], d.ev[1]));  // the draft's context rows
  CUDA_CHECK(cudaEventElapsedTime(&ms[2], d.ev[1], d.ev[2]));
  CUDA_CHECK(cudaEventElapsedTime(&ms[3], d.ev[2], d.ev[3]));
  CUDA_CHECK(cudaEventElapsedTime(&ms[4], d.ev[3], d.ev[4]));
  d.stats.ms_verify += ms[0], d.stats.ms_ctx += ms[1], d.stats.ms_accept += ms[2], d.stats.ms_commit += ms[3];
  d.stats.ms_draft += ms[4];
  d.stats.rounds++, d.stats.seqs += B, d.stats.rows += M;
  spec_k_ = kv;
  return out;
}

const float* Engine::spec_logits(int* rows) const {
  if (rows) *rows = last_rows_;
  return lg_;
}

SpecStats Engine::spec_stats(bool reset) {
  SpecStats s = dsp_->stats;
  if (reset) dsp_->stats = SpecStats{};
  return s;
}

void Engine::expert_count(bool on) {
  Dsp& d = *dsp_;
  d.ecount = on;
  CUDA_CHECK(cudaMemsetAsync(d.ecnt, 0, kLayers * 4, s_));
  d.esteps = 0;
}

std::vector<double> Engine::expert_take(long* steps) {
  Dsp& d = *dsp_;
  std::vector<int> c(kLayers);
  CUDA_CHECK(cudaMemcpyAsync(c.data(), d.ecnt, kLayers * 4, cudaMemcpyDeviceToHost, s_));
  CUDA_CHECK(cudaStreamSynchronize(s_));
  if (steps) *steps = d.esteps;
  std::vector<double> out(kLayers);
  for (int i = 0; i < kLayers; i++) out[i] = d.esteps ? double(c[i]) / d.esteps : 0.0;
  CUDA_CHECK(cudaMemsetAsync(d.ecnt, 0, kLayers * 4, s_));
  d.esteps = 0;
  return out;
}

// ---- P168 copy_draft: copy rounds (server/copy.py) -------------------------------------------------------------------
// One sequence's round whose drafts are the caller's c_1 … c_k (the tokens that followed the reply's tail where the same
// text came before): one target step over [last, c_1 … c_k] (all logits, rows q.pos … q.pos + k), then every row drawn
// as plain decoding draws that position — sample_rows with the request's settings (greedy: the argmax) and the drawn
// token's position as the Philox counter (tag 0, as run() and spec_first), penalties with the row's copied tokens
// before it as extras — and c_i kept while row i − 1's draw equals it; the first draw that differs (or row k's) is the
// next token. A point-mass proposal accepted with p(c) and replaced by a draw from p given ≠ c: the tokens of plain
// decoding itself (greedy and sampled), whatever was copied. Rollback, history, the draft's context rows and the
// penalty counts as spec_round. Then the DSpark drafts for the next round (draft_next), except when the copy goes on:
// every c_i kept and the next token == follow ≥ 0 (the token after c_k in the source) — the caller's next round copies
// again (or calls spec_draft first). q.cap (> 0) bounds k + 1 (the caller cut the copy there).
std::vector<int> Engine::spec_copy(SpecSeq& q, const std::vector<int>& copy, int follow, bool draft_next) {
  Dsp& d = *dsp_;
  const int k = int(copy.size()), M = k + 1;
  if (!patch::on(patch::copy_draft)) qlog::fail("dspark", "spec_copy with P168 copy_draft off");
  if (k < 1 || M > c_.logit_rows || M > d.roll.maxr || k > kPenMaxX)
    qlog::fail("dspark", "spec_copy: %d copied tokens (1 … %d)", k, std::min(c_.logit_rows - 1, kPenMaxX));
  if (q.cap > 0 && k > q.cap - 1) qlog::fail("dspark", "spec_copy: %d copied tokens, cap %d", k, q.cap);
  if (hist_.at(q.slot) != q.pos) qlog::fail("dspark", "slot %d: last at %d, history %d", q.slot, q.pos, hist_[q.slot]);
  if (q.pos + M > c_.ctx) qlog::fail("dspark", "slot %d: copy past the context", q.slot);
  std::vector<int> ids(M), pos(M);
  ids[0] = q.last;
  for (int i = 0; i < M; i++) pos[i] = q.pos + i;
  for (int i = 0; i < k; i++) ids[i + 1] = copy[i];
  if (pool_) kv_ensure({q.slot}, {long(q.pos) + M});  // (KvFull: the round did not happen)
  CUDA_CHECK(cudaEventRecord(d.ev[0], s_));
  const bool pen = q.penal();
  if (pen) {  // as spec_round: the round's `last` counts as generated before its rows are penalised
    const PenSeq ps{q.pen, q.rep, q.freq, q.pres};
    CUDA_CHECK(cudaMemcpyAsync(d.a_pen, &ps, sizeof ps, cudaMemcpyHostToDevice, s_));
    CUDA_CHECK(cudaMemcpyAsync(d.a_next, &q.last, 4, cudaMemcpyHostToDevice, s_));
    penalty_add(d.a_pen, 1, nullptr, nullptr, nullptr, d.a_next, kVocab, s_);
  }
  verify_ = true;
  step(ids, pos, {q.slot}, {0, M}, true);
  verify_ = false;
  CUDA_CHECK(cudaEventRecord(d.ev[1], s_));
  std::vector<float> tmp(M, q.temp), tpp(M, q.top_p), mpp(M, q.min_p);
  std::vector<int> tpk(M, q.top_k), drawn(M);
  std::vector<uint64_t> sd(M, q.seed), ct(M);
  for (int i = 0; i < M; i++) ct[i] = uint64_t(q.pos) + 1 + i;  // row i draws position q.pos + 1 + i
  CUDA_CHECK(cudaMemcpyAsync(d.a_temp, tmp.data(), M * 4, cudaMemcpyHostToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(d.a_topp, tpp.data(), M * 4, cudaMemcpyHostToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(d.a_minp, mpp.data(), M * 4, cudaMemcpyHostToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(d.a_topk, tpk.data(), M * 4, cudaMemcpyHostToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(d.a_seed, sd.data(), M * 8, cudaMemcpyHostToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(d.a_ctr, ct.data(), M * 8, cudaMemcpyHostToDevice, s_));
  if (pen) {  // row i: the copied tokens before it as extras (kept or not, as a DSpark round's drafts)
    const int st[2] = {0, M}, qst[2] = {0, k};
    CUDA_CHECK(cudaMemcpyAsync(d.a_pst, st, 8, cudaMemcpyHostToDevice, s_));
    CUDA_CHECK(cudaMemcpyAsync(d.a_qst, qst, 8, cudaMemcpyHostToDevice, s_));
    CUDA_CHECK(cudaMemcpyAsync(d.a_dr, copy.data(), k * 4, cudaMemcpyHostToDevice, s_));
    penalize_rows(lg_, kVocab, 0, kVocab, kVocab, M, d.a_pst, d.a_qst, d.a_dr, 1, d.a_pen, s_);
  }
  sample_rows(lg_, M, kVocab, d.a_temp, d.a_topk, d.a_topp, d.a_minp, d.a_seed, d.a_ctr, d.a_next, d.s_prob, s_);
  CUDA_CHECK(cudaMemcpyAsync(drawn.data(), d.a_next, M * 4, cudaMemcpyDeviceToHost, s_));
  CUDA_CHECK(cudaEventRecord(d.ev[2], s_));
  CUDA_CHECK(cudaStreamSynchronize(s_));
  int nacc = 0;
  while (nacc < k && drawn[nacc] == copy[nacc]) nacc++;
  std::vector<int> t(copy.begin(), copy.begin() + nacc);
  t.push_back(drawn[nacc]);
  const int kp = q.cap > 0 ? std::min<int>(q.cap, int(t.size())) : int(t.size());
  t.resize(kp);
  d.stats.verified += k, d.stats.accepted += nacc, d.stats.emitted += kp;
  spec_commit({kp});
  if (pen && kp > 1) {  // the kept copies count now; the last emitted token counts with its round
    const int nt = kp - 1;
    CUDA_CHECK(cudaMemcpyAsync(d.a_ntok, &nt, 4, cudaMemcpyHostToDevice, s_));
    penalty_add(d.a_pen, 1, d.a_dr, d.a_qst, d.a_ntok, nullptr, kVocab, s_);
  }
  q.pos += kp;
  q.last = t.back();
  q.drafts.clear(), q.conf.clear();
  CUDA_CHECK(cudaEventRecord(d.ev[3], s_));
  const bool on = follow >= 0 && nacc == k && kp == M && t.back() == follow;  // the copy goes on: no drafts
  if (draft_next && !on) spec_draft({&q});
  else {
    CUDA_CHECK(cudaEventRecord(d.ev[4], s_));
    CUDA_CHECK(cudaStreamSynchronize(s_));
  }
  float ms[5];
  CUDA_CHECK(cudaEventElapsedTime(&ms[0], d.ev[0], d.ev[5]));
  CUDA_CHECK(cudaEventElapsedTime(&ms[1], d.ev[5], d.ev[1]));
  CUDA_CHECK(cudaEventElapsedTime(&ms[2], d.ev[1], d.ev[2]));
  CUDA_CHECK(cudaEventElapsedTime(&ms[3], d.ev[2], d.ev[3]));
  CUDA_CHECK(cudaEventElapsedTime(&ms[4], d.ev[3], d.ev[4]));
  d.stats.ms_verify += ms[0], d.stats.ms_ctx += ms[1], d.stats.ms_accept += ms[2], d.stats.ms_commit += ms[3];
  d.stats.ms_draft += ms[4];
  d.stats.rounds++, d.stats.seqs += 1, d.stats.rows += M;
  spec_k_ = {k};
  return t;
}

}  // namespace e55
