// model.cc — the ERNIE 5.5 Flash step on this rank (contract: model.h). TP=2 placement: tools/tensor_table.py.
#include "model/model.h"

#include <malloc.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "core/log.h"
#include "kernels/misc/misc.h"
#include "kernels/sample/sample.h"
#include "model/layer.h"
#include "model/patches.h"

namespace e55 {

namespace {
template <typename T>
T* dmalloc(size_t n) {
  T* p = nullptr;
  CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
  CUDA_CHECK(cudaMemset(p, 0, n * sizeof(T)));
  return p;
}
float bf16_to_f(uint16_t b) {
  uint32_t u = uint32_t(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
uint16_t f_to_bf16_exact(float f) {  // values known to be exact in bf16
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return uint16_t(u >> 16);
}
std::vector<float> as_f32(const Weights& W, const std::string& n, std::vector<int64_t>* sh = nullptr) {
  const auto& ti = W.info(n);
  auto b = W.slice(n, sh);
  std::vector<float> v;
  if (ti.dtype == "F32") {
    v.resize(b.size() / 4);
    std::memcpy(v.data(), b.data(), b.size());
  } else if (ti.dtype == "BF16") {
    v.resize(b.size() / 2);
    for (size_t i = 0; i < v.size(); i++) v[i] = bf16_to_f(uint16_t(b[2 * i] | (b[2 * i + 1] << 8)));
  } else {
    qlog::fail("model", "%s: dtype %s is not a float", n.c_str(), ti.dtype.c_str());
  }
  return v;
}
float* up_f32(const std::vector<float>& v) {
  float* p = dmalloc<float>(v.size());
  CUDA_CHECK(cudaMemcpy(p, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
  return p;
}
float* up_f32(const Weights& W, const std::string& n) { return up_f32(as_f32(W, n)); }
// cudaFree each and null it (unset pointers are null: a no-op)
template <typename... T>
void dfree(T*&... p) {
  ((cudaFree(p), p = nullptr), ...);
}
bf16* up_bf16(const Weights& W, const std::string& n) {
  auto b = W.slice(n);
  bf16* p = dmalloc<bf16>(b.size() / 2);
  CUDA_CHECK(cudaMemcpy(p, b.data(), b.size(), cudaMemcpyHostToDevice));
  return p;
}
// VHA factor rows of this rank's heads (64 × 16 replicated → 32 × 16 fp32)
float* up_vha(const Weights& W, const std::string& n, int rank) {
  auto v = as_f32(W, n);
  if (v.size() != size_t(kHeads) * kVhaRank) qlog::fail("model", "%s shape", n.c_str());
  return up_f32(std::vector<float>(v.begin() + rank * 32 * kVhaRank, v.begin() + (rank + 1) * 32 * kVhaRank));
}
// RoPE inverse frequencies, as the reference computes them in fp32 (vLLM rotary_embedding)
std::vector<float> inv_plain(float theta) {
  std::vector<float> v(32);
  for (int i = 0; i < 32; i++) v[i] = 1.0f / powf(theta, float(2 * i) / 64.0f);
  return v;
}
std::vector<float> inv_yarn(float theta, float factor, double orig, double bfast, double bslow) {
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
}  // namespace

// P078 tp_split: rows [r0, r0 + n) of an uploaded weight as a weight of its own — a view: the same tiles in device
// memory and the same K split, so each output is the same chain of MMAs and sums as from the whole weight. Tile
// layouts: F8Lin [tile16][k/32][512 B] + UE8M0 [tile16][k/128] (fp8.h); bf16 Lin [tile16][k/32][1024 B] (linear.cu)
// and its P013 / P062 12-bit copy [tile16][k/32][768 B] + a base byte per row + per-tile offsets into one exception
// list (p013_pack12.cuh; the list itself is shared). r0 and n are multiples of the 64-row CTA tile of both kernels.
F8Lin f8_rows(const F8Lin& L, int r0, int n) {
  if (r0 % 64 || n % 64 || r0 + n > L.n || L.groups != 1) qlog::fail("model", "f8_rows %d+%d of %d", r0, n, L.n);
  F8Lin v = L;
  v.n = v.npad = n;
  v.w = L.w + size_t(r0 / 16) * (L.k / 32) * 512;
  v.s = L.s + size_t(r0 / 16) * (L.k / 128);
  return v;
}
Lin lin_rows(const Lin& L, int r0, int n) {
  if (L.fmt != WMat::BF16 || r0 % 64 || n % 64 || r0 + n > L.n) qlog::fail("model", "lin_rows %d+%d of %d", r0, n, L.n);
  Lin v = L;
  v.n = v.npad = n;
  if (L.w) v.w = static_cast<uint8_t*>(L.w) + size_t(r0 / 16) * (L.k / 32) * 1024;  // (P098: 12-bit copy only)
  if (L.pk) v.pk = L.pk + size_t(r0 / 16) * (L.k / 32) * 768, v.pk_base = L.pk_base + r0, v.pk_xo = L.pk_xo + r0 / 16;
  return v;
}

// P062 lin_pack12_rc / P098 lin12_prefill: the router's and the HCA compressor's copies (the draft's routers too) —
// none, the bf16 one + the 12-bit one, or (P098) the 12-bit one alone
int pack12_rc() {
  if (!patch::on(patch::lin_pack12_rc)) return kPack12None;
  return patch::on(patch::lin12_prefill) ? kPack12Only : kPack12Both;
}

namespace {
// P078: steps of up to kSplit078 rows read the halves (+ one all-gather per split point); larger steps the whole
// weights (no gathers). P191 tp_split64: up to kSplitMaxRows (verify / draft steps of 3–8 sequences) — the same views
// and gathers on more rows, same bits. Scratch per row for the halves: kSplitRowBytes (sp_: kSplitMaxRows rows, the
// layout strides below are by kSplitMaxRows whichever rows a step has).
constexpr int kSplit078 = 16, kSplitMaxRows = 64, kSplitRowBytes = 16384;
bool split_rows(int M) {
  if (M < 1 || !patch::on(patch::tp_split)) return false;
  return M <= kSplit078 || (M <= kSplitMaxRows && patch::on(patch::tp_split64));
}
// rows [o, o + n) of a [rows, w] buffer
template <typename T>
T* at(T* p, int o, size_t w) {
  return p + size_t(o) * w;
}
// Rows per kernel call: at most kMaxCall (mHC's workspace, the linear kernels) — the FFN of a larger step (prefill
// chunks up to max_rows) runs in parts of ≤ kMaxCall rows — and at most kMaxAttn for the attention and mHC calls (an
// 8192-row call at positions 8192… differs from two 4096-row calls from layer 8 on, measured; ≤ 4096 checked equal).
// P075 ar_lanes: steps of at least kLaneMinRows rows run in lanes — each part in ≤ kPartLanes lanes of ≥ kLaneRows
// rows (E55_LANES=n: n per part), equal sizes in multiples of 64 rows; steps above kMaxAttn rows always take lanes.
constexpr int kMaxCall = 8192, kMaxAttn = 4096, kLaneMinRows = 1024, kLaneRows = 512, kPartLanes = 4;
constexpr int kA2aAhead = 1 << 20;  // P101: attention fronts ahead of the halves (all; measured: see patches.def)
constexpr int kSpLate = 2;          // P102: lanes whose shared expert runs after the experts
constexpr int kZcRowBytes = 32768;  // P176: zr_ rows (HCA o rows; DSA's 16 KiB rows use the same row offsets)
// P136 sp_shared: where lane k of a part runs its shared expert — a letter a lane, the last one repeated (f: right after
// its FFN front, b: before the experts — cover for the last lanes' FFN gathers the experts join —, a: after them — cover
// for the latent reduce-scatters the next layer's tails join); E55_SPS overrides (measured: see patches.def)
constexpr const char* kSps = "bbaa";
char sps_at(int k) {
  const char* e = std::getenv("E55_SPS");
  const char* s = e && *e ? e : kSps;
  return s[std::min(k, int(std::strlen(s)) - 1)];
}
// P125 kv_pool: positions mapped past what a call writes (no kernel reads past a row's own position — the key tiles'
// tails are zero-size copies — this keeps any tile rounding inside mapped memory)
constexpr long kKvSlack = 256;
// → lane bounds b[0 … NL] and, per part, its first lane (parts[0 … P], parts[P] = NL)
void lane_bounds(int M, bool lanes, std::vector<int>& b, std::vector<int>& parts) {
  const char* e = std::getenv("E55_LANES");
  const int want = e ? std::max(1, std::atoi(e)) : kPartLanes;
  const int P = (M + kMaxCall - 1) / kMaxCall;
  b.assign(1, 0), parts.assign(1, 0);
  auto r64 = [](long long x) { return int((x + 32) / 64 * 64); };
  for (int q = 0; q < P; q++) {
    const int p0 = q ? r64((long long)M * q / P) : 0, p1 = q + 1 < P ? r64((long long)M * (q + 1) / P) : M;
    const int nl = std::max((p1 - p0 + kMaxAttn - 1) / kMaxAttn, lanes ? std::min(want, (p1 - p0) / kLaneRows) : 1);
    for (int k = 1; k <= nl; k++) b.push_back(k < nl ? p0 + r64((long long)(p1 - p0) * k / nl) : p1);
    parts.push_back(int(b.size()) - 1);
  }
}
// E55_SYNC (debugging, also set_sync_mode): 1 = synchronize after every stage of a step and name the first stage whose
// kernel failed; 2 = that, and add each stage's wall time (since the previous stage) to a per-line table (prof_take)
int g_sync = [] { const char* e = std::getenv("E55_SYNC"); return e ? std::atoi(e) : 0; }();
int g_layer = -1;
struct ProfEnt {
  const char* what = "";
  double ms = 0;
  long n = 0;
};
std::vector<ProfEnt> g_prof(4096);  // by source line
std::chrono::steady_clock::time_point g_t;
void chk(cudaStream_t s, const char* what, int line) {
  if (!g_sync) return;
  cudaError_t e = cudaStreamSynchronize(s);
  if (e == cudaSuccess) e = cudaGetLastError();
  if (e != cudaSuccess) qlog::fail("sync", "layer %d · %s (model.cc:%d): %s", g_layer, what, line, cudaGetErrorString(e));
  if (g_sync == 2) {
    const auto now = std::chrono::steady_clock::now();
    auto& p = g_prof.at(line);
    p.what = what, p.ms += std::chrono::duration<double, std::milli>(now - g_t).count(), p.n++;
    g_t = now;
  }
}
}  // namespace

void set_sync_mode(int m) { g_sync = m; }
std::vector<ProfLine> prof_take() {
  std::vector<ProfLine> out;
  for (int i = 0; i < int(g_prof.size()); i++)
    if (g_prof[i].n) out.push_back(ProfLine{g_prof[i].what, i, g_prof[i].ms, g_prof[i].n});
  g_prof.assign(4096, ProfEnt{});
  return out;
}

Engine::Engine(const EngineCfg& c) : c_(c) {
  auto t0 = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaStreamCreateWithFlags(&s_, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&s2_, cudaStreamNonBlocking));  // P074 hca_side
  CUDA_CHECK(cudaEventCreateWithFlags(&ev_in_, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreateWithFlags(&ev_ck_, cudaEventDisableTiming));
  w_ = std::make_unique<Weights>(c.model_dir, c.table, c.rank);
  comm_ = std::make_unique<Comm>(c.rank, c.head_ip, c.port);
  const Weights& W = *w_;
  const int r = c.rank;
  int n_hca = 0, n_dsa = 0;
  // P100 a2a_attn (fixed at load): the peer's slices of the attention's output side through a Weights view of the peer
  // rank — the same bytes, shapes and K splits as the peer's own uploads
  a2a_w_ = patch::on(patch::a2a_attn);
  shw_ = patch::on(patch::sp_shared);  // P136 (fixed at load): the peer's shared-expert halves through the same view
  std::unique_ptr<Weights> Wp;
  if (a2a_w_ || shw_) Wp = std::make_unique<Weights>(c.model_dir, c.table, 1 - r);
  for (int L = 0; L < kLayers; L++) {
    auto l = std::make_unique<Layer>();
    const std::string p = "model.layers." + std::to_string(L) + ".";
    l->ha = mhc_upload(W, p + "self_attention_hyper_connection.", p + "input_layernorm.weight");
    l->hf = mhc_upload(W, p + "mlp_hyper_connection.", p + "post_attention_layernorm.weight");
    l->dsa = is_dsa(L);
    const std::string a = p + "self_attn.";
    if (!l->dsa) {
      auto& h = l->h;
      h.qkv = f8_upload(W, {a + "linear_q_down_proj", a + "linear_kv_proj"});
      h.qup = f8_upload(W, {a + "linear_q_up_proj"});
      h.gate = f8_upload(W, {a + "gate_proj"});
      h.woa = f8_upload(W, {a + "linear_o_group_proj"}, 4);
      h.oproj = f8_upload(W, {a + "o_proj"});
      h.comp = lin_upload(W, {a + "core_attention.compressor.linear_wkv.weight",
                              a + "core_attention.compressor.linear_wgate.weight"},
                          pack12_rc());  // P062: + the 12-bit decode copy (P098: only that)
      h.qn = up_bf16(W, a + "q_layernorm.weight");
      h.kvn = up_bf16(W, a + "kv_layernorm.weight");
      h.cnw = up_bf16(W, a + "core_attention.compressor.norm.weight");
      h.ape = up_f32(W, a + "core_attention.compressor.ape");
      h.sink = up_f32(W, a + "core_attention.attn_sink");
      h.U = up_vha(W, a + "vha_postmix_U", r);
      h.V = up_vha(W, a + "vha_postmix_V", r);
      h.qv = f8_rows(h.qkv, 0, kQLora), h.kvh = f8_rows(h.qkv, kQLora + r * 256, 256);  // P078 views
      h.cvh = lin_rows(h.comp, r * 256, 256), h.cgh = lin_rows(h.comp, 512 + r * 256, 256);
      if (a2a_w_) {  // P100: + gate 4 MiB, wo_a 16 MiB, o_proj 16 MiB
        A2aW x;
        x.gate = f8_upload(*Wp, {a + "gate_proj"});
        x.woa = f8_upload(*Wp, {a + "linear_o_group_proj"}, 4);
        x.oproj = f8_upload(*Wp, {a + "o_proj"});
        x.U = up_vha(W, a + "vha_postmix_U", 1 - r), x.V = up_vha(W, a + "vha_postmix_V", 1 - r);
        a2h_.push_back(x);
      }
      n_hca++;
    } else {
      auto& d = l->d;
      d.qkv = f8_upload(W, {a + "q_a_proj", a + "kv_a_proj_with_mqa"});
      d.qb = f8_upload(W, {a + "q_b_proj"});
      d.gate = f8_upload(W, {a + "gate_proj"});
      d.oproj = f8_upload(W, {a + "o_proj"});
      const std::string ix = a + "core_attention.indexer.";
      d.iq = f8_upload(W, {ix + "wq_b"});
      d.ikw = lin_upload(W, {ix + "wk.weight", ix + "weights_proj.weight"});
      d.knw = up_bf16(W, ix + "k_norm.weight");
      d.knb = up_bf16(W, ix + "k_norm.bias");
      d.qan = up_bf16(W, a + "q_a_layernorm.weight");
      d.kvan = up_bf16(W, a + "kv_a_layernorm.weight");
      d.sink = up_f32(W, a + "core_attention.softmax_offset");
      d.U = up_vha(W, a + "vha_postmix_U", r);
      d.V = up_vha(W, a + "vha_postmix_V", r);
      if (a2a_w_) {  // P100: + gate 8 MiB, o_proj 32 MiB
        A2aW x;
        x.gate = f8_upload(*Wp, {a + "gate_proj"});
        x.oproj = f8_upload(*Wp, {a + "o_proj"});
        x.U = up_vha(W, a + "vha_postmix_U", 1 - r), x.V = up_vha(W, a + "vha_postmix_V", 1 - r);
        a2d_.push_back(x);
      }
      // kv_b_proj (this rank's 32 heads × (192 k_nope + 256 v) rows × 512): W_UKᵀ [32][512][192], W_UV [32][256][512]
      std::vector<int64_t> sh, ss;
      auto wb = W.slice(a + "kv_b_proj.weight", &sh);
      auto sc = as_f32(W, a + "kv_b_proj.weight_scale_inv", &ss);
      if (sh[0] != 32 * 448 || sh[1] != kKvLora) qlog::fail("model", "kv_b_proj shape");
      std::vector<uint16_t> uk(size_t(32) * 512 * 192), uv(size_t(32) * 256 * 512);
      std::vector<uint8_t> uk8(uk.size()), uv8(uv.size());  // P073: the same elements as e4m3 bytes
      for (int h = 0; h < 32; h++)
        for (int i = 0; i < 448; i++)
          for (int k = 0; k < 512; k++) {
            const int row = h * 448 + i;
            const float v = e4m3_to_float(wb[(size_t)row * 512 + k]) * sc[(row / 128) * 4 + k / 128];
            if (i < 192) uk[((size_t)h * 512 + k) * 192 + i] = f_to_bf16_exact(v), uk8[((size_t)h * 512 + k) * 192 + i] = wb[(size_t)row * 512 + k];
            else uv[((size_t)h * 256 + (i - 192)) * 512 + k] = f_to_bf16_exact(v), uv8[((size_t)h * 256 + (i - 192)) * 512 + k] = wb[(size_t)row * 512 + k];
          }
      d.wuk8 = dmalloc<uint8_t>(uk8.size());
      d.wuv8 = dmalloc<uint8_t>(uv8.size());
      CUDA_CHECK(cudaMemcpy(d.wuk8, uk8.data(), uk8.size(), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d.wuv8, uv8.data(), uv8.size(), cudaMemcpyHostToDevice));
      d.kvs = up_f32(sc);
      d.wuk = dmalloc<bf16>(uk.size());
      d.wuv = dmalloc<bf16>(uv.size());
      CUDA_CHECK(cudaMemcpy(d.wuk, uk.data(), uk.size() * 2, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d.wuv, uv.data(), uv.size() * 2, cudaMemcpyHostToDevice));
      n_dsa++;
    }
    auto& f = l->f;
    if (L == 0) {
      f.dense = true;
      f.gu = f8_upload(W, {p + "mlp.gate_proj", p + "mlp.up_proj"});
      f.down = f8_upload(W, {p + "mlp.down_proj"});
    } else {
      const std::string m = p + "block_sparse_moe.";
      f.router = lin_upload(W, {m + "gate.weight", m + "gate.weight_1"}, pack12_rc());  // P062 / P098
      f.bias = up_f32(W, m + "e_score_correction_bias");
      f.rscale = up_f32(W, m + "gate.routed_scaling_factor_param");
      f.fc1 = f8_upload(W, {m + "fc1_latent_proj"});
      f.fc2 = f8_upload(W, {m + "fc2_latent_proj"});
      f.lnorm = up_bf16(W, m + "latent_norm.weight");
      f.gu = f8_upload(W, {m + "shared_experts.w1", m + "shared_experts.w3"});
      f.down = f8_upload(W, {m + "shared_experts.w2"});
      if (shw_) {  // P136: + gate/up 8 MiB, down 4 MiB
        f.gup = f8_upload(*Wp, {m + "shared_experts.w1", m + "shared_experts.w3"});
        f.downp = f8_upload(*Wp, {m + "shared_experts.w2"});
      }
      f.ex = moe_upload(W, L);
      f.r0h = lin_rows(f.router, r * 256, 256), f.r1h = lin_rows(f.router, 512 + r * 256, 256);  // P078 views
      f.fc1h = f8_rows(f.fc1, r * 1024, 1024), f.fc2h = f8_rows(f.fc2, r * 2048, 2048);
    }
    L_.push_back(std::move(l));
    if (L % 6 == 5 || L == kLayers - 1)
      qlog::info("load", "rank%d · layers 0–%d · %.0f s", r, L,
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  }
  head_ = hc_head_upload(W);
  emb_half_ = patch::on(patch::embed_split);
  if (emb_half_) {  // P095 embed_split: rows rank·kVocabHalf … of the table (embed() exchanges the embedded rows)
    auto b = W.slice("model.embed_tokens.weight");
    const size_t hb = size_t(kVocabHalf) * kHidden * 2;
    if (b.size() != 2 * hb) qlog::fail("model", "embed_tokens: %zu bytes", b.size());
    embed_ = dmalloc<bf16>(hb / 2);
    CUDA_CHECK(cudaMemcpy(embed_, b.data() + size_t(r) * hb, hb, cudaMemcpyHostToDevice));
  } else {
    embed_ = up_bf16(W, "model.embed_tokens.weight");
  }
  final_norm_ = up_bf16(W, "model.norm.weight");
  // P013: lm_head's 12-bit copy for the decode kernel; P097 lm12_only: that copy alone (every lm_head read goes
  // through it: the target's R ≤ logit_rows rows, the draft's base logits with P080 / P081 off)
  lm_ = lin_upload(W, {"lm_head.weight"}, patch::on(patch::lm12_only) ? kPack12Only : kPack12Both);
  {
    auto rr = as_f32(W, "model.lm_head.multimax_ranges"), tt = as_f32(W, "model.lm_head.multimax_ts");
    std::vector<float> v(rr.begin(), rr.end());
    v.insert(v.end(), tt.begin(), tt.end());
    mm_ = up_f32(v);
  }
  inv_hca_ = up_f32(inv_yarn(160000.f, 16.f, 65536, 32, 1));
  inv_dsa_ = up_f32(inv_plain(8000000.f));
  Wp.reset();
  w_->drop_cache();
  // caches
  const int S = c.slots, maxc = c.ctx / kRatio + 1;
  if (patch::on(patch::kv_pool)) {
    // P125: the growing caches from one pool (model.h). Strides: kv_cap_ positions for the DSA rows (a multiple of 2^19,
    // so a slot's MLA (656 B) and indexer (68 B) ranges start on 2 MiB chunks), 2^18-multiples of compressed entries
    // (584 B) — the kernels see the same [slot][stride][row] layout, only the strides are longer (address space only)
    kv_cap_ = (long(c.ctx) + kKvSlack + (1 << 19) - 1) >> 19 << 19;
    const int mcomp = int((kv_cap_ / kRatio + 1 + (1 << 18) - 1) >> 18 << 18);
    std::vector<int> mla(n_dsa), idx(n_dsa), comp(n_hca);  // each group's first array
    kv_nd_ = n_dsa, kv_nh_ = n_hca;
    const long pt = c.pool_tokens > 0 ? std::min<long>(c.pool_tokens, c.ctx) : c.ctx;
    pool_ = std::make_unique<KvPool>(size_t(S) * kv_need(pt));
    kv_arr_.assign(S, {});
    for (int i = 0; i < n_dsa; i++) {
      mla[i] = pool_->add(size_t(kv_cap_) * kMlaRowBytes, S);
      idx[i] = pool_->add(size_t(kv_cap_) * kIdxRowBytes, S);
    }
    for (int i = 0; i < n_hca; i++) comp[i] = pool_->add(size_t(mcomp) * kKvRowBytes, S);
    for (int s = 0; s < S; s++) {
      for (int i = 0; i < n_dsa; i++) kv_arr_[s].push_back({mla[i] + s, 0}), kv_arr_[s].push_back({idx[i] + s, 1});
      for (int i = 0; i < n_hca; i++) kv_arr_[s].push_back({comp[i] + s, 2});
    }
    kv_cover_.assign(S, 0);
    for (int i = 0; i < n_hca; i++) {
      HcaCache h;
      h.ring = dmalloc<uint8_t>(size_t(S) * kWindow * kKvRowBytes);
      h.comp = static_cast<uint8_t*>(pool_->ptr(comp[i]));
      h.state = dmalloc<float>(size_t(S) * kRatio * 1024);
      h.max_comp = mcomp;
      hc_.push_back(h);
    }
    for (int i = 0; i < n_dsa; i++) {
      DsaCache d;
      d.mla = static_cast<uint8_t*>(pool_->ptr(mla[i]));
      d.idx = static_cast<uint8_t*>(pool_->ptr(idx[i]));
      d.ctx = int(kv_cap_);
      dc_.push_back(d);
    }
    mem_d_ = dmalloc<float>(8);
  } else {
    for (int i = 0; i < n_hca; i++) {
      HcaCache h;
      h.ring = dmalloc<uint8_t>(size_t(S) * kWindow * kKvRowBytes);
      h.comp = dmalloc<uint8_t>(size_t(S) * maxc * kKvRowBytes);
      h.state = dmalloc<float>(size_t(S) * kRatio * 1024);
      h.max_comp = maxc;
      hc_.push_back(h);
    }
    for (int i = 0; i < n_dsa; i++) {
      DsaCache d;
      d.mla = dmalloc<uint8_t>(size_t(S) * c.ctx * kMlaRowBytes);
      d.idx = dmalloc<uint8_t>(size_t(S) * c.ctx * kIdxRowBytes);
      d.ctx = c.ctx;
      dc_.push_back(d);
    }
  }
  hist_.assign(S, 0);
  // activations
  const size_t MR = c.max_rows;
  R_ = dmalloc<bf16>(MR * kHcDim);
  inp_ = dmalloc<bf16>(MR * kHidden);
  t1_ = dmalloc<bf16>(MR * 18432);
  t2_ = dmalloc<bf16>(MR * 8192);
  t3_ = dmalloc<bf16>(MR * 8192);
  t4_ = dmalloc<bf16>(MR * 8192);
  q_ = dmalloc<bf16>(MR * 16384);
  o_ = dmalloc<bf16>(MR * 16384);
  if (patch::on(patch::zc_a2a) && comm_->rdma()) {  // P176 (fixed at load): the o exchange's registered receive rows —
    // the own halves of every lane (≤ ⌈M/2⌉ + one row a lane; ≤ 16 lanes per 8192-row part)
    zr_rows_ = MR / 2 + 16 * ((MR + kMaxCall - 1) / kMaxCall) + 16;
    zr_ = static_cast<uint8_t*>(comm_->zc_alloc(zr_rows_ * kZcRowBytes));  // (both ranks: same size, same order)
    if (!zr_) zr_rows_ = 0;
  }
  aout_ = dmalloc<bf16>(MR * kHidden);
  fout_ = dmalloc<bf16>(MR * kHidden);
  mix_ = dmalloc<float>(MR * 72);
  post_ = dmalloc<float>(MR * 4);
  comb_ = dmalloc<float>(MR * 16);
  ckv_ = dmalloc<float>(MR * 1024);
  zp_ = dmalloc<float>(MR * kVhaRank * 512);
  lgr_ = dmalloc<float>(MR * 1024);
  part_ = dmalloc<float>(MR * 6144);
  {  // P094 sp_resid: in part_'s upper half (the MoE partials take its lower half, [MR, 6144] bf16, in both layouts)
    uint8_t* u = reinterpret_cast<uint8_t*>(part_) + MR * 12288;
    spq_ = reinterpret_cast<bf16*>(u), u += MR * 3200;
    spxq_ = u, u += MR * kHidden;
    spxs_ = u, u += MR * (kHidden / 128);
    sprt_ = u;  // + MR · 80 ≤ MR · 12288
  }
  d_own_ = dmalloc<int>(MR + c.logit_rows);
  wts_ = dmalloc<float>(MR * 64);
  mw_ = dmalloc<float>(MR * kTopK);
  xq_ = dmalloc<uint8_t>(MR * 8192);
  xs_ = dmalloc<uint8_t>(MR * 256);
  rows_ = dmalloc<uint8_t>(MR * kKvRowBytes);
  oq_ = dmalloc<uint8_t>(4 * MR * 4096);
  os_ = dmalloc<uint8_t>(4 * MR * 32);
  qf4_ = dmalloc<uint8_t>(MR * 64 * 64);
  qs_ = dmalloc<uint8_t>(MR * 256);
  d_ids_ = dmalloc<int>(MR);
  d_list_ = dmalloc<int>(MR * (kIdxTop + kWindow));
  d_cnt_ = dmalloc<int>(MR);
  d_moe_ids_ = dmalloc<int>(MR * kTopK);
  d_rows_ = dmalloc<int>(std::max(c.logit_rows, 1));
  d_meta_ = dmalloc<RowMeta>(MR);
  hbuf_ = dmalloc<bf16>(size_t(c.logit_rows) * kHidden);
  lg_half_ = dmalloc<float>(size_t(c.logit_rows) * kVocabHalf);
  lg2_ = dmalloc<float>(size_t(2) * c.logit_rows * kVocabHalf);
  lg_ = dmalloc<float>(size_t(c.logit_rows) * kVocab);
  cand_ = dmalloc<SparseCand>(size_t(c.logit_rows) * kSparseRec);
  cand2_ = dmalloc<SparseCand>(size_t(2) * c.logit_rows * kSparseRec);
  size_t fws = 0;
  for (auto& l : L_) {
    for (const F8Lin* f : {&l->h.qkv, &l->h.qup, &l->h.gate, &l->h.woa, &l->h.oproj, &l->d.qkv, &l->d.qb, &l->d.gate,
                           &l->d.oproj, &l->d.iq, &l->f.gu, &l->f.down, &l->f.fc1, &l->f.fc2})
      if (f->w) fws = std::max(fws, f8_ws(*f, std::min(c.max_rows, kMaxCall)));  // calls ≤ kMaxCall rows
  }
  fws_bytes_ = fws;
  fws_ = dmalloc<float>(std::max<size_t>(fws / 4, 1));
  aws_bytes_ = size_t(1) << 30;
  aws_ = dmalloc<float>(aws_bytes_ / 4);
  moe_ = moe_scratch(std::min(c.max_rows, kMaxCall));
  sp_ = dmalloc<uint8_t>(size_t(kSplitMaxRows) * kSplitRowBytes);  // P078 / P191 halves (1 MiB)
  malloc_trim(0);  // the loaders' freed host buffers back to the system (GB10: CUDA allocates from free pages)
  CUDA_CHECK(cudaDeviceSynchronize());
  size_t fr, tot;
  cudaMemGetInfo(&fr, &tot);
  if (pool_)
    qlog::ok("engine", "rank%d · %d layers (%d HCA · %d DSA) · %d slots · ≤ %d tokens a request · KV pool %zu chunks "
             "(P125) · %.0f s · %.1f GiB free", r, kLayers, n_hca, n_dsa, S, c.ctx, pool_->total_chunks(),
             std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), fr / 1073741824.0);
  else
    qlog::ok("engine", "rank%d · %d layers (%d HCA · %d DSA) · %d slots × %d tokens · %.0f s · %.1f GiB free", r, kLayers,
             n_hca, n_dsa, S, c.ctx, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
             fr / 1073741824.0);
}

// Everything the engine allocated goes back (tests/mem_report.py: "after close" ≈ "before init"): the draft, the layers'
// weights (P078 views are not owned), caches, activations, workspaces, taps, then the transports and the streams.
Engine::~Engine() {
  if (s_) cudaStreamSynchronize(s_);
  if (s2_) cudaStreamSynchronize(s2_);
  comm_.reset();  // waits for its stream; RDMA proxy, QPs, pinned staging, NCCL
  dsp_.reset();
  for (auto& l : L_) {
    mhc_free(l->ha), mhc_free(l->hf);
    auto& h = l->h;
    for (F8Lin* f : {&h.qkv, &h.qup, &h.gate, &h.woa, &h.oproj}) f8_free(*f);
    lin_free(h.comp);
    dfree(h.qn, h.kvn, h.cnw, h.ape, h.sink, h.U, h.V);
    auto& d = l->d;
    for (F8Lin* f : {&d.qkv, &d.qb, &d.gate, &d.oproj, &d.iq}) f8_free(*f);
    lin_free(d.ikw);
    dfree(d.qan, d.kvan, d.knw, d.knb, d.wuk, d.wuv, d.wuk8, d.wuv8, d.kvs, d.sink, d.U, d.V);
    auto& f = l->f;
    for (F8Lin* x : {&f.gu, &f.down, &f.fc1, &f.fc2, &f.gup, &f.downp}) f8_free(*x);
    lin_free(f.router);
    dfree(f.bias, f.rscale, f.lnorm);
    moe_free(f.ex), f.ex = nullptr;
  }
  L_.clear();
  for (auto* v : {&a2h_, &a2d_})  // P100
    for (auto& x : *v) f8_free(x.gate), f8_free(x.woa), f8_free(x.oproj), dfree(x.U, x.V);
  a2h_.clear(), a2d_.clear();
  dfree(head_.fn, head_.sb);
  lin_free(lm_);
  dfree(embed_, final_norm_, mm_, inv_hca_, inv_dsa_);
  if (pool_) {  // P125: the pool's arrays are not cudaMalloc'd — the pool gives their chunks back
    for (auto& h : hc_) h.comp = nullptr;
    for (auto& d : dc_) d.mla = d.idx = nullptr;
    pool_.reset();
  }
  dfree(mem_d_);
  for (auto& h : hc_) dfree(h.ring, h.comp, h.state);
  for (auto& d : dc_) dfree(d.mla, d.idx);
  hc_.clear(), dc_.clear();
  dfree(R_, inp_, t1_, t2_, t3_, t4_, q_, o_, aout_, fout_);
  dfree(mix_, post_, comb_, ckv_, zp_, lgr_, part_, wts_, mw_);
  dfree(xq_, xs_, rows_, oq_, os_, qf4_, qs_);
  dfree(d_ids_, d_list_, d_cnt_, d_moe_ids_, d_rows_, d_meta_);
  dfree(fws_, aws_, lg_half_, lg2_, lg_, sp_, hbuf_, cand_, cand2_);
  set_taps({});
  moe_scratch_free(moe_), moe_ = nullptr;
  w_.reset();
  if (s2_) cudaStreamDestroy(s2_);
  if (ev_in_) cudaEventDestroy(ev_in_);
  if (ev_ck_) cudaEventDestroy(ev_ck_);
  if (s_) cudaStreamDestroy(s_);
  malloc_trim(0);
}

// ---- P125 kv_pool -------------------------------------------------------------------------------------------------
namespace {
double mem_free_local() {  // MemFree (GiB): GB10's CUDA allocations come from free pages only, not the page cache
  FILE* f = std::fopen("/proc/meminfo", "r");
  if (!f) qlog::fail("kv pool", "cannot read /proc/meminfo");
  char k[64], u[16];
  long v = -1, x;
  while (std::fscanf(f, "%63s %ld %15s", k, &x, u) == 3)
    if (!std::strcmp(k, "MemFree:")) v = x;
  std::fclose(f);
  if (v < 0) qlog::fail("kv pool", "no MemFree in /proc/meminfo");
  return v / 1048576.0;
}
}  // namespace

size_t Engine::kv_bytes(int kind, long tokens) const {
  const long t = std::min(kv_cap_, tokens + kKvSlack);
  return kind == 0 ? size_t(t) * kMlaRowBytes : kind == 1 ? size_t(t) * kIdxRowBytes : size_t(t / kRatio) * kKvRowBytes;
}

size_t Engine::kv_need(long tokens) const {
  if (!kv_cap_) return 0;
  return size_t(kv_nd_) * (KvPool::chunks_for(kv_bytes(0, tokens)) + KvPool::chunks_for(kv_bytes(1, tokens))) +
         size_t(kv_nh_) * KvPool::chunks_for(kv_bytes(2, tokens));
}

size_t Engine::kv_held(int slot) const {
  if (!pool_) return 0;
  size_t n = 0;
  for (auto& a : kv_arr_.at(slot)) n += pool_->mapped(a.id) / KvPool::kChunk;
  return n;
}

void Engine::reset(int slot) {
  hist_.at(slot) = 0;
  if (!pool_ || (!kv_cover_[slot] && !ckpt_held(slot))) return;
  CUDA_CHECK(cudaStreamSynchronize(s_));  // no queued kernel still uses the chunks unmapped below
  CUDA_CHECK(cudaStreamSynchronize(s2_));
  for (auto& a : kv_arr_[slot]) pool_->shrink(a.id, 0);
  kv_cover_[slot] = 0;
  if (!ck_.empty())  // P146: the slot's checkpoints go with its caches
    for (auto& e : ck_[slot]) pool_->shrink(e.id, 0), e.pos = -1;
}

void Engine::kv_ensure(const std::vector<int>& slot, const std::vector<long>& end) {
  if (!pool_) return;
  // count first: KvFull before anything is mapped
  size_t need = 0;
  for (size_t b = 0; b < slot.size(); b++) {
    const int s = slot[b];
    if (std::min(kv_cap_, end[b] + kKvSlack) <= kv_cover_[s]) continue;
    size_t n = 0;
    for (auto& a : kv_arr_[s]) {
      const size_t want = KvPool::chunks_for(kv_bytes(a.kind, end[b])), have = pool_->mapped(a.id) / KvPool::kChunk;
      n += want > have ? want - have : 0;
    }
    if (need + n > pool_->free_chunks()) {  // (need ≤ free: checked for the sequences before)
      char m[240];
      std::snprintf(m, sizeof m, "KV pool full: slot %d needs %zu more chunks for %ld tokens, %zu of %zu free%s", s, n,
                    end[b], pool_->free_chunks() - need, pool_->total_chunks(),
                    need ? " after the step's other sequences" : "");
      qlog::warn("kv pool", "%s", m);
      throw KvFull(s, m);
    }
    need += n;
  }
  for (size_t b = 0; b < slot.size(); b++) {
    const int s = slot[b];
    const long cov = std::min(kv_cap_, end[b] + kKvSlack);
    if (cov <= kv_cover_[s]) continue;
    for (auto& a : kv_arr_[s])
      if (!pool_->grow(a.id, kv_bytes(a.kind, end[b]), s_))
        qlog::fail("kv pool", "slot %d: no chunk left (counted before)", s);
    kv_cover_[s] = cov;
  }
}

// ---- P146 prefix_ckpt (model.h) -------------------------------------------------------------------------------------
namespace {
constexpr size_t kCkRing = size_t(kWindow) * kKvRowBytes;  // one HCA layer's ring of one slot (= its 128 positions)
constexpr size_t kCkDraft = size_t(3) * kWindow * kKvRowBytes;  // the draft's 3 stages × its context rows Q−128 … Q−1
constexpr size_t kCkState = size_t(1024) * 4;  // one compressor state row (kv ‖ score + ape, fp32)
}  // namespace

bool Engine::ckpt_on() const { return pool_ && patch::on(patch::prefix_ckpt); }

size_t Engine::ckpt_bytes(long pos) const {
  return hc_.size() * (kCkRing + size_t(pos % kRatio) * kCkState) + (dsp_ ? kCkDraft : 0);
}

size_t Engine::ckpt_held(int slot) const {
  size_t n = 0;
  if (!ck_.empty())
    for (auto& e : ck_.at(slot)) n += pool_->mapped(e.id) / KvPool::kChunk;
  return n;
}

void Engine::ckpt_init(int per_slot) {
  if (!ckpt_on() || per_slot < 1) return;
  if (ck_.empty()) {  // address ranges only: an entry's chunks are mapped at its save
    const size_t most = hc_.size() * (kCkRing + size_t(kRatio - 1) * kCkState) + kCkDraft;
    const int first = pool_->add(most, c_.slots * per_slot);
    ck_.assign(c_.slots, std::vector<Ckpt>(per_slot));
    for (int s = 0; s < c_.slots; s++)
      for (int k = 0; k < per_slot; k++) ck_[s][k].id = first + s * per_slot + k;
  } else if (ckpt_cap() != per_slot) {
    qlog::fail("ckpt", "%d entries a slot asked, %d made", per_slot, ckpt_cap());
  }
  CUDA_CHECK(cudaStreamSynchronize(s_));
  for (auto& v : ck_)
    for (auto& e : v) pool_->shrink(e.id, 0), e.pos = -1;
}

void Engine::ckpt_move(int slot, long pos, uint8_t* p, bool save) {
  const size_t r = size_t(pos % kRatio);
  auto cp = [&](void* ck, const void* live, size_t n) {
    CUDA_CHECK(cudaMemcpyAsync(save ? ck : const_cast<void*>(live), save ? live : ck, n, cudaMemcpyDeviceToDevice, s_));
  };
  for (size_t i = 0; i < hc_.size(); i++) cp(p + i * kCkRing, hc_[i].ring + size_t(slot) * kCkRing, kCkRing);
  p += hc_.size() * kCkRing;
  if (dsp_) dsp_ckpt(slot, pos, p, save), p += kCkDraft;
  if (r)  // the open group's rows (state row t % 128 = position 128·⌊pos/128⌋ + t)
    for (size_t i = 0; i < hc_.size(); i++)
      cp(p + i * r * kCkState, hc_[i].state + size_t(slot) * kRatio * 1024, r * kCkState);
}

bool Engine::ckpt_save(int slot, int k) {
  auto& e = ck_.at(slot).at(k);
  const long q = hist_.at(slot);
  if (e.pos >= 0) ckpt_drop(slot, k);
  if (ckpt_need(q) > pool_->free_chunks() || !pool_->grow(e.id, ckpt_bytes(q), s_)) return false;
  ckpt_move(slot, q, static_cast<uint8_t*>(pool_->ptr(e.id)), true);
  e.pos = q, e.draft = dsp_ != nullptr;
  return true;
}

void Engine::ckpt_restore(int slot, int k) {
  auto& e = ck_.at(slot).at(k);
  const long q = e.pos;
  if (q < 0 || q > hist_.at(slot) || e.draft != (dsp_ != nullptr))
    qlog::fail("ckpt", "slot %d entry %d at %ld: history %d", slot, k, q, hist_.at(slot));
  ckpt_move(slot, q, static_cast<uint8_t*>(pool_->ptr(e.id)), false);
  hist_[slot] = int(q);
  CUDA_CHECK(cudaStreamSynchronize(s_));  // no queued kernel still uses the chunks unmapped below
  CUDA_CHECK(cudaStreamSynchronize(s2_));
  for (auto& x : ck_[slot])  // the slot rewrites its rows from q on
    if (x.pos > q) pool_->shrink(x.id, 0), x.pos = -1;
  for (auto& a : kv_arr_[slot]) pool_->shrink(a.id, kv_bytes(a.kind, q));
  kv_cover_[slot] = std::min(kv_cap_, q + kKvSlack);
}

bool Engine::ckpt_copy(int dst, int src, int k) {
  const auto& e = ck_.at(src).at(k);
  const long q = e.pos;
  if (dst == src || q < 0 || q > hist_.at(src) || e.draft != (dsp_ != nullptr))
    qlog::fail("ckpt", "copy to slot %d from slot %d entry %d at %ld", dst, src, k, q);
  if (kv_need(q) > pool_->free_chunks() + kv_held(dst) + ckpt_held(dst)) return false;
  reset(dst);
  kv_ensure({dst}, {q});
  for (size_t j = 0; j < kv_arr_[dst].size(); j++)  // rows < q (+ the slack's, never read before rewritten)
    CUDA_CHECK(cudaMemcpyAsync(pool_->ptr(kv_arr_[dst][j].id), pool_->ptr(kv_arr_[src][j].id),
                               kv_bytes(kv_arr_[dst][j].kind, q), cudaMemcpyDeviceToDevice, s_));
  ckpt_move(dst, q, static_cast<uint8_t*>(pool_->ptr(e.id)), false);
  hist_[dst] = int(q);
  return true;
}

void Engine::ckpt_drop(int slot, int k) {
  auto& e = ck_.at(slot).at(k);
  if (pool_->mapped(e.id)) {
    CUDA_CHECK(cudaStreamSynchronize(s_));
    pool_->shrink(e.id, 0);
  }
  e.pos = -1;
}

std::vector<double> Engine::mem_free_gib() {
  if (!mem_d_) mem_d_ = dmalloc<float>(8);  // (P125 off: on the first call)
  const float v[2] = {float(mem_free_local()), 0.f};
  CUDA_CHECK(cudaMemcpyAsync(mem_d_, v, 8, cudaMemcpyHostToDevice, s_));
  comm_->all_gather_f32(mem_d_, mem_d_ + 4, 2, s_);  // [rank0 ‖ rank1], 2 floats each
  float h[4];
  CUDA_CHECK(cudaMemcpyAsync(h, mem_d_ + 4, 16, cudaMemcpyDeviceToHost, s_));
  CUDA_CHECK(cudaStreamSynchronize(s_));
  return {double(h[0]), double(h[2])};
}

size_t Engine::kv_pool_init(double reserve_gib, double max_gib) {
  if (!pool_) return 0;
  for (int s = 0; s < c_.slots; s++) reset(s);
  CUDA_CHECK(cudaDeviceSynchronize());
  malloc_trim(0);
  const size_t have = pool_->total_chunks();
  const auto f = mem_free_gib();  // both ranks: the same numbers, so the same chunk count
  const double room = std::min(f[0], f[1]) - reserve_gib;
  long want = long(have) + long(std::floor(room * 1073741824.0 / KvPool::kChunk));
  if (max_gib > 0) want = std::min(want, long(max_gib * 1073741824.0 / KvPool::kChunk));
  const size_t least = kv_need(std::min<long>(c_.ctx, c_.max_rows + c_.logit_rows));
  if (want < long(least))
    qlog::fail("kv pool", "room for %.2f GiB of caches (MemFree rank0 %.1f, rank1 %.1f GiB, %.1f GiB kept free%s) — not "
               "even one %d-token sequence", want * double(KvPool::kChunk) / 1073741824.0, f[0], f[1], reserve_gib,
               max_gib > 0 ? ", capped" : "", c_.max_rows + c_.logit_rows);
  if (!pool_->resize(size_t(want))) qlog::fail("kv pool", "resize to %ld chunks: chunks still mapped", want);
  auto g = mem_free_gib();
  // a chunk takes a little more than its 2 MiB (the driver's bookkeeping: ~2 % measured, 0.23 GiB for 5,931 chunks):
  // what took a node under the reserve goes back — the same count on both ranks (both see both nodes' numbers)
  const long added = want - long(have);
  long drop = 0;
  for (int k = 0; k < 2 && added > 0; k++) {
    const double per = (f[k] - g[k]) / double(added);  // GiB a new chunk took on rank k
    if (g[k] < reserve_gib && per > 0) drop = std::max(drop, long(std::ceil((reserve_gib - g[k]) / per)));
  }
  if (drop) {
    want = std::max(long(have), want - drop);
    if (want < long(least)) qlog::fail("kv pool", "after the chunks' overhead: not even one sequence");
    if (!pool_->resize(size_t(want))) qlog::fail("kv pool", "resize to %ld chunks", want);
    g = mem_free_gib();
  }
  // one request at most: the longest sequence the whole pool holds (≤ ctx)
  long lo = 0, hi = c_.ctx;
  while (lo < hi) {
    const long m = (lo + hi + 1) / 2;
    if (kv_need(m) <= size_t(want)) lo = m;
    else hi = m - 1;
  }
  const double per_tok = kv_nd_ * double(kMlaRowBytes + kIdxRowBytes) + kv_nh_ * double(kKvRowBytes) / kRatio;
  qlog::ok("kv pool", "rank%d · %ld chunks · %.2f GiB · ≈ %.2fM tokens (%.0f B a token; one request ≤ %ld) · MemFree "
           "rank0 %.2f, rank1 %.2f GiB (before: %.2f / %.2f, %.1f kept free%s; %ld chunks back for their overhead)",
           c_.rank, want, want * double(KvPool::kChunk) / 1073741824.0, want * double(KvPool::kChunk) / per_tok / 1e6,
           per_tok, lo, g[0], g[1], f[0], f[1], reserve_gib, max_gib > 0 ? ", capped" : "", drop);
  return size_t(want);
}

F8Act Engine::act(const bf16* x, int ldx, int M, int k, int group) {
  F8Act a{xq_, xs_, k};
  if (group == 128 && f8_defer_act(M)) {  // P032 f8_fuse_act: the decode GEMMs quantize x themselves (x stays intact
    a.x = x, a.ldx = ldx;                 // until the last GEMM that reads this activation)
    return a;
  }
  act_quant(x, ldx, M, k, group, a, s_);
  return a;
}

// ---- attention: HCA -------------------------------------------------------------------------------------------------
// _a: x → q / kv / compressor → attention → VHA z partial (all-reduced); _b: z → wo_a → o_proj partial (all-reduced).
// Patches inside, per call on the lane's M rows: P078 tp_split (M ≤ 16: kv + compressor as column halves, gathered),
// P060 f8_fuse_ops (q / kv norms, σ, z ⊙ gate inside the GEMM launches), P042 hca_fuse (the 3-launch HCA chain).

int Engine::attn_hca_a(Layer& l, int il, const Lane& ln, bool async) {
  auto& h = l.h;
  const int M = ln.n, o = ln.o;
  const RowMeta* meta = d_meta_ + ln.o;
  bf16 *inp = at(inp_, o, kHidden), *t1 = at(t1_, o, 1536), *t2 = at(t2_, o, kQLora), *t3 = at(t3_, o, 4096);
  bf16 *t4 = at(t4_, o, kHeadDim), *q = at(q_, o, 16384), *ob = at(o_, o, 16384);
  float *zp = at(zp_, o, size_t(kVhaRank) * kHeadDim), *ckv = at(ckv_, o, 1024);
  uint8_t* rows = at(rows_, o, kKvRowBytes);
  // P094 sp_resid: q_down + kv (t1) and the compressor (ckv) of the lane's rows are there already (attn_front)
  F8Act xa{};
  if (!sp_on_) {
    xa = act(inp, kHidden, M, kHidden, 128);
    chk(s_, "act", __LINE__);
  }
  const bool tps = split_rows(M) && patch::on(patch::hca_fuse);
  // P074 hca_side (decode: one lane, P042 path): the compressor projection runs on s2_ beside the q / kv GEMMs and
  // norms; s_ waits for it (event) right before its rows are first read — the P078 gather, else hca_pre
  const bool side = M <= 16 && !async && !g_sync && patch::on(patch::hca_side) && patch::on(patch::hca_fuse);
  if (side) {
    CUDA_CHECK(cudaEventRecord(ev_in_, s_));
    CUDA_CHECK(cudaStreamWaitEvent(s2_, ev_in_, 0));
  }
  if (tps) {  // P078: q_down whole; kv and the compressor as this rank's column halves, gathered in one call
    bf16* kvh = reinterpret_cast<bf16*>(sp_);
    float *ckh = reinterpret_cast<float*>(sp_ + size_t(kSplitMaxRows) * 512), *cgh = ckh + size_t(kSplitMaxRows) * 256;
    if (side) {  // P074
      if (patch::on(patch::lin_pair)) {  // P114: both halves in one launch
        lin_forward_f32_pair(h.cvh, h.cgh, inp, ckh, cgh, M, s2_);
      } else {
        lin_forward_f32(h.cvh, inp, ckh, M, s2_);
        lin_forward_f32(h.cgh, inp, cgh, M, s2_);
      }
      CUDA_CHECK(cudaEventRecord(ev_ck_, s2_));
    }
    f8_forward_multi({{&h.qv, t1, 1536}, {&h.kvh, kvh, 256}}, xa, M, fws_, s_);
    chk(s_, "f8_forward", __LINE__);
    if (side) {
      CUDA_CHECK(cudaStreamWaitEvent(s_, ev_ck_, 0));
    } else if (patch::on(patch::lin_pair)) {  // P114: both halves in one launch
      lin_forward_f32_pair(h.cvh, h.cgh, inp, ckh, cgh, M, s_);
    } else {
      lin_forward_f32(h.cvh, inp, ckh, M, s_);
      lin_forward_f32(h.cgh, inp, cgh, M, s_);
    }
    chk(s_, "lin_forward_f32", __LINE__);
    const AgPiece pc[3] = {{kvh, t1 + kQLora, 512, 1536 * 2, 0, 512, 512},
                           {ckh, ckv, 1024, 4096, 0, 1024, 1024},
                           {cgh, ckv, 1024, 4096, 2048, 1024, 1024}};
    comm_->all_gather_rows(pc, 3, M, s_);
    chk(s_, "comm_->all_gather_rows", __LINE__);
  } else {
    if (side) {  // P074
      lin_forward_f32(h.comp, inp, ckv, M, s2_);
      CUDA_CHECK(cudaEventRecord(ev_ck_, s2_));
    }
    if (!sp_on_) f8_forward(h.qkv, xa, M, t1, 1536, fws_, s_);
    chk(s_, "f8_forward", __LINE__);
  }
  // P060 f8_fuse_ops: q_norm as q_up + gate's activation op, kv_norm as its side RMSNorm, σ as gate's epilogue (later:
  // z ⊙ gate as o_proj's activation op) — the misc kernels' bits, without their launches
  const bool fz = f8_fuse_ok(M);
  if (fz) {
    const F8Aux kv{t1 + kQLora, 1536, h.kvn, kHeadDim, t4, kHeadDim};
    if (a2a_on_)  // P100: the gate on the row owner (attn_own)
      f8_forward_multi({{&h.qup, q, 16384}}, f8_pro(kProRms, t1, 1536, h.qn, 0, xq_, xs_, kQLora), M, fws_, s_, &kv);
    else
      f8_forward_multi({{&h.qup, q, 16384}, {&h.gate, t3, 4096, kEpiSigmoid}},
                       f8_pro(kProRms, t1, 1536, h.qn, 0, xq_, xs_, kQLora), M, fws_, s_, &kv);
    chk(s_, "f8_forward", __LINE__);
  } else {
    rmsnorm(t1, 1536, h.qn, M, kQLora, t2, kQLora, s_);
    chk(s_, "rmsnorm", __LINE__);
    rmsnorm(t1 + kQLora, 1536, h.kvn, M, kHeadDim, t4, kHeadDim, s_);
    chk(s_, "rmsnorm", __LINE__);
    F8Act qa = act(t2, kQLora, M, kQLora, 128);
    chk(s_, "act", __LINE__);
    f8_forward_multi({{&h.qup, q, 16384}, {&h.gate, t3, 4096}}, qa, M, fws_, s_);  // P034: one launch at decode
    chk(s_, "f8_forward", __LINE__);
  }
  // P042 hca_fuse: q_prep happens inside the attention (q_ read raw); kv_rows + compress → hca_pre; the compressor
  // state update and the ring store move after the attention into hca_post (with vha_out); vha_partial is fused.
  // Lanes (hca.h row contract): every pointer here is advanced to the lane's first row o, and the lane's RowMeta carry
  // r0 − o (step(): d_meta_), so the step's keys — the window rows in rows_, a closing group's compressor rows in
  // ckv_, written by this or an earlier lane — are read at their absolute rows.
  const bool fuse = patch::on(patch::hca_fuse);
  if (!fuse) {
    hca_q_prep(q, meta, inv_hca_, M, s_);
    chk(s_, "hca_q_prep", __LINE__);
  }
  if (!fz) {
    sigmoid_inplace(t3, size_t(M) * 4096, s_);
    chk(s_, "sigmoid_inplace", __LINE__);
  }
  if (fuse) {
    if (side && !tps) CUDA_CHECK(cudaStreamWaitEvent(s_, ev_ck_, 0));  // P074: the compressor rows from s2_
    else if (!tps && !sp_on_) lin_forward_f32(h.comp, inp, ckv, M, s_);  // (P078 / P094: gathered above)
    chk(s_, "lin_forward_f32", __LINE__);
    if (verify_) dsp_stash(il, o, M);  // DSpark verify: the compressor projections, for a rejected group close
    hca_pre(t4, kHeadDim, ckv, h.ape, h.cnw, meta, inv_hca_, M, rows, hc_[il], aws_, aws_bytes_, s_);
    chk(s_, "hca_pre", __LINE__);
    hca_attn_fused(q, rows, meta, hc_[il], h.sink, inv_hca_, h.U, M, ln.max_hca, ob, zp, aws_, aws_bytes_, s_);
    chk(s_, "hca_attn_fused", __LINE__);
  } else {
    if (async) qlog::fail("model", "P075 ar_lanes needs P042 hca_fuse");
    hca_kv_rows(t4, kHeadDim, meta, inv_hca_, M, rows, s_);
    chk(s_, "hca_kv_rows", __LINE__);
    lin_forward_f32(h.comp, inp, ckv, M, s_);
    chk(s_, "lin_forward_f32", __LINE__);
    if (verify_) dsp_stash(il, o, M);
    hca_compress(ckv, h.ape, h.cnw, meta, inv_hca_, M, hc_[il], s_);
    chk(s_, "hca_compress", __LINE__);
    hca_attn(q, rows, meta, hc_[il], h.sink, M, ln.max_hca, ob, aws_, aws_bytes_, s_);
    chk(s_, "hca_attn", __LINE__);
    hca_ring_store(rows, meta, M, hc_[il], s_);
    chk(s_, "hca_ring_store", __LINE__);
    vha_partial(ob, kHeadDim, h.U, M, zp, s_);
    chk(s_, "vha_partial", __LINE__);
  }
  if (a2a_on_) {  // P100: o of the peer's rows to the peer, the peer's heads of this rank's rows into q_ (dead now)
    const int t = a2a(o_, q_, 16384 * 2, ln, async);
    chk(s_, "comm_->all_gather_rows", __LINE__);
    return t;
  }
  const size_t nz = size_t(M) * kVhaRank * kHeadDim;
  int t = -1;
  if (async) t = comm_->all_reduce_f32_async(zp, nz, s_);
  else comm_->all_reduce_f32(zp, nz, s_);
  chk(s_, "comm_->all_reduce_f32", __LINE__);
  return t;
}

int Engine::attn_hca_b(Layer& l, int il, const Lane& ln, bool async) {
  auto& h = l.h;
  const int M = ln.n, o = ln.o;
  const RowMeta* meta = d_meta_ + ln.o;
  bf16 *t2 = at(t2_, o, 4096), *t3 = at(t3_, o, 4096), *t4 = at(t4_, o, 4096), *ob = at(o_, o, 16384);
  bf16* aout = at(aout_, o, kHidden);
  const float* zp = at(zp_, o, size_t(kVhaRank) * kHeadDim);
  if (patch::on(patch::hca_fuse)) {
    hca_post(ob, zp, h.V, meta, inv_hca_, at(rows_, o, kKvRowBytes), at(ckv_, o, 1024), h.ape, M, hc_[il], oq_, os_, s_);
    chk(s_, "hca_post", __LINE__);
  } else {
    hca_vha_out(ob, zp, h.V, meta, inv_hca_, M, oq_, os_, s_);
    chk(s_, "hca_vha_out", __LINE__);
  }
  F8Act ga{oq_, os_, 4096};
  if (f8_fuse_ok(M) && f8_epi_fused(h.woa, M, kEpiMulQ128, fws_)) {
    // P104 f8_tile_epi: z ⊙ gate quantized as o_proj's FP8 activation in wo_a's store (z itself not written)
    f8_forward_out({&h.woa, nullptr, 4096, kEpiMulQ128, t3, 4096, xq_, xs_}, ga, M, fws_, s_, long(M) * 4096,
                   long(M) * 32);
    chk(s_, "f8_forward", __LINE__);
    f8_forward(h.oproj, F8Act{xq_, xs_, 4096}, M, aout, kHidden, fws_, s_);
    chk(s_, "f8_forward", __LINE__);
  } else if (f8_fuse_ok(M)) {  // P060: z ⊙ gate as o_proj's activation op
    f8_forward(h.woa, ga, M, t4, 4096, fws_, s_, long(M) * 4096, long(M) * 32);
    chk(s_, "f8_forward", __LINE__);
    f8_forward_out({&h.oproj, aout, kHidden}, f8_pro(kProMul, t4, 4096, t3, 4096, xq_, xs_, 4096), M, fws_, s_);
    chk(s_, "f8_forward", __LINE__);
  } else {
    f8_forward(h.woa, ga, M, t4, 4096, fws_, s_, long(M) * 4096, long(M) * 32);
    chk(s_, "f8_forward", __LINE__);
    mul_bf16(t4, 4096, t3, 4096, M, 4096, t2, 4096, s_);
    chk(s_, "mul_bf16", __LINE__);
    F8Act za = act(t2, 4096, M, 4096, 128);
    chk(s_, "act", __LINE__);
    f8_forward(h.oproj, za, M, aout, kHidden, fws_, s_);
    chk(s_, "f8_forward", __LINE__);
  }
  int t = -1;
  if (sp_on_) t = sp_rs(aout, size_t((M + 1) / 2) * kHidden, async);  // P094: each rank's rows of the sum
  else if (async) t = comm_->all_reduce_bf16_async(aout, size_t(M) * kHidden, s_);
  else comm_->all_reduce_bf16(aout, size_t(M) * kHidden, s_);
  chk(s_, "comm_->all_reduce_bf16", __LINE__);
  return t;
}

// ---- attention: DSA -------------------------------------------------------------------------------------------------
// _a: x → q / kv / indexer → top-2048 select → MLA attention → VHA z partial (all-reduced); _b: z → gate → o_proj
// partial (all-reduced). P060 inside (q_a / kv_a norms, σ), per call on the lane's M rows.

int Engine::attn_dsa_a(Layer& l, int id, const Lane& ln, bool async) {
  auto& d = l.d;
  const int M = ln.n, o = ln.o;
  const RowMeta* meta = d_meta_ + ln.o;
  // (P094 sp_resid: q_a ‖ kv_a gathered into spq_ — t1_ holds the earlier lanes' q576 then — and the indexer's k ‖
  // weights into kw, both by attn_front)
  bf16 *inp = at(inp_, o, kHidden), *t1 = at(sp_on_ ? spq_ : t1_, o, 1600), *t2 = at(t2_, o, kQLora);
  bf16* t3 = at(t3_, o, 8192);
  bf16 *t4 = at(t4_, o, kKvLora), *q = at(q_, o, 8192), *qi = at(o_, o, 8192), *kw = at(aout_, o, 192);
  bf16 *q576 = at(t1_, o, 32 * 576), *olat = at(o_, o, 32 * 512);
  uint8_t *qf4 = at(qf4_, o, 64 * 64), *qs = at(qs_, o, 256);
  float *wts = at(wts_, o, 64), *zp = at(zp_, o, size_t(kVhaRank) * kMlaV);
  int *list = at(d_list_, o, kIdxTop + kWindow), *cnt = d_cnt_ + o;
  // P140 dsa_sel_split (P094 lanes with a row of more than kIdxTop candidates): this rank selects for its own rows
  // [a, a + na) of the lane only; the peer's rows [b, b + nb) get their first kIdxTop list entries from the peer (an
  // in-place all-gather of each half's rows) and the rest + counts from dsa_select_tail. An odd lane: rank 1's half
  // ends in the scratch row (not selected; what it sends lands in rank 0's scratch row). P141 dsa_q_own: the indexer's
  // query (wq_b + its prep) only for rows [q0, q0 + nq) — the own rows, or none if no row of the lane selects (no
  // select reads a query there); its GEMM on the q_b group's quantized activation rows (16-byte aligned offsets)
  const bool sel_any = ln.max_pos - kWindow + 1 > kIdxTop;
  const bool split = sp_on_ && sel_any && patch::on(patch::dsa_sel_split);
  const int h = (M + 1) / 2, a = split ? c_.rank * h : 0, na = split ? (c_.rank ? M - h : h) : M;
  const int b = split ? (1 - c_.rank) * h : 0, nb = M - na;
  int q0 = 0, nq = M;
  if (sp_on_ && patch::on(patch::dsa_sel_split) && patch::on(patch::dsa_q_own)) {
    if (!sel_any) nq = 0;
    else if (a % 4 == 0) q0 = a, nq = na;
  }
  const bool iq_in = nq == M;  // wq_b in the q_b group (all rows)
  if (!sp_on_) {
    F8Act xa = act(inp, kHidden, M, kHidden, 128);
    chk(s_, "act", __LINE__);
    f8_forward(d.qkv, xa, M, t1, 1600, fws_, s_);
    chk(s_, "f8_forward", __LINE__);
  }
  if (f8_fuse_ok(M)) {  // P060 f8_fuse_ops: q_a norm as the activation op, kv_a norm as the side RMSNorm, σ as gate's epilogue
    const F8Aux kv{t1 + kQLora, 1600, d.kvan, kKvLora, t4, kKvLora};
    const F8Act qa = f8_pro(kProRms, t1, 1600, d.qan, 0, xq_, xs_, kQLora);
    if (a2a_on_ && iq_in)  // P100: the gate on the row owner (attn_own)
      f8_forward_multi({{&d.qb, q, 8192}, {&d.iq, qi, 8192}}, qa, M, fws_, s_, &kv);
    else if (a2a_on_)  // (P141: wq_b below)
      f8_forward_multi({{&d.qb, q, 8192}}, qa, M, fws_, s_, &kv);
    else if (iq_in)
      f8_forward_multi({{&d.qb, q, 8192}, {&d.gate, t3, 8192, kEpiSigmoid}, {&d.iq, qi, 8192}}, qa, M, fws_, s_, &kv);
    else
      f8_forward_multi({{&d.qb, q, 8192}, {&d.gate, t3, 8192, kEpiSigmoid}}, qa, M, fws_, s_, &kv);
    chk(s_, "f8_forward", __LINE__);
  } else {
    rmsnorm(t1, 1600, d.qan, M, kQLora, t2, kQLora, s_);
    chk(s_, "rmsnorm", __LINE__);
    rmsnorm(t1 + kQLora, 1600, d.kvan, M, kKvLora, t4, kKvLora, s_);
    chk(s_, "rmsnorm", __LINE__);
    F8Act qa = act(t2, kQLora, M, kQLora, 128);
    chk(s_, "act", __LINE__);
    // q_b, gate, indexer q [M, 64, 128] — P034: one launch at decode
    if (iq_in) f8_forward_multi({{&d.qb, q, 8192}, {&d.gate, t3, 8192}, {&d.iq, qi, 8192}}, qa, M, fws_, s_);
    else f8_forward_multi({{&d.qb, q, 8192}, {&d.gate, t3, 8192}}, qa, M, fws_, s_);
    chk(s_, "f8_forward", __LINE__);
    sigmoid_inplace(t3, size_t(M) * 8192, s_);
    chk(s_, "sigmoid_inplace", __LINE__);
  }
  if (!iq_in && nq > 0) {  // P141: wq_b on rows q0 … of the activation the q_b group quantized into xq_ / xs_ (M > 8)
    f8_forward(d.iq, F8Act{xq_ + size_t(q0) * kQLora, xs_ + size_t(q0) * (kQLora / 128), kQLora}, nq,
               at(qi, q0, 8192), 8192, fws_, s_);
    chk(s_, "f8_forward", __LINE__);
  }
  if (!sp_on_) lin_forward(d.ikw, inp, kw, M, s_);  // indexer k ‖ weights [M, 192]
  dsa_mla_rows(t4, kKvLora, t1 + 1536, 1600, meta, inv_dsa_, M, dc_[id], s_);
  chk(s_, "dsa_mla_rows", __LINE__);
  if (iq_in) {
    dsa_idx_prep(kw, qi, d.knw, d.knb, meta, inv_dsa_, M, dc_[id], qf4, qs, wts, s_);
  } else {  // P141: the query blocks for rows q0 … q0 + nq, the K rows (and weights) for the others
    auto kprep = [&](int r0, int n) {
      dsa_idx_prep(at(kw, r0, 192), nullptr, d.knw, d.knb, meta + r0, inv_dsa_, n, dc_[id], nullptr, nullptr,
                   at(wts, r0, 64), s_, false);
    };
    kprep(0, q0);
    dsa_idx_prep(at(kw, q0, 192), at(qi, q0, 8192), d.knw, d.knb, meta + q0, inv_dsa_, nq, dc_[id], at(qf4, q0, 4096),
                 at(qs, q0, 256), at(wts, q0, 64), s_);
    kprep(q0 + nq, M - q0 - nq);
  }
  chk(s_, "dsa_idx_prep", __LINE__);
  auto q_side = [&] {  // q576 = [q_nope · W_UK ‖ RoPE'd q_pe]
    dsa_q_pe(q, meta, inv_dsa_, M, q576, s_);
    chk(s_, "dsa_q_pe", __LINE__);
    if (!dsa_bmm_f8(q, 8192, 256, d.wuk8, d.kvs, 0, 32, kKvLora, kMlaNope, M, q576, 32 * 576, 576, s_))  // P073
      dsa_bmm(q, 8192, 256, d.wuk, 32, kKvLora, kMlaNope, M, q576, 32 * 576, 576, s_);
    chk(s_, "dsa_bmm", __LINE__);
  };
  if (split) {  // P140: select the own rows, exchange, q576 + the own rows' attention under it, then the peer's rows
    dsa_select(at(qf4, a, 4096), at(qs, a, 256), at(wts, a, 64), meta + a, dc_[id], na, ln.max_pos,
               at(list, a, kIdxTop + kWindow), cnt + a, aws_, aws_bytes_, s_);
    chk(s_, "dsa_select", __LINE__);
    dsa_select_tail(meta + b, nb, at(list, b, kIdxTop + kWindow), cnt + b, s_);
    chk(s_, "dsa_select_tail", __LINE__);
    const AgPiece pc = sp_piece(list, Lane{0, M}, (kIdxTop + kWindow) * 4);  // (rows of 8704 B, the first 8192 sent)
    const AgPiece pk{pc.src, pc.dst, pc.src_ld, pc.dst_ld, pc.off, pc.stride, kIdxTop * 4};
    const int tk = sp_gather(&pk, 1, ln, async);
    chk(s_, "comm_->all_gather_rows", __LINE__);
    q_side();
    dsa_attn(at(q576, a, 32 * 576), at(list, a, kIdxTop + kWindow), cnt + a, meta + a, dc_[id], d.sink, na, ln.max_dsa,
             at(olat, a, 32 * 512), aws_, aws_bytes_, s_);
    chk(s_, "dsa_attn", __LINE__);
    comm_->join(tk, s_);
    dsa_attn(at(q576, b, 32 * 576), at(list, b, kIdxTop + kWindow), cnt + b, meta + b, dc_[id], d.sink, nb, ln.max_dsa,
             at(olat, b, 32 * 512), aws_, aws_bytes_, s_);
    chk(s_, "dsa_attn peer", __LINE__);
  } else {
    q_side();
    dsa_select(qf4, qs, wts, meta, dc_[id], M, ln.max_pos, list, cnt, aws_, aws_bytes_, s_);
    chk(s_, "dsa_select", __LINE__);
    dsa_attn(q576, list, cnt, meta, dc_[id], d.sink, M, ln.max_dsa, olat, aws_, aws_bytes_, s_);
    chk(s_, "dsa_attn", __LINE__);
  }
  if (!dsa_bmm_f8(olat, 32 * 512, 512, d.wuv8, d.kvs, 1, 32, kMlaV, kKvLora, M, q, 8192, 256, s_))  // P073
    dsa_bmm(olat, 32 * 512, 512, d.wuv, 32, kMlaV, kKvLora, M, q, 8192, 256, s_);
  chk(s_, "dsa_bmm", __LINE__);
  if (a2a_on_) {  // P100: o (q_) of the peer's rows to the peer, the peer's heads of this rank's rows into q_'s upper half
    const int t = a2a(q_, q_ + size_t(c_.max_rows) * 8192, 8192 * 2, ln, async);
    chk(s_, "comm_->all_gather_rows", __LINE__);
    return t;
  }
  vha_partial(q, kMlaV, d.U, M, zp, s_);
  chk(s_, "vha_partial", __LINE__);
  const size_t nz = size_t(M) * kVhaRank * kMlaV;
  int t = -1;
  if (async) t = comm_->all_reduce_f32_async(zp, nz, s_);
  else comm_->all_reduce_f32(zp, nz, s_);
  chk(s_, "comm_->all_reduce_f32", __LINE__);
  return t;
}

int Engine::attn_dsa_b(Layer& l, int id, const Lane& ln, bool async) {
  auto& d = l.d;
  const int M = ln.n, o = ln.o;
  bf16 *q = at(q_, o, 8192), *t2 = at(t2_, o, 8192), *t3 = at(t3_, o, 8192), *aout = at(aout_, o, kHidden);
  const float* zp = at(zp_, o, size_t(kVhaRank) * kMlaV);
  dsa_vha_gate(q, zp, d.V, t3, M, t2, s_);
  chk(s_, "dsa_vha_gate", __LINE__);
  F8Act ya = act(t2, 8192, M, 8192, 128);
  chk(s_, "act", __LINE__);
  f8_forward(d.oproj, ya, M, aout, kHidden, fws_, s_);
  chk(s_, "f8_forward", __LINE__);
  int t = -1;
  if (sp_on_) t = sp_rs(aout, size_t((M + 1) / 2) * kHidden, async);  // P094
  else if (async) t = comm_->all_reduce_bf16_async(aout, size_t(M) * kHidden, s_);
  else comm_->all_reduce_bf16(aout, size_t(M) * kHidden, s_);
  chk(s_, "comm_->all_reduce_bf16", __LINE__);
  return t;
}

// ---- FFN ------------------------------------------------------------------------------------------------------------
// ffn_a: router → routing, fc1 (latent) + shared gate/up, experts, shared down → all-reduce(s); ffn_b (per lane, after
// the all-reduce): latent_norm → fc2, + shared. Per call on the part's / lane's M rows: P078 tp_split (M ≤ 16: router,
// fc1, fc2 as column halves, gathered), P060 f8_fuse_ops (the latent's MXFP8 copy, SiTU, latent_norm, + shared inside
// the GEMM launches).

// the MoE's all-reduced partials (bf16 inside the fp32 buffer part_): one lane — rows of [latent 2048 ‖ shared 4096]
// (one all-reduce); lanes — latent [rows, 2048] and shared [rows, 4096] in two blocks (separate all-reduces)
void Engine::ffn_bufs(bool lanes, bf16*& pl, int& ldl, bf16*& ps, int& lds) const {
  pl = reinterpret_cast<bf16*>(part_);
  if (lanes) ldl = kLatent, ps = pl + size_t(c_.max_rows) * kLatent, lds = kHidden;
  else ldl = 6144, ps = pl + kLatent, lds = 6144;
}

void Engine::ffn_a(Layer& l, const Lane& pt, const Lane* ln, int nl, bool lanes, bool async, int* tsh, int* tl) {
  auto& f = l.f;
  const int M = pt.n, o = pt.o;
  bf16 *inp = at(inp_, o, kHidden), *t1 = at(t1_, o, f.dense ? 14336 : 2048), *t2 = at(t2_, o, f.dense ? 7168 : kLatent);
  bf16 *t3 = at(t3_, o, 1024), *fout = at(fout_, o, kHidden);
  float *lgr = at(lgr_, o, 1024), *mw = at(mw_, o, kTopK);
  int* ids = at(d_moe_ids_, o, kTopK);
  uint8_t *lq = at(qf4_, o, kLatent), *ls = at(qs_, o, kLatent / 32);  // the latent as MXFP8 (qf4_/qs_ are free here)
  auto ar = [&](bf16* p, size_t n) {  // lanes: one all-reduce of rows, async (P075) or on s_
    if (async) return comm_->all_reduce_bf16_async(p, n, s_);
    comm_->all_reduce_bf16(p, n, s_);
    return -1;
  };
  if (f.dense) {
    F8Act xa = act(inp, kHidden, M, kHidden, 128);
    f8_forward(f.gu, xa, M, t1, 14336, fws_, s_);
    situ_bf16(t1, M, 7168, t2, s_);
    F8Act ha = act(t2, 7168, M, 7168, 128);
    f8_forward(f.down, ha, M, fout, kHidden, fws_, s_);
    if (!lanes) return comm_->all_reduce_bf16(fout, size_t(M) * kHidden, s_);
    for (int k = 0; k < nl; k++) tl[k] = ar(at(fout_, ln[k].o, kHidden), size_t(ln[k].n) * kHidden);
    return;
  }
  // P060 f8_fuse_ops: the latent's MXFP8 copy as fc1's epilogue, SiTU as the shared down's activation op (ffn_b:
  // latent_norm as fc2's and + shared as its epilogue)
  const bool fz = f8_fuse_ok(M);
  const bool fused = moe_fused_ok(M);  // P066: routing + (the latent's MXFP8) + experts + combine in one launch below
  if (split_rows(M)) {  // P078: router and fc1 as this rank's column halves, gathered in one call, then the routing
    float *r0h = reinterpret_cast<float*>(sp_), *r1h = r0h + size_t(kSplitMaxRows) * 256;
    bf16* lath = reinterpret_cast<bf16*>(r1h + size_t(kSplitMaxRows) * 256);
    if (patch::on(patch::lin_pair)) {  // P114: both halves in one launch
      lin_forward_f32_pair(f.r0h, f.r1h, inp, r0h, r1h, M, s_);
    } else {
      lin_forward_f32(f.r0h, inp, r0h, M, s_);
      lin_forward_f32(f.r1h, inp, r1h, M, s_);
    }
    chk(s_, "lin_forward_f32", __LINE__);
    F8Act xa = act(inp, kHidden, M, kHidden, 128);
    chk(s_, "act", __LINE__);
    if (fz) {  // + P060: the half's MXFP8 copy (groups of 32 inside the half: the whole row's bytes) is what is gathered
      uint8_t *lqh = reinterpret_cast<uint8_t*>(lath + size_t(kSplitMaxRows) * 1024), *lsh = lqh + kSplitMaxRows * 1024;
      f8_forward_multi({{&f.fc1h, lath, 1024, kEpiQ32, nullptr, 0, lqh, lsh}, {&f.gu, t1, 2048}}, xa, M, fws_, s_);
      chk(s_, "f8_forward", __LINE__);
      const AgPiece pc[4] = {{r0h, lgr, 1024, 4096, 0, 1024, 1024}, {r1h, lgr, 1024, 4096, 2048, 1024, 1024},
                             {lqh, lq, 1024, kLatent, 0, 1024, 1024}, {lsh, ls, 32, kLatent / 32, 0, 32, 32}};
      comm_->all_gather_rows(pc, 4, M, s_);
    } else {
      f8_forward_multi({{&f.fc1h, lath, 1024}, {&f.gu, t1, 2048}}, xa, M, fws_, s_);  // P034: one launch
      chk(s_, "f8_forward", __LINE__);
      const AgPiece pc[3] = {{r0h, lgr, 1024, 4096, 0, 1024, 1024}, {r1h, lgr, 1024, 4096, 2048, 1024, 1024},
                             {lath, t2, 2048, kLatent * 2, 0, 2048, 2048}};
      comm_->all_gather_rows(pc, 3, M, s_);
    }
    chk(s_, "comm_->all_gather_rows", __LINE__);
    if (!fused) {
      moe_route(lgr, f.bias, f.rscale, M, ids, mw, s_);
      chk(s_, "moe_route", __LINE__);
    }
  } else {
    lin_forward_f32(f.router, inp, lgr, M, s_);
    chk(s_, "lin_forward_f32", __LINE__);
    if (!fused) {
      moe_route(lgr, f.bias, f.rscale, M, ids, mw, s_);
      chk(s_, "moe_route", __LINE__);
    }
    F8Act xa = act(inp, kHidden, M, kHidden, 128);
    chk(s_, "act", __LINE__);
    if (fz) f8_forward_multi({{&f.fc1, t2, kLatent, kEpiQ32, nullptr, 0, lq, ls}, {&f.gu, t1, 2048}}, xa, M, fws_, s_);
    else f8_forward_multi({{&f.fc1, t2, kLatent}, {&f.gu, t1, 2048}}, xa, M, fws_, s_);  // P034: one launch at decode
    chk(s_, "f8_forward", __LINE__);
  }
  if (!fz && !fused) {  // (P066 quantizes the bf16 latent itself)
    act_quant(t2, kLatent, M, kLatent, 32, F8Act{lq, ls, kLatent}, s_);
    chk(s_, "act_quant", __LINE__);
  }
  // The combine as vLLM at TP = 2: this rank's routed latent and shared-expert output leave in bf16, one bf16
  // all-reduce of [latent 2048 ‖ shared 4096] (= vLLM's two), latent_norm → fc2, then bf16 + bf16. Lanes: the shared
  // expert first, its all-reduce under the experts, then the latent's per lane (the same sums, element by element).
  bf16 *pl, *ps;
  int ldl, lds;
  ffn_bufs(lanes, pl, ldl, ps, lds);
  pl = at(pl, o, ldl), ps = at(ps, o, lds);
  auto shared = [&] {
    if (fz) {
      f8_forward_out({&f.down, ps, lds}, f8_pro(kProSitu, t1, 2048, nullptr, 0, xq_, xs_, 1024), M, fws_, s_);
      chk(s_, "f8_forward", __LINE__);
      return;
    }
    situ_bf16(t1, M, 1024, t3, s_);
    chk(s_, "situ_bf16", __LINE__);
    F8Act ha = act(t3, 1024, M, 1024, 128);
    chk(s_, "act", __LINE__);
    f8_forward(f.down, ha, M, ps, lds, fws_, s_);
    chk(s_, "f8_forward", __LINE__);
  };
  if (lanes) {
    shared();
    *tsh = ar(ps, size_t(M) * kHidden);
  }
  if (fused)  // the latent as fc1's MXFP8 copy (P060) or bf16 (quantized inside, act_quant's arithmetic)
    moe_fused(f.ex, moe_, lgr, f.bias, f.rscale, fz ? nullptr : t2, kLatent, fz ? lq : nullptr, fz ? ls : nullptr, M, ids,
              mw, pl, ldl, s_);
  else
    moe_experts_bf16(f.ex, moe_, lq, ls, ids, mw, M, pl, ldl, s_);
  chk(s_, "moe_experts", __LINE__);
  if (lanes) {
    for (int k = 0; k < nl; k++) tl[k] = ar(at(pl, ln[k].o - o, kLatent), size_t(ln[k].n) * kLatent);
    return;
  }
  shared();
  comm_->all_reduce_bf16(pl, size_t(M) * 6144, s_);
  chk(s_, "comm_->all_reduce_bf16", __LINE__);
}

void Engine::ffn_b(Layer& l, const Lane& ln, bool lanes) {
  auto& f = l.f;
  if (f.dense) return;
  const int M = ln.n, o = ln.o;
  bf16 *pl, *ps;
  int ldl, lds;
  ffn_bufs(lanes, pl, ldl, ps, lds);
  pl = at(pl, o, ldl), ps = at(ps, o, lds);
  bf16 *t2 = at(t2_, o, kLatent), *t4 = at(t4_, o, kHidden), *fout = at(fout_, o, kHidden);
  const bool fz = f8_fuse_ok(M);  // P060: latent_norm as fc2's activation op, + shared as its epilogue
  const F8Act rn = f8_pro(kProRms, pl, ldl, f.lnorm, 0, xq_, xs_, kLatent);
  F8Act ra{};
  if (!fz) {
    rmsnorm(pl, ldl, f.lnorm, M, kLatent, t2, kLatent, s_);
    chk(s_, "rmsnorm", __LINE__);
    ra = act(t2, kLatent, M, kLatent, 128);
    chk(s_, "act", __LINE__);
  }
  if (split_rows(M)) {  // P078: this rank's half of fc2's outputs and of the sum with the shared expert, gathered
    bf16 *yh = reinterpret_cast<bf16*>(sp_), *fh = yh + size_t(kSplitMaxRows) * 2048;
    const bf16* sh = ps + c_.rank * 2048;
    if (fz) {
      f8_forward_out({&f.fc2h, fh, 2048, kEpiAdd, sh, lds}, rn, M, fws_, s_);
      chk(s_, "f8_forward", __LINE__);
    } else {
      f8_forward(f.fc2h, ra, M, yh, 2048, fws_, s_);
      chk(s_, "f8_forward", __LINE__);
      add_bf16(yh, sh, lds, M, 2048, fh, s_);
      chk(s_, "add_bf16", __LINE__);
    }
    const AgPiece pc{fh, fout, 4096, kHidden * 2, 0, 4096, 4096};
    comm_->all_gather_rows(&pc, 1, M, s_);
    chk(s_, "comm_->all_gather_rows", __LINE__);
    return;
  }
  if (fz) {
    f8_forward_out({&f.fc2, fout, kHidden, kEpiAdd, ps, lds}, rn, M, fws_, s_);
    chk(s_, "f8_forward", __LINE__);
  } else {
    f8_forward(f.fc2, ra, M, t4, kHidden, fws_, s_);
    chk(s_, "f8_forward", __LINE__);
    add_bf16(t4, ps, lds, M, kHidden, fout, s_);
    chk(s_, "add_bf16", __LINE__);
  }
}

// ---- P094 sp_resid: sequence-parallel residual (prefill lanes) -----------------------------------------------------
// Each lane's rows are split in halves of h = ⌈n/2⌉ rows: rank r owns [o + r·h, o + r·h + h) — its rows of the residual
// R, the mHC post / pre, and the per-row work both ranks did for every row: the attention's q_down + kv (HCA) / q_a +
// kv_a (DSA) and the compressor / the indexer's k ‖ weights; the FFN's FP8 input copy, router + routing, fc1 and the
// latent's MXFP8 copy; after the reduce, latent_norm → fc2 + the shared sum. An odd lane (the step's last, M odd) gives
// rank 1 one scratch row past the lane (row M < max_rows): computed from whatever is there, never read. The
// tensor-parallel parts (heads, groups, intermediates) still run on every row of the lane: what they read is
// all-gathered as the bytes the owner wrote (deterministic per-row kernels — the bytes either rank computes); where the
// step all-reduced partials (o_proj, the MoE latent and shared expert, the dense FFN) a reduce-scatter leaves each rank
// exactly the all-reduce's value for its rows (Comm::reduce_scatter_bf16). Logit rows and DSpark's context rows are
// exchanged from their owners at the end (sp_rows).
Engine::Lane Engine::sp_own(const Lane& ln) const {
  Lane m = ln;
  m.n = (ln.n + 1) / 2;
  m.o = ln.o + c_.rank * m.n;
  return m;
}
int Engine::sp_owner(int row) const {
  for (int k = 0; k + 1 < int(sp_lb_.size()); k++)
    if (row < sp_lb_[k + 1]) return row - sp_lb_[k] < (sp_lb_[k + 1] - sp_lb_[k] + 1) / 2 ? 0 : 1;
  return 0;
}
AgPiece Engine::sp_piece(void* base, const Lane& ln, int w) const {
  const long long h = (ln.n + 1) / 2;
  char* d = static_cast<char*>(base) + size_t(ln.o) * w;
  return AgPiece{d + c_.rank * h * w, d, w, w, 0, h * w, w};
}
int Engine::sp_gather(const AgPiece* p, int np, const Lane& ln, bool async) {
  const int h = (ln.n + 1) / 2;
  if (async) return comm_->all_gather_rows_async(p, np, h, s_);
  comm_->all_gather_rows(p, np, h, s_);
  return -1;
}
int Engine::sp_rs(bf16* lane, size_t half, bool async) {
  if (async) return comm_->reduce_scatter_bf16_async(lane, half, s_);
  comm_->reduce_scatter_bf16(lane, half, s_);
  return -1;
}

int Engine::attn_front(Layer& l, const Lane& ln, bool async) {
  const Lane me = sp_own(ln);
  const int M = me.n, o = me.o;
  const bf16* inp = at(inp_, o, kHidden);
  F8Act xa = act(inp, kHidden, M, kHidden, 128);
  chk(s_, "act", __LINE__);
  AgPiece pc[2];
  if (!l.dsa) {
    f8_forward(l.h.qkv, xa, M, at(t1_, o, 1536), 1536, fws_, s_);
    chk(s_, "f8_forward", __LINE__);
    lin_forward_f32(l.h.comp, inp, at(ckv_, o, 1024), M, s_);
    chk(s_, "lin_forward_f32", __LINE__);
    pc[0] = sp_piece(t1_, ln, 1536 * 2), pc[1] = sp_piece(ckv_, ln, 1024 * 4);
  } else {
    f8_forward(l.d.qkv, xa, M, at(spq_, o, 1600), 1600, fws_, s_);
    chk(s_, "f8_forward", __LINE__);
    lin_forward(l.d.ikw, inp, at(aout_, o, 192), M, s_);  // indexer k ‖ weights (attn_dsa_a's kw)
    chk(s_, "lin_forward", __LINE__);
    pc[0] = sp_piece(spq_, ln, 1600 * 2), pc[1] = sp_piece(aout_, ln, 192 * 2);
  }
  const int t = sp_gather(pc, 2, ln, async);
  chk(s_, "comm_->all_gather_rows", __LINE__);
  return t;
}

int Engine::ffn_front(Layer& l, const Lane& ln, bool async) {
  auto& f = l.f;
  const Lane me = sp_own(ln);
  const int M = me.n, o = me.o;
  const bf16* inp = at(inp_, o, kHidden);
  const F8Act xa{at(spxq_, o, kHidden), at(spxs_, o, kHidden / 128), kHidden};  // act()'s bytes, kept per row
  act_quant(inp, kHidden, M, kHidden, 128, xa, s_);
  chk(s_, "act", __LINE__);
  AgPiece pc[5] = {sp_piece(spxq_, ln, kHidden), sp_piece(spxs_, ln, kHidden / 128)};
  int np = 2;
  if (!f.dense) {
    float* lgr = at(lgr_, o, 1024);
    int* ids = at(d_moe_ids_, o, kTopK);
    float* mw = at(mw_, o, kTopK);
    lin_forward_f32(f.router, inp, lgr, M, s_);
    chk(s_, "lin_forward_f32", __LINE__);
    moe_route(lgr, f.bias, f.rscale, M, ids, mw, s_);
    chk(s_, "moe_route", __LINE__);
    bf16* t2 = at(t2_, o, kLatent);
    uint8_t *lq = at(qf4_, o, kLatent), *ls = at(qs_, o, kLatent / 32);
    if (f8_fuse_ok(M)) {  // P060: the latent's MXFP8 copy as fc1's epilogue
      f8_forward_out({&f.fc1, t2, kLatent, kEpiQ32, nullptr, 0, lq, ls}, xa, M, fws_, s_);
      chk(s_, "f8_forward", __LINE__);
    } else {
      f8_forward(f.fc1, xa, M, t2, kLatent, fws_, s_);
      chk(s_, "f8_forward", __LINE__);
      act_quant(t2, kLatent, M, kLatent, 32, F8Act{lq, ls, kLatent}, s_);
      chk(s_, "act_quant", __LINE__);
    }
    // routing rows [ids 10 ‖ weights 10] (80 B: a gather piece)
    uint8_t* rt = sprt_ + size_t(o) * 80;
    CUDA_CHECK(cudaMemcpy2DAsync(rt, 80, ids, kTopK * 4, kTopK * 4, M, cudaMemcpyDeviceToDevice, s_));
    CUDA_CHECK(cudaMemcpy2DAsync(rt + 40, 80, mw, kTopK * 4, kTopK * 4, M, cudaMemcpyDeviceToDevice, s_));
    pc[2] = sp_piece(qf4_, ln, kLatent), pc[3] = sp_piece(qs_, ln, kLatent / 32), pc[4] = sp_piece(sprt_, ln, 80);
    np = 5;
    if (shs_on_) pc[0] = pc[2], pc[1] = pc[3], pc[2] = pc[4], np = 3;  // P136: the shared expert reads this rank's rows
  }
  const int t = sp_gather(pc, np, ln, async);
  chk(s_, "comm_->all_gather_rows", __LINE__);
  return t;
}

void Engine::ffn_tp(Layer& l, const Lane& pt, const Lane* ln, int nl, bool async, int* tf, int* tsh, int* tl) {
  auto& f = l.f;
  const int M = pt.n, o = pt.o;
  if (f.dense) {
    for (int k = 0; k < nl; k++) comm_->join(tf[k], s_), tf[k] = -1;
    const F8Act xa{at(spxq_, o, kHidden), at(spxs_, o, kHidden / 128), kHidden};
    bf16 *t1 = at(t1_, o, 14336), *t2 = at(t2_, o, 7168), *fout = at(fout_, o, kHidden);
    f8_forward(f.gu, xa, M, t1, 14336, fws_, s_);
    situ_bf16(t1, M, 7168, t2, s_);
    F8Act ha = act(t2, 7168, M, 7168, 128);
    f8_forward(f.down, ha, M, fout, kHidden, fws_, s_);
    chk(s_, "f8_forward", __LINE__);
    for (int k = 0; k < nl; k++) tl[k] = sp_rs(at(fout_, ln[k].o, kHidden), size_t((ln[k].n + 1) / 2) * kHidden, async);
    chk(s_, "comm_->reduce_scatter_bf16", __LINE__);
    return;
  }
  const bool fz = f8_fuse_ok(M);
  bf16 *pl, *ps;
  int ldl, lds;
  ffn_bufs(true, pl, ldl, ps, lds);
  // P102 sp_late_shared: the last `late` lanes' shared experts after the experts — the latent reduce-scatters, issued
  // right after the experts, then move under them (the next layer's first tail waits for its lane's latent otherwise)
  const int late = [&] {
    const char* e = std::getenv("E55_SP_LATE");
    return std::min(nl, std::max(0, e ? std::atoi(e) : patch::on(patch::sp_late_shared) ? kSpLate : 0));
  }();
  auto shared = [&](int k) {
    if (shs_on_) return ffn_shared(l, ln[k]);  // P136: no reduce-scatter (tsh[k] stays −1)
    const Lane& x = ln[k];
    const F8Act xa{at(spxq_, x.o, kHidden), at(spxs_, x.o, kHidden / 128), kHidden};
    bf16 *t1 = at(t1_, x.o, 2048), *t3 = at(t3_, x.o, 1024), *pk = at(ps, x.o, lds);
    if (fz && f8_epi_fused(f.gu, x.n, kEpiSituQ128, fws_)) {  // P134: SiTU + down's FP8 activation in gate/up's store
      f8_forward_out({&f.gu, nullptr, 1024, kEpiSituQ128, nullptr, 0, xq_, xs_}, xa, x.n, fws_, s_);
      chk(s_, "f8_forward", __LINE__);
      f8_forward(f.down, F8Act{xq_, xs_, 1024}, x.n, pk, lds, fws_, s_);
      chk(s_, "f8_forward", __LINE__);
      tsh[k] = sp_rs(pk, size_t((x.n + 1) / 2) * kHidden, async);
      return;
    }
    f8_forward(f.gu, xa, x.n, t1, 2048, fws_, s_);
    chk(s_, "f8_forward", __LINE__);
    if (fz) {
      f8_forward_out({&f.down, pk, lds}, f8_pro(kProSitu, t1, 2048, nullptr, 0, xq_, xs_, 1024), x.n, fws_, s_);
    } else {
      situ_bf16(t1, x.n, 1024, t3, s_);
      F8Act ha = act(t3, 1024, x.n, 1024, 128);
      f8_forward(f.down, ha, x.n, pk, lds, fws_, s_);
    }
    chk(s_, "f8_forward", __LINE__);
    tsh[k] = sp_rs(pk, size_t((x.n + 1) / 2) * kHidden, async);
  };
  for (int k = 0; k < nl; k++) {  // per lane (its gather in): the peer's routing rows, the shared expert, its reduce
    const Lane& x = ln[k];
    const int h = (x.n + 1) / 2, po = x.o + (1 - c_.rank) * h;
    comm_->join(tf[k], s_), tf[k] = -1;
    CUDA_CHECK(cudaMemcpy2DAsync(at(d_moe_ids_, po, kTopK), kTopK * 4, sprt_ + size_t(po) * 80, 80, kTopK * 4, h,
                                 cudaMemcpyDeviceToDevice, s_));
    CUDA_CHECK(cudaMemcpy2DAsync(at(mw_, po, kTopK), kTopK * 4, sprt_ + size_t(po) * 80 + 40, 80, kTopK * 4, h,
                                 cudaMemcpyDeviceToDevice, s_));
    if (shs_on_ ? sps_at(k) == 'b' : k < nl - late) shared(k);
  }
  comm_->mark("X", g_layer, s_);  // (E55_COMM_TRACE: the experts' span)
  moe_experts_bf16(f.ex, moe_, at(qf4_, o, kLatent), at(qs_, o, kLatent / 32), at(d_moe_ids_, o, kTopK), at(mw_, o, kTopK),
                   M, at(pl, o, ldl), ldl, s_);
  chk(s_, "moe_experts", __LINE__);
  comm_->mark("Y", g_layer, s_);
  for (int k = 0; k < nl; k++) tl[k] = sp_rs(at(pl, ln[k].o, ldl), size_t((ln[k].n + 1) / 2) * kLatent, async);
  chk(s_, "comm_->reduce_scatter_bf16", __LINE__);
  for (int k = 0; k < nl; k++)
    if (shs_on_ ? sps_at(k) == 'a' : k >= nl - late) shared(k);
}

// P136 sp_shared: the shared expert of this rank's rows of lane ln with both intermediate halves — gate/up of this
// rank's rows and of the peer's (one activation: act()'s bytes of these rows, spxq_), each half's down as that rank's
// GEMM makes it (bf16: the reduce-scatter's operands, row by row — the GEMMs are batch-invariant), the peer's half's
// down adding this rank's partial in its epilogue (kEpiAdd bf16(h + e) = the reduce-scatter's bf16(a + b)) → the rows of
// the shared sum (ps) the reduce-scatter leaves this rank. Scratch t2_ / t4_ (free between the lanes' output sides).
void Engine::ffn_shared(Layer& l, const Lane& ln) {
  auto& f = l.f;
  const Lane me = sp_own(ln);
  const int M = me.n;
  bf16 *pl, *ps;
  int ldl, lds;
  ffn_bufs(true, pl, ldl, ps, lds);
  const F8Act xa{at(spxq_, me.o, kHidden), at(spxs_, me.o, kHidden / 128), kHidden};
  bf16 *t = t2_, *pm = t4_, *pk = at(ps, me.o, lds);
  f8_forward_multi({{&f.gu, t, 4096}, {&f.gup, t + 2048, 4096}}, xa, M, fws_, s_);
  chk(s_, "f8_forward", __LINE__);
  f8_forward_out({&f.down, pm, kHidden}, f8_pro(kProSitu, t, 4096, nullptr, 0, xq_, xs_, 1024), M, fws_, s_);
  f8_forward_out({&f.downp, pk, lds, kEpiAdd, pm, kHidden}, f8_pro(kProSitu, t + 2048, 4096, nullptr, 0, xq_, xs_, 1024),
                 M, fws_, s_);
  chk(s_, "f8_forward", __LINE__);
}

// ---- P100 a2a_attn: the attention's output side on the row owner -------------------------------------------------
// The attention core stays heads-split on every row of the lane; its output o (this rank's 32 heads) goes to each row's
// owner (sp_own), which then has all 64 heads of its rows and runs, for them only, what both ranks ran for every row:
// z = the two ranks' VHA partials (this rank's as its kernel made it, the peer's by the same chain) added as the
// all-reduce adds them; per rank's half of the heads VHA out (its V) → its 4 wo_a groups, its gate columns, its K half
// of o_proj (each partial bf16 as that rank's GEMM), the two partials added as the reduce-scatter adds them → aout's
// rows bit for bit. hca_post's ring / state writes stay on every row (KV is replicated). No z all-reduce (32 KiB a row
// fp32), no o_proj reduce-scatter: one exchange of half the lane's o rows (HCA 32 KiB, DSA 16 KiB a row).
int Engine::a2a(const void* send, void* recv, int w, const Lane& ln, bool async) {
  const int h = (ln.n + 1) / 2, r = c_.rank;
  const char* sr = static_cast<const char*>(send) + size_t(ln.o + (1 - r) * h) * w;  // the peer's half: sent
  char* rv = static_cast<char*>(recv) + size_t(ln.o + r * h) * w;                     // this rank's half: received
  if (zr_on_) {  // P176: the peer's half (staged from send) straight into the peer's receive rows of this lane — the
    // same offset in its zr_ as here (a2a_rx: both ranks count the lanes' own halves alike); this rank's half arrives
    // the same way at a2a_rx(ln) here (attn_own reads it from there)
    const ZcPut x{sr, zr_ + a2a_rx(ln) * kZcRowBytes, size_t(h) * w};
    if (async) return comm_->put_async(&x, 1, s_);
    comm_->put(&x, 1, s_);
    return -1;
  }
  // an all-gather piece whose own part lands in place (rdma_ar skips it): rank r's part at dst + r · stride = sr, the
  // peer's at dst + (1 − r) · stride = rv
  char* s0 = const_cast<char*>(sr);
  const AgPiece pc = r == 0 ? AgPiece{sr, s0, w, w, 0, rv - s0, w} : AgPiece{sr, rv, w, w, 0, s0 - rv, w};
  if (async) return comm_->all_gather_rows_async(&pc, 1, h, s_);
  comm_->all_gather_rows(&pc, 1, h, s_);
  return -1;
}

size_t Engine::a2a_rx(const Lane& ln) const {
  size_t c = 0;
  for (int k = 0; k + 1 < int(sp_lb_.size()) && sp_lb_[k] < ln.o; k++) c += size_t(sp_lb_[k + 1] - sp_lb_[k] + 1) / 2;
  return c;
}

// P101 a2a_sched: the gates of this rank's rows of lane ln (both halves, σ) into g [rows][2 · half] (HCA half 4096, DSA
// 8192): their inputs are this rank's own rows (q_lora / q_a, no exchange), so they run before the exchanges' joins
void Engine::attn_gate(Layer& l, int i, const Lane& ln, bf16* g) {
  const Lane me = sp_own(ln);
  const int M = me.n, o = me.o;
  if (!l.dsa) {
    f8_forward_multi({{&l.h.gate, g, 4096, kEpiSigmoid}, {&a2h_.at(i).gate, g + size_t(M) * 4096, 4096, kEpiSigmoid}},
                     f8_pro(kProRms, at(t1_, o, 1536), 1536, l.h.qn, 0, xq_, xs_, kQLora), M, fws_, s_);
  } else {
    f8_forward_multi({{&l.d.gate, g, 8192, kEpiSigmoid}, {&a2d_.at(i).gate, g + size_t(M) * 8192, 8192, kEpiSigmoid}},
                     f8_pro(kProRms, at(spq_, o, 1600), 1600, l.d.qan, 0, xq_, xs_, kQLora), M, fws_, s_);
  }
  chk(s_, "f8_forward", __LINE__);
}

void Engine::attn_own(Layer& l, int i, const Lane& ln, bf16* g) {
  const Lane me = sp_own(ln);
  const int M = me.n, o = me.o;
  const RowMeta* meta = d_meta_ + o;
  bf16* aout = at(aout_, o, kHidden);
  const size_t MR = c_.max_rows;
  if (!l.dsa) {
    auto& h = l.h;
    const A2aW& p = a2h_.at(i);
    if (!g)  // (P101: run for every lane before the joins)
      hca_post_state(d_meta_ + ln.o, at(rows_, ln.o, kKvRowBytes), at(ckv_, ln.o, 1024), h.ape, ln.n, hc_[i], s_);
    chk(s_, "hca_post", __LINE__);
    const float* zp = at(zp_, o, size_t(kVhaRank) * kHeadDim);
    const bf16 *om = at(o_, o, 16384), *op = at(q_, o, 16384);  // o of this rank's heads · of the peer's (received)
    if (zr_on_) op = reinterpret_cast<const bf16*>(zr_ + a2a_rx(ln) * kZcRowBytes);  // (P176)
    uint8_t *qm = oq_, *qp = oq_ + size_t(4) * M * 4096, *sm = os_, *sq = os_ + size_t(4) * M * 32;
    hca_vha_own(om, op, zp, p.U, h.V, p.V, meta, inv_hca_, M, qm, sm, qp, sq, s_);  // z, VHA out of all 64 heads
    chk(s_, "hca_vha_out", __LINE__);
    bf16 *t4m = t4_, *t4p = t4_ + size_t(M) * 4096, *pm = t2_;
    f8_forward(h.woa, F8Act{qm, sm, 4096}, M, t4m, 4096, fws_, s_, long(M) * 4096, long(M) * 32);
    f8_forward(p.woa, F8Act{qp, sq, 4096}, M, t4p, 4096, fws_, s_, long(M) * 4096, long(M) * 32);
    chk(s_, "f8_forward", __LINE__);
    // the gates of these rows: q_norm + quantization (attn_hca_a's prologue, per row), each half with σ (P101: given)
    if (!g) attn_gate(l, i, ln, g = t3_);
    bf16 *gm = g, *gp = g + size_t(M) * 4096;
    // o_proj: this rank's K half → bf16 partial; the peer's K half + that partial (P060 kEpiAdd: bf16(h + e))
    f8_forward_out({&h.oproj, pm, kHidden}, f8_pro(kProMul, t4m, 4096, gm, 4096, xq_, xs_, 4096), M, fws_, s_);
    f8_forward_out({&p.oproj, aout, kHidden, kEpiAdd, pm, kHidden}, f8_pro(kProMul, t4p, 4096, gp, 4096, xq_, xs_, 4096),
                   M, fws_, s_);
    chk(s_, "f8_forward", __LINE__);
    return;
  }
  auto& d = l.d;
  const A2aW& p = a2d_.at(i);
  float* z = at(zp_, o, size_t(kVhaRank) * kMlaV);
  const bf16 *om = at(q_, o, 8192), *op = q_ + (MR + o) * 8192;
  if (zr_on_) op = reinterpret_cast<const bf16*>(zr_ + a2a_rx(ln) * kZcRowBytes);  // (P176)
  vha_z(op, p.U, om, d.U, nullptr, kMlaV, M, z, s_);  // both partials (this rank's: vha_partial's chain)
  chk(s_, "vha_z", __LINE__);
  if (!g) attn_gate(l, i, ln, g = t3_);
  bf16 *gm = g, *gp = g + size_t(M) * 8192, *ym = t2_, *yp = t2_ + size_t(M) * 8192, *pm = t4_;
  dsa_vha_gate(om, z, d.V, gm, M, ym, s_);
  dsa_vha_gate(op, z, p.V, gp, M, yp, s_);
  chk(s_, "dsa_vha_gate", __LINE__);
  F8Act ya = act(ym, 8192, M, 8192, 128);
  f8_forward(d.oproj, ya, M, pm, kHidden, fws_, s_);
  F8Act yb = act(yp, 8192, M, 8192, 128);
  f8_forward_out({&p.oproj, aout, kHidden, kEpiAdd, pm, kHidden}, yb, M, fws_, s_);
  chk(s_, "f8_forward", __LINE__);
}

// P094: rows whose bytes only their owner has (the others' are stale) → right on both ranks: all-gathered side by side
// into q_ (free at the step's end), then each row taken from its owner
void Engine::sp_rows(void* buf, long ld, int w, const int* own, int n) {
  if (n <= 0) return;
  const AgPiece pc{buf, q_, ld, 2LL * w, 0, w, w};
  comm_->all_gather_rows(&pc, 1, n, s_);
  pick_rows(q_, 2L * w, w, own, n, buf, ld, s_);
  chk(s_, "comm_->all_gather_rows", __LINE__);
}

void Engine::embed(int M) {
  if (!emb_half_) return embed4(embed_, d_ids_, M, R_, s_);
  // P095 embed_split: streams 2 / 3 of each row of R_ take the row as rank 0 / 1 embeds it (its half's tokens; the
  // rest stale), exchanged in place, then the owner's copy into all four streams
  bf16 *E = R_ + 2 * kHidden, *mine = E + c_.rank * kHidden;
  embed_half(embed_, d_ids_, M, c_.rank * kVocabHalf, kVocabHalf, mine, kHcDim, s_);
  const AgPiece pc{mine, E, kHcDim * 2, kHcDim * 2, 0, kHidden * 2, kHidden * 2};
  comm_->all_gather_rows(&pc, 1, M, s_);
  embed4_pick(d_ids_, M, kVocabHalf, R_, s_);
}

void Engine::set_taps(const std::vector<int>& rows) {
  taps_ = rows;
  cudaFree(d_taps_), cudaFree(tap_r_), cudaFree(tap_a_), cudaFree(tap_f_), cudaFree(tap_route_), cudaFree(tap_m_);
  cudaFree(tap_p_), cudaFree(tap_c_);
  d_taps_ = nullptr, tap_r_ = tap_a_ = tap_f_ = tap_m_ = nullptr, tap_route_ = nullptr, tap_p_ = tap_c_ = nullptr;
  if (rows.empty()) return;
  const size_t n = rows.size();
  d_taps_ = dmalloc<int>(n);
  CUDA_CHECK(cudaMemcpy(d_taps_, rows.data(), n * 4, cudaMemcpyHostToDevice));
  tap_r_ = dmalloc<bf16>(kLayers * n * kHcDim);
  tap_a_ = dmalloc<bf16>(kLayers * n * kHidden);
  tap_f_ = dmalloc<bf16>(kLayers * n * kHidden);
  tap_route_ = dmalloc<int>(kLayers * n * kTopK);
  tap_m_ = dmalloc<bf16>(kLayers * n * kHcDim);
  tap_p_ = dmalloc<float>(kLayers * n * 4);
  tap_c_ = dmalloc<float>(kLayers * n * 16);
}

void Engine::taps_mid(int layer, int M) {
  if (taps_.empty()) return;
  const int n = int(taps_.size());
  gather_rows(R_, kHcDim * 2, d_taps_, n, M, tap_m_ + size_t(layer) * n * kHcDim, s_);
  gather_rows(post_, 4 * 4, d_taps_, n, M, tap_p_ + size_t(layer) * n * 4, s_);
  gather_rows(comb_, 16 * 4, d_taps_, n, M, tap_c_ + size_t(layer) * n * 16, s_);
}

void Engine::taps_copy(int layer, int M) {
  if (taps_.empty()) return;
  const int n = int(taps_.size());
  gather_rows(R_, kHcDim * 2, d_taps_, n, M, tap_r_ + size_t(layer) * n * kHcDim, s_);
  gather_rows(aout_, kHidden * 2, d_taps_, n, M, tap_a_ + size_t(layer) * n * kHidden, s_);
  gather_rows(fout_, kHidden * 2, d_taps_, n, M, tap_f_ + size_t(layer) * n * kHidden, s_);
  if (layer > 0) gather_rows(d_moe_ids_, kTopK * 4, d_taps_, n, M, tap_route_ + size_t(layer) * n * kTopK, s_);
}

const float* Engine::step(const std::vector<int>& ids, const std::vector<int>& pos, const std::vector<int>& slot,
                          const std::vector<int>& seq_start, bool all_logits, const std::vector<int>& rows,
                          bool gather, bool prefill) {
  if (g_sync == 2) g_t = std::chrono::steady_clock::now();
  const int M = int(ids.size()), B = int(slot.size());
  if (M < 1 || M > c_.max_rows || int(pos.size()) != M || int(seq_start.size()) != B + 1 || seq_start[B] != M)
    qlog::fail("step", "bad step shape (M %d, B %d)", M, B);
  if (prefill && verify_) qlog::fail("step", "a verify step is a decode step (prefill = true)");
  // the step's kind for every kernel call below (core/step_kind.h): the caller's prefill flag, never M; back to the
  // previous kind (decode outside steps) when the step returns or throws
  const StepKind kind(prefill);
  std::vector<RowMeta> meta(M);
  int max_hca = 0, max_pos = 0, max_dsa = 0;
  for (int b = 0; b < B; b++) {
    const int r0 = seq_start[b], n = seq_start[b + 1] - r0, s = slot[b], p0 = pos[r0];
    if (n < 1 || s < 0 || s >= c_.slots) qlog::fail("step", "sequence %d: %d rows, slot %d", b, n, s);
    if (p0 != hist_[s]) qlog::fail("step", "slot %d continues at %d, history is %d", s, p0, hist_[s]);
    if (p0 + n > c_.ctx) qlog::fail("step", "slot %d: position %d past the context %d", s, p0 + n - 1, c_.ctx);
    for (int i = 0; i < n; i++) {
      const int p = pos[r0 + i];
      if (p != p0 + i) qlog::fail("step", "positions of sequence %d are not consecutive", b);
      meta[r0 + i] = RowMeta{p, s, r0, p0, p0 + n - 1};
      max_hca = std::max(max_hca, (p + 1) / kRatio + std::min(p + 1, kWindow));
      max_dsa = std::max(max_dsa, std::min(p + 1, kIdxTop + kWindow));
      max_pos = std::max(max_pos, p);
    }
  }
  if (pool_) {  // P125: the rows this step writes mapped (KvFull before anything changes)
    std::vector<long> end(B);
    for (int b = 0; b < B; b++) end[b] = long(pos[seq_start[b + 1] - 1]) + 1;
    kv_ensure(slot, end);
  }
  std::vector<int> lrows;
  if (!rows.empty()) {
    if (int(rows.size()) > c_.logit_rows) qlog::fail("step", "%zu logit rows > %d", rows.size(), c_.logit_rows);
    lrows = rows;
  } else if (all_logits) {
    if (M > c_.logit_rows) qlog::fail("step", "all_logits with M %d > logit_rows %d", M, c_.logit_rows);
    for (int i = 0; i < M; i++) lrows.push_back(i);
  } else {
    for (int b = 0; b < B; b++) lrows.push_back(seq_start[b + 1] - 1);
    if (B > c_.logit_rows) qlog::fail("step", "%d sequences > logit_rows %d", B, c_.logit_rows);
  }
  const int R = int(lrows.size());
  // P075 ar_lanes (prefill): rows in lanes; per layer, lane by lane: [the previous FFN's tail → this attention up to its
  // VHA z all-reduce], then lane by lane [z → o_proj all-reduce], then lane by lane the post + FFN pre, then the FFN
  // part by part (≤ kMaxCall rows each: its shared-expert all-reduce under the experts, the latent's per lane). Every
  // all-reduce runs on the comm stream under the other lanes' compute; the lanes' kernels run in row order on s_ (a
  // later lane's attention reads the earlier lanes' KV rows / compressed entries; all attention reads of a layer come
  // before its ring / state writes, as in one call). Same kernels on the same rows → same bits. Steps above kMaxAttn
  // rows take the same path with P075 off (lanes of ≤ kMaxAttn rows, all-reduces on s_).
  const bool async = patch::on(patch::ar_lanes) && patch::on(patch::hca_fuse);
  const bool lanes = (async && M >= kLaneMinRows && taps_.empty()) || M > kMaxAttn;
  if (M > kMaxAttn && (!taps_.empty() || !patch::on(patch::hca_fuse)))
    qlog::fail("step", "%d rows > %d per call: taps / the unfused HCA path take one call", M, kMaxAttn);
  std::vector<int> lb{0, M}, parts{0, 1};  // lane bounds, each part's first lane (one lane: every row)
  if (lanes) lane_bounds(M, async && M >= kLaneMinRows, lb, parts);
  // P094 sp_resid (lanes): rows split between the ranks within each lane (an odd last lane needs the scratch row M)
  const bool sp = lanes && patch::on(patch::sp_resid) && patch::on(patch::hca_fuse) && (M % 2 == 0 || M < c_.max_rows);
  sp_on_ = sp;
  if (sp) sp_lb_ = lb;
  // P100 a2a_attn (SP lanes, P060's fused activations; weights loaded at start)
  a2a_on_ = sp && a2a_w_ && patch::on(patch::a2a_attn) && f8_fuse_ok(1);
  // P136 sp_shared (SP lanes, P060's fused activations; weights loaded at start)
  shs_on_ = sp && shw_ && patch::on(patch::sp_shared) && f8_fuse_ok(1);
  // P176 zc_a2a: this step's o exchanges as puts into the peer's registered receive rows (both ranks decide alike)
  zr_on_ = a2a_on_ && zr_ && patch::on(patch::zc_a2a) && comm_->zc_ready();
  if (zr_on_) {
    size_t own = 0;
    for (int k = 0; k + 1 < int(lb.size()); k++) own += size_t(lb[k + 1] - lb[k] + 1) / 2;
    zr_on_ = own <= zr_rows_;
  }
  // P101 a2a_sched: the attention fronts run `ahead` lanes ahead of the attention halves (P100 alone: all of them);
  // every lane's gates and ring / state writes before the first exchange join (E55_A2A_AHEAD=n: n lanes ahead)
  const bool sched = a2a_on_ && patch::on(patch::a2a_sched);
  const int ahead = [&] {
    const char* e = std::getenv("E55_A2A_AHEAD");
    return std::max(1, e ? std::atoi(e) : sched ? kA2aAhead : 1 << 20);
  }();
  // device RowMeta: r0 relative to the row's lane (hca.h: the HCA kernels index the step's keys by r0 relative to the
  // pointers passed, which each lane advances to its first row; negative when the sequence began in an earlier lane)
  std::vector<RowMeta> dmeta = meta;
  for (int k = 0; k + 1 < int(lb.size()); k++)
    for (int i = lb[k]; i < lb[k + 1]; i++) dmeta[i].r0 -= lb[k];
  CUDA_CHECK(cudaMemcpyAsync(d_ids_, ids.data(), M * 4, cudaMemcpyHostToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(d_meta_, dmeta.data(), M * sizeof(RowMeta), cudaMemcpyHostToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(d_rows_, lrows.data(), R * 4, cudaMemcpyHostToDevice, s_));
  if (sp) {  // P094: the owner of each logit row
    std::vector<int> own(R);
    for (int j = 0; j < R; j++) own[j] = sp_owner(lrows[j]);
    CUDA_CHECK(cudaMemcpyAsync(d_own_, own.data(), R * 4, cudaMemcpyHostToDevice, s_));
  }
  g_layer = -1;
  chk(s_, "uploads", __LINE__);
  if (dsp_) {  // DSpark: the draft's context rows of this step; a verify's rollback state (one lane: M ≤ logit_rows)
    if (verify_ && lanes) qlog::fail("step", "a verify step of %d rows in lanes", M);
    dsp_step_begin(meta, slot, seq_start);
  }
  embed(M);
  chk(s_, "embed4", __LINE__);
  int il = 0, id = 0;
  auto lane = [&](int o, int n) {
    Lane ln{o, n, 0, 0, 0};
    for (int i = o; i < o + n; i++) {
      const int p = meta[i].pos;
      ln.max_hca = std::max(ln.max_hca, (p + 1) / kRatio + std::min(p + 1, kWindow));
      ln.max_dsa = std::max(ln.max_dsa, std::min(p + 1, kIdxTop + kWindow));
      ln.max_pos = std::max(ln.max_pos, p);
    }
    return ln;
  };
  // post (+ the next sublayer's pre) on the rows of a lane
  auto post_pre = [&](const MhcW* next, const bf16* out, const Lane& ln) {
    const int o = ln.o;
    if (next) mhc_post_pre(*next, at(out, o, kHidden), at(R_, o, kHcDim), ln.n, at(mix_, o, 72), at(post_, o, 4),
                           at(comb_, o, 16), at(inp_, o, kHidden), s_);
    else mhc_post(at(out, o, kHidden), at(R_, o, kHcDim), at(post_, o, 4), at(comb_, o, 16), ln.n, at(R_, o, kHcDim), s_);
  };
  if (!lanes) {  // one lane: every row in each call (decode, small prefill chunks, P075 off)
    const Lane all{0, M, max_hca, max_dsa, max_pos};
    for (int L = 0; L < kLayers; L++) {
      Layer& l = *L_[L];
      g_layer = L;
      if (L == 0) {  // the first pre: R = [e, e, e, e], broadcast mixer (every later pre is fused with a post)
        mhc_pre_embed(l.ha, R_, M, mix_, post_, comb_, inp_, s_);
        chk(s_, "mhc_pre0", __LINE__);
      }
      if (l.dsa) attn_dsa_a(l, id, all, false), attn_dsa_b(l, id++, all, false);
      else attn_hca_a(l, il, all, false), attn_hca_b(l, il++, all, false);
      mhc_post_pre(l.hf, aout_, R_, M, mix_, post_, comb_, inp_, s_);  // attention's post + the FFN's pre
      chk(s_, "mhc_post_pre attn", __LINE__);
      taps_mid(L, M);
      ffn_a(l, all, &all, 1, false, false, nullptr, nullptr);
      if (dsp_) dsp_count(L, M);
      ffn_b(l, all, false);
      if (L + 1 < kLayers) mhc_post_pre(L_[L + 1]->ha, fout_, R_, M, mix_, post_, comb_, inp_, s_);  // + next attn's pre
      else mhc_post(fout_, R_, post_, comb_, M, R_, s_);
      chk(s_, "mhc_post_pre ffn", __LINE__);
      taps_copy(L, M);  // R_ is this layer's output residual either way (the fused pre does not change it)
      chk(s_, "taps_copy", __LINE__);
      if (dsp_) dsp_after_layer(L, M);  // DSpark: the stream mean of R_ after layers 40, 41, 42
    }
  } else if (sp) {
    // P094 sp_resid: lanes as P075, each rank doing the per-row work for its half of every lane (sp_own). Per layer:
    // lane by lane [the previous FFN's tail (reduce-scatters joined, fc2 + shared, post + pre) → the attention's
    // replicated projections → their all-gather], each lane's attention up to the VHA z all-reduce running one lane
    // behind (its z all-reduce starts early: the attention phase is bound by the serial z all-reduces — all fronts
    // first measured 1435 vs 1419 ms a 4096-row chunk); lane by lane [z → o_proj → reduce-scatter]; lane by lane [post
    // + FFN pre → the FFN's per-row work → its all-gather]; the FFN's tensor-parallel part per part (shared expert per
    // lane, experts per part, reduce-scatters per lane under the next layer).
    const int NL = int(lb.size()) - 1, NP = int(parts.size()) - 1;
    std::vector<Lane> ln(NL), pt(NP);
    std::vector<int> part_first(NL);  // each lane's part's first lane
    for (int k = 0; k < NL; k++) ln[k] = lane(lb[k], lb[k + 1] - lb[k]);
    for (int q = 0; q < NP; q++) {
      pt[q] = lane(lb[parts[q]], lb[parts[q + 1]] - lb[parts[q]]);
      for (int k = parts[q]; k < parts[q + 1]; k++) part_first[k] = parts[q];
    }
    std::vector<int> tg(NL, -1), tz(NL, -1), to(NL, -1), tf(NL, -1), tsh(NL, -1), tl(NL, -1);
    auto tail = [&](int Lprev, int k, const MhcW* next) {
      comm_->join(tl[k], s_), tl[k] = -1;
      comm_->join(tsh[k], s_), tsh[k] = -1;
      const Lane me = sp_own(ln[k]);
      ffn_b(*L_[Lprev], me, true);
      post_pre(next, fout_, me);
      chk(s_, "mhc_post_pre ffn", __LINE__);
    };
    for (int L = 0; L < kLayers; L++) {
      Layer& l = *L_[L];
      g_layer = L;
      comm_->mark("L", L, s_);  // (E55_COMM_TRACE: the layer's start on s_)
      // P100: the fronts (gathers) `ahead` lanes ahead of the attention halves (exchanges): all of them first by default —
      // the exchanges are not needed before the lanes' output sides below; interleaved, the gathers queued behind them
      // (45 ms of joins a 4096-row chunk); P101 picks the lead
      const int A = a2a_on_ ? std::min(NL, ahead) : 1;
      for (int k = 0; k < NL + A; k++) {
        if (a2a_on_) {
          if (k < NL) {
            if (L == 0) {
              const Lane me = sp_own(ln[k]);
              mhc_pre_embed(l.ha, at(R_, me.o, kHcDim), me.n, at(mix_, me.o, 72), at(post_, me.o, 4),
                            at(comb_, me.o, 16), at(inp_, me.o, kHidden), s_);
              chk(s_, "mhc_pre0", __LINE__);
            } else {
              tail(L - 1, k, &l.ha);
            }
            tg[k] = attn_front(l, ln[k], async);
          }
          if (k >= A) {
            comm_->join(tg[k - A], s_), tg[k - A] = -1;
            tz[k - A] = l.dsa ? attn_dsa_a(l, id, ln[k - A], async) : attn_hca_a(l, il, ln[k - A], async);
          }
          continue;
        }
        if (k > NL) break;
        if (k < NL) {
          if (L == 0) {
            const Lane me = sp_own(ln[k]);
            mhc_pre_embed(l.ha, at(R_, me.o, kHcDim), me.n, at(mix_, me.o, 72), at(post_, me.o, 4),
                          at(comb_, me.o, 16), at(inp_, me.o, kHidden), s_);
            chk(s_, "mhc_pre0", __LINE__);
          } else {
            tail(L - 1, k, &l.ha);
          }
          tg[k] = attn_front(l, ln[k], async);
        }
        if (k > 0) {
          comm_->join(tg[k - 1], s_), tg[k - 1] = -1;
          tz[k - 1] = l.dsa ? attn_dsa_a(l, id, ln[k - 1], async) : attn_hca_a(l, il, ln[k - 1], async);
        }
      }
      if (dsp_ && L > 0) dsp_after_layer(L - 1, M);  // (each rank's rows of R_; sp_rows at the end for the rest)
      comm_->mark("C", L, s_);  // (E55_COMM_TRACE: the attention cores done)
      if (a2a_on_) {  // P100: lane by lane [o exchange in → the output side on this rank's rows → post + FFN pre → …]
        std::vector<bf16*> gk(NL, nullptr);
        if (sched) {  // P101: first, for every lane, what needs no exchange: the ring / state writes and the gates
          size_t c = 0;
          for (int k = 0; k < NL; k++) {
            if (!l.dsa)
              hca_post_state(d_meta_ + ln[k].o, at(rows_, ln[k].o, kKvRowBytes), at(ckv_, ln[k].o, 1024), l.h.ape,
                             ln[k].n, hc_[il], s_);
            gk[k] = t3_ + c * (l.dsa ? 16384 : 8192);  // this rank's rows of the lanes, packed: ≤ ⌈M/2⌉ rows
            attn_gate(l, l.dsa ? id : il, ln[k], gk[k]);
            c += sp_own(ln[k]).n;
          }
        }
        for (int k = 0; k < NL; k++) {
          comm_->join(tz[k], s_), tz[k] = -1;
          attn_own(l, l.dsa ? id : il, ln[k], gk[k]);
          post_pre(&l.hf, aout_, sp_own(ln[k]));
          chk(s_, "mhc_post_pre attn", __LINE__);
          tf[k] = ffn_front(l, ln[k], async);
          if (shs_on_ && !l.f.dense && sps_at(k - part_first[k]) == 'f') ffn_shared(l, ln[k]);  // P136
        }
      } else {
        for (int k = 0; k < NL; k++) {
          comm_->join(tz[k], s_), tz[k] = -1;
          to[k] = l.dsa ? attn_dsa_b(l, id, ln[k], async) : attn_hca_b(l, il, ln[k], async);
        }
        for (int k = 0; k < NL; k++) {
          comm_->join(to[k], s_), to[k] = -1;
          post_pre(&l.hf, aout_, sp_own(ln[k]));
          chk(s_, "mhc_post_pre attn", __LINE__);
          tf[k] = ffn_front(l, ln[k], async);
          if (shs_on_ && !l.f.dense && sps_at(k - part_first[k]) == 'f') ffn_shared(l, ln[k]);  // P136
        }
      }
      if (l.dsa) id++;
      else il++;
      comm_->mark("F", L, s_);  // (E55_COMM_TRACE: the FFN fronts done)
      for (int q = 0; q < NP; q++)
        ffn_tp(l, pt[q], &ln[parts[q]], parts[q + 1] - parts[q], async, &tf[parts[q]], &tsh[parts[q]], &tl[parts[q]]);
      if (dsp_) dsp_count(L, M);
    }
    g_layer = kLayers - 1;
    for (int k = 0; k < NL; k++) tail(kLayers - 1, k, nullptr);
    if (dsp_) dsp_after_layer(kLayers - 1, M);
  } else {
    const int NL = int(lb.size()) - 1, NP = int(parts.size()) - 1;
    std::vector<Lane> ln(NL), pt(NP);
    std::vector<int> part_of(NL);
    for (int k = 0; k < NL; k++) ln[k] = lane(lb[k], lb[k + 1] - lb[k]);
    for (int q = 0; q < NP; q++) {
      pt[q] = lane(lb[parts[q]], lb[parts[q + 1]] - lb[parts[q]]);
      for (int k = parts[q]; k < parts[q + 1]; k++) part_of[k] = q;
    }
    // tickets: the previous FFN's all-reduces (shared part per part, latent per lane), this layer's z / o_proj per lane
    std::vector<int> tsh(NP, -1), tl(NL, -1), tz(NL, -1), to(NL, -1);
    auto ffn_tail = [&](int Lprev, int k, const MhcW* next) {
      comm_->join(tl[k], s_), tl[k] = -1;
      comm_->join(tsh[part_of[k]], s_), tsh[part_of[k]] = -1;
      ffn_b(*L_[Lprev], ln[k], true);
      post_pre(next, fout_, ln[k]);
      chk(s_, "mhc_post_pre ffn", __LINE__);
    };
    for (int L = 0; L < kLayers; L++) {
      Layer& l = *L_[L];
      g_layer = L;
      for (int k = 0; k < NL; k++) {
        if (L == 0) {
          mhc_pre_embed(l.ha, at(R_, ln[k].o, kHcDim), ln[k].n, at(mix_, ln[k].o, 72), at(post_, ln[k].o, 4),
                        at(comb_, ln[k].o, 16), at(inp_, ln[k].o, kHidden), s_);
          chk(s_, "mhc_pre0", __LINE__);
        } else {
          ffn_tail(L - 1, k, &l.ha);
        }
        tz[k] = l.dsa ? attn_dsa_a(l, id, ln[k], async) : attn_hca_a(l, il, ln[k], async);
      }
      // DSpark: every lane's R_ is layer L − 1's output now (the attention halves do not write R_; its post below does)
      if (dsp_ && L > 0) dsp_after_layer(L - 1, M);
      for (int k = 0; k < NL; k++) {
        comm_->join(tz[k], s_);
        to[k] = l.dsa ? attn_dsa_b(l, id, ln[k], async) : attn_hca_b(l, il, ln[k], async);
      }
      for (int k = 0; k < NL; k++) {
        comm_->join(to[k], s_);
        post_pre(&l.hf, aout_, ln[k]);
        chk(s_, "mhc_post_pre attn", __LINE__);
      }
      if (l.dsa) id++;
      else il++;
      for (int q = 0; q < NP; q++)
        ffn_a(l, pt[q], &ln[parts[q]], parts[q + 1] - parts[q], true, async, &tsh[q], &tl[parts[q]]);
      if (dsp_) dsp_count(L, M);
    }
    g_layer = kLayers - 1;
    for (int k = 0; k < NL; k++) ffn_tail(kLayers - 1, k, nullptr);
    if (dsp_) dsp_after_layer(kLayers - 1, M);
  }
  g_layer = kLayers;
  hc_head(head_, R_, d_rows_, R, hbuf_, s_);
  chk(s_, "hc_head", __LINE__);
  if (sp) sp_rows(hbuf_, kHidden * 2, kHidden * 2, d_own_, R);  // P094: each logit row as its owner has it
  rmsnorm(hbuf_, kHidden, final_norm_, R, kHidden, t2_, kHidden, s_);
  lin_forward_f32(lm_, t2_, lg_half_, R, s_);
  chk(s_, "lm_head", __LINE__);
  multimax(lg_half_, R, kVocabHalf, kVocabHalf, mm_, s_);
  if (gather) logits_gather(R);  // else (P063) the caller samples from the halves' records, or calls gather_logits()
  chk(s_, "logits", __LINE__);
  if (dsp_) dsp_step_end(M);  // DSpark: feature_projection → the draft's context rows
  sp_on_ = a2a_on_ = shs_on_ = zr_on_ = false;
  for (int b = 0; b < B; b++) hist_[slot[b]] += seq_start[b + 1] - seq_start[b];
  last_rows_ = R;
  return gather ? lg_ : lg_half_;
}

const SparseCand* Engine::sparse_cands() {  // P063 sparse_wire
  const int R = last_rows_;
  sample_cands(lg_half_, R, kVocabHalf, c_.rank * kVocabHalf, cand_, s_);
  comm_->all_gather_f32(reinterpret_cast<const float*>(cand_), reinterpret_cast<float*>(cand2_),
                        size_t(R) * kSparseRec * sizeof(SparseCand) / 4, s_);
  return cand2_;
}

const float* Engine::gather_logits() {  // P063: the full rows after a step with gather = false
  logits_gather(last_rows_);
  return lg_;
}

void Engine::logits_gather(int R) {
  if (patch::on(patch::rdma_gather) && comm_->rdma()) {  // P076: each rank's half lands in place in the rows of lg_
    const AgPiece pc{lg_half_, lg_, kVocabHalf * 4, kVocab * 4, 0, kVocabHalf * 4, kVocabHalf * 4};
    comm_->all_gather_rows(&pc, 1, R, s_);
  } else {
    comm_->all_gather_f32(lg_half_, lg2_, size_t(R) * kVocabHalf, s_);
    halves_to_rows(lg2_, R, kVocabHalf, lg_, s_);
  }
}

}  // namespace e55
