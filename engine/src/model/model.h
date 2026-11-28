// model.h — one step of ERNIE 5.5 Flash on this rank (TP=2; docs/model.md, docs/design.md), kernels from src/kernels/*.
// A step takes M = Σ_b T_b rows: rows of sequence b are consecutive positions starting at the slot's history length
// (caches are positional; the caller continues a slot's sequence or resets it). Every rank makes the same calls in the
// same order (collectives in every layer). Output: fp32 logits [R, 201216] (multimax applied) on both ranks.
#pragma once
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/comm.h"
#include "core/common.h"
#include "core/kv_pool.h"
#include "core/weights.h"
#include "kernels/attn/dsa.h"
#include "kernels/attn/hca.h"
#include "kernels/gemm/fp8.h"
#include "kernels/linear/linear.h"
#include "kernels/mhc/mhc.h"
#include "kernels/moe/moe.h"
#include "kernels/sample/sample.h"

namespace e55 {

// Debugging (model.cc chk): 0 off · 1 synchronize after every stage and name a failing kernel · 2 also time each
// stage (wall time since the previous one, synchronized) into a per-source-line table that prof_take returns and clears
struct ProfLine {
  std::string what;
  int line;
  double ms;
  long n;
};
void set_sync_mode(int m);
std::vector<ProfLine> prof_take();

// One sequence in speculative generation (DSpark, model/dspark.cc). pos = position of `last` (emitted, not yet through
// the target); drafts = the drafter's proposals for positions pos + 1 … (made by spec_draft / the previous spec_round),
// conf = their confidence logits (the draft's confidence head: logit of P(draft j accepted | drafts before it were)).
// depth (1 … 15): the drafter's block size; a round verifies min(verify(), drafts) of them (P121 conf_gate; off: all) —
// all, or with gate > 0 the confidence-gated count: drafts j while Π_{i≤j} σ(conf_i) ≥ gate; gate < 0: the count n that
// maximizes the expected tokens per ms, (1 + Σ_{j<n} Π_{i≤j} σ(conf_i)) / cost[n], cost = Engine::spec_cost's table
// (ms of a round verifying n drafts) — with P143 verify_joint and a spec_round_cost table, the round's gate < 0
// sequences' counts together instead (Engine::spec_round_cost). cap (> 0): emit at most cap tokens this round (the rest of an accepted block is
// dropped; the server's budgets) — and, P122 cap_verify, verify at most cap − 1 drafts. Sampling as kernels/sample: temp 0 = greedy, top_k,
// top_p, min_p (target rows only; drafts are drawn with temp / top_k / top_p), penalties (target rows only; state: pen,
// owned by the caller, cnt == null = off), seed (Philox key; the round's draws are keyed by positions).
struct SpecSeq {
  int slot = 0, pos = 0, last = 0, depth = 4, cap = 0;
  float gate = 0.f;
  float temp = 0.f, top_p = 1.f, min_p = 0.f;
  int top_k = 0;
  float rep = 1.f, freq = 0.f, pres = 0.f;
  PenSeen pen{nullptr, nullptr, nullptr};
  uint64_t seed = 0;
  std::vector<int> drafts;
  std::vector<float> conf;
  int verify(const std::vector<float>* cost = nullptr) const;  // drafts the next round verifies
  bool penal() const { return pen.cnt && (rep != 1.f || freq != 0.f || pres != 0.f); }
};
// spec_round wall time per phase (cudaEvents, ms summed over rounds): verify = the target step (with the draft's
// context rows), accept = penalties + spec_accept, commit = rollback + history, draft = the drafter for the next round
struct SpecStats {
  double ms_verify = 0, ms_ctx = 0, ms_accept = 0, ms_commit = 0, ms_draft = 0;
  long rounds = 0, seqs = 0, rows = 0, verified = 0, accepted = 0, emitted = 0;
};

// P125 kv_pool: a step (or a spec round) needs pool chunks the pool does not have. Thrown before anything changed; both
// ranks throw at the same call (their pools change in step). The server admits requests by the pool, so this comes only
// when its accounting was wrong: it ends that slot's request and goes on.
struct KvFull : std::runtime_error {
  int slot;
  KvFull(int s, const std::string& m) : std::runtime_error(m), slot(s) {}
};

struct EngineCfg {
  std::string model_dir, table;
  int rank = 0;
  std::string head_ip;   // the head's CX7 address (NCCL bootstrap)
  int port = 29611;
  int slots = 4;         // sequences with resident caches
  int ctx = 32768;       // tokens per slot (P125 kv_pool: per request at most — the address ranges; memory as it grows)
  int max_rows = 4096;   // M per step (prefill chunk)
  int logit_rows = 64;   // rows that can get logits in one step
  int pool_tokens = 0;   // P125: the pool at init holds every slot's first pool_tokens positions (0: ctx — what the
                         // per-slot caches took); kv_pool_init then sizes it to the room left
};

class Engine {
 public:
  explicit Engine(const EngineCfg& c);
  ~Engine();
  // ids / pos [M], slot [B], seq_start [B+1] (host). Returns device fp32 [R, 201216] (valid until the next step):
  // R = M if all_logits (M ≤ logit_rows), else the last row of each sequence (R = B); `rows` (non-empty) names the
  // logit rows instead (the gate).
  // gather = false (P063 sparse_wire): no logits all-gather — returns this rank's vocabulary half [R, kVocabHalf]
  // (ids rank·kVocabHalf …, multimax applied); then sparse_cands() for the sparse sampler or gather_logits() for the
  // full rows (both collectives: every rank calls them in the same order).
  const float* step(const std::vector<int>& ids, const std::vector<int>& pos, const std::vector<int>& slot,
                    const std::vector<int>& seq_start, bool all_logits, const std::vector<int>& rows = {},
                    bool gather = true, bool prefill = false);  // prefill: the step's kind (core/step_kind.h), set by the caller
  // P063: both ranks' top-kSparseK records of the last step's rows (its half as it is now — penalties applied by the
  // caller go in): device [2][R][kSparseRec]
  const SparseCand* sparse_cands();
  // P063: the last step's full rows [R, kVocab] from the halves as they are now
  const float* gather_logits();
  int logit_rows_last() const { return last_rows_; }
  void reset(int slot);  // history 0 (P125: the slot's pool chunks go back)
  int history(int slot) const { return hist_.at(slot); }
  // Gate taps: rows (of the next steps, step-local indices) whose residual after every layer and sublayer outputs
  // are kept: resid [43][n][4][4096], attn / ffn [43][n][4096] (device bf16; filled by each step with taps on)
  void set_taps(const std::vector<int>& rows);
  const bf16* tap_resid() const { return tap_r_; }
  const bf16* tap_attn() const { return tap_a_; }
  const bf16* tap_ffn() const { return tap_f_; }
  const int* tap_route() const { return tap_route_; }  // [43][n][10] expert ids (layer 0: unset)
  // mid-layer state, so the gate can rebuild the fused mHC's unrounded fp32 residual exactly: the residual after the
  // attention post [43][n][4][4096] bf16 and the FFN pre's post [43][n][4] / comb [43][n][16] (fp32)
  const bf16* tap_mid() const { return tap_m_; }
  const float* tap_post() const { return tap_p_; }
  const float* tap_comb() const { return tap_c_; }
  cudaStream_t stream() const { return s_; }
  int rank() const { return c_.rank; }
  const EngineCfg& cfg() const { return c_; }

  // ---- P125 kv_pool: the sequence caches from one shared pool (core/kv_pool.h) --------------------------------------
  // What grows with a sequence — the DSA MLA rows (656 B a position) and indexer rows (68 B), the HCA compressed entries
  // (584 B per 128 positions) — lives in one KvPool of 2 MiB chunks shared by all slots: each (layer, kind) is one
  // address reservation for all slots (stride: ctx rounded up so a slot's range starts on a chunk; the kernels index
  // slot · stride from one base, as in the cudaMalloc'd arrays), mapped as a slot's positions grow (a step, a spec
  // round: positions < their last + 1, + kKvSlack, mapped zeroed before any kernel; KvFull if the pool has too few) and
  // given back by reset(). The fixed per-slot state (HCA window ring, compressor state) stays as it was. The same
  // kernels read the same bytes: the same bits.
  // kv_pool_init (both ranks, before the first request, after dspark_init and a warm-up of the serving paths so their
  // lazy allocations are already taken): every slot reset, then the pool grows (or shrinks) to its chunks + the room
  // left = min over both nodes of MemFree − reserve_gib (GB10: CUDA allocates from free pages), at most max_gib in all
  // (> 0) — the same chunk count on both ranks. Returns the chunk count (0: P125 off).
  size_t kv_pool_init(double reserve_gib, double max_gib);
  bool kv_pool() const { return pool_ != nullptr; }
  size_t kv_need(long tokens) const;  // chunks one sequence of `tokens` positions maps (slack included)
  size_t kv_held(int slot) const;     // chunks the slot maps now
  size_t kv_free() const { return pool_ ? pool_->free_chunks() : 0; }
  size_t kv_total() const { return pool_ ? pool_->total_chunks() : 0; }
  // both nodes' MemFree in GiB [rank0, rank1] (a collective: every rank calls it in the same order)
  std::vector<double> mem_free_gib();

  // ---- P146 prefix_ckpt: prefix checkpoints (needs P125) ------------------------------------------------------------
  // The positional caches (DSA MLA / indexer rows, HCA compressed entries) stay valid below any position; what else a
  // slot's history at Q is made of only moves forward: every HCA layer's window ring (positions Q−128 … Q−1, by
  // position % 128), the compressor's open group (its state rows 0 … Q % 128 − 1: positions 128·⌊Q/128⌋ … Q−1; none
  // when Q % 128 = 0) and the draft's context rows (Q−128 … Q−1 of its 256-ring). A checkpoint copies those at the
  // slot's history Q into one of the slot's entries — pool chunks, mapped at the save, given back at a drop / reset —
  // [rings nh × 128 × 584 B][draft 3 × 128 × 584 B][state nh × (Q % 128) × 4096 B]. Restoring it copies them back, sets
  // the history to Q, drops the slot's entries above Q (their rows get rewritten) and gives the pool back the caches'
  // chunks above Q (+ slack): the slot is then exactly a slot whose first Q positions just ran — the same bytes wherever
  // a later step, verify or draft reads (positional rows above the history are never read before rewritten), so the
  // same bits. ckpt_copy: an empty slot takes another slot's first Q positions (the cache rows copied) and that slot's
  // entry. Every call is the same on both ranks (the pools change in step).
  bool ckpt_on() const;
  void ckpt_init(int per_slot);  // entries per slot (address ranges only); a later call drops every entry
  int ckpt_cap() const { return ck_.empty() ? 0 : int(ck_[0].size()); }
  size_t ckpt_bytes(long pos) const;  // one checkpoint at history pos
  size_t ckpt_need(long pos) const { return KvPool::chunks_for(ckpt_bytes(pos)); }  // pool chunks it maps
  bool ckpt_save(int slot, int k);     // entry k ← the slot's state at its history; false: too few free chunks
  void ckpt_restore(int slot, int k);  // the slot back to entry k's position
  bool ckpt_copy(int dst, int src, int k);  // false (nothing changed): the pool cannot hold dst's rows
  void ckpt_drop(int slot, int k);
  long ckpt_pos(int slot, int k) const { return ck_.empty() ? -1 : ck_.at(slot).at(k).pos; }  // −1: empty
  size_t ckpt_held(int slot) const;  // chunks the slot's entries map

  // ---- DSpark speculative decoding (model/dspark.cc) ---------------------------------------------------------------
  // dspark_init: load the draft (model.mtp.*; the tensor table must place them) and its state. From then on every step
  // also writes the draft's context rows of its last ≤ 128 rows per sequence (the residual after layers 40–42 →
  // feature_projection → each stage's context KV), so plain steps (prefill chunks, decode) and spec rounds mix freely.
  void dspark_init();
  bool dspark() const { return dsp_ != nullptr; }
  // spec_first: sample the first token of a sequence whose prompt ran in the last step (its logits row lrow), as
  // run() samples (ctr = q.pos = the token's position); sets q.last, q.pos. No drafts yet (spec_draft).
  int spec_first(SpecSeq& q, int lrow);
  // spec_draft: the drafter over the sequences' anchors (q.last at q.pos), q.depth rows each → q.drafts, q.conf
  void spec_draft(const std::vector<SpecSeq*>& seqs);
  // spec_round: verify [last, drafts…] of every sequence in one target step, accept (greedy: the target's argmax;
  // temp > 0: spec_accept / P016 block_verify on the draft rows the drafts were drawn from), roll the caches back to
  // the kept rows, then draft the next round. Returns per sequence the emitted tokens (1 … verify()+1, ≤ cap).
  // Every cache and the history end bitwise as plain decoding of the emitted tokens would have left them.
  std::vector<std::vector<int>> spec_round(const std::vector<SpecSeq*>& seqs, bool draft_next = true);
  // spec_verify + spec_commit: spec_round's two halves for tests (verify returns n_acc / next per sequence; commit
  // keeps keep[b] rows of each sequence of the last verify step)
  void spec_commit(const std::vector<int>& keep);
  const float* spec_logits(int* rows) const;  // the last verify step's logits [rows, V] (until the next step)
  SpecStats spec_stats(bool reset);
  // gate < 0: ms of a round by verified drafts — the same numbers on both ranks (the verify count drives collectives)
  void spec_cost(const std::vector<float>& ms) { spec_cost_ = ms; }
  // P143 verify_joint: ms of a whole round (the verify step with the draft's context rows, accept, commit, the next
  // drafts) by its sequences B = 1 … and total verify rows R = 0 … rmax, ms[(B − 1) · (rmax + 1) + R]; a round's
  // gate < 0 sequences then get their verify lengths together: the counts that maximize the round's expected emitted
  // tokens Σ_b (1 + Σ_{j ≤ k_b} Π_{i ≤ j} σ(conf_b,i)) / ms(B, Σ_b (1 + k_b)) (spec_round). Both ranks take rank 0's
  // table (a collective: every rank calls it at the same point, with the same size). Empty: off.
  void spec_round_cost(const std::vector<float>& ms, int rmax);
  const std::vector<int>& spec_verified() const { return spec_k_; }  // drafts each sequence of the last round verified
  // P168 copy_draft (dspark.cc): one sequence's round verifying the caller's tokens `copy` (1 … logit_rows − 1, ≤ q.cap
  // − 1) in place of drafts — each verify row drawn as plain decoding draws it, a copied token kept while the draw equals
  // it (plain decoding's tokens) — then the DSpark drafts (draft_next) unless every copied token and then `follow` (≥ 0)
  // came out. Returns the emitted tokens; caches, history and q as after spec_round.
  std::vector<int> spec_copy(SpecSeq& q, const std::vector<int>& copy, int follow, bool draft_next);
  // test hooks: the draft's context rows of slot from given aux features (device bf16 [K, 3·4096]) at positions pos;
  // distinct routed experts per layer, summed over the steps since the last call (on: count in every step)
  void dspark_ctx(int slot, const bf16* aux, const std::vector<int>& pos);
  void set_history(int slot, int n) { hist_.at(slot) = n; }  // test hook: the caller vouches for the caches
  void expert_count(bool on);
  std::vector<double> expert_take(long* steps);

  struct HcaW;
  struct DsaW;
  struct FfnW;
  struct Layer;

 private:
  EngineCfg c_;
  std::unique_ptr<Weights> w_;
  std::unique_ptr<Comm> comm_;
  cudaStream_t s_ = nullptr;
  cudaStream_t s2_ = nullptr;                         // P074 hca_side: the compressor projection's stream
  cudaEvent_t ev_in_ = nullptr, ev_ck_ = nullptr;     // P074: layer input ready (s_) · compressor rows ready (s2_)
  std::vector<std::unique_ptr<Layer>> L_;
  HcHead head_;
  Lin lm_;
  bf16 *embed_ = nullptr, *final_norm_ = nullptr;  // embed_: P095 embed_split — this rank's vocabulary half only
  bool emb_half_ = false;                          // P095 (fixed at load)
  float* mm_ = nullptr;  // multimax r0…r3, t0…t3
  float *inv_hca_ = nullptr, *inv_dsa_ = nullptr;
  std::vector<HcaCache> hc_;  // per HCA layer (all slots)
  std::vector<DsaCache> dc_;  // per DSA layer
  std::vector<int> hist_;
  // P125 kv_pool: per slot its arrays (kind 0 MLA rows, 1 indexer rows, 2 HCA compressed entries) and the positions
  // they all cover now; kv_cap_: positions an array's range holds (≥ ctx + kKvSlack)
  struct KvArr {
    int id, kind;
  };
  std::unique_ptr<KvPool> pool_;
  std::vector<std::vector<KvArr>> kv_arr_;
  std::vector<long> kv_cover_;
  long kv_cap_ = 0;
  int kv_nd_ = 0, kv_nh_ = 0;  // DSA / HCA layers
  float* mem_d_ = nullptr;  // mem_free_gib's exchange [8]
  size_t kv_bytes(int kind, long tokens) const;  // an array's bytes for positions < tokens (+ slack, ≤ kv_cap_)
  // positions < end[b] of slot[b] mapped (each slot once); KvFull (nothing mapped) if the pool has too few chunks
  void kv_ensure(const std::vector<int>& slot, const std::vector<long>& end);
  // P146: per slot its checkpoint entries (pool array id, position or −1, with the draft's rows)
  struct Ckpt {
    int id = -1;
    long pos = -1;
    bool draft = false;
  };
  std::vector<std::vector<Ckpt>> ck_;
  void ckpt_move(int slot, long pos, uint8_t* buf, bool save);  // the slot's per-slot state ⇄ an entry's bytes
  void dsp_ckpt(int slot, long pos, uint8_t* buf, bool save);   // (dspark.cc) the draft's context rows pos−128 … pos−1
  MoeScratch* moe_ = nullptr;
  // activations [max_rows, …]
  bf16 *R_, *inp_, *t1_, *t2_, *t3_, *t4_, *q_, *o_, *aout_, *fout_;
  float *mix_, *post_, *comb_, *ckv_, *zp_, *lgr_, *part_, *wts_, *mw_;
  uint8_t *xq_, *xs_, *rows_, *oq_, *os_, *qf4_, *qs_;
  int *d_ids_, *d_list_, *d_cnt_, *d_moe_ids_, *d_rows_;
  RowMeta* d_meta_ = nullptr;
  float *fws_ = nullptr, *aws_ = nullptr, *lg_half_ = nullptr, *lg2_ = nullptr, *lg_ = nullptr;
  uint8_t* sp_ = nullptr;  // P078 tp_split: this rank's column halves before their all-gather
  // P094 sp_resid (in the upper half of part_): DSA q_a ‖ kv_a rows [MR, 1600] bf16, the FFN input's FP8 rows
  // [MR, 4096] + scales [MR, 32], the routing [MR][ids 10 ‖ weights 10]; owners of the logit rows [logit_rows] and of
  // the draft's context rows [max_rows]
  bf16* spq_ = nullptr;
  uint8_t *spxq_ = nullptr, *spxs_ = nullptr, *sprt_ = nullptr;
  int* d_own_ = nullptr;
  bool sp_on_ = false;       // this step runs P094
  // P100 a2a_attn: the peer's halves of the attention's output side per HCA / DSA layer (its gate rows, wo_a groups,
  // o_proj K half — as the peer uploads them — and its VHA U / V rows), loaded when the patch is on at load
  struct A2aW {
    F8Lin gate, woa, oproj;  // (woa: HCA only)
    float *U = nullptr, *V = nullptr;
  };
  std::vector<A2aW> a2h_, a2d_;
  bool a2a_w_ = false, a2a_on_ = false;  // P100 weights loaded · this step runs P100
  bool shw_ = false, shs_on_ = false;    // P136 sp_shared: the peer's shared halves loaded · this step runs P136
  // P176 zc_a2a: the o exchange's receive rows, NIC-registered (Comm::zc_alloc; null: none) — each lane's own half
  // compact at a2a_rx(ln) rows (≤ zr_rows_ × 32 KiB); zr_on_: this step's exchanges land there (puts)
  uint8_t* zr_ = nullptr;
  size_t zr_rows_ = 0;
  bool zr_on_ = false;
  std::vector<int> sp_lb_;   // its lane bounds
  size_t fws_bytes_ = 0, aws_bytes_ = 0;
  bf16* hbuf_ = nullptr;  // head rows [logit_rows, 4096]
  SparseCand *cand_ = nullptr, *cand2_ = nullptr;  // P063: this rank's records [logit_rows][…], gathered [2][…][…]
  int last_rows_ = 0;
  // taps
  std::vector<int> taps_;
  int* d_taps_ = nullptr;
  bf16 *tap_r_ = nullptr, *tap_a_ = nullptr, *tap_f_ = nullptr, *tap_m_ = nullptr;
  float *tap_p_ = nullptr, *tap_c_ = nullptr;
  int* tap_route_ = nullptr;

  // DSpark (model/dspark.cc)
  struct Dsp;
  static void dsp_delete(Dsp* d);
  std::unique_ptr<Dsp, void (*)(Dsp*)> dsp_{nullptr, &dsp_delete};
  bool verify_ = false;  // the next step is a verify: rollback state is saved (spec_commit)
  std::vector<float> spec_cost_;
  std::vector<float> rcost_;  // P143: spec_round_cost's table [B][rmax + 1]
  int rcost_rmax_ = 0;
  std::vector<int> spec_k_;   // the last spec_round's verify lengths
  std::vector<int> spec_lengths(const std::vector<SpecSeq*>& seqs) const;
  void dsp_step_begin(const std::vector<RowMeta>& meta, const std::vector<int>& slot, const std::vector<int>& seq_start);
  void dsp_after_layer(int L, int M);
  void dsp_stash(int il, int o, int M);
  void dsp_step_end(int M);
  void dsp_context(int K);
  void dsp_forward(const std::vector<SpecSeq*>& seqs, int M);
  void dsp_count(int L, int M);

  // Rows [o, o + n) of the step (P075 ar_lanes: lanes at prefill, else one lane with every row) and their key
  // bounds. Each attention is two halves, each ending in an all-reduce: _a up to the VHA z partial, _b from the
  // reduced z to o_proj's partial; async: the all-reduce goes to the comm stream and its ticket is returned (else −1).
  // Every kernel call inside works on the lane's rows (M = n), so each per-M switch (P032 / P060 f8_fuse_ok / P078
  // split_rows / the kernels' own decode-prefill choices) decides per lane.
  struct Lane {
    int o = 0, n = 0, max_hca = 0, max_dsa = 0, max_pos = 0;
  };
  int attn_hca_a(Layer& l, int il, const Lane& ln, bool async);
  int attn_hca_b(Layer& l, int il, const Lane& ln, bool async);
  int attn_dsa_a(Layer& l, int id, const Lane& ln, bool async);
  int attn_dsa_b(Layer& l, int id, const Lane& ln, bool async);
  // FFN of the rows of part pt (≤ 8192) up to its all-reduce. One lane (!lanes): [latent ‖ shared] in one all-reduce
  // on s_. Lanes (the nl lanes ln of this part): the shared expert's output all-reduced first (*tsh, under the
  // experts), then the latent per lane (tl[k]); async: on the comm stream (tickets), else on s_ (−1).
  void ffn_a(Layer& l, const Lane& pt, const Lane* ln, int nl, bool lanes, bool async, int* tsh, int* tl);
  // after the all-reduce, rows of a lane: latent_norm → fc2, + shared → fout_ (dense FFN: nothing)
  void ffn_b(Layer& l, const Lane& ln, bool lanes);
  void ffn_bufs(bool lanes, bf16*& pl, int& ldl, bf16*& ps, int& lds) const;
  // P094 sp_resid (model.cc): lane ln's rows of this rank (h = ⌈n/2⌉ rows from o + rank·h); the owner of a step row;
  // a [rows, w bytes] buffer's lane rows as an in-place all-gather piece; the gather / reduce-scatter of a lane
  Lane sp_own(const Lane& ln) const;
  int sp_owner(int row) const;
  AgPiece sp_piece(void* base, const Lane& ln, int w) const;
  int sp_gather(const AgPiece* p, int np, const Lane& ln, bool async);
  int sp_rs(bf16* lane, size_t half, bool async);
  // the attention's replicated projections on this rank's rows of the lane, then their all-gather (ticket); _a then
  // skips them
  int attn_front(Layer& l, const Lane& ln, bool async);
  // P100 a2a_attn: rows of this rank's half of lane ln of a [rows, w bytes] buffer arrive from the peer into recv (the
  // same rows), the peer's half of send goes to it (ticket); then the attention's output side of this rank's rows with
  // all 64 heads (VHA z, VHA out, wo_a, gate, o_proj as both K halves) → aout's rows (layer index il / id)
  int a2a(const void* send, void* recv, int w, const Lane& ln, bool async);
  size_t a2a_rx(const Lane& ln) const;  // P176: lane ln's first row in zr_ (the own halves of the earlier lanes)
  void attn_own(Layer& l, int i, const Lane& ln, bf16* g = nullptr);  // g: the gates (P101), else made here
  void attn_gate(Layer& l, int i, const Lane& ln, bf16* g);
  // the FFN's per-row work on this rank's rows (input FP8 copy; MoE: router, routing, fc1 + the latent's MXFP8 copy),
  // then their all-gather (ticket)
  int ffn_front(Layer& l, const Lane& ln, bool async);
  // the FFN's tensor-parallel part over the part's rows (after each lane's gather tf[k]): MoE — per lane the shared
  // expert and its reduce-scatter (tsh[k]; P136: on this rank's rows, no reduce-scatter), the experts over the part,
  // per lane the latent's reduce-scatter (tl[k]); dense — gate/up, down, per lane the reduce-scatter (tl[k])
  void ffn_tp(Layer& l, const Lane& pt, const Lane* ln, int nl, bool async, int* tf, int* tsh, int* tl);
  // P136 sp_shared: the shared expert on this rank's rows of lane ln with both intermediate halves → its sum's rows
  void ffn_shared(Layer& l, const Lane& ln);
  // rows [n] of buf (stride ld, w bytes each) right only on their owner (own: device [n]) → right on both ranks
  void sp_rows(void* buf, long ld, int w, const int* own, int n);
  void embed(int M);  // ids d_ids_ [M] → R_ (P095: each rank embeds its half's tokens, the rows exchanged)
  void logits_gather(int R);  // lg_half_ of both ranks → lg_ [R, kVocab] (P076: in place over RDMA)
  F8Act act(const bf16* x, int ldx, int M, int k, int group);
  void taps_copy(int layer, int M);
  void taps_mid(int layer, int M);
};

}  // namespace e55
