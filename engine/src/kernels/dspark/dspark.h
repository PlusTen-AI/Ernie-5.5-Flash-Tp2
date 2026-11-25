// dspark.h — kernels of the DSpark draft (the checkpoint's model.mtp.*, docs/model.md "DSpark draft") and of the
// speculative verify's state rollback. The draft's linears, MoE, mHC and VHA run through the target's kernels
// (kernels/gemm, moe, mhc, attn); here only what the draft has of its own:
//   aux       the stream mean of the target's bf16 residual after layers 40, 41, 42 (vLLM-ERNIE's aux hidden states)
//   ctx rows  the draft's context KV: per stage the HCA-format row (584 B) of RMSNorm_kv(linear_kv_proj(main_x)), RoPE
//             at the row's position, in a per-slot ring of 256 rows (by position % 256; a draft at anchor P reads
//             P−128 … P−1, so rows a rejected verify left past the kept ones are never read before they are rewritten)
//   attention the block's queries (anchor + N−1 noise rows, positions P … P+N−1) over the context rows max(P−128, 0) …
//             P−1 and all N block rows (non-causal), softmax with the per-head sink, scale 512^-0.5
//   Markov    draft logits of step i: multimax(base_i + markov_w2 · markov_w1[prev]) on this rank's vocabulary half, its
//             argmax candidate exchanged between the ranks (greedy drafts)
//   rollback  HCA window ring rows (and, when the block closes a compressor group, the open group's state rows) saved
//             before a verify step and put back for the rejected rows after it
#pragma once
#include "core/common.h"
#include "kernels/attn/hca.h"
#include "kernels/sample/sample.h"

namespace e55 {

constexpr int kDspStages = 3, kDspRing = 256, kDspCtx = 128, kDspMaxN = 16, kDspRank = 256, kDspNoise = 1024;
constexpr int kDspAux = 3 * 4096;  // features: cat(aux40, aux41, aux42)

// aux[k][which·4096 + h] = bf16((((R0 + R1) + R2) + R3) · 0.25) of step row rows[k] (R [M, 4, 4096] bf16)
void dsp_aux(const bf16* R, const int* rows, int K, int which, bf16* aux, cudaStream_t s);
// kv [K, 1536] bf16 (the three stages' RMSNorm_kv outputs side by side) → ring rows of position pos[k], slot slot[k]:
// ring [3][slots][256][584]
void dsp_ctx_rows(const bf16* kv, int ldkv, const int* pos, const int* slot, int K, const float* inv, uint8_t* ring,
                  int slots, cudaStream_t s);
// q [M, 32, 512] (prepped: per-head RMS, RoPE), rows [M][584] the block's own KV rows, meta: pos, slot, r0 (first row
// of the block), p0 (anchor position), pend (its last position); ring: this stage's [slots][256][584] → o [M, 32, 512]
void dsp_attn(const bf16* q, const uint8_t* rows, const RowMeta* meta, const uint8_t* ring, const float* sink, int M,
              bf16* o, cudaStream_t s);
// P129 draft_attn: hca_q_prep (draft RoPE inv) + hca_kv_rows of the block (kv [M, 512] bf16 after kv_layernorm, row
// stride ldkv) + dsp_attn + vha_partial (U: this rank's [32][16]) in one launch, programmatic dependent — q raw from
// q_up → o [M, 32, 512] bf16 and zp [M, 16, 512] fp32, the bytes of the four base launches (rows_ is not written)
void dsp_attn_fused(const bf16* q, const bf16* kv, int ldkv, const RowMeta* meta, const uint8_t* ring, const float* sink,
                    const float* inv, const float* U, int M, bf16* o, float* zp, cudaStream_t s);

// One sequential step of the Markov head for K items: item k reads base row brow[k] of base [*, Vh] (fp32, this
// rank's half of h·lm_headᵀ), the previous token prev[seq[k]] and writes out row orow[k] of out [*, Vh] =
// multimax(base + w2 · w1t[prev]) (w1t [V, 256] bf16, w2 [Vh, 256] bf16, both fp32 products / sums). Greedy: part
// [K][nb] (value, id) — each CTA's first maximum (global id off + v).
int dsp_markov_blocks(int Vh);
int dsp_markov_rows();  // vocabulary rows a sweep CTA covers (block vb: rows vb · this … )
// w2q / w2s (P082 markov8, else null): w2 as e4m3 [Vh, 256] + UE8M0 [Vh, 8] instead of bf16. g0 / ng: this launch is
// CTAs g0 … g0 + ng − 1 of the sweep's dsp_markov_blocks(Vh) (ng < 0: to the end); back: CTA g sweeps vocabulary block
// nb − 1 − g instead of g (P161 markov_back / P163 markov_pipe) — the same values in every element of out and part
void dsp_markov(const float* base, const int* brow, const int* seq, const int* prev, const bf16* w1t, const bf16* w2,
                const uint8_t* w2q, const uint8_t* w2s, int Vh, int off, const float* mm, int K, float* out,
                const int* orow, float2* part, cudaStream_t s, int g0 = 0, int ng = -1, bool back = false);
// P083 markov_topk: item k's row brow[k] of cands [*][kSparseRec] (this rank's top-64 base logits, sample_cands):
// y_j = multimax(x_j + w2[id_j − off] · w1t[prev]); greedy: exch [2][K][2] as dsp_argmax writes it (this rank's first
// maximum); sampled (oc != null): oc [K][64] the candidates with y (each rank's, gathered by the caller)
void dsp_markov_topk(const SparseCand* cands, const int* brow, const int* seq, const int* prev, const bf16* w1t,
                     const bf16* w2, int off, const float* mm, int K, int rank, float* exch, SparseCand* oc,
                     cudaStream_t s);
// sampled drafts from candidates: q rows [K, V] = −inf except both ranks' candidates (oc2 [2][K][64])
void dsp_cand_rows(const SparseCand* oc2, int K, float* q, cudaStream_t s);
// part → exch [2][K][2] fp32: this rank's (max, id) at [rank], zeros at the other rank (all-reduced next)
void dsp_argmax(const float2* part, int nb, int K, int rank, float* exch, cudaStream_t s);
// after the all-reduce: token = the first maximum of the two candidates (rank 0's ids are lower); prev[seq[k]] = token,
// drafts[seq[k] · kDspMaxN + step] = token
void dsp_pick(const float* exch, int K, const int* seq, int step, int* prev, int* drafts, cudaStream_t s);
// P130 draft_chain (greedy, markov8's w2q / w2s): step `step` of the chain in one launch — the previous step's pick
// (k_pick's rule on its all-reduced exchange rows pex [2][pK][2], items' sequences pseq [pK]; step 0: pex = null, the
// previous token is last[seq]) with CTA 0 writing that step's drafts, dsp_markov's sweep over this rank's half, and
// dsp_argmax's candidate per item in ex [2][K][2] by the last CTA (cnt: this step's counter, 0 between launches;
// part [K][dsp_markov_blocks(Vh)]). Then the all-reduce of ex; after the last step dsp_pick.
void dsp_chain(const float* base, const int* brow, const int* seq, int K, const int* pseq, int pK, const float* pex,
               const int* last, int* drafts, int step, const bf16* w1t, const uint8_t* w2q, const uint8_t* w2s, int Vh,
               int off, const float* mm, float2* part, unsigned* cnt, int rank, float* ex, cudaStream_t s);
// sampled drafts: prev[seq[k]] = tok[k], drafts[seq[k] · kDspMaxN + step] = tok[k]
void dsp_put(const int* tok, int K, const int* seq, int step, int* prev, int* drafts, cudaStream_t s);

// confidence logits of the draft rows: conf[r] = w · [h_r, w1t[prev_r]] (fp32; w [4096 + 256] fp32, h [M, 4096] the
// head's input before its norm), prev_r = r's index i in its block: i = 0 → last[seq], else drafts[seq · 16 + i − 1]
void dsp_conf(const bf16* h, const int* rseq, const int* ridx, const int* last, const int* drafts, const bf16* w1t,
              const float* w, int M, float* conf, cudaStream_t s);
// distinct ids among ids[0 … n) (< 512) added to *cnt (one CTA)
void dsp_experts(const int* ids, int n, int* cnt, cudaStream_t s);

// ---- verify rollback (HCA layers l < nl; tables: device arrays of the per-layer cache pointers) ---------------------
struct DspRoll {
  uint8_t* const* ring;   // [nl] HcaCache::ring  [slots][128][584]
  float* const* state;    // [nl] HcaCache::state [slots][128][1024]
  const float* const* ape;  // [nl] compressor ape [128][512]
  uint8_t* bk_ring;       // [nl][maxr][584]
  float* bk_state;        // [nl][maxr][1024]
  float* stash;           // [nl][maxr][1024]  the step's compressor projections (crossing steps)
  int nl, maxr;
};
// before a verify step of M rows: ring row (slot, pos % 128) of every row; the state row (slot, pos % 128) of the rows
// after cross[r] (the position closing a compressor group inside the row's block, −1 if none)
void dsp_backup(const RowMeta* meta, const int* cross, int M, DspRoll r, cudaStream_t s);
// after the accept: info[r] = (cross, kend = first position not kept): rows at pos ≥ kend get their ring row back; if
// the group-closing row was not kept (cross ≥ kend), the rows after it give back their state rows and the kept rows
// of the block write their state rows (the stashed projection, + ape on the score half) as a plain step would have
void dsp_restore(const RowMeta* meta, const int2* info, int M, DspRoll r, cudaStream_t s);

}  // namespace e55
