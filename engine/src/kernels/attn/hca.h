// hca.h — Heavily Compressed Attention of ERNIE 5.5 (37 layers; docs/model.md "HCA attention") on this rank's 32 heads.
//
// Per slot and layer: the window ring (the last 128 KV rows by position % 128), the compressed entries (one per 128
// tokens) and the compressor state (kv ‖ score+ape of the current group's tokens, fp32, by position % 128).
// KV row (584 B, vLLM fp8_ds_mla V4 content): nope 448 as e4m3 with one UE8M0 per 64 (values rounded to bf16 first,
// amax ≥ 1e-4), rope 64 as bf16; byte layout here: [448 e4m3][64 bf16][7 UE8M0][pad].
// Step rows: rows of one sequence are consecutive positions; RowMeta per row = {pos, slot, first row of its sequence
// in this step, that row's position, the sequence's last position in this step}.
// Row indexing contract: per-row arrays (q, o, zp, rows written by kv_rows / hca_pre, ckv read by the state update,
// the wo_a activation) are indexed by the row's index r in [0, M) relative to the pointers passed; the KEYS of this
// step (the window rows in `rows`, the group's compressor rows in `ckv`) are indexed by RowMeta.r0 + (t − p0) relative
// to the same pointers. A caller that runs a step's rows in pieces (row lanes) with pointers advanced by L rows must
// pass RowMeta with r0 − L (negative allowed), and run the pieces' chains in row order (a later piece's hca_pre /
// attention reads the earlier pieces' KV rows and compressed entries; its hca_post overwrites ring slots the earlier
// pieces' attention reads).
#pragma once
#include "core/common.h"

namespace e55 {

struct RowMeta {
  int pos, slot, r0, p0, pend;  // pend: the sequence's last position in this step
};

struct HcaCache {
  uint8_t* ring = nullptr;  // [slots][128][584]
  uint8_t* comp = nullptr;  // [slots][max_comp][584]
  float* state = nullptr;   // [slots][128][1024]
  int max_comp = 0;
};

// q [M, 32·512] bf16 in place: per head x · rsqrt(mean(x²) + eps), then RoPE (interleaved) on dims 448…511 at pos
void hca_q_prep(bf16* q, const RowMeta* meta, const float* inv_freq, int M, cudaStream_t s);
// kv [M, 512] bf16 (after kv_layernorm) → rows [M][584]: RoPE, bf16, FP8
void hca_kv_rows(const bf16* kv, int ldkv, const RowMeta* meta, const float* inv_freq, int M, uint8_t* rows,
                 cudaStream_t s);
// ckv fp32 [M, 1024] = x · [linear_wkv ‖ linear_wgate]ᵀ; every row that closes a group (pos % 128 == 127) writes its
// compressed entry: softmax over the group's 128 scores (+ ape) per dim, Σ p·kv, RMSNorm(norm), RoPE at 128·group, FP8.
// Rows of the group before this step come from the state. Then the state takes the rows of each sequence's open group.
void hca_compress(const float* ckv, const float* ape, const bf16* normw, const RowMeta* meta, const float* inv_freq,
                  int M, HcaCache c, cudaStream_t s);
// attention: o [M, 32, 512] bf16. Keys of row (pos p): compressed 0 … ⌊(p+1)/128⌋ − 1 and window max(0, p−127) … p
// (rows of this step from `rows`, older ones from the ring). Softmax with the per-head sink (fp32 [32]), scale 512^-0.5.
// Keys are taken in fixed chunks of 256 (fp32 partials, merged in chunk order): a row's result depends on its keys
// only. max_keys: the most keys any row has (host); the rows run in passes whose partials fit ws (ws_bytes).
void hca_attn(const bf16* q, const uint8_t* rows, const RowMeta* meta, HcaCache c, const float* sink, int M,
              int max_keys, bf16* o, float* ws, size_t ws_bytes, cudaStream_t s);
// after attention: the ring takes each sequence's last ≤ 128 rows of this step
void hca_ring_store(const uint8_t* rows, const RowMeta* meta, int M, HcaCache c, cudaStream_t s);

// VHA, this rank's heads h0 … h0+31 of 64: zp fp32 [M, 16, D] = Σ_h U[h] · o_h (partial; all-reduced by the caller)
void vha_partial(const bf16* o, int D, const float* U, int M, float* zp, cudaStream_t s);
// o' = bf16(o + Σ_r V[h][r] z_r) for this rank's heads; HCA: inverse RoPE at pos on dims 448…, then FP8 1×128 into
// the group-major activation of wo_a: q [4][M][4096] e4m3, sc [4][M][32] UE8M0
void hca_vha_out(const bf16* o, const float* z, const float* V, const RowMeta* meta, const float* inv_freq, int M,
                 uint8_t* q, uint8_t* sc, cudaStream_t s);

// ---- P042 hca_fuse (patches.def; model.cc calls these instead of the nine stages above when it is on) --------------
// hca_pre: hca_kv_rows + the group-closing rows of hca_compress (the compressor state update moves to hca_post);
// ws: scratch of ≥ M · 64 B (P044 at decode, P089 at prefill)
void hca_pre(const bf16* kv, int ldkv, const float* ckv, const float* ape, const bf16* normw, const RowMeta* meta,
             const float* inv_freq, int M, uint8_t* rows, HcaCache c, float* ws, size_t ws_bytes, cudaStream_t s);
// hca_attn_fused: hca_q_prep on the raw q (q is read, not written back) + hca_attn + vha_partial (zp [M, 16, 512])
void hca_attn_fused(const bf16* q, const uint8_t* rows, const RowMeta* meta, HcaCache c, const float* sink,
                    const float* inv_freq, const float* U, int M, int max_keys, bf16* o, float* zp, float* ws,
                    size_t ws_bytes, cudaStream_t s);
// P150 hca_attn_long (tests / tuning): ≤ G chunks per work item, rows of ≥ min_ch chunks (≤ 0: unchanged)
void hca_attn_long_tune(int G, int min_ch);
// hca_post: hca_vha_out + hca_ring_store + the state update of hca_compress (rows, ckv, ape as given to hca_pre)
void hca_post(const bf16* o, const float* z, const float* V, const RowMeta* meta, const float* inv_freq,
              const uint8_t* rows, const float* ckv, const float* ape, int M, HcaCache c, uint8_t* q, uint8_t* sc,
              cudaStream_t s);
// P100 a2a_attn: hca_post's ring store + state update alone (every row; the VHA part: hca_vha_out on the row owner)
void hca_post_state(const RowMeta* meta, const uint8_t* rows, const float* ckv, const float* ape, int M, HcaCache c,
                    cudaStream_t s);
// P100: the row owner's VHA of all 64 heads: z = fl(zp + Σ_h Up[h]·op[h]) (the peer's partial by vha_partial's chain,
// added as the all-reduce adds), then hca_vha_out of this rank's heads (om, Vm → qm / sm) and the peer's (op, Vp → qp /
// sp), z never stored; om / op [M, 32, 512], zp [M, 16, 512] (this rank's partial)
void hca_vha_own(const bf16* om, const bf16* op, const float* zp, const float* Up, const float* Vm, const float* Vp,
                 const RowMeta* meta, const float* inv_freq, int M, uint8_t* qm, uint8_t* sm, uint8_t* qp, uint8_t* sp,
                 cudaStream_t s);

}  // namespace e55
