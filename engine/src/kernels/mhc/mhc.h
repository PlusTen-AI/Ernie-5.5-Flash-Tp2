// mhc.h — ERNIE's manifold-constrained hyper-connections (docs/model.md "mHC") with the numerics of vLLM-ERNIE's fused
// post + pre path (vllm_ernie/models/ernie5_5/mhc.py mhc_fused_post_pre_tilelang → vLLM mhc_fused_tilelang + the ERNIE
// big-fuse kernel), used for every M (batch-invariant: one fixed fp32 order, no TF32 path). Residual R [M, 4, H] bf16.
//
// post (one sublayer's output `out`, the previous pre's post / comb; comb[i][j] = comb[4i + j]):
//   r_j  = fma(c[3][j], R_3, fma(c[2][j], R_2, fma(c[1][j], R_1, fma(c[0][j], R_0, post_j · out))))     fp32
//   R'_j = bf16(r_j)                                                                                   (stored)
// pre (of the next sublayer) on the UNROUNDED r: with x_q = r_j[h] at q = 4h + j (h-major), fp32 fn on CUDA cores,
//   64 chunks c of 256 q (h ∈ [64c, 64c+64), all 4 streams); chunk partial P_c[n] = sequential fmaf from 0 over its q
//   of fn[n][j·H+h]·x_q (n < 24) and x_q·x_q (n = 24, Σr²); y[n] = P_0[n], then y[n] += P_c[n] for c = 1 … 63.
//   mixes = y[0:24] · 1/(sqrt(y[24]/16384) + 1e-6)   (ERNIE's RMS factor)
//   pre = σ(mixes[0:4]·s0 + b) + 1e-6 · post = 2σ(mixes[4:8]·s1 + b) · comb = softmax_row(mixes[8:24]·s2 + b) + 1e-6,
//   comb /= colsum + 1e-6, 19× { /= rowsum + 1e-6; /= colsum + 1e-6 }   (IEEE divisions)
//   input = RMSNorm_w(bf16(Σ_j pre_j R'_j))   on the stored bf16 R' (the sublayer's input / post-attention norm)
// Layer 0's first pre (R = [e, e, e, e] from embed4): fn_b[n][h] = ((fn[n][h] + fn[n][H+h]) + fn[n][2H+h]) + fn[n][3H+h]
//   (vLLM-ERNIE finalize_mhc_broadcast_weights), y and Σe² over h in 64 chunks of 64 h (same chunk rule), mixes =
//   y · 1/(sqrt(Σe²/4096) + 1e-6); the rest as above with R'_j = e.
// The last layer's post alone (mhc_post) uses the post expression above; hc_head is unchanged.
//
// Base: one CTA per row (kernels in mhc.cu). Patches: P050 mhc_dec (M ≤ 16: chunk-parallel over 64 CTAs, the last CTA
// finishes the row at M = 1; P107 mhc_rows_pdl: its row kernel at M ≥ 2 lets the next kernel launch early), P051
// mhc_big (M > 16: the same over blocks of 16 rows), P072 mhc_fuse (M ≥ 256 in one launch), P106 mhc_prefill (post +
// pre at M ≥ 256: 2 row blocks a CTA, 4-row chains, deferred finishes), P164 mhc_verify (M ≤ 32: one CTA a chunk in one
// wave, register-blocked chunk partials) — bitwise the base.
// mix: kept in the interface (scratch), not used by these kernels.
#pragma once
#include <string>

#include "core/common.h"
#include "core/weights.h"

namespace e55 {

struct MhcW {
  float* fnq = nullptr;     // device fp32 [16384 q][24]: fn in chunk order (q = 4h + j); 24 KB per chunk of 256 q
  float* fnb = nullptr;     // device fp32 [4096 h][24]: the broadcast mixer (layer 0's attention pre only)
  float* sb = nullptr;      // device [27]: base[24], scale[3]
  bf16* normw = nullptr;    // device [4096] the fused RMSNorm's weight
  float* ws = nullptr;      // P050: chunk partials [64][16][25]
  unsigned* cnt = nullptr;  // P050: arrival counter (0 between calls)
};
// prefix: "model.layers.N.self_attention_hyper_connection." / "...mlp_hyper_connection."; norm: the RMSNorm weight.
// Layer 0's attention mixer also gets fnb.
MhcW mhc_upload(const Weights& W, const std::string& prefix, const std::string& norm);
void mhc_free(MhcW& m);

struct HcHead {
  float* fn = nullptr;  // device [4, 16384] fp32
  float* sb = nullptr;  // device [5]: base[4], scale
};
HcHead hc_head_upload(const Weights& W, const std::string& prefix = "model.mhc_contract.");  // or the draft's

// Shapes: R bf16 [M, 4, 4096]; out bf16 [M, 4096]; post fp32 [M, 4]; comb fp32 [M, 16]; inp bf16 [M, 4096];
// mix fp32 scratch [M, 72].
// Layer 0's first pre: R = [e, e, e, e] (embed4) → post, comb, inp (broadcast mixer; m must be layer 0's attention).
void mhc_pre_embed(const MhcW& m, const bf16* R, int M, float* mix, float* post, float* comb, bf16* inp, cudaStream_t s);
// The post of one sublayer and the pre of the next (every boundary but the first and the last): R ← R' in place;
// post / comb are read (this post) and then overwritten (the next pre); inp = the next sublayer's input.
void mhc_post_pre(const MhcW& m, const bf16* out, bf16* R, int M, float* mix, float* post, float* comb, bf16* inp,
                  cudaStream_t s);
// A pre on a stored residual (x_q = the bf16 R itself) — vLLM-ERNIE's unfused mhc_pre; not used by the step.
void mhc_pre(const MhcW& m, const bf16* R, int M, float* mix, float* post, float* comb, bf16* inp, cudaStream_t s);
// The last layer's post alone (before hc_head): Rout = R' (Rout may be R)
void mhc_post(const bf16* out, const bf16* R, const float* post, const float* comb, int M, bf16* Rout, cudaStream_t s);
// rows[r] of R → h bf16 [R, 4096] (rows: device int [n])
void hc_head(const HcHead& h, const bf16* R, const int* rows, int n, bf16* out, cudaStream_t s);
// Tests only: P164's rows a warp chain (0 = from M; 1, 2, 4, 8 forced — same bits)
void mhc_p164_rows(int r);

}  // namespace e55
