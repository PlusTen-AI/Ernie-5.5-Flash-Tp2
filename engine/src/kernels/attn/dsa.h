// dsa.h — ERNIE 5.5's MLA-DSA attention (6 layers; docs/model.md "DSA attention") on this rank's 32 heads.
//
// Per slot and layer, one row per position:
//   MLA cache  656 B: kv_c (after kv_a_layernorm) as e4m3 with one power-of-two fp32 scale per 128
//              (2^ceil(log2(max(amax/448, 1e-4)))), then k_pe (RoPE'd) as bf16 — [512 e4m3][4 fp32][64 bf16]
//   indexer K   68 B: LayerNorm(wk x) → NeoX RoPE on dims 0…63 → Hadamard128 → MXFP4 (e2m1 pairs, low nibble = even
//              dim, UE8M0 per 32 = 2^(floor(log2(amax rounded)) − 2)) — [64 B codes][4 UE8M0]
// Indexer logits of a query at p over candidates t ≤ p − 128: Σ_h relu(q_h · k_t) · w_h, with q in MXFP4 too, on
// tensor cores (e2m1 × e2m1, UE8M0 per 32, fp32 sums); the top 2048 (all if fewer) plus the window p−127 … p are
// the keys of the sparse MLA attention: per head q = [q_nope·W_UK (512) ‖ q_pe (64)], k = [kv_c ‖ k_pe], v = kv_c,
// scale 256^-0.5, the per-head sink. Keys are taken in position order, in fixed chunks of 256.
#pragma once
#include "core/common.h"
#include "kernels/attn/hca.h"  // RowMeta

namespace e55 {

struct DsaCache {
  uint8_t* mla = nullptr;  // [slots][ctx][656]
  uint8_t* idx = nullptr;  // [slots][ctx][68]
  int ctx = 0;
};

// kvl [M, ld] bf16: cols 0…511 kv_c after kv_a_layernorm (kvn, own buffer) and the raw k_pe at col off_pe of kvl
void dsa_mla_rows(const bf16* kvn, int ldn, const bf16* kpe, int ldpe, const RowMeta* meta, const float* inv8, int M,
                  DsaCache c, cudaStream_t s);
// q [M, 32, 256] (q_b_proj out) → q576 [M, 32, 576]: cols 512… = RoPE'd q_pe (cols 0…511 written by dsa_bmm)
void dsa_q_pe(const bf16* q, const RowMeta* meta, const float* inv8, int M, bf16* q576, cudaStream_t s);
// Y[m][h][n] = Σ_k X[m][h][k] · W[h][n][k] (bf16, fp32 sums): X row m head h at x + m·xm + h·xh, Y at y + m·ym + h·yh
void dsa_bmm(const bf16* x, long xm, long xh, const bf16* w, int H, int N, int K, int M, bf16* y, long ym, long yh,
             cudaStream_t s);
// P073 dsa_bmm_f8 (decode, M ≤ 16): dsa_bmm with W from kv_b_proj's own FP8 bytes — w8 [H][N][K] e4m3 (mode 0 W_UK:
// the transposed bytes [h][k_lat][i] = kv_b_proj[448h + i][k_lat]; mode 1 W_UV: rows 448h + 192 + n as stored) and
// its block scales sc (fp32 [rows / 128][4], this rank's rows). Same bits as dsa_bmm on the bf16 copy. false: not run
// (patch off, M > 16, other shape) — the caller runs dsa_bmm.
bool dsa_bmm_f8(const bf16* x, long xm, long xh, const uint8_t* w8, const float* sc, int mode, int H, int N, int K,
                int M, bf16* y, long ym, long yh, cudaStream_t s);
// indexer: kw [M, 192] bf16 = x · [wk ‖ weights_proj]ᵀ; qi [M, 64·128] bf16 = wq_b(q_c). Writes the K rows and the
// query's MXFP4 codes / scales (qf4 [M, 64·64] bytes, qs [M, 64·4]) and weights fp32 [M, 64]. q = false (P141
// dsa_q_own): the K rows and weights only (qi, qf4, qs not touched; may be null) — the same launch's K blocks
void dsa_idx_prep(const bf16* kw, const bf16* qi, const bf16* knw, const bf16* knb, const RowMeta* meta,
                  const float* inv8, int M, DsaCache c, uint8_t* qf4, uint8_t* qs, float* wts, cudaStream_t s,
                  bool q = true);
// keys of each row → list [M, 2176] int32 (position order) + count [M]. Rows with more than 2048 candidates get the
// indexer's top 2048 (logits in ws, rows in passes that fit ws_bytes)
void dsa_select(const uint8_t* qf4, const uint8_t* qs, const float* wts, const RowMeta* meta, DsaCache c, int M,
                int max_pos, int* list, int* cnt, float* ws, size_t ws_bytes, cudaStream_t s);
// P140 dsa_sel_split: the part of dsa_select's output that is a function of a row's position alone — list entries
// 2048 … count − 1 (the window p − 127 … p of a row with more than 2048 candidates; positions 2048 … p of one with
// fewer) and the count. With entries 0 … 2047 from the rank that selected the row: dsa_select's list and count.
void dsa_select_tail(const RowMeta* meta, int M, int* list, int* cnt, cudaStream_t s);
// sparse MLA: q576 [M, 32, 576] → olat [M, 32, 512] bf16
void dsa_attn(const bf16* q576, const int* list, const int* cnt, const RowMeta* meta, DsaCache c, const float* sink,
              int M, int max_keys, bf16* olat, float* ws, size_t ws_bytes, cudaStream_t s);
// o' = bf16(o + Σ_r V[h][r] z_r) · gate → y [M, 32·256] bf16 (o [M, 32, 256], z fp32 [M, 16, 256], gate [M, 8192])
void dsa_vha_gate(const bf16* o, const float* z, const float* V, const bf16* gate, int M, bf16* y, cudaStream_t s);

}  // namespace e55
