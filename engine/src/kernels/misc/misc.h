// misc.h — small elementwise / row kernels of the ERNIE step (docs/model.md): embedding into the 4 mHC streams,
// RMSNorm, the output gates, SiTU-GLU for the bf16 FFN paths, multimax on the logits. fp32 math, one rounding to bf16.
#pragma once
#include "core/common.h"

namespace e55 {

// ids [M] (device) → R [M, 4, 4096] = the token's embedding row in all four streams
void embed4(const bf16* table, const int* ids, int M, bf16* R, cudaStream_t s);
// y [M, n] (row stride ldy) = bf16(x · rsqrt(mean(x²) + eps) · w), x row stride ldx; n % 8 == 0
void rmsnorm(const bf16* x, int ldx, const bf16* w, int M, int n, bf16* y, int ldy, cudaStream_t s);
// y = bf16(σ(x)) in place, [M, n] contiguous
void sigmoid_inplace(bf16* x, size_t n, cudaStream_t s);
// y [M, n] = bf16(a · b) (a row stride lda, b row stride ldb)
void mul_bf16(const bf16* a, int lda, const bf16* b, int ldb, int M, int n, bf16* y, int ldy, cudaStream_t s);
// gu [M, 2n] = [g | u] bf16 → h [M, n] = bf16(2·tanh(g/4)·(1 + tanh(g/2)) · 25·tanh(u·fl(1/25))) — vLLM's situ_and_mul
void situ_bf16(const bf16* gu, int M, int n, bf16* h, cudaStream_t s);
// logits fp32 [M, V] in place: x + t0·relu(r0−x) + t1·relu(x−r1) + t2·relu(r2−x)² + t3·relu(x−r3)²; rt = device [8]
// (r0…r3, t0…t3)
void multimax(float* lg, int M, int V, int ld, const float* rt, cudaStream_t s);
// y [M, n] = bf16(a + b), b row stride ldb (fc2 output + the shared expert's all-reduced output, both bf16 as vLLM)
void add_bf16(const bf16* a, const bf16* b, int ldb, int M, int n, bf16* y, cudaStream_t s);
// dst[j] = src row rows[j] (row_bytes each; rows ≥ M are skipped) — the gate's taps
void gather_rows(const void* src, int row_bytes, const int* rows, int n, int M, void* dst, cudaStream_t s);
// src [2][R][Vh] (the all-gathered vocabulary halves) → dst [R][2·Vh]
void halves_to_rows(const float* src, int R, int Vh, float* dst, cudaStream_t s);
// P094 sp_resid: rows of both ranks side by side (src row i = [rank 0's w bytes | rank 1's w bytes], stride sld) → dst
// row i (stride dld) = rank own[i]'s bytes (w, strides, addresses multiples of 16)
void pick_rows(const void* src, long sld, int w, const int* own, int n, void* dst, long dld, cudaStream_t s);
// P095 embed_split: half = rows [v0, v0 + vh) of the table; E row i (stride ld elements) = the row of ids[i] if it is in
// the half (else untouched)
void embed_half(const bf16* half, const int* ids, int M, int v0, int vh, bf16* E, long ld, cudaStream_t s);
// P095: R [M, 4, 4096] in place — streams 2 / 3 of row i hold the row as embedded by rank 0 / 1; all four streams ←
// the one from the rank whose half holds ids[i] (ids[i] / vh) — embed4's bytes
void embed4_pick(const int* ids, int M, int vh, bf16* R, cudaStream_t s);
// P100 a2a_attn: the VHA z of rows whose 64 heads' o are here: z [M, 16, D] = fl(b + a), a = Σ_h U[h][r]·o[m][h][d]
// over o's 32 heads (hca.cu k_vha_partial's fma chain from 0), b = zo (the other heads' partial; may be z itself) or,
// zo null, the same chain over (o2, U2) — the two ranks' partials added as the VHA all-reduce adds them (one fp32 add,
// commutative). D % 128 == 0.
void vha_z(const bf16* o, const float* U, const bf16* o2, const float* U2, const float* zo, int D, int M, float* z,
           cudaStream_t s);

}  // namespace e55
