// fp8.h — ERNIE's FP8 block linears natively: y[M, n] = x[M, k] · Wᵀ with W e4m3 [n, k] as stored (fp32 pow2 scale per
// 128×128 block) and x quantized the reference's way (e4m3, one UE8M0 scale per 1×128 group, 2^ceil(log2(amax/448))).
// Tensor cores: mma.sync kind::mxf8f6f4.block_scale m16n8k32 e4m3 × e4m3, UE8M0 per 32 — the 128×128 block scale is
// exact as UE8M0 (every checkpoint scale is a power of two, checked at upload), so each 32-wide k step takes the
// weight block's and the activation group's scale in hardware and accumulates in fp32.
// Swap-AB: weight rows are the MMA M side (16-row tiles, fragments retiled at upload so a lane reads one 16-byte word
// per k32 step), tokens the N side (8 per MMA, staged in shared memory). Decode (few tokens) and prefill (many) are one
// kernel with different token tiles: every output element is the same chain of MMAs over k in the same order, and K
// splits are fixed per weight at upload (partials summed in fixed order) — a row's result never depends on M or on
// the other rows (deterministic, batch-invariant).
// Grouped weights (HCA wo_a, block-diagonal over output groups): n rows in `groups` equal groups, group j reads its
// own activation block x + j·gx (and scales + j·gs).
// Patches (same bits as this base, patches.def): P030 f8_decode / P031 f8_prefill (faster schedules for M ≤ 32 / M > 32,
// p030_f8_decode.cuh / p031_f8_prefill.cuh), P032 f8_fuse_act (F8Act::x), P033 f8_pdl, P034 f8_multi (f8_forward_multi),
// P060 f8_fuse_ops (F8Act::pro, F8Out::epi, F8Aux: the decode path's norms / gates / SiTU / sum inside the GEMM launch),
// P103 f8_tile (prefill M ≥ 256: 128 × 128 tiles, K splits as clusters, p103_f8_tile.cuh), P104 f8_tile_epi (the
// M > 32 epilogues in P103's store, kEpiMulQ128), P133 f8_tile_seq (prefill M ≥ 256 with a K split: 128-row tiles,
// the chains one after the other in the same warps, p133_f8_seq.cuh), P200 f8_mid (33 ≤ M ≤ 128: P030's chain and
// cluster sum with 32 / 64-row token tiles a CTA, the activation through a cp.async ring, p200_f8_mid.cuh).
#pragma once
#include <initializer_list>
#include <string>
#include <vector>

#include "core/common.h"
#include "core/weights.h"

namespace e55 {

struct F8Lin {
  int n = 0, k = 0;      // output rows (multiple of 16), inputs (multiple of 256)
  int npad = 0;          // n rounded up to the CTA's 64 rows (zero tiles)
  int split = 1;         // K splits, fixed at upload
  int groups = 1;        // block-diagonal output groups (wo_a): rows [j·n/groups, …) read activation block j
  uint8_t* w = nullptr;  // device: [tile16][k/32][lane 32][16 B] — lane (g,t): rows 16T+g / +8, k 4t… and 16+4t…
  uint8_t* s = nullptr;  // device: UE8M0 [tile16][k/128]
};

// Activation quantized for the FP8 GEMMs: q e4m3 [M, ld] (row stride ld bytes), s UE8M0 [M, ld/G] (G = 128 here,
// 32 for the MXFP8 activations of the experts). Buffers owned by the caller.
struct F8Act {
  uint8_t* q = nullptr;
  uint8_t* s = nullptr;
  int ld = 0;  // elements (bytes) per row; scales per row = ld / group
  // P032 f8_fuse_act: when set, q / s are not written yet — the activation is bf16 x (row stride ldx elements, group
  // 128) and the GEMM quantizes it itself (decode kernel) or runs act_quant into q / s first (every other path)
  const bf16* x = nullptr;
  int ldx = 0;
  // P060 f8_fuse_ops (f8_fuse_ok): the activation is computed from x — kProRms bf16(x · rs · pw) with rs as misc
  // rmsnorm over the k columns (pw = norm weight [k]); kProMul bf16(x · pw) (pw row stride pld, misc mul_bf16);
  // kProSitu bf16(SiTU(x[:, c], x[:, k + c])) (misc situ_bf16) — then quantized: in the GEMM's prologue at M ≤ 8, else
  // by one launch into q / s (row stride ld ≥ k, caller's buffers) before the GEMM.
  int pro = 0;
  const bf16* pw = nullptr;
  int pld = 0;
};
enum F8Pro : int { kProNone = 0, kProRms = 1, kProMul = 2, kProSitu = 3 };

// Upload this rank's slices of `names` (FP8 weights; "<name>.weight" + "<name>.weight_scale_inv", or the bare name +
// "<name>_scale_inv" for linear_o_group_proj), concatenated along n in the given order.
F8Lin f8_upload(const Weights& W, const std::vector<std::string>& names, int groups = 1);
// From host bytes: w e4m3 [n, k] row-major, s fp32 [cdiv(n_part,128), k/128] per part (parts = row counts, sum n).
F8Lin f8_make(int n, int k, const uint8_t* w, const std::vector<const float*>& s, const std::vector<int>& parts,
              int groups = 1, int split = 0);
void f8_free(F8Lin& L);

// x bf16 [M, k] (row stride ldx elements) → a.q / a.s, one scale per `group` columns (128 or 32).
void act_quant(const bf16* x, int ldx, int M, int k, int group, F8Act a, cudaStream_t s);
// y [M, n] (row stride ldy) = x · Wᵀ; x quantized with group 128. ws: fp32 workspace for split > 1, ≥ f8_ws(L, M).
// Grouped: activation block j at a.q + j·gx bytes / a.s + j·gs bytes (rows of a block still a.ld apart).
void f8_forward(const F8Lin& L, const F8Act& a, int M, bf16* y, int ldy, float* ws, cudaStream_t s,
                long gx = 0, long gs = 0);
void f8_forward_f32(const F8Lin& L, const F8Act& a, int M, float* y, int ldy, float* ws, cudaStream_t s,
                    long gx = 0, long gs = 0);
size_t f8_ws(const F8Lin& L, int M);  // bytes
// P034 f8_multi: several GEMMs on the same activation (same k), y_i = x · W_iᵀ — at decode one launch for all (patches
// on, M ≤ 32), otherwise one f8_forward each. Same bits as separate calls.
struct F8Out {
  const F8Lin* L;
  bf16* y;
  int ldy;
  // P060 f8_fuse_ops (f8_fuse_ok): epilogue on the bf16 output h = bf16(acc) — kEpiSigmoid y = bf16(σ(h)) (misc
  // sigmoid_inplace); kEpiAdd y = bf16(h + e) (e row stride lde, misc add_bf16); kEpiQ32 y = h and also its MXFP8 copy
  // (group 32, act_quant's bytes) into q [M, ldy] / qs [M, ldy / 32]. In the GEMM's store on the decode kernel's path
  // (M ≤ 32), else one launch after the GEMM.
  int epi = 0;
  const bf16* e = nullptr;
  int lde = 0;
  uint8_t* q = nullptr;
  uint8_t* qs = nullptr;
};
// P104 f8_tile_epi: kEpiMulQ128 v = bf16(h ⊙ e) (e row stride lde, misc mul_bf16's bits) quantized as the next GEMM's
// activation (group 128, act_quant's bytes) into q [M, ldy] / qs [M, ldy / 128]; y (h) written only when non-null —
// only in P103's store (f8_epi_fused), there is no separate launch for it.
// P134 f8_situ_epi: kEpiSituQ128 on a weight whose rows are [gate n/2 ‖ up n/2] (the shared expert's gate/up): v =
// bf16(SiTU(h[:, c], h[:, n/2 + c])) (k_prep's kProSitu bits) quantized as the down's FP8 activation (group 128,
// act_quant's bytes) into q [M, ldy = n/2] / qs [M, ldy / 128]; no bf16 output (y unused) — only in P133's store.
enum F8Epi : int { kEpiNone = 0, kEpiSigmoid = 1, kEpiAdd = 2, kEpiQ32 = 3, kEpiMulQ128 = 4, kEpiSituQ128 = 5 };
// P060: a side RMSNorm in the same launch as the GEMM (M ≤ 8) or as its activation op (misc rmsnorm's bits): y [M, n]
// (stride ldy) = bf16(x · rs · w), x stride ldx; n ∈ {512, 1024, 2048}
struct F8Aux {
  const bf16* x;
  int ldx;
  const bf16* w;
  int n;
  bf16* y;
  int ldy;
};
void f8_forward_multi(std::initializer_list<F8Out> outs, const F8Act& a, int M, float* ws, cudaStream_t s,
                      const F8Aux* aux = nullptr);
// one weight with F8Out's epilogue (P060; P104 in P103's store); grouped weights take gx / gs as f8_forward
void f8_forward_out(const F8Out& o, const F8Act& a, int M, float* ws, cudaStream_t s, long gx = 0, long gs = 0);
// P104: does f8_forward_out run epilogue `epi` of weight L at M rows inside the GEMM's store (P103 f8_tile +
// P104 f8_tile_epi on, the shape taken, ws the workspace that will be passed)? kEpiMulQ128 runs only there.
bool f8_epi_fused(const F8Lin& L, int M, int epi, const float* ws);
// P032: may a group-128 activation of M rows be handed to the GEMMs unquantized (F8Act::x)? (patches on, M ≤ 8)
bool f8_defer_act(int M);
// P060 f8_fuse_ops: may the caller use F8Act::pro, F8Out::epi and F8Aux (any M ≥ 1; patch on)?
bool f8_fuse_ok(int M);
// P060 activation: op on x (pw, pld as F8Act), quantized into q / s (row stride ld) when M > 8
inline F8Act f8_pro(int op, const bf16* x, int ldx, const bf16* pw, int pld, uint8_t* q, uint8_t* s, int ld) {
  F8Act a{q, s, ld};
  a.x = x, a.ldx = ldx, a.pro = op, a.pw = pw, a.pld = pld;
  return a;
}
int f8_test_mid(int mt);  // tests: P200's token tile 32 / 64 / 128 for every shape it runs (0: the measured rule); returns the previous one
int f8_test_pfn(int n);  // tests: P064's L2 prefetch depth in k32 blocks per warp (0: none); returns the previous one
int f8_test_seq(int c);  // tests: P133's tile config 0–3 for every shape it can take (−1: the measured choice)
const char* f8_test_path(const F8Lin& L, int M, int epi);  // tests: which kernel f8_forward(_out) takes (P031, P103 …)

}  // namespace e55
