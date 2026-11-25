// linear.h — y[M,N] = x[M,K] · Wᵀ. M ≤ 64 (decode): skinny GEMM, weights streamed once per call. 64 < M ≤ 8192
// (prefill): tiled tensor-core GEMM over the same packed weights (linear_prefill.cuh) whose every output is bitwise the
// decode kernel's (same MMA operands, chain and split order) — the switch at M = 64 changes no value. Patch P004
// gemm_tune (p004_gemm_tune.cuh) replaces the prefill tiles of split-1 shapes with faster ones, same bits.
// Formats: MXFP8 (W8A16: bf16 activations, weights dequantized exactly e4m3·2^(s−127), fp32 accumulate) and bf16.
// Patch P013 lin_pack12 (p013_pack12.cuh): a bf16 Lin may also hold a lossless 12-bit copy (pack12 at upload) that the
// decode kernel reads instead (M ≤ 64, same MMA operands → same bits, 25 % fewer bytes); M > 64 reads the bf16 copy.
// Patches P097 lm12_only / P098 lin12_prefill (p098_lin12_prefill.cuh): a Lin uploaded with kPack12Only keeps only the
// 12-bit copy (no bf16 copy: L.w == nullptr) and every M reads it — M ≤ 64 the decode kernel, M > 64 a tiled prefill
// kernel that unpacks the same bf16 operands in registers (same MMAs, chains and split sums → every output bitwise the
// bf16 copy's). With P098 on, a Lin that has both copies also takes that prefill kernel at M > 64.
// Several checkpoint tensors can be fused along N at upload (vLLM's local output order):
//   GDN qkvz = in_proj_qkv(seg slice) ‖ in_proj_z   ·  GDN ba = in_proj_b ‖ in_proj_a
//   QSA qkv  = q_proj ‖ k_proj ‖ v_proj ‖ indexer.index_qk_proj   ·  shared gate_up = gate_proj ‖ up_proj
// Contract (all kernels in this engine): deterministic and batch-invariant — a row's result never depends on M or on
// the other rows (fixed reduction order, no float atomics).
#pragma once
#include <string>
#include <vector>

#include "core/common.h"
#include "core/weights.h"

namespace e55 {

constexpr int kMaxRows = 8192;   // M limit of lin_forward
constexpr int kDecodeRows = 64;  // M ≤ this: skinny decode kernel; above: tiled prefill GEMM (same per-row arithmetic)

struct Lin {
  int fmt = WMat::BF16;  // WMat::BF16 | WMat::MXFP8 | WMat::NVFP4 (lin_make_nvfp4, decode only)
  int n = 0, k = 0;
  void* w = nullptr;      // device, kernel layout (owned); null for a 12-bit-only Lin (kPack12Only)
  uint8_t* s = nullptr;   // device scales, kernel layout (MXFP8)
  int npad = 0;           // n rounded up to the 64-row CTA tile (zero rows)
  int split = 1;          // K splits (fixed per shape at upload, summed in fixed order; never depends on M)
  float gs = 1.f;         // NVFP4: global scale · 2^8
  // P013 lin_pack12 (bf16 only): 12-bit copy for the decode kernel, layout in p013_pack12.cuh; null when not built
  uint8_t* pk = nullptr;       // device: 768 B per (16-row tile, k32 block)
  uint8_t* pk_base = nullptr;  // device: per row exponent window base, npad bytes
  uint32_t* pk_xo = nullptr;   // device: per tile first exception [npad/16 + 1]
  uint32_t* pk_x = nullptr;    // device: exceptions (kb, lane, slot, exact high byte), sorted per tile
  int pk_nx = 0;               // exceptions in pk_x
};

// pack12 modes of lin_upload / lin_make (bf16 only): no 12-bit copy · both copies (P013) · the 12-bit copy only (P097 /
// P098: the bf16 copy is never uploaded; every M reads the 12-bit one)
enum : int { kPack12None = 0, kPack12Both = 1, kPack12Only = 2 };
// Upload this rank's slices of `names` (weights; for MXFP8 the matching `<name>_scale` = `...weight_scale` is used),
// concatenated along N in the given order. pack12 (bf16 only): also build the P013 12-bit copy (checked bit-exact).
Lin lin_upload(const Weights& W, const std::vector<std::string>& names, int pack12 = kPack12None);
// Same from host row-major bytes: w [n,k] (e4m3 or bf16), s [n,k/32] ue8m0 (MXFP8). split 0 = chosen from the shape.
Lin lin_make(int fmt, int n, int k, const uint8_t* w, const uint8_t* s, int split = 0, int pack12 = kPack12None);
// Tests: the P013 copy unpacked on the device to bf16 [n, k] row-major (same unpacking as the decode kernel).
void lin_pack12_unpack(const Lin& L, bf16* out, cudaStream_t s);
// NVFP4 copy of a bf16 [n, k] matrix (quantized here; k % 256 == 0), for draft-only weights: decode kernel only (M ≤ 64).
Lin lin_make_nvfp4(int n, int k, const bf16* w);
void lin_free(Lin& L);

// x [M, k] bf16 row-major, 16-byte aligned; M ≤ kMaxRows. No workspace or global state (split-K partials are reduced
// inside a thread-block cluster), so calls may run concurrently on any streams.
void lin_forward(const Lin& L, const bf16* x, bf16* y, int M, cudaStream_t s);    // y bf16 [M, n]
void lin_forward_f32(const Lin& L, const bf16* x, float* y, int M, cudaStream_t s);  // y fp32 [M, n] (logits)
// P114 lin_pair: two Lins on the same x (same K and split, e.g. P078's column halves) as one launch when the patch is
// on and the shapes allow (M ≤ 32, one wave); otherwise lin_forward_f32(A), then lin_forward_f32(B). Same bits.
void lin_forward_f32_pair(const Lin& A, const Lin& B, const bf16* x, float* ya, float* yb, int M, cudaStream_t s);

// Tests only: force the P004 gemm_tune tile (-1 = automatic, 0 = 128×128, 1 = 128×64); applies while the patch is on.
void lin_p004_tile(int t);
// Tests only: force the P098 prefill tile (-1 = automatic, 0 = 128×128, 1 = 128×64).
void lin_p098_tile(int t);
// Tests only: P114 / P115 warps a CTA (0 = automatic; 1, 2, 4 forced — same bits).
void lin_p114_warps(int w);

}  // namespace e55
