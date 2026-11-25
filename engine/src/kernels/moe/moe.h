// moe.h — ERNIE 5.5's latent MoE on this rank (docs/model.md "FFN"), native W4A8:
//   router   l = x · [gate.weight ‖ gate.weight_1]ᵀ (bf16 weights, fp32 out; kernels/linear) — s = σ(l0) + σ(l1),
//            ids = top10(s + e_score_correction_bias), w = s[ids] / max(Σ, 1e-20) · routed_scaling_factor_param[ids]
//   experts  in the 2048 latent: lat (MXFP8: e4m3, UE8M0 per 32) → up = lat · W13ᵀ (this rank's 1024 gate + 1024 up
//            rows) → h = SiTU(g, u) → MXFP8 per 32 → y = h · W2ᵀ (this rank's 1024 columns) — weights e2m1 + UE8M0
//            per 32 as stored, tensor cores mma.sync kind::mxf8f6f4 e2m1 × e4m3 (W4A8, the reference's numerics)
//   combine  part[row] = Σ_slot w · y (fp32, slot order) — this rank's partial latent (the caller all-reduces, then
//            latent_norm → fc2_latent_proj, kernels/gemm); moe_experts_bf16: that fp32 sum rounded once to bf16 (vLLM
//            TP2 rounds each rank's routed output to bf16 before its all-reduce)
//   slots    moe_route returns the 10 experts in torch.topk(sorted=False) order (vLLM-ERNIE's routing kernel):
//            above the 10th score by id, then those equal to it by id; renormalized by the 16-slot tree sum
// Routing plan: the (row, slot) pairs of each expert in (row, slot) order, built without atomics; every pair's output
// is the same chain of MMAs whatever M or the other rows are (deterministic, batch-invariant).
#pragma once
#include <string>

#include "core/common.h"
#include "core/weights.h"

namespace e55 {

struct MoeExperts;  // opaque (moe.cu): the 512 experts' retiled weights for this rank

MoeExperts* moe_upload(const Weights& W, int layer);
// the same from "<prefix><e>.w1.weight" … (the DSpark draft stages: "model.mtp.<i>.block_sparse_moe.experts.")
MoeExperts* moe_upload(const Weights& W, const std::string& prefix);
void moe_free(MoeExperts* e);

// Scratch for M ≤ max_rows rows (one per engine; calls stream-ordered).
struct MoeScratch;
MoeScratch* moe_scratch(int max_rows);
void moe_scratch_free(MoeScratch* s);

// logits fp32 [M, 1024] (router GEMM output), bias / scale fp32 [512] → ids int32 [M, 10], w fp32 [M, 10]
void moe_route(const float* logits, const float* bias, const float* scale, int M, int* ids, float* w, cudaStream_t s);
// lat_q / lat_s: the latent as MXFP8 [M, 2048] / [M, 64]; ids / w from moe_route → part fp32 [M, ld] (first 2048)
void moe_experts(const MoeExperts* e, MoeScratch* sc, const uint8_t* lat_q, const uint8_t* lat_s, const int* ids,
                 const float* w, int M, float* part, int ld, cudaStream_t s);
// same, part bf16 [M, ld] (first 2048) = bf16(the fp32 combine above)
void moe_experts_bf16(const MoeExperts* e, MoeScratch* sc, const uint8_t* lat_q, const uint8_t* lat_s, const int* ids,
                      const float* w, int M, bf16* part, int ld, cudaStream_t s);

// P066 moe_fused (decode / verify, M ≤ 32): moe_route + (the latent's MXFP8 act_quant) + moe_experts_bf16 as one
// launch, same bits — logits fp32 [M, 1024]; the latent either bf16 lat [M, ldl] (quantized inside, act_quant's
// arithmetic; lat_q = lat_s = null) or already MXFP8 lat_q [M, 2048] / lat_s [M, 64] (lat = null); ids / w as
// moe_route writes them. moe_fused_ok(M): the patch is on and M fits (otherwise call moe_route / act_quant /
// moe_experts_bf16).
bool moe_fused_ok(int M);
void moe_fused(const MoeExperts* e, MoeScratch* sc, const float* logits, const float* bias, const float* scale,
               const bf16* lat, int ldl, const uint8_t* lat_q, const uint8_t* lat_s, int M, int* ids, float* w,
               bf16* part, int ld, cudaStream_t s);
void moe_fused_f32(const MoeExperts* e, MoeScratch* sc, const float* logits, const float* bias, const float* scale,
                   const bf16* lat, int ldl, const uint8_t* lat_q, const uint8_t* lat_s, int M, int* ids, float* w,
                   float* part, int ld, cudaStream_t s);

}  // namespace e55
