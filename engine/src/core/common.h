// common.h — shared types and number formats for the ERNIE 5.5 Flash engine (GB10, sm_121a, TP=2). Math: docs/model.md.
//
// Weight formats as stored in the checkpoint (row-major [N, K], N = output rows, K = input):
//   FP8 block : E4M3 [N, K] + fp32 scale_inv [N/128, K/128], every scale a power of two (checked at load) — uploaded
//               as MXFP8 (E4M3 + UE8M0 [N, K/32], each 128-block scale repeated): w = e4m3 · 2^(s−127), exact in bf16
//   MXFP4     : routed experts, packed e2m1 [N, K/2] (low nibble = even k) + UE8M0 [N, K/32]: w = e2m1 · 2^(s−127)
//   bf16      : embedding, lm_head, routers, compressor, indexer wk / weights_proj, VHA, norms; fp32: mHC, sinks, ape
// Quantized GEMMs keep the reference's activation quantization: the activation is put on its FP8 grid first (W8A8:
// 1×128 UE8M0 groups; W4A8 experts: 1×32) and the product runs as bf16 × bf16 → fp32 on tensor cores — the same
// products as an FP8 MMA (both factors are exact in bf16), fp32 sums.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "core/step_kind.h"  // step_prefill(): the running step's kind (prefill chunk vs decode / verify / draft)

namespace e55 {

using bf16 = __nv_bfloat16;

// ---- model constants (config.json; checked against the config at load) -------------------------------------------
constexpr int kHidden = 4096, kLayers = 43, kStreams = 4, kHcDim = kStreams * kHidden, kHcMix = 24;
constexpr int kVocab = 201216, kVocabHalf = 100608;
// HCA attention (37 layers): 64 heads × 512 (448 nope + 64 rope), one shared KV row, q_lora 1024, 8 groups × 1024
constexpr int kHeads = 64, kHeadDim = 512, kNope = 448, kRope = 64, kQLora = 1024, kGroups = 8, kOLora = 1024;
constexpr int kWindow = 128, kRatio = 128;
constexpr int kKvRowBytes = 584;  // 448 e4m3 + 64 bf16 + 7 UE8M0 + 1 pad (vLLM fp8_ds_mla, DeepSeek-V4)
// DSA attention (6 layers): MLA, kv_lora 512 + rope 64, q 192 nope + 64 rope, v 256; indexer 64 × 128, top 2048
constexpr int kKvLora = 512, kMlaNope = 192, kMlaV = 256, kIdxHeads = 64, kIdxDim = 128, kIdxTop = 2048;
constexpr int kMlaRowBytes = 656;  // 512 e4m3 + 4 fp32 pow2 scales + 64 bf16 (vLLM fp8_ds_mla, DeepSeek-V3.2)
constexpr int kIdxRowBytes = 68;   // 64 B packed e2m1 + 4 UE8M0 (MXFP4)
// FFN
constexpr int kDenseInter = 14336, kExperts = 512, kTopK = 10, kLatent = 2048, kExpInter = 2048, kSharedInter = 2048;
constexpr float kEps = 1e-6f;
constexpr int kVhaRank = 16;
// config.json compress_ratios: -2 (DSA) at layers 7, 14, 21, 28, 35, 42; 128 (HCA) elsewhere (checked at load)
inline bool is_dsa(int i) { return i > 0 && i % 7 == 0; }

// ---- number formats (device) -------------------------------------------------------------------------------------
__host__ __device__ inline float e2m1_to_float(uint32_t c) {  // 4-bit code: s eem
  const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  float v = mag[c & 7];
  return (c & 8) ? -v : v;
}
__host__ __device__ inline float e4m3_to_float(uint8_t b) {
#ifdef __CUDA_ARCH__
  __nv_fp8_e4m3 x;
  x.__x = b;
  return float(x);
#else
  int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
  float v = (e == 0) ? (m / 8.f) * (1.f / 64.f) : (1.f + m / 8.f) * float(1 << e) / 128.f;  // bias 7
  if (e == 15 && m == 7) v = __builtin_nanf("");
  return s ? -v : v;
#endif
}
__host__ __device__ inline float ue8m0_to_float(uint8_t s) {  // 2^(s-127)
#ifdef __CUDA_ARCH__
  return __int_as_float(s == 0 ? 0x00400000 : (int(s) << 23));
#else
  union { uint32_t u; float f; } c;
  c.u = s == 0 ? 0x00400000u : (uint32_t(s) << 23);
  return c.f;
#endif
}

// ---- small helpers -----------------------------------------------------------------------------------------------
__host__ __device__ constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

// A weight matrix resident on one rank, in the checkpoint's row-major layout unless a kernel says otherwise.
struct WMat {
  enum Fmt : int { BF16 = 0, MXFP8 = 1, NVFP4 = 2 } fmt = BF16;
  int n = 0, k = 0;              // logical rows (outputs) × cols (inputs) on this rank
  const void* w = nullptr;       // bf16 [n,k] | e4m3 [n,k] | packed e2m1 [n,k/2]
  const uint8_t* s = nullptr;    // MXFP8: ue8m0 [n,k/32] | NVFP4: e4m3 [n,k/16]
  float g = 1.f;                 // NVFP4 global scale (weight_scale_2)
};

}  // namespace e55
