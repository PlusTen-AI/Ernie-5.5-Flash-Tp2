// p051_mhc_big.cuh — P051 mhc_big: mHC pre / post + pre / layer-0 pre for M > 16 rows (prefill).
// Included by mhc.cu inside its anonymous namespace after p050_mhc_dec.cuh.
//
// Why: the base's one CTA per row re-reads the whole fp32 mixer (1.57 MB) for every row (6.4 GB of L2 at M = 4096).
// Here P050's kernels run over a grid of 64 chunks × blocks of 16 rows: each CTA brings its chunk's 24 KB of fnq once
// (an L2 hit after the first block) and computes the chunk partials of 16 rows exactly as the base; the partials go to
// a workspace shared by all mixers ([64][M][25] fp32, 52 MB for 8192 rows, allocated once at the first upload; calls
// with M > 16 must not run concurrently on two streams); then one CTA per row sums them in order and finishes the row
// (p050::k_rows). Every value is the base's.
#pragma once

namespace p051 {

constexpr int kMaxRowsWs = 8192;  // the engine's row limit (kernels/linear kMaxRows)
float* g_ws = nullptr;

void init() {
  if (!g_ws) CUDA_CHECK(cudaMalloc(&g_ws, size_t(kChunks) * kMaxRowsWs * kOut * 4));
}

template <int MODE>
void run(const MhcW& m, const float* w, bf16* R, const bf16* out, int M, float* post, float* comb, bf16* inp,
         cudaStream_t s) {
  if (M > kMaxRowsWs) qlog::fail("mhc", "M %d > %d", M, kMaxRowsWs);
  p050::run<MODE>(m, w, R, out, M, post, comb, inp, s, g_ws);
}

}  // namespace p051
