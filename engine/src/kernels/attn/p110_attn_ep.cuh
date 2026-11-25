// p110_attn_ep.cuh — P110 dsa_attn_ep: prefill dsa_attn = P092 with each score warp publishing a tile's P and f to the
// P·V warps right after storing them, before its tile sum. Included by dsa.cu inside its anonymous namespace after
// p092_attn_pf.cuh (P092's kernel, instantiated with EP = true), selected by patch::on(dsa_attn_ep).
//
// Bitwise contract — P092's (= the base's): nothing computed changes. The tile sum (Σ p in key order from 0), the
// running sum l = l·f + sum handed between the two score groups, the chunk partials and the merge are the same
// operations on the same values; only the barrier arrival that releases the P·V warps for tile j moves ahead of the
// sum (the P·V warps read P and f only). Measured on GB10 (clock64 timeline of one CTA, 16K context): a tile's
// dependency loop through the two K buffers — QK(j) → softmax(j) → P·V(j) frees k[j & 1] → K(j+2) stored → QK(j+2) —
// held the 32-term sum and the l hand-off (~1,000 clk of ~7,800 a two-tile period).
#pragma once

namespace p110 {

inline bool wanted(int M, int max_keys, int nsm) { return p092::wanted(M, max_keys, nsm); }
inline void run(const bf16* q576, const int* list, const int* cnt, const RowMeta* meta, DsaCache c, const float* sink,
                int M, int max_keys, bf16* olat, float* ws, size_t ws_bytes, cudaStream_t s) {
  p092::run<true>(q576, list, cnt, meta, c, sink, M, max_keys, olat, ws, ws_bytes, s);
}

}  // namespace p110
