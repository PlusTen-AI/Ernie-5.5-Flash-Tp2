// p093_bmm_pf.cuh — P093 dsa_bmm_pf: the per-head W_UK / W_UV GEMMs of dsa_bmm at prefill sizes (M > 16), selected
// by patch::on(dsa_bmm_pf). Included by dsa.cu inside its anonymous namespace (runs its k_bmm template).
//
// Bitwise contract — the base kernel itself (dsa.cu k_bmm<MT>): every output element is the same m16n8k16 chain over
// the k16 steps in ascending order from 0, bf16 out; only the row-tile height differs. The base uses 64-row tiles for
// both; measured on GB10 (memory-bound shapes): W_UK (K = 192, N = 512) is faster with 128-row tiles (each W tile
// read half as often), W_UV (K = 512, N = 256) with 32-row tiles (twice the CTAs in flight).
#pragma once

namespace p093 {

// false: not handled (decode sizes, other shapes) — the caller runs the base
inline bool run(const bf16* x, long xm, long xh, const bf16* w, int H, int N, int K, int M, bf16* y, long ym, long yh,
                cudaStream_t s) {
  if (M <= 16) return false;
  if (K == kMlaNope) k_bmm<128><<<dim3(H, N / 64, cdiv(M, 128)), 128, 0, s>>>(x, xm, xh, w, N, K, M, y, ym, yh);
  else if (K == kKvLora) k_bmm<32><<<dim3(H, N / 64, cdiv(M, 32)), 128, 0, s>>>(x, xm, xh, w, N, K, M, y, ym, yh);
  else return false;
  CUDA_CHECK(cudaGetLastError());
  return true;
}

}  // namespace p093
