// p192_mhc_verify64.cuh — P192 mhc_verify64: P164 mhc_verify's one-CTA-a-chunk mHC pre / post + pre / layer-0 pre for
// 33 ≤ M ≤ 64 rows (DSpark verify / draft steps of 5–8 sequences), where P051 runs 64 chunks × ⌈M/16⌉ row blocks and
// re-reads each chunk's 24 KB of fn per block. Included by mhc.cu after p164_mhc_verify.cuh (its chains, rows_of,
// k_rows, kPass, kXs; P050's Args and helpers).
//
// P164 holds every row's x unit (8 dims of the chunk) in registers across its passes: 8·M units, one a thread, ≤ 32
// rows in its 288 threads. Here a pass's 16 rows are made right before the pass (threads 0 … 8·mp − 1 make unit
// (row p0 + tid / 8, dims 8·(tid % 8) …)), so any M fits; the residual-line / post / comb L2 prefetches loop over the
// rows. The units are made by the same expressions (MODE 1: the post written back to R exactly once a unit), the chunk
// partials by P164's chains, the rows finished by P164's k_rows (P051's shared workspace: M > 16): the same bits as
// P164 / P051 / the base (test_mhc.py --v64 compares them over 33 … 64 rows).
#pragma once

namespace p192 {

constexpr int kMaxM = 64;

template <int MODE, int R>
__global__ void __launch_bounds__(p050::kNT, 2) k_chunk64(p050::Args a) {
  using namespace p050;
  constexpr int K = MODE == 2 ? kCh : kCq;
  __shared__ __align__(128) float W[kCq * kHcMix];        // the chunk's fn (24 KB; MODE 2: 6 KB used)
  __shared__ __align__(16) float xs[p164::kPass][p164::kXs];  // x_q of a pass's rows
  __shared__ __align__(8) uint64_t bar;
  const int c = blockIdx.x, tid = threadIdx.x, M = a.M;
  if (tid == 0) {
    mbar_init(&bar);
    tma_load(W, a.w + (size_t)c * K * kHcMix, K * kHcMix * 4, &bar);
  } else if (tid == 32) {  // what finishing a row reads first: the norm weights (64 × 128 B) and sb
    asm volatile("prefetch.global.L2 [%0];\n" ::"l"(a.nw + c * 64));
    if (c == 0) asm volatile("prefetch.global.L2 [%0];\n" ::"l"(a.sb));
    if (c == 1) asm volatile("prefetch.global.L2 [%0];\n" ::"l"(a.sb + 26));
  } else if (MODE == 1 && tid > 32 && tid < 64) {  // comb [M][16] and post [M][4] into L2
    const int nc = (M * 16 + 31) / 32, np = (M * 4 + 31) / 32;
    for (int i = tid - 33; i < nc + np; i += 31)
      asm volatile("prefetch.global.L2 [%0];\n" ::"l"(i < nc ? a.comb + i * 32 : a.post + (i - nc) * 32));
  } else if (tid >= 64) {  // this chunk's residual lines (128 B a row and stream) into L2
    for (int i = tid - 64; i < 4 * M; i += kNT - 64) {
      const int m = i >> 2, j = i & 3;
      if (MODE != 2 || j == 0)
        asm volatile("prefetch.global.L2 [%0];\n" ::"l"(a.R + (size_t)m * kHcDim + j * kHidden + c * kCh));
    }
  }
  griddep_wait();  // R, out, post, comb come from the previous kernels
  const int ms = M > kDecM ? M : kDecM, hl = (tid & 7) * 8, h0 = c * kCh + hl;
  for (int p0 = 0; p0 < M; p0 += p164::kPass) {
    const int mp = min(p164::kPass, M - p0);
    if (p0) __syncthreads();  // the previous pass's chains have read xs
    if (tid < 8 * mp) {       // unit: row p0 + tid / 8, dims h0 … h0+7 of the chunk (P164's / P050's arithmetic)
      const int um = p0 + (tid >> 3);
      float v[32];
      bf16* r = a.R + (size_t)um * kHcDim;
      if constexpr (MODE == 1) {
        float cm[16], p[4], xi[4][8], o[8];
#pragma unroll
        for (int i = 0; i < 16; i++) cm[i] = a.comb[(size_t)um * 16 + i];
#pragma unroll
        for (int i = 0; i < 4; i++) p[i] = a.post[(size_t)um * 4 + i];
#pragma unroll
        for (int j = 0; j < 4; j++) unpack8(*reinterpret_cast<const uint4*>(r + j * kHidden + h0), xi[j]);
        unpack8(*reinterpret_cast<const uint4*>(a.out + (size_t)um * kHidden + h0), o);
#pragma unroll
        for (int j = 0; j < 4; j++) {
          uint32_t wv[4];
#pragma unroll
          for (int k = 0; k < 8; k += 2) {
            v[4 * k + j] = post1(cm, p, j, xi[0][k], xi[1][k], xi[2][k], xi[3][k], o[k]);
            v[4 * (k + 1) + j] = post1(cm, p, j, xi[0][k + 1], xi[1][k + 1], xi[2][k + 1], xi[3][k + 1], o[k + 1]);
            wv[k / 2] = pack_bf16x2(v[4 * k + j], v[4 * (k + 1) + j]);
          }
          *reinterpret_cast<uint4*>(r + j * kHidden + h0) = make_uint4(wv[0], wv[1], wv[2], wv[3]);
        }
      } else if constexpr (MODE == 0) {
        float xi[4][8];
#pragma unroll
        for (int j = 0; j < 4; j++) unpack8(*reinterpret_cast<const uint4*>(r + j * kHidden + h0), xi[j]);
#pragma unroll
        for (int k = 0; k < 8; k++) v[4 * k] = xi[0][k], v[4 * k + 1] = xi[1][k], v[4 * k + 2] = xi[2][k], v[4 * k + 3] = xi[3][k];
      } else {
        unpack8(*reinterpret_cast<const uint4*>(r + h0), v);
      }
      float* xr = xs[um - p0];
      if constexpr (MODE == 2) {
#pragma unroll
        for (int k = 0; k < 8; k += 4) *reinterpret_cast<float4*>(xr + hl + k) = make_float4(v[k], v[k + 1], v[k + 2], v[k + 3]);
      } else {
#pragma unroll
        for (int k = 0; k < 8; k++)
          *reinterpret_cast<float4*>(xr + xsw(4 * hl + 4 * k)) = make_float4(v[4 * k], v[4 * k + 1], v[4 * k + 2], v[4 * k + 3]);
      }
    }
    __syncthreads();
    if (p0 == 0) mbar_wait(&bar, 0);
    p164::chains<MODE, R>(W, xs, a.ws, c, ms, p0, mp);
  }
  griddep_launch();  // the row kernel (its prologue: the norm weights) under the chains' tail
}

template <int MODE>
void run(const MhcW& m, const float* w, bf16* R, const bf16* out, int M, float* post, float* comb, bf16* inp,
         cudaStream_t s) {
  if (M <= kDecM || M > kMaxM) qlog::fail("mhc", "mhc_verify64: M %d", M);
  p050::Args a{w, m.sb, m.normw, R, out, post, comb, inp, p051::g_ws, m.cnt, M, M};
  switch (p164::rows_of(M)) {
    case 1: p050::launch_pdl(k_chunk64<MODE, 1>, dim3(kChunks), s, a); break;
    case 2: p050::launch_pdl(k_chunk64<MODE, 2>, dim3(kChunks), s, a); break;
    case 4: p050::launch_pdl(k_chunk64<MODE, 4>, dim3(kChunks), s, a); break;
    default: p050::launch_pdl(k_chunk64<MODE, 8>, dim3(kChunks), s, a); break;
  }
  p050::launch_pdl(p164::k_rows<MODE>, dim3(M), s, a);
}

}  // namespace p192
