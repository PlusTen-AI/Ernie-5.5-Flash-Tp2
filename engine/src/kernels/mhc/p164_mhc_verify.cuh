// p164_mhc_verify.cuh — P164 mhc_verify: mHC pre / post + pre / layer-0 pre for M ≤ 32 rows (decode, DSpark verify).
// Included by mhc.cu inside its anonymous namespace after p050_mhc_dec.cuh / p051_mhc_big.cuh (uses p050's Args,
// helpers, launch_pdl, and mhc.cu's post1 / collapse / sigm …).
//
// Why (measured on P050 / P051 with globaltimer stamps, behind a 10 µs programmatic predecessor — fn copied under it, as
// under the comm kernel in the engine): the chunk partials took 1.2 µs at M = 1 and 3.4 / 5.2 / 10.6 µs at M = 5 / 8 /
// 16 — every warp re-read the chunk's 24 KB of fn from shared memory for each of its rows (one LDS a FMA: shared-memory
// issue, not FMAs, bound the chains); at M = 1 P050's chunk kernel (the row finish inlined: 128 registers) fit one CTA an
// SM, its 64 CTAs in two waves (+2.4 µs); at M = 17–32 P051's 128 CTAs (2 row blocks a chunk) did not fit one wave
// either; and the row finish requested the 32 KB residual row before the 64 partials the coefficients wait on (1 µs).
// Here:
//   · one CTA a chunk for every M ≤ 32 (rows in passes of 16 through a 16-row x buffer: 41 KB of shared memory,
//     __launch_bounds__(288, 2): 2 CTAs an SM, the 64 CTAs in one wave);
//   · the chunk partials register-blocked: warp w takes R consecutive rows (lanes 0–23 = mixer row n), loads each fn
//     value once for its R rows into R accumulators, the next 4 q's loads in flight under the current FMAs; the Σx²
//     chains (n = 24) on warp 8, a lane a row; the x rows padded to 260 floats (those row-parallel loads conflict-free;
//     P050's xsw swizzle inside a row);
//   · before griddepcontrol.wait, besides fn (TMA) and P050's norm / sb prefetches, the rows' residual lines of the
//     chunk and their post / comb are prefetched into L2 (prefetches only: no value is read before the wait);
//   · the rows finished by a second, programmatic launch at every M (one CTA a row; its dependent launched right after
//     its wait, as P107): P050's finish with the norm weights and sb loaded before the wait, only the 64 partials before
//     the coefficient chain (sum, softmax, Sinkhorn passes: the critical path), the residual row loaded by the collapse
//     warps after it. Partials: MhcW::ws (M ≤ 16) or P051's shared workspace (M > 16: no concurrent calls on 2 streams).
// Bitwise: every chunk partial is the base's chain (sequential fmaf from 0 over the chunk's q in order), the partials
// are summed in order and the rows finished with P050's arithmetic (the same expressions, sigm, collapse).
#pragma once

namespace p164 {

constexpr int kMaxM = 32;           // rows a call (one CTA a chunk)
constexpr int kPass = 16;           // rows a pass through the x buffer
constexpr int kXs = kCq + 4;        // x buffer row stride (floats)
int g_rows = 0;                     // tests: force R (rows a warp chain: 1, 2, 4, 8); 0 = from M

// The chunk partials of rows p0 … p0 + mp − 1 (x in xs[0 … mp−1]). Warps 0 … ⌈mp/R⌉−1: lane n < 24 = mixer row n,
// R consecutive rows a warp — each fn value loaded once for its R rows (R accumulators), the next 4 q's loads in
// flight under this block's FMAs. Warp 8: the Σx² chains (lane = row). ws[(c · ms + row) · 25 + n].
template <int MODE, int R>
__device__ __forceinline__ void chains(const float* __restrict__ W, const float (*xs)[kXs], float* ws, int c, int ms,
                                       int p0, int mp) {
  constexpr int K = MODE == 2 ? kCh : kCq;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  auto xo = [](int q) { return MODE == 2 ? q : xsw(q); };
  if (warp < 8) {
    const int r0 = warp * R;
    if (r0 >= mp || lane >= kHcMix) return;
    const float* xr[R];
#pragma unroll
    for (int r = 0; r < R; r++) xr[r] = xs[min(r0 + r, mp - 1)];
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; r++) acc[r] = 0.f;
    float w[4];
    float4 x[R];
#pragma unroll
    for (int i = 0; i < 4; i++) w[i] = W[i * kHcMix + lane];
#pragma unroll
    for (int r = 0; r < R; r++) x[r] = *reinterpret_cast<const float4*>(xr[r] + xo(0));
#pragma unroll 4
    for (int q = 0; q < K; q += 4) {
      const int qn = q + 4 < K ? q + 4 : q;  // the next block's loads (the last block re-reads its own)
      float wn[4];
      float4 xn[R];
#pragma unroll
      for (int i = 0; i < 4; i++) wn[i] = W[(qn + i) * kHcMix + lane];
#pragma unroll
      for (int r = 0; r < R; r++) xn[r] = *reinterpret_cast<const float4*>(xr[r] + xo(qn));
#pragma unroll
      for (int r = 0; r < R; r++) {
        acc[r] = __fmaf_rn(w[0], x[r].x, acc[r]);
        acc[r] = __fmaf_rn(w[1], x[r].y, acc[r]);
        acc[r] = __fmaf_rn(w[2], x[r].z, acc[r]);
        acc[r] = __fmaf_rn(w[3], x[r].w, acc[r]);
      }
#pragma unroll
      for (int i = 0; i < 4; i++) w[i] = wn[i];
#pragma unroll
      for (int r = 0; r < R; r++) x[r] = xn[r];
    }
#pragma unroll
    for (int r = 0; r < R; r++)
      if (r0 + r < mp) ws[((size_t)c * ms + p0 + r0 + r) * kOut + lane] = acc[r];
  } else if (lane < mp) {  // warp 8: Σx² of row lane
    const float* xr = xs[lane];
    float acc = 0.f;
    float4 x = *reinterpret_cast<const float4*>(xr + xo(0));
#pragma unroll 4
    for (int q = 0; q < K; q += 4) {
      const float4 xn = *reinterpret_cast<const float4*>(xr + xo(q + 4 < K ? q + 4 : q));
      acc = __fmaf_rn(x.x, x.x, acc);
      acc = __fmaf_rn(x.y, x.y, acc);
      acc = __fmaf_rn(x.z, x.z, acc);
      acc = __fmaf_rn(x.w, x.w, acc);
      x = xn;
    }
    ws[((size_t)c * ms + p0 + lane) * kOut + kHcMix] = acc;
  }
}

// The row finish: P050's finish (the same arithmetic, the same functions) with the loads reordered — only the 64
// partials before the coefficients (their chain — the sum, softmax and Sinkhorn passes — is the critical path), the
// residual row loaded by the collapse warps after it (the collapse finishes well before the Sinkhorn passes); the norm
// weights and sb passed in (the row kernel loads them before griddepcontrol.wait: no kernel writes them).
struct Wts {
  float w[16], s24, sbi[4];     // collapse warps
  float t25, t26, sbp, sbc;     // warp 8
};
__device__ __forceinline__ void load_wts(const p050::Args& a, Wts& t) {
  const int tid = threadIdx.x;
  if (tid < 256) {
    load_nw(a.nw, t.w);
    t.s24 = a.sb[24];
#pragma unroll
    for (int i = 0; i < 4; i++) t.sbi[i] = a.sb[i];
  } else {
    t.t25 = a.sb[25], t.t26 = a.sb[26], t.sbp = a.sb[4 + (tid & 3)], t.sbc = a.sb[8 + (tid & 15)];
  }
}

template <int MODE>
__device__ __forceinline__ void finish(const p050::Args& a, int m, float* vmix, float* red, float* part, const Wts& t) {
  using namespace p050;
  const int tid = threadIdx.x;
  const bool cw = tid < 256;
  constexpr int PER = (kChunks * kOut + kNT - 1) / kNT;
  float pv[PER];
#pragma unroll
  for (int k = 0; k < PER; k++) {
    const int i = tid + k * kNT;
    pv[k] = i < kChunks * kOut ? __ldcg(a.ws + ((size_t)(i / kOut) * a.ms + m) * kOut + i % kOut) : 0.f;
  }
#pragma unroll
  for (int k = 0; k < PER; k++)
    if (tid + k * kNT < kChunks * kOut) part[tid + k * kNT] = pv[k];
  __syncthreads();
  if (tid < kOut) {
    float y = part[tid];
    for (int c = 1; c < kChunks; c++) y = __fadd_rn(y, part[c * kOut + tid]);
    vmix[tid] = y;
  }
  __syncthreads();
  const float f = __fdiv_rn(1.0f, sqrtf(__fdiv_rn(vmix[24], float(MODE == 2 ? kHidden : kHcDim))) + kEps);
  if (!cw) {  // warp 8: post (lanes 0–3) and comb (lane L = element L; lanes 16–31 duplicate 0–15)
    const unsigned full = 0xffffffffu;
    const int lane = tid & 31, L = lane & 15, i4 = L & ~3, jc = L & 3;
    const float mp = vmix[4 + (lane & 3)] * f;
    const float po = sigm(mp * t.t25 + t.sbp) * 2.0f;
    const float mc = vmix[8 + L] * f;
    const float l = mc * t.t26 + t.sbc;
    float mx = -INFINITY;
#pragma unroll
    for (int k = 0; k < 4; k++) mx = fmaxf(mx, __shfl_sync(full, l, i4 + k));
    const float e = expf(l - mx);
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < 4; k++) s += __shfl_sync(full, e, i4 + k);
    float cm = __fdiv_rn(e, s) + kEps;
    auto colnorm = [&] {
      const float c0 = __shfl_sync(full, cm, jc), c1 = __shfl_sync(full, cm, 4 + jc);
      const float c2 = __shfl_sync(full, cm, 8 + jc), c3 = __shfl_sync(full, cm, 12 + jc);
      cm = __fdiv_rn(cm, ((c0 + c1) + c2) + c3 + kEps);
    };
    auto rownorm = [&] {
      const float c0 = __shfl_sync(full, cm, i4), c1 = __shfl_sync(full, cm, i4 + 1);
      const float c2 = __shfl_sync(full, cm, i4 + 2), c3 = __shfl_sync(full, cm, i4 + 3);
      cm = __fdiv_rn(cm, ((c0 + c1) + c2) + c3 + kEps);
    };
    colnorm();
    for (int it = 0; it < 19; it++) rownorm(), colnorm();
    if (lane < 4) a.post[(size_t)m * 4 + lane] = po;
    if (lane < 16) a.comb[(size_t)m * 16 + L] = cm;
  } else {
    float pre[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const float mi = vmix[i] * f;
      pre[i] = sigm(mi * t.s24 + t.sbi[i]) + kEps;
    }
    float x[4][16];
    const bf16* r = a.R + (size_t)m * kHcDim;
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
      for (int c8 = 0; c8 < 2; c8++)
        unpack8(__ldcg(reinterpret_cast<const uint4*>(r + j * kHidden + tid * 16 + c8 * 8)), &x[j][c8 * 8]);
    collapse<true>(x, pre, t.w, red, a.inp + (size_t)m * kHidden);
  }
}

// one CTA a row, launched as a programmatic dependent of the chunk kernel (the norm weights and sb loaded under it); its
// own dependent launched right after the wait (as P107)
template <int MODE>
__global__ void __launch_bounds__(p050::kNT) k_rows(p050::Args a) {
  using namespace p050;
  __shared__ float vmix[kOut], red[8], part[kChunks * kOut];
  Wts t;
  load_wts(a, t);
  p050::griddep_wait();
  p050::griddep_launch();
  finish<MODE>(a, blockIdx.x, vmix, red, part, t);
}

// CTA c = chunk c, all M rows
template <int MODE, int R>
__global__ void __launch_bounds__(p050::kNT, 2) k_chunk(p050::Args a) {
  using namespace p050;
  constexpr int K = MODE == 2 ? kCh : kCq;
  __shared__ __align__(128) float W[kCq * kHcMix];  // the chunk's fn (24 KB; MODE 2: 6 KB used)
  __shared__ __align__(16) float xs[kPass][kXs];    // x_q of a pass's rows
  __shared__ __align__(8) uint64_t bar;
  const int c = blockIdx.x, tid = threadIdx.x, M = a.M;
  if (tid == 0) {
    mbar_init(&bar);
    tma_load(W, a.w + (size_t)c * K * kHcMix, K * kHcMix * 4, &bar);
  } else if (tid == 32) {  // what finishing a row reads first: the norm weights (64 × 128 B) and sb
    asm volatile("prefetch.global.L2 [%0];\n" ::"l"(a.nw + c * 64));
    if (c == 0) asm volatile("prefetch.global.L2 [%0];\n" ::"l"(a.sb));
    if (c == 1) asm volatile("prefetch.global.L2 [%0];\n" ::"l"(a.sb + 26));
  } else if (tid >= 64 && tid < 64 + 4 * M) {  // this chunk's residual lines (128 B a row and stream) into L2
    const int m = (tid - 64) >> 2, j = (tid - 64) & 3;
    if (MODE != 2 || j == 0)
      asm volatile("prefetch.global.L2 [%0];\n" ::"l"(a.R + (size_t)m * kHcDim + j * kHidden + c * kCh));
  } else if (MODE == 1 && tid >= 192 && tid < 192 + (M * 16 + 31) / 32) {  // comb [M][16] into L2
    asm volatile("prefetch.global.L2 [%0];\n" ::"l"(a.comb + (tid - 192) * 32));
  } else if (MODE == 1 && tid >= 224 && tid < 224 + (M * 4 + 31) / 32) {  // post [M][4]
    asm volatile("prefetch.global.L2 [%0];\n" ::"l"(a.post + (tid - 224) * 32));
  }
  griddep_wait();  // R, out, post, comb come from the previous kernels
  // x: unit u = tid < 8M — row u / 8, dims h0 … h0+7 of the chunk (P050's arithmetic), kept in registers
  float v[32];
  const int um = tid >> 3, hl = (tid & 7) * 8, h0 = c * kCh + hl;
  if (tid < 8 * M) {
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
  }
  const int ms = M > kDecM ? M : kDecM;
  for (int p0 = 0; p0 < M; p0 += kPass) {
    const int mp = min(kPass, M - p0);
    if (p0) __syncthreads();  // the previous pass's chains have read xs
    if (tid < 8 * M && um >= p0 && um < p0 + mp) {
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
    chains<MODE, R>(W, xs, a.ws, c, ms, p0, mp);
  }
  griddep_launch();  // the row kernel (its prologue: the norm weights) under the chains' tail
}

// rows a warp chain from M (measured, see test_mhc.py; g_rows forces R): the 8 warps cover ⌈rows a pass / R⌉
inline int rows_of(int M) {
  const int mp = min(M, kPass);
  if (g_rows) return max(g_rows, (mp + 7) / 8);
  return M == 1 ? 1 : M <= 8 ? 2 : 4;
}

template <int MODE>
void run(const MhcW& m, const float* w, bf16* R, const bf16* out, int M, float* post, float* comb, bf16* inp,
         cudaStream_t s) {
  float* ws = M > kDecM ? p051::g_ws : m.ws;
  p050::Args a{w, m.sb, m.normw, R, out, post, comb, inp, ws, m.cnt, M, M > kDecM ? M : kDecM};
  switch (rows_of(M)) {
    case 1: p050::launch_pdl(k_chunk<MODE, 1>, dim3(kChunks), s, a); break;
    case 2: p050::launch_pdl(k_chunk<MODE, 2>, dim3(kChunks), s, a); break;
    case 4: p050::launch_pdl(k_chunk<MODE, 4>, dim3(kChunks), s, a); break;
    default: p050::launch_pdl(k_chunk<MODE, 8>, dim3(kChunks), s, a); break;
  }
  p050::launch_pdl(k_rows<MODE>, dim3(M), s, a);
}

}  // namespace p164
