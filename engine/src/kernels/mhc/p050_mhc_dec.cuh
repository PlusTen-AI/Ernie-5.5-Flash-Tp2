// p050_mhc_dec.cuh — P050 mhc_dec: mHC pre / post + pre / layer-0 pre for M ≤ 16 rows (decode, speculative verify).
// Included by mhc.cu inside its anonymous namespace (uses sigm, post1, unpack8, load_nw, collapse, kChunks …).
//
// Why: the base runs one CTA per row, which streams the whole fp32 mixer (1.57 MB) through one SM.
// Here the 64 chunks of the fixed order (mhc.h) are 64 CTAs: CTA c loads its 24 KB of fnq with one bulk copy (TMA,
// issued before griddepcontrol.wait — weights do not depend on the previous kernel; the launch is programmatic, so it
// overlaps the previous kernel's tail), computes x_q of its 64 dims h for every row (post: r from R, out, post, comb in
// the base's FMA order; R'_j = bf16(r_j) written back in place — chunks own disjoint dims, so no CTA reads another's
// output; x_q kept in shared memory in xsw order), and runs the chunk partials exactly as the base: lane n < 24 = mixer
// row n, lane 24 = Σx², sequential fmaf from 0 over the chunk's q (warp w: rows w, w + 9 of the CTA's block of 16).
// Partials go to MhcW::ws [64][16][25] (P051: the same kernels over blocks of 16 rows, ws = a shared [64][M][25]).
//   M = 1: the last CTA to arrive (one acq_rel atomic ticket after a barrier; partials read with ld.cg after it) finishes
//          the row; it triggers the dependent launch first, so the next kernel's prologue overlaps this tail.
//   M 2–16: a second, programmatic launch: one CTA per row does the same.
// Finishing a row = the base's arithmetic, scheduled for latency: one round trip brings the 64 × 25 partials (summed
// in order y = P0, y += Pc), the row's residual, the norm weights and the coefficients' constants; then a 9th warp
// computes post and comb (softmax, the 20 Sinkhorn passes with one element per lane, every row / column sum gathered by
// shuffles and added in the base's order, IEEE divisions) while warps 0–7 compute pre themselves and the collapse (the
// base's reduction on named barrier 2). sb and the norm weights are prefetched into L2 when the chunk CTAs start.
#pragma once

namespace p050 {

constexpr int kNT = 288;  // 8 collapse warps (the base's 256-thread layout) + the coefficient warp

struct Args {
  const float* w;  // fnq (or fnb)
  const float* sb;
  const bf16* nw;
  bf16* R;
  const bf16* out;
  float* post;
  float* comb;
  bf16* inp;
  float* ws;      // chunk partials [64][ms][25]
  unsigned* cnt;
  int M;
  int ms;         // ws rows (MhcW::ws: 16; P051's shared workspace: M)
};

__device__ __forceinline__ uint32_t su32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void griddep_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
__device__ __forceinline__ void griddep_launch() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }
__device__ __forceinline__ void mbar_init(uint64_t* b) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(su32(b)) : "memory");
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
__device__ __forceinline__ void tma_load(void* dst, const void* src, uint32_t bytes, uint64_t* b) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(su32(b)), "r"(bytes) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(su32(dst)),
               "l"(src), "r"(bytes), "r"(su32(b))
               : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t phase) {
  asm volatile(
      "{\n .reg .pred P1;\n WAIT:\n mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n @P1 bra DONE;\n bra WAIT;\n"
      " DONE:\n }\n" ::"r"(su32(b)),
      "r"(phase)
      : "memory");
}
__device__ __forceinline__ unsigned atom_add_acq_rel(unsigned* p, unsigned v) {
  unsigned r;
  asm volatile("atom.add.acq_rel.gpu.global.u32 %0, [%1], %2;\n" : "=r"(r) : "l"(p), "r"(v) : "memory");
  return r;
}

// Row m: y (WS: the 64 chunk partials summed in order; else P051's y in mix[m][0..24]), post / comb (warp 8), pre and
// the collapse (warps 0–7). All kNT threads; part: shared [64 × 25] (WS).
template <int MODE, bool WS = true>
__device__ __forceinline__ void finish(const Args& a, int m, float* vmix, float* red, float* part,
                                       const float* mix = nullptr) {
  const int tid = threadIdx.x;
  const bool cw = tid < 256;
  float x[4][16], w[16], s24 = 0.f, sbi[4] = {0.f, 0.f, 0.f, 0.f};  // collapse warps
  float t25 = 0.f, t26 = 0.f, sbp = 0.f, sbc = 0.f;                  // warp 8 (lane L: element L of comb)
  const bf16* r = a.R + (size_t)m * kHcDim;
  if (cw) {
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
      for (int c8 = 0; c8 < 2; c8++)
        unpack8(__ldcg(reinterpret_cast<const uint4*>(r + j * kHidden + tid * 16 + c8 * 8)), &x[j][c8 * 8]);
    load_nw(a.nw, w);
    s24 = a.sb[24];
#pragma unroll
    for (int i = 0; i < 4; i++) sbi[i] = a.sb[i];
  } else {
    t25 = a.sb[25], t26 = a.sb[26], sbp = a.sb[4 + (tid & 3)], sbc = a.sb[8 + (tid & 15)];
  }
  if constexpr (WS) {
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
  } else {
    if (tid < kOut) vmix[tid] = mix[(size_t)m * 72 + tid];
  }
  __syncthreads();
  const float f = __fdiv_rn(1.0f, sqrtf(__fdiv_rn(vmix[24], float(MODE == 2 ? kHidden : kHcDim))) + kEps);
  if (!cw) {  // warp 8: post (lanes 0–3) and comb (lane L = element L; lanes 16–31 duplicate 0–15)
    const unsigned full = 0xffffffffu;
    const int lane = tid & 31, L = lane & 15, i4 = L & ~3, jc = L & 3;
    const float mp = vmix[4 + (lane & 3)] * f;
    const float po = sigm(mp * t25 + sbp) * 2.0f;
    const float mc = vmix[8 + L] * f;
    const float l = mc * t26 + sbc;
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
      pre[i] = sigm(mi * s24 + sbi[i]) + kEps;
    }
    collapse<true>(x, pre, w, red, a.inp + (size_t)m * kHidden);
  }
}

// CTA c = chunk c (MODE 2: 64 h of e; else 256 q = 64 h × 4 streams); LAST: M = 1, the last CTA finishes the row
template <int MODE, bool LAST>
__global__ void __launch_bounds__(kNT) k_chunk(Args a) {
  constexpr int K = MODE == 2 ? kCh : kCq;
  __shared__ __align__(128) float W[kCq * kHcMix];  // the chunk's fn (24 KB; MODE 2: 6 KB used)
  __shared__ __align__(16) float xs[kDecM][kCq];    // x_q of the chunk per row
  __shared__ __align__(8) uint64_t bar;
  __shared__ float vmix[kOut], red[8];
  __shared__ unsigned ticket;
  const int c = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int m0 = blockIdx.y * kDecM, M = min(kDecM, a.M - m0);  // this CTA's row block
  if (tid == 0) {
    mbar_init(&bar);
    tma_load(W, a.w + (size_t)c * K * kHcMix, K * kHcMix * 4, &bar);
  } else if (tid == 32) {  // what finishing the row reads first: the norm weights (64 × 128 B) and sb
    asm volatile("prefetch.global.L2 [%0];\n" ::"l"(a.nw + c * 64));
    if (c == 0) asm volatile("prefetch.global.L2 [%0];\n" ::"l"(a.sb));
    if (c == 1) asm volatile("prefetch.global.L2 [%0];\n" ::"l"(a.sb + 26));
  }
  griddep_wait();  // R, out, post, comb come from the previous kernels
  for (int u = tid; u < M * 8; u += kNT) {  // unit: row m0 + m, 8 dims h0 … h0+7 of the chunk
    const int m = u >> 3, hl = (u & 7) * 8, h0 = c * kCh + hl, mg = m0 + m;
    bf16* r = a.R + (size_t)mg * kHcDim;
    if constexpr (MODE == 1) {
      float cm[16], p[4], xi[4][8], o[8], v[32];  // v: x_q for q = 4·hl … 4·hl+31
#pragma unroll
      for (int i = 0; i < 16; i++) cm[i] = a.comb[(size_t)mg * 16 + i];
#pragma unroll
      for (int i = 0; i < 4; i++) p[i] = a.post[(size_t)mg * 4 + i];
#pragma unroll
      for (int j = 0; j < 4; j++) unpack8(*reinterpret_cast<const uint4*>(r + j * kHidden + h0), xi[j]);
      unpack8(*reinterpret_cast<const uint4*>(a.out + (size_t)mg * kHidden + h0), o);
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
#pragma unroll
      for (int k = 0; k < 8; k++)
        *reinterpret_cast<float4*>(&xs[m][xsw(4 * hl + 4 * k)]) = make_float4(v[4 * k], v[4 * k + 1], v[4 * k + 2], v[4 * k + 3]);
    } else if constexpr (MODE == 0) {
      float xi[4][8];
#pragma unroll
      for (int j = 0; j < 4; j++) unpack8(*reinterpret_cast<const uint4*>(r + j * kHidden + h0), xi[j]);
#pragma unroll
      for (int k = 0; k < 8; k++)
        *reinterpret_cast<float4*>(&xs[m][xsw(4 * hl + 4 * k)]) = make_float4(xi[0][k], xi[1][k], xi[2][k], xi[3][k]);
    } else {
      float xi[8];
      unpack8(*reinterpret_cast<const uint4*>(r + h0), xi);
#pragma unroll
      for (int k = 0; k < 8; k++) xs[m][hl + k] = xi[k];
    }
  }
  __syncthreads();
  mbar_wait(&bar, 0);
  if (lane < kOut) {  // the chunk partials (the base's chain), 32 terms per register block
    const int n = lane < kHcMix ? lane : 0;
    const bool ssl = lane >= kHcMix;
    for (int m = warp; m < M; m += kNT / 32) {
      float acc = 0.f;
#pragma unroll 1
      for (int q0 = 0; q0 < K; q0 += 32) {
        float xv[32], wv[32];
#pragma unroll
        for (int k = 0; k < 32; k += 4) {
          const float4 v = *reinterpret_cast<const float4*>(&xs[m][MODE == 2 ? q0 + k : xsw(q0 + k)]);
          xv[k] = v.x, xv[k + 1] = v.y, xv[k + 2] = v.z, xv[k + 3] = v.w;
        }
#pragma unroll
        for (int k = 0; k < 32; k++) wv[k] = W[(q0 + k) * kHcMix + n];
#pragma unroll
        for (int k = 0; k < 32; k++) acc = __fmaf_rn(ssl ? xv[k] : wv[k], xv[k], acc);
      }
      a.ws[((size_t)c * a.ms + m0 + m) * kOut + lane] = acc;
    }
  }
  if constexpr (LAST) {
    __syncthreads();
    if (tid == 0) ticket = atom_add_acq_rel(a.cnt, 1u);
    __syncthreads();
    if (ticket != kChunks - 1) return;
    griddep_launch();
    finish<MODE>(a, 0, vmix, red, W);  // W is free after the chain
    if (tid == 0) *a.cnt = 0u;
  } else {
    griddep_launch();
  }
}

// EARLY (P107 mhc_rows_pdl): the dependent launch is allowed once the rows' inputs are visible, so the next kernel's
// prologue (the next mixer's fn bulk copies, weights) runs under the row finish, as on P050's M = 1 path
template <int MODE, bool EARLY = false>
__global__ void __launch_bounds__(kNT) k_rows(Args a) {
  __shared__ float vmix[kOut], red[8], part[kChunks * kOut];
  griddep_wait();
  if constexpr (EARLY) griddep_launch();
  finish<MODE>(a, blockIdx.x, vmix, red, part);
}

template <class... P, class... A>
void launch_pdl(void (*k)(P...), dim3 grid, cudaStream_t s, A... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = dim3(kNT);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = s;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, k, args...));
}

// M ≤ 16: MhcW::ws (LAST at M = 1); larger M (P051): ws = a shared workspace [64][M][25], rows in blocks of 16
template <int MODE>
void run(const MhcW& m, const float* w, bf16* R, const bf16* out, int M, float* post, float* comb, bf16* inp,
         cudaStream_t s, float* ws = nullptr) {
  Args a{w, m.sb, m.normw, R, out, post, comb, inp, ws ? ws : m.ws, m.cnt, M, ws ? M : kDecM};
  if (M == 1) {
    launch_pdl(k_chunk<MODE, true>, dim3(kChunks), s, a);
  } else {
    launch_pdl(k_chunk<MODE, false>, dim3(kChunks, cdiv(M, kDecM)), s, a);
    if (!ws && patch::on(patch::mhc_rows_pdl)) launch_pdl(k_rows<MODE, true>, dim3(M), s, a);  // P107 (M ≤ 16)
    else launch_pdl(k_rows<MODE>, dim3(M), s, a);
  }
}

}  // namespace p050
