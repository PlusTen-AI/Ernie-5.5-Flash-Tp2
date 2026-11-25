// p106_mhc_prefill.cuh — P106 mhc_prefill: mHC post + pre at prefill sizes (M ≥ 256) in one launch.
// Included by mhc.cu inside its anonymous namespace after p072_mhc_fuse.cuh (uses p050 / p051 / p072 helpers).
//
// P072's grid (64 chunks × blocks of 16 rows; the last chunk CTA of a block finishes its rows; P051's workspace for the
// chunk partials), with what costs P072 time redone (P072 at M = 512 measured with parts switched off: 301 µs, the
// chunk partials ~90 of it, the finish ~50, the rest the stream):
//  · A CTA takes 2 row blocks of its chunk: fn is brought once (one 24 KB bulk copy before griddepcontrol.wait), and the
//    second block's inputs (R's 4 streams, out, comb, post) are in flight while the first block's partials run.
//    Warps 0–3: x_q and R' (P072's code), then the next block's loads; warps 4–7: the partials; tid 128 (no loads in
//    flight) takes the block's ticket (bar.sync, one thread's fence and acq_rel atomic, as P050 / CUTLASS).
//  · Partials: a lane keeps 4 rows' chains (warp 4 + k → rows 4k … 4k+3; lane n < 24 = mixer row n, lane 24 = Σx²):
//    one fn load serves 4 rows and the 4 independent chains overlap. Each chain is P072's: acc = 0, then
//    acc = fma(fn[q][n], x_q, acc) over the chunk's q in order (Σx²: fma(x_q, x_q, acc)).
//  · A block the CTA arrived last at is finished after the CTA's own blocks (finishing it in between would make the
//    CTA late, and so last, for its next block too). The finish: every (row, n) sum has its 64 loads in flight (y = P_0,
//    y += P_c in chunk order); post / comb / pre are P072's code; the norm weights go to shared memory once; the
//    collapse runs 4 rows at a time: 4 groups of 64 threads, each thread doing the work of 4 of the base's 256 threads
//    (base thread e = 32·(2k + rw) + lane, k = 0..3, rw = the group's warp) — the same 16 dims a thread, the same
//    sequential Σol² over them, the same warp butterflies (a base warp's lanes are this warp's lanes), the same final
//    butterfly over the 8 warp sums (lanes < 8, zeros above), the same output expression.
// Every value is the base's (same expressions, same orders) → bit-identical to P072 / P051 / the base.
#pragma once

namespace p106 {

constexpr int kRB = kDecM;  // rows per block (16)
constexpr int kNT = 256;    // 8 warps; two CTAs an SM (42 KB of shared memory each)
constexpr int kNB = 2;      // row blocks per CTA (4: slower at M = 512, 8: slower everywhere)
constexpr int kMinM = 256;  // P072's range

// the base's collapse of row mg: group g (64 threads: warps 2g, 2g+1), thread t of the group; nws: the norm weights
// (shared memory)
__device__ __forceinline__ void collapse4(const p050::Args& a, int mg, const float (&pre)[4], float* red, int t, int g,
                                          const bf16* nws) {
  const int rw = t >> 5, l = t & 31;
  const bf16* r = a.R + (size_t)mg * kHcDim;
  float s[4];
  uint32_t pk[4][8];  // bf16(ol) of the 4 base threads' 16 dims
#pragma unroll
  for (int k2 = 0; k2 < 4; k2 += 2) {
    uint4 u[2][4][2];
#pragma unroll
    for (int kk = 0; kk < 2; kk++) {
      const int e = 32 * (2 * (k2 + kk) + rw) + l;
#pragma unroll
      for (int j = 0; j < 4; j++)
#pragma unroll
        for (int c8 = 0; c8 < 2; c8++) u[kk][j][c8] = __ldcg(reinterpret_cast<const uint4*>(r + j * kHidden + e * 16 + c8 * 8));
    }
#pragma unroll
    for (int kk = 0; kk < 2; kk++) {
      float x[4][16];
#pragma unroll
      for (int j = 0; j < 4; j++)
#pragma unroll
        for (int c8 = 0; c8 < 2; c8++) unpack8(u[kk][j][c8], &x[j][c8 * 8]);
      float ol[16], s2 = 0.f;
#pragma unroll
      for (int i = 0; i < 16; i++) {
        ol[i] = ((pre[0] * x[0][i] + pre[1] * x[1][i]) + pre[2] * x[2][i]) + pre[3] * x[3][i];
        s2 += ol[i] * ol[i];
      }
      s[k2 + kk] = s2;
#pragma unroll
      for (int i = 0; i < 8; i++) pk[k2 + kk][i] = pack_bf16x2(ol[2 * i], ol[2 * i + 1]);
    }
  }
#pragma unroll
  for (int k = 0; k < 4; k++) s[k] = warp_sum(s[k]);  // base warp 2k + rw
  if (l == 0) {
#pragma unroll
    for (int k = 0; k < 4; k++) red[2 * k + rw] = s[k];
  }
  asm volatile("bar.sync %0, 64;\n" ::"r"(1 + g) : "memory");
  float v = l < 8 ? red[l] : 0.f;  // the base's warp 0 over the 8 warp sums
  v = warp_sum(v);
  const float rs = rsqrtf(__fdiv_rn(v, float(kHidden)) + kEps);
#pragma unroll
  for (int k = 0; k < 4; k++) {
    const int e = 32 * (2 * k + rw) + l;
    float w[16];
    unpack8(*reinterpret_cast<const uint4*>(nws + e * 16), w);
    unpack8(*reinterpret_cast<const uint4*>(nws + e * 16 + 8), w + 8);
    uint32_t o[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const float x0 = bf16_lo(pk[k][i]) * rs * w[2 * i];
      const float x1 = bf16_hi(pk[k][i]) * rs * w[2 * i + 1];
      o[i] = pack_bf16x2(x0, x1);
    }
    uint4* d = reinterpret_cast<uint4*>(a.inp + (size_t)mg * kHidden + e * 16);
    d[0] = make_uint4(o[0], o[1], o[2], o[3]);
    d[1] = make_uint4(o[4], o[5], o[6], o[7]);
  }
}

// the inputs of one unit (row mg, dims h0 … h0+7 of the chunk): R's 4 streams, out, comb, post
struct Unit {
  uint4 r[4], o;
  float4 cm[4], p;
};
__device__ __forceinline__ void unit_load(const p050::Args& a, int mg, int h0, Unit& d) {
  const bf16* r = a.R + (size_t)mg * kHcDim;
#pragma unroll
  for (int j = 0; j < 4; j++) d.r[j] = *reinterpret_cast<const uint4*>(r + j * kHidden + h0);
  d.o = *reinterpret_cast<const uint4*>(a.out + (size_t)mg * kHidden + h0);
#pragma unroll
  for (int i = 0; i < 4; i++) d.cm[i] = *reinterpret_cast<const float4*>(a.comb + (size_t)mg * 16 + 4 * i);
  d.p = *reinterpret_cast<const float4*>(a.post + (size_t)mg * 4);
}

struct Smem {
  float W[kCq * kHcMix];  // the chunk's fn (24 KB), loaded once
  float xs[kRB][kCq];     // x_q of the chunk per row (xsw order); in the finish: the norm weights (bf16)
  float vmix[kRB * kOut], pre_s[kRB * 4], red[4][2][8];
  uint64_t bar;
  unsigned ticket;
};

// the finish of row block b (this CTA arrived last at it)
__device__ __noinline__ void finish(const p050::Args& a, Smem& s, int b, unsigned* cnt) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int m0 = b * kRB, M = min(kRB, a.M - m0);
  bf16* nws = reinterpret_cast<bf16*>(&s.xs[0][0]);
  reinterpret_cast<uint4*>(nws)[tid] = reinterpret_cast<const uint4*>(a.nw)[tid];
  reinterpret_cast<uint4*>(nws)[tid + kNT] = reinterpret_cast<const uint4*>(a.nw)[tid + kNT];
  for (int i = tid; i < M * kOut; i += kNT) {  // y = P_0, y += P_c in chunk order; the 64 loads in flight
    const int m = i / kOut, n = i % kOut;
    const float* src = a.ws + (size_t)(m0 + m) * kOut + n;
    float pv[kChunks];
#pragma unroll
    for (int k = 0; k < kChunks; k++) pv[k] = __ldcg(src + (size_t)k * a.ms * kOut);
    float y = pv[0];
#pragma unroll
    for (int k = 1; k < kChunks; k++) y = __fadd_rn(y, pv[k]);
    s.vmix[i] = y;
  }
  __syncthreads();
  {  // coefficients (P072): warp w → rows 2w (lanes 0–15), 2w + 1 (lanes 16–31); pre for every row
    const float t25 = a.sb[25], t26 = a.sb[26], sbp = a.sb[4 + (lane & 3)], sbc = a.sb[8 + (lane & 15)];
    const int hb = lane & 16, m = 2 * warp + (hb >> 4);
    const int mm = min(m, M - 1);  // rows past the block compute a copy (every lane joins the shuffles)
    const float* vm = s.vmix + mm * kOut;
    const float f = __fdiv_rn(1.0f, sqrtf(__fdiv_rn(vm[24], float(kHcDim))) + kEps);
    float po, cm;
    p072::coeffs(vm, f, t25, t26, sbp, sbc, hb, po, cm);
    if (m < M) {
      if ((lane & 15) < 4) a.post[(size_t)(m0 + m) * 4 + (lane & 3)] = po;
      a.comb[(size_t)(m0 + m) * 16 + (lane & 15)] = cm;
    }
    if (tid < M * 4) {
      const int pm = tid >> 2, i = tid & 3;
      const float* v2 = s.vmix + pm * kOut;
      const float f2 = __fdiv_rn(1.0f, sqrtf(__fdiv_rn(v2[24], float(kHcDim))) + kEps);
      const float mi = v2[i] * f2;
      s.pre_s[tid] = sigm(mi * a.sb[24] + a.sb[i]) + kEps;
    }
  }
  __syncthreads();
  {  // the collapse: group g (warps 2g, 2g+1) → rows g, g + 4, g + 8, g + 12
    const int g = tid >> 6, t = tid & 63;
#pragma unroll 1
    for (int m = g, it = 0; m < M; m += 4, it++) {
      const float pr[4] = {s.pre_s[4 * m], s.pre_s[4 * m + 1], s.pre_s[4 * m + 2], s.pre_s[4 * m + 3]};
      collapse4(a, m0 + m, pr, s.red[g][it & 1], t, g, nws);
    }
  }
  if (tid == 0) cnt[b] = 0u;
  __syncthreads();  // xs / vmix / pre_s / red reuse
}

// grid (64 chunks, ⌈M / 32⌉ row groups of kNB blocks of 16 rows), kNT threads; programmatic launch
// (griddepcontrol.wait before R / out / post / comb; fn, a weight, before it)
__global__ void __launch_bounds__(kNT, 2) k_post_pre(p050::Args a, unsigned* __restrict__ cnt) {
  __shared__ __align__(128) Smem s;
  const int c = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int nblk = cdiv(a.M, kRB), b0 = blockIdx.y * kNB, b1 = min(b0 + kNB, nblk);
  if (tid == 0) {
    p050::mbar_init(&s.bar);
    p050::tma_load(s.W, a.w + (size_t)c * kCq * kHcMix, kCq * kHcMix * 4, &s.bar);
  }
  p050::griddep_wait();
  const int um = tid >> 3, hl = (tid & 7) * 8, h0 = c * kCh + hl;  // warps 0–3: unit (row um, dims h0 …)
  Unit d;
  if (tid < 128 && b0 * kRB + um < a.M) unit_load(a, b0 * kRB + um, h0, d);
  unsigned pend = 0;  // blocks (bit b - b0) to finish
#pragma unroll 1
  for (int b = b0; b < b1; b++) {
    const int m0 = b * kRB, M = min(kRB, a.M - m0);
    if (tid < 128 && um < M) {  // ---- x_q and R' (P072's arithmetic) ----
      const int mg = m0 + um;
      bf16* r = a.R + (size_t)mg * kHcDim;
      float cm[16], p[4], xi[4][8], o[8], v[32];
#pragma unroll
      for (int i = 0; i < 4; i++) cm[4 * i] = d.cm[i].x, cm[4 * i + 1] = d.cm[i].y, cm[4 * i + 2] = d.cm[i].z, cm[4 * i + 3] = d.cm[i].w;
      p[0] = d.p.x, p[1] = d.p.y, p[2] = d.p.z, p[3] = d.p.w;
#pragma unroll
      for (int j = 0; j < 4; j++) unpack8(d.r[j], xi[j]);
      unpack8(d.o, o);
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
        *reinterpret_cast<float4*>(&s.xs[um][xsw(4 * hl + 4 * k)]) = make_float4(v[4 * k], v[4 * k + 1], v[4 * k + 2], v[4 * k + 3]);
    }
    __syncthreads();  // [A] xs ready
    if (tid < 128) {
      if (b + 1 < b1 && m0 + kRB + um < a.M) unit_load(a, m0 + kRB + um, h0, d);  // the next block's inputs in flight
    } else if (lane < kOut && 4 * (warp - 4) < M) {  // ---- chunk partials: warp 4 + k → rows 4k … 4k+3 ----
      if (b == b0) p050::mbar_wait(&s.bar, 0);
      const int n = lane < kHcMix ? lane : 0, r0 = 4 * (warp - 4);
      const bool ssl = lane >= kHcMix;
      const float* x0 = s.xs[r0];
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 2
      for (int q0 = 0; q0 < kCq; q0 += 8) {
        float wv[8];
#pragma unroll
        for (int k = 0; k < 8; k++) wv[k] = s.W[(q0 + k) * kHcMix + n];
#pragma unroll
        for (int rr = 0; rr < 4; rr++) {
          const float4 xa = *reinterpret_cast<const float4*>(x0 + rr * kCq + xsw(q0));
          const float4 xb = *reinterpret_cast<const float4*>(x0 + rr * kCq + xsw(q0 + 4));
          const float xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
          for (int k = 0; k < 8; k++) acc[rr] = __fmaf_rn(ssl ? xv[k] : wv[k], xv[k], acc[rr]);
        }
      }
#pragma unroll
      for (int rr = 0; rr < 4; rr++)
        if (r0 + rr < M) a.ws[((size_t)c * a.ms + m0 + r0 + rr) * kOut + lane] = acc[rr];
    }
    __syncthreads();  // [B] partials stored, xs free
    if (tid == 128) {  // release (bar.sync, then one thread's fence and atomic)
      __threadfence();
      s.ticket = p050::atom_add_acq_rel(cnt + b, 1u);
    }
    __syncthreads();  // [C]
    if (s.ticket == kChunks - 1) pend |= 1u << (b - b0);
  }
  p050::griddep_launch();
  if (!pend) return;
  __threadfence();
#pragma unroll 1
  for (int b = b0; b < b1; b++)
    if (pend >> (b - b0) & 1) finish(a, s, b, cnt);
}

void run(const MhcW& m, const float* w, bf16* R, const bf16* out, int M, float* post, float* comb, bf16* inp,
         cudaStream_t s) {
  if (M > p051::kMaxRowsWs) qlog::fail("mhc", "M %d > %d", M, p051::kMaxRowsWs);
  p050::Args a{w, m.sb, m.normw, R, out, post, comb, inp, p051::g_ws, m.cnt, M, M};
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(kChunks, cdiv(cdiv(M, kRB), kNB));
  cfg.blockDim = dim3(kNT);
  cfg.stream = s;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, k_post_pre, a, p072::g_cnt));
}

}  // namespace p106
