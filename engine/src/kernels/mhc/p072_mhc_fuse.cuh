// p072_mhc_fuse.cuh — P072 mhc_fuse: mHC pre / post + pre / layer-0 pre for M > 16 rows (prefill) in ONE launch.
// Included by mhc.cu inside its anonymous namespace after p051_mhc_big.cuh (uses p050's chunk code and helpers).
//
// Why: P051 runs P050's chunk kernel over all row blocks, then a second kernel finishes every row — by then the 128 MB
// of R' (M = 4096) has left L2, so the finish reads it back from DRAM (plus the partials): ~45 % of the call's bytes.
// Here the chunk CTAs are P050's (same x_q, same R' bytes, same chunk partials into P051's workspace), and the LAST
// chunk CTA of each block of 16 rows (an acq_rel ticket per block, after every CTA's fence) finishes those rows right
// away, while their R' and partials are still in L2:
//   y[n] = P_0[n], y += P_c[n] (c = 1 … 63) — one thread per (row, n), the base's order;
//   post / comb: P050's coefficient warp code, two rows per warp (lanes 0–15 / 16–31), all 9 warps;
//   pre (the base's expression) and the collapse with the base's 256-thread layout and reduction (named barrier 2),
//   row after row, the next row's R' fetched during the current row's reduction.
// Used from M ≥ 256 (fewer row blocks: P051 — one CTA finishing 16 rows is then the longer path).
// Every value is the base's (same expressions in the same order) → bit-identical to P051 / the base.
#pragma once

namespace p072 {

constexpr int kRB = kDecM;  // rows per block (P050's 16)
constexpr int kNT = p050::kNT;  // 288 (9 warps): two CTAs an SM (512 threads, one CTA an SM, measured slower)
constexpr int kMinM = 256;      // below: P051 (few row blocks — one CTA finishing 16 rows is the longer path)
unsigned* g_cnt = nullptr;  // per row block: chunk CTAs done (zero between calls; reset by the finishing CTA)

void init() {
  if (!g_cnt) {
    CUDA_CHECK(cudaMalloc(&g_cnt, (p051::kMaxRowsWs / kRB) * sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(g_cnt, 0, (p051::kMaxRowsWs / kRB) * sizeof(unsigned)));
  }
}

// one row's coefficients on 16 lanes (base / P050 finish warp-8 arithmetic); hb = 0 or 16: the lane half of this row
__device__ __forceinline__ void coeffs(const float* vm, float f, float t25, float t26, float sbp, float sbc, int hb,
                                       float& po, float& cm) {
  const unsigned full = 0xffffffffu;
  const int lane = threadIdx.x & 31, L = lane & 15, i4 = L & ~3, jc = L & 3;
  const float mp = vm[4 + (lane & 3)] * f;
  po = sigm(mp * t25 + sbp) * 2.0f;
  const float mc = vm[8 + L] * f;
  const float l = mc * t26 + sbc;
  float mx = -INFINITY;
#pragma unroll
  for (int k = 0; k < 4; k++) mx = fmaxf(mx, __shfl_sync(full, l, hb + i4 + k));
  const float e = expf(l - mx);
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < 4; k++) s += __shfl_sync(full, e, hb + i4 + k);
  cm = __fdiv_rn(e, s) + kEps;
  auto colnorm = [&] {
    const float c0 = __shfl_sync(full, cm, hb + jc), c1 = __shfl_sync(full, cm, hb + 4 + jc);
    const float c2 = __shfl_sync(full, cm, hb + 8 + jc), c3 = __shfl_sync(full, cm, hb + 12 + jc);
    cm = __fdiv_rn(cm, ((c0 + c1) + c2) + c3 + kEps);
  };
  auto rownorm = [&] {
    const float c0 = __shfl_sync(full, cm, hb + i4), c1 = __shfl_sync(full, cm, hb + i4 + 1);
    const float c2 = __shfl_sync(full, cm, hb + i4 + 2), c3 = __shfl_sync(full, cm, hb + i4 + 3);
    cm = __fdiv_rn(cm, ((c0 + c1) + c2) + c3 + kEps);
  };
  colnorm();
  for (int it = 0; it < 19; it++) rownorm(), colnorm();
}

// the base's collapse (mhc.cu collapse / block_sum256) for 256 threads with local index lt on named barrier bid
__device__ __forceinline__ void collapse_g(const float (&x)[4][16], const float (&pre)[4], const float (&w)[16],
                                           float* red, bf16* inp_row, int lt, int bid) {
  float ol[16], s2 = 0.f;
#pragma unroll
  for (int i = 0; i < 16; i++) {
    ol[i] = ((pre[0] * x[0][i] + pre[1] * x[1][i]) + pre[2] * x[2][i]) + pre[3] * x[3][i];
    s2 += ol[i] * ol[i];
  }
  {  // block_sum256: warp butterfly, warp 0 over the 8 warp sums
    float v = warp_sum(s2);
    const int wl = lt >> 5, l = lt & 31;
    asm volatile("bar.sync %0, 256;\n" ::"r"(bid) : "memory");
    if (l == 0) red[wl] = v;
    asm volatile("bar.sync %0, 256;\n" ::"r"(bid) : "memory");
    if (wl == 0) {
      v = l < 8 ? red[l] : 0.f;
      v = warp_sum(v);
      if (l == 0) red[0] = v;
    }
    asm volatile("bar.sync %0, 256;\n" ::"r"(bid) : "memory");
    s2 = red[0];
  }
  const float rs = rsqrtf(__fdiv_rn(s2, float(kHidden)) + kEps);
  uint32_t o[8];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const float a = __bfloat162float(__float2bfloat16_rn(ol[2 * i])) * rs * w[2 * i];
    const float b = __bfloat162float(__float2bfloat16_rn(ol[2 * i + 1])) * rs * w[2 * i + 1];
    o[i] = pack_bf16x2(a, b);
  }
  uint4* d = reinterpret_cast<uint4*>(inp_row + lt * 16);
  d[0] = make_uint4(o[0], o[1], o[2], o[3]);
  d[1] = make_uint4(o[4], o[5], o[6], o[7]);
}

// grid (64 chunks, ⌈M / 16⌉ row blocks), kNT threads; programmatic launch (griddepcontrol.wait before R / out)
template <int MODE>
__global__ void __launch_bounds__(kNT) k_fuse(p050::Args a, unsigned* __restrict__ cnt) {
  constexpr int K = MODE == 2 ? kCh : kCq;
  __shared__ __align__(128) float W[kCq * kHcMix];  // the chunk's fn; after the chain: the finish's scratch
  __shared__ __align__(16) float xs[kDecM][kCq];
  __shared__ __align__(8) uint64_t bar;
  __shared__ float red[1][8];
  __shared__ unsigned ticket;
  const int c = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int m0 = blockIdx.y * kRB, M = min(kRB, a.M - m0);
  // ---- the chunk: p050::k_chunk's body (same x_q, R' and partials) ------------------------------------------------
  if (tid == 0) {
    p050::mbar_init(&bar);
    p050::tma_load(W, a.w + (size_t)c * K * kHcMix, K * kHcMix * 4, &bar);
  }
  p050::griddep_wait();
  for (int u = tid; u < M * 8; u += kNT) {
    const int m = u >> 3, hl = (u & 7) * 8, h0 = c * kCh + hl, mg = m0 + m;
    bf16* r = a.R + (size_t)mg * kHcDim;
    if constexpr (MODE == 1) {
      float cm[16], p[4], xi[4][8], o[8], v[32];
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
  p050::mbar_wait(&bar, 0);
  if (lane < kOut) {
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
  // ---- the last chunk CTA of the row block finishes its rows ----------------------------------------------------------
  __threadfence();
  __syncthreads();
  if (tid == 0) ticket = p050::atom_add_acq_rel(cnt + blockIdx.y, 1u);
  __syncthreads();
  p050::griddep_launch();
  if (ticket != kChunks - 1) return;
  __threadfence();
  float* vmix = W;                // [16][25]
  float* pre_s = W + kRB * kOut;  // [16][4]
  // y = P_0, y += P_c in chunk order: thread → (row, n)
  for (int i = tid; i < M * kOut; i += kNT) {
    const int m = i / kOut, n = i % kOut;
    const float* src = a.ws + (size_t)(m0 + m) * kOut + n;
    float y = 0.f;
#pragma unroll
    for (int c0 = 0; c0 < kChunks; c0 += 16) {  // 16 loads in flight, then their adds in chunk order
      float pv[16];
#pragma unroll
      for (int k = 0; k < 16; k++) pv[k] = __ldcg(src + (size_t)(c0 + k) * a.ms * kOut);
#pragma unroll
      for (int k = 0; k < 16; k++) y = c0 + k == 0 ? pv[0] : __fadd_rn(y, pv[k]);
    }
    vmix[i] = y;
  }
  __syncthreads();
  {  // coefficients: warp w → rows 2w (lanes 0–15), 2w + 1 (lanes 16–31); pre for every row
    const float t25 = a.sb[25], t26 = a.sb[26], sbp = a.sb[4 + (lane & 3)], sbc = a.sb[8 + (lane & 15)];
    const int hb = lane & 16, m = 2 * warp + (hb >> 4);
    const int mm = min(m, M - 1);  // rows past the block compute a copy (every lane joins the shuffles)
    const float* vm = vmix + mm * kOut;
    const float f = __fdiv_rn(1.0f, sqrtf(__fdiv_rn(vm[24], float(MODE == 2 ? kHidden : kHcDim))) + kEps);
    float po, cm;
    coeffs(vm, f, t25, t26, sbp, sbc, hb, po, cm);
    if (m < M) {
      if ((lane & 15) < 4) a.post[(size_t)(m0 + m) * 4 + (lane & 3)] = po;
      a.comb[(size_t)(m0 + m) * 16 + (lane & 15)] = cm;
    }
    if (tid < M * 4) {  // pre (warps 0–7 of P050's finish compute the same expression)
      const int pm = tid >> 2, i = tid & 3;
      const float* v2 = vmix + pm * kOut;
      const float f2 = __fdiv_rn(1.0f, sqrtf(__fdiv_rn(v2[24], float(MODE == 2 ? kHidden : kHcDim))) + kEps);
      const float mi = v2[i] * f2;
      pre_s[tid] = sigm(mi * a.sb[24] + a.sb[i]) + kEps;
    }
  }
  __syncthreads();
  if (tid < 256) {  // the collapse, row after row (the base's layout: thread → dims 16·tid …), next row's R' in flight
    float w[16];
    unpack8(*reinterpret_cast<const uint4*>(a.nw + tid * 16), w);
    unpack8(*reinterpret_cast<const uint4*>(a.nw + tid * 16 + 8), w + 8);
    uint4 cur[4][2], nxt[4][2];
    auto fetch = [&](int m, uint4 (&d)[4][2]) {
      const bf16* r = a.R + (size_t)(m0 + m) * kHcDim;
#pragma unroll
      for (int j = 0; j < 4; j++)
#pragma unroll
        for (int c8 = 0; c8 < 2; c8++) d[j][c8] = __ldcg(reinterpret_cast<const uint4*>(r + j * kHidden + tid * 16 + c8 * 8));
    };
    fetch(0, cur);
#pragma unroll 1
    for (int m = 0; m < M; m++) {
      float x[4][16];
#pragma unroll
      for (int j = 0; j < 4; j++)
#pragma unroll
        for (int c8 = 0; c8 < 2; c8++) unpack8(cur[j][c8], &x[j][c8 * 8]);
      if (m + 1 < M) fetch(m + 1, nxt);
      const float pr[4] = {pre_s[4 * m], pre_s[4 * m + 1], pre_s[4 * m + 2], pre_s[4 * m + 3]};
      collapse_g(x, pr, w, red[0], a.inp + (size_t)(m0 + m) * kHidden, tid, 2);
#pragma unroll
      for (int j = 0; j < 4; j++) cur[j][0] = nxt[j][0], cur[j][1] = nxt[j][1];
    }
  }
  if (tid == 0) cnt[blockIdx.y] = 0u;
}

template <int MODE>
void run(const MhcW& m, const float* w, bf16* R, const bf16* out, int M, float* post, float* comb, bf16* inp,
         cudaStream_t s) {
  if (M > p051::kMaxRowsWs) qlog::fail("mhc", "M %d > %d", M, p051::kMaxRowsWs);
  p050::Args a{w, m.sb, m.normw, R, out, post, comb, inp, p051::g_ws, m.cnt, M, M};
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(kChunks, cdiv(M, kRB));
  cfg.blockDim = dim3(kNT);
  cfg.stream = s;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, k_fuse<MODE>, a, g_cnt));
}

}  // namespace p072
