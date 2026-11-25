// p042_hca_fuse.cuh — P042 hca_fuse (included by hca.cu): the HCA chain in three kernels instead of nine (model.cc):
//   hca_pre        = hca_kv_rows + the group-closing rows of hca_compress (one launch; the state update moves to
//                    hca_post, after every reader of the old state);
//   hca_attn_fused = hca_q_prep (on load, q is not written back) + hca_attn (P041) + vha_partial;
//   hca_post       = hca_vha_out (P040) + hca_ring_store + the state update.
// Same arithmetic as each base stage → bit-identical outputs and caches.
#pragma once

namespace e55 {
namespace p042 {

// the body of k_compress for one row (512 threads, thread = dim); same code
__device__ __forceinline__ void compress_row(const float* __restrict__ ckv, const float* __restrict__ ape,
                                             const bf16* __restrict__ nw, const RowMeta& mt,
                                             const float* __restrict__ inv, HcaCache c) {
  __shared__ float red[16], cn[512];
  const int p = mt.pos;
  const int d = threadIdx.x, g = p / kRatio, t0 = g * kRatio;
  const float* st = c.state + (size_t)mt.slot * kRatio * 1024;
  auto score = [&](int t) {
    return t >= mt.p0 ? ckv[(size_t)(mt.r0 + t - mt.p0) * 1024 + 512 + d] + ape[(t % kRatio) * 512 + d]
                      : st[(t % kRatio) * 1024 + 512 + d];
  };
  auto kvv = [&](int t) {
    return t >= mt.p0 ? ckv[(size_t)(mt.r0 + t - mt.p0) * 1024 + d] : st[(t % kRatio) * 1024 + d];
  };
  float mx = -INFINITY;
  for (int t = t0; t <= p; t++) mx = fmaxf(mx, score(t));
  float sum = 0.f;
  for (int t = t0; t <= p; t++) sum += expf(score(t) - mx);
  float cv = 0.f;
  for (int t = t0; t <= p; t++) cv += __fdiv_rn(expf(score(t) - mx), sum) * kvv(t);
  float ss = warp_sum(cv * cv);
  if ((d & 31) == 0) red[d >> 5] = ss;
  __syncthreads();
  if (d < 32) {
    float x = d < 16 ? red[d] : 0.f;
    x = warp_sum(x);
    if (d == 0) red[0] = x;
  }
  __syncthreads();
  const float rs = rsqrtf(__fdiv_rn(red[0], 512.f) + kEps);
  cn[d] = cv * rs * __bfloat162float(nw[d]);
  __syncthreads();
  if (d < 64) {
    const int d0 = 8 * d;
    float v[8];
#pragma unroll
    for (int j = 0; j < 8; j++) v[j] = cn[d0 + j];
    if (d0 >= kNope) rope_pairs8(v, d0 - kNope, t0, inv, false);
    store_kv_row8(c.comp + ((size_t)mt.slot * c.max_comp + g) * kKvRowBytes, d0, v);
  }
}

// blocks [0, nkv): KV rows, 8 rows per block (64 threads each, k_kv_rows); blocks [nkv, nkv + M): k_compress rows
__global__ void __launch_bounds__(512) k_pre(const bf16* __restrict__ kv, int ld, const float* __restrict__ ckv,
                                             const float* __restrict__ ape, const bf16* __restrict__ nw,
                                             const RowMeta* __restrict__ meta, const float* __restrict__ inv, int M,
                                             int nkv, uint8_t* rows, HcaCache c) {
  if (int(blockIdx.x) < nkv) {
    const int r = blockIdx.x * 8 + (threadIdx.x >> 6), i = threadIdx.x & 63, d0 = 8 * i;
    if (r >= M) return;
    const int p = meta[r].pos;
    float v[8];
    load_bf16x8(kv + (size_t)r * ld + d0, v);
    if (d0 >= kNope) rope_pairs8(v, d0 - kNope, p, inv, false);
    store_kv_row8(rows + (size_t)r * kKvRowBytes, d0, v);
    return;
  }
  const RowMeta mt = meta[blockIdx.x - nkv];
  if ((mt.pos + 1) % kRatio) return;
  compress_row(ckv, ape, nw, mt, inv, c);
}

// P100 a2a_attn: hca_post's ring store and compressor state update alone, for every row of the step (KV is replicated:
// both ranks keep it); the VHA part runs on each row's owner (hca_vha_out). The code of p040's POST CTA: same bytes.
__global__ void __launch_bounds__(64) k_post_state(const RowMeta* __restrict__ meta, const uint8_t* __restrict__ rows,
                                                   const float* __restrict__ ckv, const float* __restrict__ ape,
                                                   HcaCache c) {
  const int m = blockIdx.x, i = threadIdx.x;
  const RowMeta mt = meta[m];
  if (mt.pos > mt.pend - kWindow) {  // k_ring
    const uint2* src = reinterpret_cast<const uint2*>(rows + (size_t)m * kKvRowBytes);
    uint2* dst = reinterpret_cast<uint2*>(c.ring + ((size_t)mt.slot * kWindow + mt.pos % kWindow) * kKvRowBytes);
    for (int k = i; k < kKvRowBytes / 8; k += 64) dst[k] = src[k];
  }
  if (mt.pos >= (mt.pend + 1) / kRatio * kRatio) {  // k_state
    float* st = c.state + ((size_t)mt.slot * kRatio + mt.pos % kRatio) * 1024;
    const float* src = ckv + (size_t)m * 1024;
    for (int d = i; d < 1024; d += 64) st[d] = d < 512 ? src[d] : src[d] + ape[(mt.pos % kRatio) * 512 + d - 512];
  }
}

// P100 a2a_attn: the row owner's VHA for all 64 heads, one CTA per row (thread i: dims 8i … 8i+7). z = fl(zp + Σ_h
// U_peer[h]·o_peer[h]) — hca.cu k_vha_partial's fma chain over the peer's 32 heads, then the VHA all-reduce's one fp32
// add — kept in registers, then p040 k_vha_out's per-head body (HPB 32: V from shared memory) for this rank's heads
// (V_own → q_own groups) and the peer's (V_peer → q_peer): the bytes of misc vha_z + hca_vha_out twice, without z's
// DRAM round trip. q_* [4][M][4096] e4m3, s_* [4][M][32] UE8M0 (wo_a's group-major activations).
__global__ void __launch_bounds__(64) k_vha_own(const bf16* __restrict__ om, const bf16* __restrict__ op,
                                                const float* __restrict__ zp, const float* __restrict__ Up,
                                                const float* __restrict__ Vm, const float* __restrict__ Vp,
                                                const RowMeta* __restrict__ meta, const float* __restrict__ inv, int M,
                                                uint8_t* __restrict__ qm, uint8_t* __restrict__ sm,
                                                uint8_t* __restrict__ qp, uint8_t* __restrict__ sp) {
  constexpr int kLH = kHeads / 2;
  __shared__ float Us[kLH * kVhaRank], Vs[2][kLH * kVhaRank];
  const int m = blockIdx.x, i = threadIdx.x, d0 = 8 * i;
  const RowMeta mt = meta[m];
  for (int k = i; k < kLH * kVhaRank; k += 64) Us[k] = Up[k], Vs[0][k] = Vm[k], Vs[1][k] = Vp[k];
  __syncthreads();
  float zr[kVhaRank][8];
#pragma unroll
  for (int r = 0; r < kVhaRank; r++)
#pragma unroll
    for (int j = 0; j < 8; j++) zr[r][j] = 0.f;
#pragma unroll 1
  for (int h = 0; h < kLH; h++) {  // the peer's partial: per (r, d) the chain over h from 0
    float x[8];
    p040::ld8(op + ((size_t)m * kLH + h) * kHeadDim + d0, x);
#pragma unroll
    for (int r = 0; r < kVhaRank; r++) {
      const float u = Us[h * kVhaRank + r];
#pragma unroll
      for (int j = 0; j < 8; j++) zr[r][j] += u * x[j];
    }
  }
#pragma unroll
  for (int r = 0; r < kVhaRank; r++) {
    const float4* zq = reinterpret_cast<const float4*>(zp + ((size_t)m * kVhaRank + r) * kHeadDim + d0);
    const float4 a = zq[0], b = zq[1];
    const float own[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
    for (int j = 0; j < 8; j++) zr[r][j] = __fadd_rn(own[j], zr[r][j]);
  }
  float cs[4] = {1.f, 1.f, 1.f, 1.f}, sn[4] = {0.f, 0.f, 0.f, 0.f};
  if (d0 >= kNope) {  // p040: fl(pos · inv), sincosf, inverse → −sin
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const float a = __fmul_rn(float(mt.pos), inv[(d0 - kNope) / 2 + j]);
      sincosf(a, &sn[j], &cs[j]);
      sn[j] = -sn[j];
    }
  }
#pragma unroll 1
  for (int set = 0; set < 2; set++) {
    const bf16* o = set ? op : om;
    uint8_t *q = set ? qp : qm, *sc = set ? sp : sm;
#pragma unroll 1
    for (int h = 0; h < kLH; h++) {  // p040 k_vha_out's body
      float v[8], vr[kVhaRank];
      p040::ld8(o + ((size_t)m * kLH + h) * kHeadDim + d0, v);
#pragma unroll
      for (int r = 0; r < kVhaRank; r++) vr[r] = Vs[set][h * kVhaRank + r];
#pragma unroll
      for (int j = 0; j < 8; j++) {
        float a = 0.f;
#pragma unroll
        for (int r = 0; r < kVhaRank; r++) a += vr[r] * zr[r][j];
        v[j] = __bfloat162float(__float2bfloat16_rn(v[j] + a));
      }
      if (d0 >= kNope) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const float e = v[2 * j], od = v[2 * j + 1];
          v[2 * j] = __fsub_rn(__fmul_rn(e, cs[j]), __fmul_rn(od, sn[j]));
          v[2 * j + 1] = __fadd_rn(__fmul_rn(od, cs[j]), __fmul_rn(e, sn[j]));
        }
      }
      float amax = 0.f;
#pragma unroll
      for (int j = 0; j < 8; j++) amax = fmaxf(amax, fabsf(v[j]));
#pragma unroll
      for (int off = 8; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
      const uint32_t e = act_scale_e8(amax, 1e-10f);
      const int gi = h >> 3, col = (h & 7) * kHeadDim + d0;
      uint32_t lo = 0, hi = 0;
#pragma unroll
      for (int j = 0; j < 4; j++) lo |= quant_e4m3(v[j], e) << (8 * j), hi |= quant_e4m3(v[4 + j], e) << (8 * j);
      *reinterpret_cast<uint2*>(q + ((size_t)gi * M + m) * 4096 + col) = make_uint2(lo, hi);
      if ((d0 & 127) == 0) sc[((size_t)gi * M + m) * 32 + col / 128] = uint8_t(e);
    }
  }
}

// ---- P044 hca_compress_dec: decode (M ≤ 64) pre stage as one 8-CTA cluster per row -------------------------------
// CTA x = 0 also writes the row's KV row (k_kv_rows). A group-closing row's compressor runs over the cluster: CTA x owns
// dims 64x … 64x+63 (= base warps 2x, 2x+1 of k_compress), stages the group's 128 scores / kv of its dims in shared
// memory, and keeps every per-dim chain of the base in order (max, Σ expf sequential, Σ (e / sum) · kv sequential); the
// expf / divisions in between run on all threads (same values). The RMS needs the 16 warp sums of all dims: they meet in
// L2 behind one cluster barrier, then CTA x writes part x of the compressed row (its 64 dims). Same bits as k_compress.
constexpr int kPreDecMaxM = 64;
// attn_common.cuh store_kv_row8 for the 8 lanes 0–7 of a warp alone (shuffle mask 0xff): the same bytes
__device__ __forceinline__ void store_kv_row8_l8(uint8_t* row, int d0, const float in[8]) {
  float v[8], amax = 0.f;
#pragma unroll
  for (int j = 0; j < 8; j++) v[j] = __bfloat162float(__float2bfloat16_rn(in[j])), amax = fmaxf(amax, fabsf(v[j]));
#pragma unroll
  for (int off = 4; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffu, amax, off));
  if (d0 < kNope) {
    const uint32_t e = act_scale_e8(amax, 1e-4f);
    uint32_t lo = 0, hi = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) lo |= quant_e4m3(v[j], e) << (8 * j), hi |= quant_e4m3(v[4 + j], e) << (8 * j);
    *reinterpret_cast<uint2*>(row + d0) = make_uint2(lo, hi);
    if ((d0 & 63) == 0) row[576 + d0 / 64] = uint8_t(e);
    if (d0 == 0) row[583] = 0;
  } else {
    uint2* dst = reinterpret_cast<uint2*>(row + kNope + 2 * (d0 - kNope));
    dst[0] = make_uint2(pack_bf16x2(v[0], v[1]), pack_bf16x2(v[2], v[3]));
    dst[1] = make_uint2(pack_bf16x2(v[4], v[5]), pack_bf16x2(v[6], v[7]));
  }
}
struct CompSmem {
  float sc[kRatio][64];  // score + ape → e → e / sum
  float kv[kRatio][64];
  float mx[64], sum[64], cv[64];
  float red[16];
};
__global__ void __launch_bounds__(512) k_pre_dec(const bf16* __restrict__ kv, int ld, const float* __restrict__ ckv,
                                                 const float* __restrict__ ape, const bf16* __restrict__ nw,
                                                 const RowMeta* __restrict__ meta, const float* __restrict__ inv,
                                                 uint8_t* rows, HcaCache c, float* __restrict__ xs) {
  const int x = blockIdx.x, r = blockIdx.y, tid = threadIdx.x;
  const RowMeta mt = meta[r];
  if (x == 0 && tid < 64) {  // k_kv_rows
    const int d0 = 8 * tid;
    float v[8];
    load_bf16x8(kv + (size_t)r * ld + d0, v);
    if (d0 >= kNope) rope_pairs8(v, d0 - kNope, mt.pos, inv, false);
    store_kv_row8(rows + (size_t)r * kKvRowBytes, d0, v);
  }
  const int p = mt.pos;
  if ((p + 1) % kRatio) return;  // the whole cluster leaves together (no barrier reached)
  extern __shared__ __align__(16) uint8_t smraw[];
  CompSmem& S = *reinterpret_cast<CompSmem*>(smraw);
  const int g = p / kRatio, t0 = g * kRatio, dx = 64 * x;
  const float* st = c.state + (size_t)mt.slot * kRatio * 1024;
  // stage tokens t0 … p (all 128: the group closes at p) for dims dx …: 128 × 16 float4 per array, 4 per thread
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const int i = tid + 512 * j, tt = i >> 4, c4 = (i & 15) * 4, t = t0 + tt, d = dx + c4;
    float4 s4, k4;
    if (t >= mt.p0) {
      const float* src = ckv + (size_t)(mt.r0 + t - mt.p0) * 1024;
      const float4 a = *reinterpret_cast<const float4*>(src + 512 + d);
      const float4 e = *reinterpret_cast<const float4*>(ape + (t % kRatio) * 512 + d);
      s4 = make_float4(a.x + e.x, a.y + e.y, a.z + e.z, a.w + e.w);
      k4 = *reinterpret_cast<const float4*>(src + d);
    } else {
      s4 = *reinterpret_cast<const float4*>(st + (t % kRatio) * 1024 + 512 + d);
      k4 = *reinterpret_cast<const float4*>(st + (t % kRatio) * 1024 + d);
    }
    *reinterpret_cast<float4*>(&S.sc[tt][c4]) = s4;
    *reinterpret_cast<float4*>(&S.kv[tt][c4]) = k4;
  }
  __syncthreads();
  const int dd = tid & 63, q8 = tid >> 6;  // 8 threads per dim, 16 tokens each
  {  // max over the group (exact in any order)
    float m = -INFINITY;
#pragma unroll
    for (int k = 0; k < 16; k++) m = fmaxf(m, S.sc[16 * q8 + k][dd]);
    __shared__ float pm[8][64];
    pm[q8][dd] = m;
    __syncthreads();
    if (tid < 64) {
      float mm = -INFINITY;
      for (int k = 0; k < 8; k++) mm = fmaxf(mm, pm[k][tid]);
      S.mx[tid] = mm;
    }
    __syncthreads();
  }
  {  // e = expf(score − max), all threads
    const float m = S.mx[dd];
#pragma unroll
    for (int k = 0; k < 16; k++) S.sc[16 * q8 + k][dd] = expf(S.sc[16 * q8 + k][dd] - m);
  }
  __syncthreads();
  if (tid < 64) {  // Σ e in token order (k_compress's sum)
    float sum = 0.f;
#pragma unroll 16
    for (int t = 0; t < kRatio; t++) sum += S.sc[t][tid];
    S.sum[tid] = sum;
  }
  __syncthreads();
  {
    const float sm = S.sum[dd];
#pragma unroll
    for (int k = 0; k < 16; k++) S.sc[16 * q8 + k][dd] = __fdiv_rn(S.sc[16 * q8 + k][dd], sm);
  }
  __syncthreads();
  if (tid < 64) {  // Σ (e / sum) · kv in token order, then the base warps' Σ cv² (lanes = dims of warp 2x + tid / 32)
    float cv = 0.f;
#pragma unroll 16
    for (int t = 0; t < kRatio; t++) cv += S.sc[t][tid] * S.kv[t][tid];
    S.cv[tid] = cv;
    const float ss = warp_sum(cv * cv);
    if ((tid & 31) == 0) __stcg(xs + (size_t)r * 16 + 2 * x + (tid >> 5), ss);
  }
  asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
  asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
  if (tid < 32) {  // k_compress's second level over the 16 warp sums, every CTA alike
    float v = tid < 16 ? __ldcg(xs + (size_t)r * 16 + tid) : 0.f;
    v = warp_sum(v);
    if (tid == 0) S.red[0] = v;
  }
  __syncthreads();
  if (tid < 8) {  // part x of the compressed row: dims dx + 8·tid …
    const float rs = rsqrtf(__fdiv_rn(S.red[0], 512.f) + kEps);
    const int d0 = dx + 8 * tid;
    float v[8];
#pragma unroll
    for (int j = 0; j < 8; j++) v[j] = S.cv[8 * tid + j] * rs * __bfloat162float(nw[d0 + j]);
    if (d0 >= kNope) rope_pairs8(v, d0 - kNope, t0, inv, false);
    store_kv_row8_l8(c.comp + ((size_t)mt.slot * c.max_comp + g) * kKvRowBytes, d0, v);
  }
}

}  // namespace p042
}  // namespace e55
