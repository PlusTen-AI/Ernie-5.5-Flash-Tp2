// p129_draft_attn.cuh — P129 draft_attn (patches.def; included by dspark.cu inside namespace e55): a draft stage's
// attention as one launch, k_fattn, instead of hca_q_prep + hca_kv_rows + dsp_attn + vha_partial (dspark.cc then also
// takes P060's fused GEMM calls around it, as the target's attn_hca_a / _b). One CTA (8 warps) per block row, as k_attn:
//   - raw q and the block's kv rows copied into shared memory (cp.async, all in flight); q prep in place with k_q_prep's
//     arithmetic (64 threads a head, the same partial sums, the same two-warp sum); the block's KV rows built in shared
//     memory with k_kv_rows' bytes (store_kv_row8) by each CTA of the block (≤ 15 rows) — no launches, no round trips
//   - k_attn's loop over the same 32-key tiles in the same order with the same MMAs per output: the next tile's context
//     rows are loaded into registers while the current tile is computed (tile 0's before griddepcontrol.wait — the ring
//     is written by the target's steps, never inside the pass); a warp dequantizes one 64-dim part of all 32 keys
//     (conflict-free 16-byte stores; k_attn's 8 threads a key hit one bank group); Q·Kᵀ loads step k+1's fragments
//     before step k's MMA, P·V loads a tile's fragments before its MMAs (every accumulator still takes them in order);
//     a tile's online softmax by 8 warps (4 heads each, a lane per key: the same max, expf values and bf16 P), each
//     head's sum still added key by key in order by one lane, then k_attn's l / m / fac updates
//   - the output: fac · o into shared memory (fp32), then per dim bf16(· / l) (k_attn's division) stored and fed to
//     vha_partial's fma chain over the 32 heads in order
//   - the code that runs once a launch is kept in compact loops: on a CTA per row its instruction fetch is the cost
// Same bytes as the four base launches (o and the VHA partial z).
#pragma once

namespace p129 {

constexpr int kNT = 256;
constexpr int kLH = kHeads / 2;
constexpr int kKT = 32;
constexpr int kQS = kHeadDim + 8;
constexpr int kRowB = kKvRowBytes;
constexpr int kPS = kHeadDim + 4;  // fp32 output rows over q ‖ k (conflict-free float2 stores)

struct Smem {
  bf16 q[kLH][kQS];  // raw q, prepped in place · after the loop (with k) the unnormalized output, fp32 [32][kPS]
  bf16 k[kKT][kQS];  // the block's kv rows (staged) · then the K tile
  float s[kLH][kKT + 1];
  bf16 p[kLH][kKT + 8];
  float m[kLH], l[kLH], fac[kLH];
  alignas(16) float U[kLH * kVhaRank];
  float red[kLH / 4][8];  // q prep: [round][warp] partial sums
  int bpos[kDspMaxN];     // the block rows' positions
  alignas(16) uint8_t brow[kDspMaxN][kRowB];  // the block's KV rows (584 B each: 8-byte aligned)
};
static_assert(kLH * kPS * 4 <= (kLH + kKT) * kQS * 2, "the fp32 output rows fit over q and k");
static_assert(sizeof(Smem) <= 99 * 1024, "shared memory");

// load_bf16x8's unpacking of a loaded 16-byte word
__device__ __forceinline__ void unpack8(const uint4& u, float v[8]) {
  const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
  for (int i = 0; i < 4; i++) v[2 * i] = bf16_lo(w[i]), v[2 * i + 1] = bf16_hi(w[i]);
}

// one thread's share of a K tile: key kk = lane, part = warp (64 dims) — a context row's raw bytes in registers
struct Raw {
  uint2 w[16];
  uint32_t sc;
  int src;  // 0: past the keys (zeros) · 1: context row (w, sc) · 2: block row (shared memory)
};

__device__ __forceinline__ void fetch(Raw& R, const uint8_t* __restrict__ ring, const RowMeta& mt, int nctx, int nk,
                                      int key, int part) {
  if (key >= nk) {
    R.src = 0;
    return;
  }
  if (key >= nctx) {
    R.src = 2;
    return;
  }
  R.src = 1;
  const int t = mt.p0 - nctx + key;
  const uint8_t* row = ring + ((size_t)mt.slot * kDspRing + t % kDspRing) * kRowB;
  if (part < 7) {
    const uint2* src = reinterpret_cast<const uint2*>(row + part * 64);
#pragma unroll
    for (int j = 0; j < 8; j++) R.w[j] = src[j];
    R.sc = row[576 + part];
  } else {
    const uint2* src = reinterpret_cast<const uint2*>(row + kNope);
#pragma unroll
    for (int j = 0; j < 16; j++) R.w[j] = src[j];
  }
}

// dequant_kv_part's values from the registers (or from the block row in shared memory)
__device__ __forceinline__ void put(const Raw& R, Smem& S, int kk, int part, int nctx, int key) {
  bf16* dst = &S.k[kk][part * 64];
  if (R.src == 0) {
#pragma unroll
    for (int j = 0; j < 64; j += 8) *reinterpret_cast<uint4*>(dst + j) = make_uint4(0, 0, 0, 0);
  } else if (R.src == 2) {
    dequant_kv_part(S.brow[key - nctx], part, dst);
  } else if (part < 7) {
    const float s = ue8m0_f32(R.sc);
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const uint2 u = R.w[j];
      float v[8];
#pragma unroll
      for (int i = 0; i < 4; i++) {
        __nv_fp8_e4m3 a, b;
        a.__x = (u.x >> (8 * i)) & 0xFF, b.__x = (u.y >> (8 * i)) & 0xFF;
        v[i] = float(a) * s, v[4 + i] = float(b) * s;
      }
      store_bf16x8(dst + 8 * j, v);
    }
  } else {
#pragma unroll
    for (int j = 0; j < 8; j++)
      reinterpret_cast<uint4*>(dst)[j] = make_uint4(R.w[2 * j].x, R.w[2 * j].y, R.w[2 * j + 1].x, R.w[2 * j + 1].y);
  }
}

__global__ void __launch_bounds__(kNT) k_fattn(const bf16* __restrict__ q, const bf16* __restrict__ kv, int ldkv,
                                               const RowMeta* __restrict__ meta, const uint8_t* __restrict__ ring,
                                               const float* __restrict__ sink, const float* __restrict__ inv,
                                               const float* __restrict__ U, bf16* out, float* zp) {
  extern __shared__ __align__(16) uint8_t smraw[];
  Smem& S = *reinterpret_cast<Smem*>(smraw);
  const int r = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  // meta (copied in before the pass's first kernel), the ring and U are not written by the preceding kernels
  const RowMeta mt = meta[r];
  const int nctx = min(mt.p0, kDspCtx), nblk = mt.pend - mt.p0 + 1, nk = nctx + nblk;
  const int kk = tid & 31, part = tid >> 5;  // warp = part: a quarter-warp's 16-byte stores hit 8 distinct bank groups
  const int g = tid >> 6, i64 = tid & 63, d0 = 8 * i64;
  Raw R;
  fetch(R, ring, mt, nctx, nk, kk, part);
  for (int i = tid; i < kLH * kVhaRank; i += kNT) S.U[i] = U[i];
  if (tid < nblk) S.bpos[tid] = meta[mt.r0 + tid].pos;
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // the next (the z all-reduce) waits for our end
  // raw q and the block's kv rows into shared memory (all copies in flight), then compact loops: code that runs once
  // per launch costs its instruction fetch, so nothing here is unrolled beyond a round
#pragma unroll 1
  for (int i = tid; i < kLH * (kHeadDim / 8); i += kNT)
    cp_async16(&S.q[i >> 6][8 * (i & 63)], q + ((size_t)r * kLH + (i >> 6)) * kHeadDim + 8 * (i & 63), true);
#pragma unroll 1
  for (int i = tid; i < nblk * (kHeadDim / 8); i += kNT)
    cp_async16(&S.k[i >> 6][8 * (i & 63)], kv + (size_t)(mt.r0 + (i >> 6)) * ldkv + 8 * (i & 63), true);
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();
  // q prep (k_q_prep's arithmetic): head 4·t + g in round t, 64 threads a head; one barrier for the 8 rounds' sums
#pragma unroll 1
  for (int t = 0; t < kLH / 4; t++) {
    float v[8], ss = 0.f;
    load_bf16x8(&S.q[4 * t + g][d0], v);
#pragma unroll
    for (int j = 0; j < 8; j++) ss += v[j] * v[j];
    ss = warp_sum(ss);
    if (lane == 0) S.red[t][warp] = ss;
  }
  __syncthreads();
#pragma unroll 1
  for (int t = 0; t < kLH / 4; t++) {
    float v[8];
    load_bf16x8(&S.q[4 * t + g][d0], v);
    const float rs = rsqrtf(__fdiv_rn(S.red[t][2 * g] + S.red[t][2 * g + 1], float(kHeadDim)) + kEps);
#pragma unroll
    for (int j = 0; j < 8; j++) v[j] *= rs;
    if (d0 >= kNope) rope_pairs8(v, d0 - kNope, mt.pos, inv, false);
    store_bf16x8(&S.q[4 * t + g][d0], v);
  }
  // the block's KV rows (k_kv_rows' bytes): row 4·t + g in round t (≤ 15 rows)
#pragma unroll 1
  for (int j = g; j < nblk; j += 4) {
    float v[8];
    load_bf16x8(&S.k[j][d0], v);
    if (d0 >= kNope) rope_pairs8(v, d0 - kNope, S.bpos[j], inv, false);
    store_kv_row8(S.brow[j], d0, v);
  }
  if (tid < kLH) S.m[tid] = -INFINITY, S.l[tid] = 0.f;
  float o[2][8][4];
#pragma unroll
  for (int a = 0; a < 2; a++)
#pragma unroll
    for (int b = 0; b < 8; b++) o[a][b][0] = o[a][b][1] = o[a][b][2] = o[a][b][3] = 0.f;
  const float scale = rsqrtf(float(kHeadDim));
  for (int k0 = 0; k0 < nk; k0 += kKT) {
    __syncthreads();
    put(R, S, kk, part, nctx, k0 + kk);
    if (k0 + kKT < nk) fetch(R, ring, mt, nctx, nk, k0 + kKT + kk, part);
    __syncthreads();
    {  // S = Q·Kᵀ (k_attn): warp → heads 16·(warp/4) …, keys 8·(warp%4) …
      const int mt16 = warp >> 2, nt8 = warp & 3;
      float sacc[4] = {0.f, 0.f, 0.f, 0.f};
      // the same 32 MMAs in the same order; step k16 + 1's fragments are loaded before step k16's MMA
      const bf16* qa = &S.q[16 * mt16 + (lane & 15)][(lane >> 4) * 8];
      const bf16* kb = &S.k[8 * nt8 + (lane & 7)][((lane >> 3) & 1) * 8];
      uint32_t fa[2][4], fb[2][2];
      ldsm_x4(fa[0][0], fa[0][1], fa[0][2], fa[0][3], qa);
      ldsm_x2(fb[0][0], fb[0][1], kb);
#pragma unroll 8
      for (int k16 = 0; k16 < kHeadDim / 16; k16++) {
        const int c = k16 & 1, n = c ^ 1;
        if (k16 + 1 < kHeadDim / 16) {
          ldsm_x4(fa[n][0], fa[n][1], fa[n][2], fa[n][3], qa + 16 * (k16 + 1));
          ldsm_x2(fb[n][0], fb[n][1], kb + 16 * (k16 + 1));
        }
        mma_bf16(sacc, fa[c][0], fa[c][1], fa[c][2], fa[c][3], fb[c][0], fb[c][1]);
      }
      const int gg = lane >> 2, t = lane & 3;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int h = 16 * mt16 + gg + (i >= 2 ? 8 : 0), kc = 8 * nt8 + 2 * t + (i & 1);
        S.s[h][kc] = k0 + kc < nk ? sacc[i] * scale : -INFINITY;
      }
    }
    __syncthreads();
    {  // online softmax: warp → heads 4·warp …, lane → key; each head's sum in key order by lane hh
      float mxh[4];
#pragma unroll
      for (int hh = 0; hh < 4; hh++) {
        const int h = 4 * warp + hh;
        const float sv = S.s[h][lane];
        float mx = sv;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
        mx = fmaxf(S.m[h], mx);
        const float e = expf(sv - mx);
        S.p[h][lane] = __float2bfloat16_rn(e);
        S.s[h][lane] = e;
        mxh[hh] = mx;
      }
      __syncwarp();
      if (lane < 4) {
        const int h = 4 * warp + lane;
        const float mx = lane == 0 ? mxh[0] : lane == 1 ? mxh[1] : lane == 2 ? mxh[2] : mxh[3];
        const float f = expf(S.m[h] - mx);
        float sum = 0.f;
        for (int j = 0; j < kKT; j++) sum += S.s[h][j];
        S.l[h] = S.l[h] * f + sum;
        S.m[h] = mx;
        S.fac[h] = f;
      }
    }
    __syncthreads();
    {  // O = O·fac + P·V (k_attn): warp → dims 64·warp …
      const int gg = lane >> 2;
#pragma unroll
      for (int a = 0; a < 2; a++) {
        const float f0 = S.fac[16 * a + gg], f1 = S.fac[16 * a + gg + 8];
#pragma unroll
        for (int b = 0; b < 8; b++) o[a][b][0] *= f0, o[a][b][1] *= f0, o[a][b][2] *= f1, o[a][b][3] *= f1;
      }
#pragma unroll
      for (int ks = 0; ks < kKT / 16; ks++) {  // every o[a][b] still takes ks = 0, then 1; fragments loaded first
        uint32_t pa[2][4], vb[8][2];
#pragma unroll
        for (int a = 0; a < 2; a++)
          ldsm_x4(pa[a][0], pa[a][1], pa[a][2], pa[a][3], &S.p[16 * a + (lane & 15)][16 * ks + (lane >> 4) * 8]);
#pragma unroll
        for (int b = 0; b < 8; b++) ldsm_x2_t(vb[b][0], vb[b][1], &S.k[16 * ks + (lane & 15)][64 * warp + 8 * b]);
#pragma unroll
        for (int a = 0; a < 2; a++)
#pragma unroll
          for (int b = 0; b < 8; b++) mma_bf16(o[a][b], pa[a][0], pa[a][1], pa[a][2], pa[a][3], vb[b][0], vb[b][1]);
      }
    }
  }
  __syncthreads();
  if (tid < kLH) {  // the sink (k_attn)
    const float sk = sink[tid], Mx = fmaxf(S.m[tid], sk), f = expf(S.m[tid] - Mx);
    S.fac[tid] = f, S.l[tid] = f * S.l[tid] + expf(sk - Mx);
  }
  __syncthreads();
  float* P = reinterpret_cast<float*>(&S.q[0][0]);  // fp32 [32][kPS]: fac · o (q and k are free now)
  {
    const int gg = lane >> 2, t = lane & 3;
#pragma unroll
    for (int a = 0; a < 2; a++)
#pragma unroll
      for (int b = 0; b < 8; b++) {
        const int h0 = 16 * a + gg, d = 64 * warp + 8 * b + 2 * t;
        *reinterpret_cast<float2*>(P + h0 * kPS + d) = make_float2(S.fac[h0] * o[a][b][0], S.fac[h0] * o[a][b][1]);
        *reinterpret_cast<float2*>(P + (h0 + 8) * kPS + d) =
            make_float2(S.fac[h0 + 8] * o[a][b][2], S.fac[h0 + 8] * o[a][b][3]);
      }
  }
  __syncthreads();
  // per dim: o = bf16(fac·o / l) (k_attn's output) and vha_partial's zp[r][rk][d] = Σ_h U[h][rk] · o[h][d], h in order
#pragma unroll 1
  for (int d = tid; d < kHeadDim; d += kNT) {
    float acc[kVhaRank];
#pragma unroll
    for (int rk = 0; rk < kVhaRank; rk++) acc[rk] = 0.f;
#pragma unroll 4
    for (int h = 0; h < kLH; h++) {
      const bf16 ob = __float2bfloat16_rn(__fdiv_rn(P[h * kPS + d], S.l[h]));
      out[((size_t)r * kLH + h) * kHeadDim + d] = ob;
      const float x = __bfloat162float(ob);
      const float4* u4 = reinterpret_cast<const float4*>(&S.U[h * kVhaRank]);
#pragma unroll
      for (int c = 0; c < kVhaRank / 4; c++) {
        const float4 u = u4[c];
        acc[4 * c] += u.x * x, acc[4 * c + 1] += u.y * x, acc[4 * c + 2] += u.z * x, acc[4 * c + 3] += u.w * x;
      }
    }
#pragma unroll
    for (int rk = 0; rk < kVhaRank; rk++) zp[((size_t)r * kVhaRank + rk) * kHeadDim + d] = acc[rk];
  }
}

inline void launch(const bf16* q, const bf16* kv, int ldkv, const RowMeta* meta, const uint8_t* ring,
                   const float* sink, const float* inv, const float* U, int M, bf16* o, float* zp, cudaStream_t s) {
  static bool attr = false;
  if (!attr) {
    CUDA_CHECK(cudaFuncSetAttribute(k_fattn, cudaFuncAttributeMaxDynamicSharedMemorySize, int(sizeof(Smem))));
    attr = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(M);
  cfg.blockDim = dim3(kNT);
  cfg.dynamicSmemBytes = sizeof(Smem);
  cfg.stream = s;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, k_fattn, q, kv, ldkv, meta, ring, sink, inv, U, o, zp));
}

}  // namespace p129
