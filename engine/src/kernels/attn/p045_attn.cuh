// p045_attn.cuh — P045 dsa_attn_fast: the sparse MLA attention of dsa_attn (base: dsa.cu k_mla_attn + k_mla_merge),
// selected by patch::on(dsa_attn_fast). Included by dsa.cu inside its anonymous namespace (uses its constants, the
// MMA / ldmatrix wrappers and pack_bf16x2).
//
// Bitwise contract — every output element is the base's: per (row, 256-key chunk) the same 32-key tiles in list order,
// the same scores (the same bf16 operands — e4m3 latent dequantized exactly, q576 — through the same m16n8k16 MMA
// chain over the 36 k-steps in ascending order, × 256^-0.5), the same online softmax per tile (max, f = expf(m_old −
// mx), p = expf(s − mx), P = bf16(p), the tile sum Σ p in key order from 0, l = l·f + sum, o = o·f then the two k16
// P·V MMAs), and the same chunk merge (M = max(sink, m_c), den = Σ_c expf(m_c − M)·l_c in chunk order + expf(sink −
// M), out = Σ_c expf(m_c − M)·o_c in chunk order / den, bf16). A head's numbers never depend on other heads, so a CTA
// may hold all 32 local heads or 16 of them. Only scheduling differs:
//   · scores as Sᵀ = K·Qᵀ (keys on the MMA's M side, heads on N): the same k-ordered chain of the same products, with
//     q's operand fragments held in registers for the whole chunk (the swap of the MMA operand roles was checked
//     bit-identical on 2.8e8 real and adversarial score elements, test_dsa.py / the bring-up probe). That frees the
//     shared memory of q for a second K tile: the scores of tile j+1 run interleaved with the P·V of tile j;
//   · the tile after next is gathered into registers during the current tile; the e4m3 → bf16 dequantization writes
//     whole 512-byte rows per warp instruction (no bank conflicts);
//   · the softmax runs in the score registers (cross-warp max through shared memory); only the 32-term tile sum stays
//     a sequential chain (one lane per head);
//   · prefill: a CTA runs 3 chunks of a row as one pipelined tile stream (state reset at each chunk start, partial
//     written at each chunk end); decode (few CTAs): one chunk and 16 heads per CTA, twice the CTAs;
//   · CTAs are ordered row-major (a row's chunks run together); the last CTA of a row (integer counter, reset by that
//     CTA) merges the chunk partials right away — read back from L2, then discarded there (no DRAM write-back) —
//     instead of a second kernel over partials that went through DRAM.
#pragma once
#include <cuda_fp16.h>

namespace p045 {

constexpr int kThreads = 256;
constexpr int kCpc = 3;  // most chunks per CTA (prefill)
__host__ __device__ constexpr int slab(int nh) { return 2 * nh + nh * kKvLora; }  // floats per (row, chunk, set)

template <int NH>
struct Smem {
  bf16 k[2][kKT][kQS];  // K tiles (k[1] stages q at the start)
  float e[NH][kKT + 1];
  bf16 p[NH][kKT + 8];
  float red[2][NH];
  float m[2][NH];
  float fac[NH];
  float l[NH];
  int key[kCpc * kChunk];
  float mf[10][NH];  // merge: expf(m_c − M) per chunk (≤ 9) and head
  float den[NH];
  int last;
};

// two e4m3 bytes (low byte = lower dim) · sc → bf16x2, exact
__device__ __forceinline__ uint32_t e4m3x2_bf16x2(uint32_t v16, float sc) {
  const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(v16), __NV_E4M3);
  const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h));
  return pack_bf16x2(f.x * sc, f.y * sc);
}

// a thread's share of one 32-key tile: latent bytes 16·lane … (16 dims) of keys warp + 8j, j < 4, with their scales;
// k_pe unit tid % 8 (8 dims) of key tid / 8
struct Raw {
  uint4 kv[4];
  float sc[4];
  uint4 pe;
};

__device__ __forceinline__ void discard_l2(const void* p) {
  asm volatile("discard.global.L2 [%0], 128;\n" ::"l"(p) : "memory");
}

// grid: n rows × ⌈nch / cpc⌉ chunk groups × (32 / NH) head sets (row-major); a CTA runs the tiles of its cpc chunks as
// one pipelined stream (online state reset at each chunk start, partial written at each chunk end). ws: per (row,
// chunk, set) slab [m NH][l NH][o NH × 512] fp32; ctr: per (row, set) count of finished chunks (zero at launch, zero
// again after the merge)
template <int NH>
__global__ void __launch_bounds__(kThreads, 1)
    k_attn(const bf16* __restrict__ q, const int* __restrict__ list, const int* __restrict__ cnt,
           const RowMeta* __restrict__ meta, DsaCache c, const float* __restrict__ sink, int row0, int nch, int cpc,
           float* ws, int* ctr, bf16* __restrict__ olat) {
  constexpr int HS = 32 / NH, NQ = NH / 16;  // head sets per row, m16 head tiles in P·V
  constexpr int NHG = NH / 8, NQW = 2 * NHG;  // n8 head groups; score warps (2 key halves × head groups)
  extern __shared__ __align__(16) uint8_t smraw[];
  Smem<NH>& S = *reinterpret_cast<Smem<NH>*>(smraw);
  const int ngrp = (nch + cpc - 1) / cpc;
  const int hs = blockIdx.x % HS, cg = (blockIdx.x / HS) % ngrp, ri = blockIdx.x / (HS * ngrp), r = row0 + ri;
  const int hb = hs * NH;
  const int nk = cnt[r], kbeg = cg * cpc * kChunk;
  if (kbeg >= nk) return;
  const int kend = min(nk, kbeg + cpc * kChunk), ntile = (kend - kbeg + kKT - 1) / kKT;
  constexpr int TPC = kChunk / kKT;  // tiles per chunk
  constexpr int SL = slab(NH);
  const int nchr = (nk + kChunk - 1) / kChunk;
  const RowMeta mt = meta[r];
  const uint8_t* base = c.mla + (size_t)mt.slot * c.ctx * kMR;
  const int* lst = list + (size_t)r * kMaxKeys;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int u8 = tid & 7, k8 = tid >> 3;
  const bool qkw = warp < NQW;
  const int ng = warp % NHG, kh = warp / NHG;  // score warp: heads 8·ng …, keys 16·kh …

  Raw raw;
  auto load = [&](int k0, const int* keys) {  // keys[i] = list position kbeg + i (global or shared)
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int kk = k0 + warp + 8 * j;
      if (kk < kend) {
        const uint8_t* row = base + (size_t)keys[kk - kbeg] * kMR;
        raw.kv[j] = __ldg(reinterpret_cast<const uint4*>(row) + lane);
        raw.sc[j] = __ldg(reinterpret_cast<const float*>(row + kKvLora) + (lane >> 3));
      }
    }
    const int kk = k0 + k8;
    if (kk < kend) raw.pe = __ldg(reinterpret_cast<const uint4*>(base + (size_t)keys[kk - kbeg] * kMR + kKvLora + 16) + u8);
  };
  auto store = [&](int k0, int buf) {
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int kl = warp + 8 * j;
      uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;
      if (k0 + kl < kend) {
        const float sc = raw.sc[j];
        const uint4 x = raw.kv[j];
        v0 = make_uint4(e4m3x2_bf16x2(x.x & 0xFFFF, sc), e4m3x2_bf16x2(x.x >> 16, sc), e4m3x2_bf16x2(x.y & 0xFFFF, sc),
                        e4m3x2_bf16x2(x.y >> 16, sc));
        v1 = make_uint4(e4m3x2_bf16x2(x.z & 0xFFFF, sc), e4m3x2_bf16x2(x.z >> 16, sc), e4m3x2_bf16x2(x.w & 0xFFFF, sc),
                        e4m3x2_bf16x2(x.w >> 16, sc));
      }
      uint4* d = reinterpret_cast<uint4*>(&S.k[buf][kl][16 * lane]);
      d[0] = v0, d[1] = v1;
    }
    *reinterpret_cast<uint4*>(&S.k[buf][k8][kKvLora + 8 * u8]) = k0 + k8 < kend ? raw.pe : make_uint4(0, 0, 0, 0);
  };
  // Sᵀ chain of one tile on buffer buf: keys 16·kh + (g, g+8) × heads 8·ng + (2t, 2t+1)
  uint32_t qb[kQK / 16][2];
  auto score_step = [&](float* acc, int buf, int kk) {
    uint32_t a0, a1, a2, a3;
    ldsm_x4(a0, a1, a2, a3, &S.k[buf][16 * kh + (lane & 15)][16 * kk + (lane >> 4) * 8]);
    mma_bf16(acc, a0, a1, a2, a3, qb[kk][0], qb[kk][1]);
  };

  // ---- prologue: q staged through k[1] into the score warps' registers; tile 0 → k[0]; tile 1 → k[1]
  load(kbeg, lst + kbeg);
  {  // keys and q: every load in flight before the first store
    constexpr int NQ8 = NH * kQK / 8, NQV = (NQ8 + kThreads - 1) / kThreads, NKV = kCpc * kChunk / kThreads;
    uint4 qv[NQV];
    int kv[NKV];
#pragma unroll
    for (int j = 0; j < NKV; j++) {
      const int i = tid + j * kThreads;
      kv[j] = i < kend - kbeg ? __ldg(lst + kbeg + i) : 0;
    }
#pragma unroll
    for (int j = 0; j < NQV; j++) {
      const int i = tid + j * kThreads, h = i / (kQK / 8), c8 = i % (kQK / 8);
      if (i < NQ8) qv[j] = __ldg(reinterpret_cast<const uint4*>(q + ((size_t)r * kLH + hb + h) * kQK) + c8);
    }
#pragma unroll
    for (int j = 0; j < NKV; j++) S.key[tid + j * kThreads] = kv[j];
#pragma unroll
    for (int j = 0; j < NQV; j++) {
      const int i = tid + j * kThreads, h = i / (kQK / 8), c8 = i % (kQK / 8);
      if (i < NQ8) *reinterpret_cast<uint4*>(&S.k[1][h][c8 * 8]) = qv[j];
    }
  }
  store(kbeg, 0);
  __syncthreads();
  if (qkw) {
#pragma unroll
    for (int kk = 0; kk < kQK / 16; kk++)
      ldsm_x2(qb[kk][0], qb[kk][1], &S.k[1][8 * ng + (lane & 7)][16 * kk + ((lane >> 3) & 1) * 8]);
  }
  __syncthreads();
  if (ntile > 1) load(kbeg + kKT, S.key);
  const float scale = 0.0625f;  // 256^-0.5
  const int ha = 8 * ng + 2 * t, kc = 16 * kh + g;  // this lane's heads ha, ha+1 and keys kc, kc+8 (tile-relative)
  // scores of the tile at k0 (Sᵀ fragment acc) → sv (scaled, masked) and this key half's per-head maxima → red
  float sv[4];
  auto maxima = [&](const float* acc, int k0) {
#pragma unroll
    for (int i = 0; i < 4; i++) sv[i] = k0 + kc + (i >> 1) * 8 < kend ? acc[i] * scale : -INFINITY;
    float mA = fmaxf(sv[0], sv[2]), mB = fmaxf(sv[1], sv[3]);
#pragma unroll
    for (int off = 4; off < 32; off <<= 1)
      mA = fmaxf(mA, __shfl_xor_sync(0xffffffffu, mA, off)), mB = fmaxf(mB, __shfl_xor_sync(0xffffffffu, mB, off));
    if (g == 0) S.red[kh][ha] = mA, S.red[kh][ha + 1] = mB;
  };
  if (qkw) {
    float sacc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int kk = 0; kk < kQK / 16; kk++) score_step(sacc, 0, kk);
    maxima(sacc, kbeg);
  }
  if (ntile > 1) store(kbeg + kKT, 1);
  __syncthreads();

  float o[NQ][8][4];
#pragma unroll
  for (int a = 0; a < NQ; a++)
#pragma unroll
    for (int b = 0; b < 8; b++) o[a][b][0] = o[a][b][1] = o[a][b][2] = o[a][b][3] = 0.f;

  for (int ti = 0; ti < ntile; ti++) {
    const int k0 = kbeg + ti * kKT, par = ti & 1, cur = ti & 1;
    const bool first = ti % TPC == 0, lastt = ti % TPC == TPC - 1 || ti == ntile - 1;  // chunk start / end
    if (ti + 2 < ntile) load(k0 + 2 * kKT, S.key);
    // ---- (B) softmax of the tile in the score registers
    if (qkw) {
#pragma unroll
      for (int hh = 0; hh < 2; hh++) {
        const int h = ha + hh;
        const float mo = first ? -INFINITY : S.m[par][h];
        const float mx = fmaxf(mo, fmaxf(S.red[0][h], S.red[1][h]));
        const float e0 = expf(sv[hh] - mx), e1 = expf(sv[2 + hh] - mx);
        S.e[h][kc] = e0, S.e[h][kc + 8] = e1;
        S.p[h][kc] = __float2bfloat16_rn(e0), S.p[h][kc + 8] = __float2bfloat16_rn(e1);
        if (kh == 0 && g == 0) S.fac[h] = expf(mo - mx), S.m[par ^ 1][h] = mx;
      }
    }
    __syncthreads();  // (B)
    // ---- (C) o = o·f + P·V of tile ti, interleaved with the scores of tile ti+1; l of tile ti
    if (warp == 7 && lane < NH) {  // the tile sum in key order (base), then l·f + sum
      const int h = lane;
      float sum = 0.f;
#pragma unroll
      for (int j = 0; j < kKT; j++) sum += S.e[h][j];
      S.l[h] = (first ? 0.f : S.l[h]) * S.fac[h] + sum;
    }
#pragma unroll
    for (int a = 0; a < NQ; a++) {
      const float f0 = S.fac[16 * a + g], f1 = S.fac[16 * a + g + 8];
#pragma unroll
      for (int b = 0; b < 8; b++) o[a][b][0] *= f0, o[a][b][1] *= f0, o[a][b][2] *= f1, o[a][b][3] *= f1;
    }
    const bool nxt = qkw && ti + 1 < ntile;
    float nacc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int ks = 0; ks < kKT / 16; ks++) {
#pragma unroll
      for (int a = 0; a < NQ; a++) {
        uint32_t a0, a1, a2, a3;
        ldsm_x4(a0, a1, a2, a3, &S.p[16 * a + (lane & 15)][16 * ks + (lane >> 4) * 8]);
#pragma unroll
        for (int b = 0; b < 8; b++) {
          uint32_t b0, b1;
          ldsm_x2_t(b0, b1, &S.k[cur][16 * ks + (lane & 15)][64 * warp + 8 * b]);
          mma_bf16(o[a][b], a0, a1, a2, a3, b0, b1);
          const int kk = (ks * NQ + a) * 8 + b;  // 0 … 8·2·NQ − 1: one score step per P·V MMA
          if (nxt && kk < kQK / 16) score_step(nacc, cur ^ 1, kk);
        }
      }
    }
    if (nxt) {
#pragma unroll
      for (int kk = 16 * NQ; kk < kQK / 16; kk++) score_step(nacc, cur ^ 1, kk);
      maxima(nacc, k0 + kKT);  // (A) of tile ti+1: red is free (read by (B) before the barrier)
    }
    __syncthreads();  // (C) k[cur], P, e free; maxima of tile ti+1 posted
    if (lastt) {  // ---- partial of this chunk → workspace slab; fresh state for the next chunk
      float* w = ws + (((size_t)ri * nch + kbeg / kChunk + ti / TPC) * HS + hs) * SL;
      if (tid < NH) w[tid] = S.m[par ^ 1][tid], w[NH + tid] = S.l[tid];
      float* wo = w + 2 * NH;
#pragma unroll
      for (int a = 0; a < NQ; a++)
#pragma unroll
        for (int b = 0; b < 8; b++) {
          const int hr = 16 * a + g, d = 64 * warp + 8 * b + 2 * t;
          *reinterpret_cast<float2*>(wo + hr * kKvLora + d) = make_float2(o[a][b][0], o[a][b][1]);
          *reinterpret_cast<float2*>(wo + (hr + 8) * kKvLora + d) = make_float2(o[a][b][2], o[a][b][3]);
          o[a][b][0] = o[a][b][1] = o[a][b][2] = o[a][b][3] = 0.f;
        }
    }
    // ---- (D) tile ti+2 → k[cur]
    if (ti + 2 < ntile) store(k0 + 2 * kKT, cur);
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    const int mine = (ntile + TPC - 1) / TPC;
    S.last = atomicAdd(ctr + ri * HS + hs, mine) + mine == nchr;
  }
  __syncthreads();
  if (!S.last) return;
  __threadfence();
  // ---- the last CTA of (row, head set): merge the chunk partials (base k_mla_merge arithmetic)
  const float* wr = ws + (size_t)ri * nch * HS * SL + (size_t)hs * SL;
  const size_t cs = (size_t)HS * SL;  // chunk stride
  if (tid < NH) {
    const int h = tid;
    const float sk = sink[hb + h];
    float Mx = sk;
    for (int cc = 0; cc < nchr; cc++) Mx = fmaxf(Mx, __ldcg(wr + cc * cs + h));
    float den = 0.f;
    for (int cc = 0; cc < nchr; cc++) den += expf(__ldcg(wr + cc * cs + h) - Mx) * __ldcg(wr + cc * cs + NH + h);
    den += expf(sk - Mx);
    for (int cc = 0; cc < nchr; cc++) S.mf[cc][h] = expf(__ldcg(wr + cc * cs + h) - Mx);
    S.den[h] = den;
  }
  __syncthreads();
  for (int i = tid; i < NH * kKvLora / 4; i += kThreads) {
    const int h = (4 * i) / kKvLora;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    for (int cc = 0; cc < nchr; cc++) {
      const float f = S.mf[cc][h];
      const float4 x = __ldcg(reinterpret_cast<const float4*>(wr + cc * cs + 2 * NH) + i);
      a0 += f * x.x, a1 += f * x.y, a2 += f * x.z, a3 += f * x.w;
    }
    const float den = S.den[h];
    *reinterpret_cast<uint2*>(olat + ((size_t)r * kLH + hb) * kKvLora + 4 * i) =
        make_uint2(pack_bf16x2(__fdiv_rn(a0, den), __fdiv_rn(a1, den)), pack_bf16x2(__fdiv_rn(a2, den), __fdiv_rn(a3, den)));
  }
  __syncthreads();
  for (int cc = 0; cc < nchr; cc++) {
    const char* wb = reinterpret_cast<const char*>(wr + cc * cs);
    for (int i = tid; i < SL * 4 / 128; i += kThreads) discard_l2(wb + (size_t)i * 128);
  }
  if (tid == 0) ctr[ri * HS + hs] = 0;
}

template <int NH>
void launch(const bf16* q576, const int* list, const int* cnt, const RowMeta* meta, DsaCache c, const float* sink,
            int M, int nch, int cpc, bf16* olat, float* ws, size_t ws_bytes, int* ctr, cudaStream_t s) {
  constexpr int HS = 32 / NH;
  const size_t per_row = (size_t)nch * HS * slab(NH) * 4;
  if (ws_bytes < per_row) qlog::fail("dsa", "attention workspace too small");
  const int pass = int(std::min<size_t>(M, ws_bytes / per_row));
  static bool attr = false;
  if (!attr) {
    CUDA_CHECK(cudaFuncSetAttribute(k_attn<NH>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(sizeof(Smem<NH>))));
    attr = true;
  }
  for (int r0 = 0; r0 < M; r0 += pass) {
    const int n = std::min(pass, M - r0);
    k_attn<NH><<<n * ((nch + cpc - 1) / cpc) * HS, kThreads, sizeof(Smem<NH>), s>>>(q576, list, cnt, meta, c, sink, r0,
                                                                                    nch, cpc, ws, ctr, olat);
    CUDA_CHECK(cudaGetLastError());
  }
}

inline void run(const bf16* q576, const int* list, const int* cnt, const RowMeta* meta, DsaCache c, const float* sink,
                int M, int max_keys, bf16* olat, float* ws, size_t ws_bytes, cudaStream_t s) {
  // per (row, head set) chunk counters: zero when allocated, every merge resets its own — they stay zero between
  // launches, so no memset launch (sized for the largest pass: M rows × 2 sets)
  static int* ctr = nullptr;
  static size_t nctr = 0;
  static int nsm = 0;
  if (!nsm) CUDA_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  if ((size_t)M * 2 > nctr) {
    if (ctr) CUDA_CHECK(cudaFree(ctr));
    nctr = std::max<size_t>((size_t)M * 2, 1 << 14);
    CUDA_CHECK(cudaMalloc(&ctr, nctr * 4));
    CUDA_CHECK(cudaMemset(ctr, 0, nctr * 4));
  }
  const int nch = std::max(1, (max_keys + kChunk - 1) / kChunk);
  // few CTAs (decode): 16 heads per CTA; many (prefill): 3 chunks per CTA (the prologue amortized over 24 tiles)
  const int cpc = M * nch >= 8 * nsm ? kCpc : 1;
  if (2 * M * nch <= nsm) launch<16>(q576, list, cnt, meta, c, sink, M, nch, 1, olat, ws, ws_bytes, ctr, s);
  else launch<32>(q576, list, cnt, meta, c, sink, M, nch, cpc, olat, ws, ws_bytes, ctr, s);
}

}  // namespace p045
