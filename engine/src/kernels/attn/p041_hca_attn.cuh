// p041_hca_attn.cuh — P041 hca_attn_tc (included by hca.cu): HCA attention with the base kernel's arithmetic, bit for
// bit (k_attn + k_merge: same fixed chunks of 256 keys, same 32-key softmax tiles, same mma.sync m16n8k16 chains in
// the same order, same merge order), restructured for speed:
//   - one CTA = one query row × all 32 heads × a slice of DS output dims (M ≤ 8: DS = 64, 8 CTAs per row; else
//     DS = 512, persistent CTAs over rows); the chunks of a row run inside the CTA and are merged there in chunk order
//     (earlier chunks' fp32 partials go to a per-CTA workspace slot, L2-resident), so no partial round trip for
//     single-chunk rows;
//   - Q·Kᵀ takes K straight from the 584-byte rows (e4m3 → bf16 · scale in registers, the base's dequantization), 128
//     keys per pass, one 8-key n-tile per warp for both 16-head m-tiles; scores to shared memory;
//   - the 32-key tiles' online-softmax statistics of a pass are computed by all threads at once (prefix max over
//     tiles; the per-tile sum is the base's sequential sum, handed lane to lane);
//   - P·V over the slice from dequantized V rows in shared memory;
//   - optional fusions (P042): q_prep on load (q read raw, never written back) and vha_partial on the merged output.
// Helpers copied from attn_common.cuh (rope, KV row parts) so this file does not depend on its later changes.
#pragma once

namespace e55 {
namespace p041 {

constexpr int kLH = 32;           // local heads
constexpr int kD = kHeadDim;      // 512
constexpr int kCh = 256;          // keys per chunk (base kChunk)
constexpr int kKT = 32;           // keys per softmax tile (base kKT)
constexpr int kBK = 128;          // keys per pass (Q·Kᵀ / softmax granularity) = 4 tiles
constexpr int kTh = 512;          // threads per CTA (16 warps)
constexpr int kQS = kD + 8;       // Q smem row, bf16 (1040 B: conflict-free ldmatrix)
constexpr int kSS = kBK + 4;      // score row, fp32
constexpr int kPS = kBK + 8;      // P row, bf16
constexpr int kRB = kKvRowBytes;  // 584

template <int DS>
struct Cfg {
  static constexpr int VK = DS <= 128 ? kBK : kKT;  // keys in the V buffer: a whole pass (narrow slices) or one tile
  static constexpr int VS = DS + 8;                 // V smem row, bf16
  static constexpr int ACC = DS / 64;               // accumulator tiles (16 × 8) per warp
  static constexpr int SEG = DS / 16;               // 16-dim segments per key
  static constexpr int VJOBS = VK * SEG / kTh;      // V segments per thread per fill (≥ 1)
};

template <int DS>
struct Smem {
  bf16 q[kLH][kQS];  // prepped Q; after the last pass: the merged output (bf16) for the VHA partial
  float s[kLH][kSS];
  bf16 p[kLH][kPS];
  bf16 v[Cfg<DS>::VK][Cfg<DS>::VS];
  float fac[kLH][4];
  float m[kLH], l[kLH];  // running statistics of the current chunk
  float M[kLH], den[kLH], flast[kLH];
  float red[64];
  float rcs[32], rsn[32];  // the row's RoPE angles (q prep)
  float U[kLH * kVhaRank];
};

// ---- copies of attn_common.cuh helpers (same arithmetic) -------------------------------------------------------
__device__ __forceinline__ void ld_bf16x8(const bf16* p, float v[8]) {
  const uint4 u = *reinterpret_cast<const uint4*>(p);
  const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
  for (int i = 0; i < 4; i++) v[2 * i] = bf16_lo(w[i]), v[2 * i + 1] = bf16_hi(w[i]);
}
__device__ __forceinline__ void st_bf16x8(bf16* p, const float v[8]) {
  *reinterpret_cast<uint4*>(p) =
      make_uint4(pack_bf16x2(v[0], v[1]), pack_bf16x2(v[2], v[3]), pack_bf16x2(v[4], v[5]), pack_bf16x2(v[6], v[7]));
}
// attn_common.cuh rope_pairs8 with the angles' cos / sin from a table (pairs i0 … i0+3): the same products
__device__ __forceinline__ void rope8t(float v[8], int i0, const float* cs, const float* sn) {
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const float c = cs[i0 + j], s = sn[i0 + j];
    const float e = v[2 * j], o = v[2 * j + 1];
    v[2 * j] = __fsub_rn(__fmul_rn(e, c), __fmul_rn(o, s));
    v[2 * j + 1] = __fadd_rn(__fmul_rn(o, c), __fmul_rn(e, s));
  }
}
__device__ __forceinline__ void ldsm4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(a));
}
__device__ __forceinline__ void ldsm2t(uint32_t& r0, uint32_t& r1, const void* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(a));
}
// two e4m3 codes (low byte first) · scale → bf16 pair: the base dequantization (float(e4m3) · 2^e, exact, bf16 RN)
__device__ __forceinline__ uint32_t e4m3x2_bf16x2(uint32_t two, float s) {
  __nv_fp8_e4m3 a, b;
  a.__x = two & 0xFF, b.__x = (two >> 8) & 0xFF;
  return pack_bf16x2(float(a) * s, float(b) * s);
}

// the lane's bytes of one K row for the Q·Kᵀ B fragments (lane (g, t4) of an n-tile: key g, k-step dims 2t4, 2t4+1,
// 2t4+8, 2t4+9): nope k-step kk → one word {e4m3 pair at 16kk + 2t4, pair at 16kk + 2t4 + 8}; rope → the bf16 pairs;
// the 7 scales. Every load is issued before any is used. No row (key past the chunk) → zeros, which convert to the
// same +0 the base's zero-filled tile gives.
struct KRaw {
  uint32_t nope[kNope / 16], rope[8];
  uint2 sc;
  __device__ __forceinline__ void load(const uint8_t* row, int t4) {
    if (row) {
      sc = *reinterpret_cast<const uint2*>(row + 576);
#pragma unroll
      for (int kk = 0; kk < kNope / 16; kk++)
        nope[kk] = uint32_t(*reinterpret_cast<const uint16_t*>(row + 16 * kk + 2 * t4)) |
                   uint32_t(*reinterpret_cast<const uint16_t*>(row + 16 * kk + 2 * t4 + 8)) << 16;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const uint8_t* rp = row + kNope + 2 * (16 * j + 2 * t4);
        rope[2 * j] = *reinterpret_cast<const uint32_t*>(rp);
        rope[2 * j + 1] = *reinterpret_cast<const uint32_t*>(rp + 16);
      }
    } else {
      sc = make_uint2(0, 0);
#pragma unroll
      for (int kk = 0; kk < kNope / 16; kk++) nope[kk] = 0;
#pragma unroll
      for (int j = 0; j < 8; j++) rope[j] = 0;
    }
  }
  __device__ __forceinline__ void frag(int kk, uint32_t& b0, uint32_t& b1) const {
    if (kk < kNope / 16) {
      const int part = kk >> 2;
      const float s = ue8m0_f32(((part < 4 ? sc.x : sc.y) >> (8 * (part & 3))) & 0xFF);
      b0 = e4m3x2_bf16x2(nope[kk] & 0xFFFF, s);
      b1 = e4m3x2_bf16x2(nope[kk] >> 16, s);
    } else {
      b0 = rope[2 * (kk - kNope / 16)], b1 = rope[2 * (kk - kNope / 16) + 1];
    }
  }
};

// sacc[m] = Σ_kk Q[m-tile m0 + m] · Kᵀ over the 32 k-steps in order (the base's chain), A / B one k-step ahead
template <int NM>
__device__ __forceinline__ void qk_chain(const bf16 (*qs)[kQS], int m0, const KRaw& kr, int lane, float (&sacc)[NM][4]) {
  uint32_t bb[2][2], aa[2][NM][4];
  kr.frag(0, bb[0][0], bb[0][1]);
#pragma unroll
  for (int m = 0; m < NM; m++)
    ldsm4(aa[0][m][0], aa[0][m][1], aa[0][m][2], aa[0][m][3], &qs[16 * (m0 + m) + (lane & 15)][(lane >> 4) * 8]);
#pragma unroll
  for (int kk = 0; kk < kD / 16; kk++) {
    const int cu = kk & 1;
    if (kk + 1 < kD / 16) {
      kr.frag(kk + 1, bb[cu ^ 1][0], bb[cu ^ 1][1]);
#pragma unroll
      for (int m = 0; m < NM; m++)
        ldsm4(aa[cu ^ 1][m][0], aa[cu ^ 1][m][1], aa[cu ^ 1][m][2], aa[cu ^ 1][m][3],
              &qs[16 * (m0 + m) + (lane & 15)][16 * (kk + 1) + (lane >> 4) * 8]);
    }
#pragma unroll
    for (int m = 0; m < NM; m++)
      mma_bf16(sacc[m], aa[cu][m][0], aa[cu][m][1], aa[cu][m][2], aa[cu][m][3], bb[cu][0], bb[cu][1]);
  }
}

// the row of key k (0 … nk−1) of a query at position p: compressed entries, then the window (this step's rows or ring)
__device__ __forceinline__ const uint8_t* key_row(int k, int ncomp, int nwin, int p, const RowMeta& mt,
                                                  const uint8_t* rows, const HcaCache& c) {
  if (k < ncomp) return c.comp + ((size_t)mt.slot * c.max_comp + k) * kRB;
  const int t = p - nwin + 1 + (k - ncomp);
  return t >= mt.p0 ? rows + (size_t)(mt.r0 + t - mt.p0) * kRB
                    : c.ring + ((size_t)mt.slot * kWindow + t % kWindow) * kRB;
}

// V rows of keys [k0, k0 + VK) restricted to dims [d0, d0 + DS) → registers (raw) → shared (bf16), keys ≥ kend zero
template <int DS>
struct VFill {
  uint2 u[Cfg<DS>::VJOBS][4];
  float sc[Cfg<DS>::VJOBS];
  __device__ __forceinline__ void load(int k0, int kend, int d0, int ncomp, int nwin, int p, const RowMeta& mt,
                                       const uint8_t* rows, const HcaCache& c) {
#pragma unroll
    for (int j = 0; j < Cfg<DS>::VJOBS; j++) {
      const int job = threadIdx.x + j * kTh, kk = job / Cfg<DS>::SEG, d = d0 + 16 * (job % Cfg<DS>::SEG);
      const int key = k0 + kk;
      if (key < kend) {
        const uint8_t* row = key_row(key, ncomp, nwin, p, mt, rows, c);
        if (d < kNope) {
          const uint2* src = reinterpret_cast<const uint2*>(row + d);
          u[j][0] = src[0], u[j][1] = src[1];
          sc[j] = ue8m0_f32(row[576 + d / 64]);
        } else {
          const uint2* src = reinterpret_cast<const uint2*>(row + kNope + 2 * (d - kNope));
          u[j][0] = src[0], u[j][1] = src[1], u[j][2] = src[2], u[j][3] = src[3];
        }
      }
    }
  }
  __device__ __forceinline__ void store(Smem<DS>& S, int k0, int kend, int d0) {
#pragma unroll
    for (int j = 0; j < Cfg<DS>::VJOBS; j++) {
      const int job = threadIdx.x + j * kTh, kk = job / Cfg<DS>::SEG, seg = job % Cfg<DS>::SEG, d = d0 + 16 * seg;
      uint4* dst = reinterpret_cast<uint4*>(&S.v[kk][16 * seg]);
      if (k0 + kk >= kend) {
        dst[0] = dst[1] = make_uint4(0, 0, 0, 0);
      } else if (d < kNope) {
        const float s = sc[j];
        float v[16];
#pragma unroll
        for (int h = 0; h < 2; h++)
#pragma unroll
          for (int i = 0; i < 4; i++) {
            __nv_fp8_e4m3 a, b;
            a.__x = (u[j][h].x >> (8 * i)) & 0xFF, b.__x = (u[j][h].y >> (8 * i)) & 0xFF;
            v[8 * h + i] = float(a) * s, v[8 * h + 4 + i] = float(b) * s;
          }
        dst[0] = make_uint4(pack_bf16x2(v[0], v[1]), pack_bf16x2(v[2], v[3]), pack_bf16x2(v[4], v[5]),
                            pack_bf16x2(v[6], v[7]));
        dst[1] = make_uint4(pack_bf16x2(v[8], v[9]), pack_bf16x2(v[10], v[11]), pack_bf16x2(v[12], v[13]),
                            pack_bf16x2(v[14], v[15]));
      } else {
        dst[0] = make_uint4(u[j][0].x, u[j][0].y, u[j][1].x, u[j][1].y);
        dst[1] = make_uint4(u[j][2].x, u[j][2].y, u[j][3].x, u[j][3].y);
      }
    }
  }
};

// grid (row CTAs, 512 / DS slices); CTA x handles rows x, x + gridDim.x, …
template <int DS, bool PREP, bool VHA>
__global__ void __launch_bounds__(kTh, 1)
    k_attn(const bf16* __restrict__ q, const uint8_t* __restrict__ rows, const RowMeta* __restrict__ meta, HcaCache c,
           const float* __restrict__ sink, const float* __restrict__ inv, const float* __restrict__ U, int M,
           bf16* __restrict__ o, float* __restrict__ zp, float* __restrict__ ws, size_t ws_slot) {
  using C = Cfg<DS>;
  extern __shared__ __align__(16) uint8_t smraw[];
  Smem<DS>& S = *reinterpret_cast<Smem<DS>*>(smraw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
  const int d0 = blockIdx.y * DS;
  float* wsl = ws + ws_slot * (blockIdx.x * gridDim.y + blockIdx.y);
  if (VHA)
    for (int i = tid; i < kLH * kVhaRank; i += kTh) S.U[i] = U[i];
  const float scale = rsqrtf(float(kD));

  for (int r = blockIdx.x; r < M; r += gridDim.x) {
    const RowMeta mt = meta[r];
    const int p = mt.pos, ncomp = (p + 1) / kRatio, nwin = min(p + 1, kWindow), nk = ncomp + nwin;
    const int nch = (nk + kCh - 1) / kCh;
    __syncthreads();  // the previous row is done with shared memory
    // ---- Q (32 heads × 512) → shared, prepped like k_q_prep when PREP --------------------------------------------
    {
      const bf16* qr = q + (size_t)r * kLH * kD;
      const int i = tid & 63, d = 8 * i;
      float v[4][8];
#pragma unroll
      for (int it = 0; it < 4; it++) ld_bf16x8(qr + (it * 8 + (tid >> 6)) * kD + d, v[it]);
      if (r + (int)gridDim.x < M && tid < kLH * kD * 2 / 128)  // the CTA's next row's q → L2
        asm volatile("prefetch.global.L2 [%0];" ::"l"(q + (size_t)(r + gridDim.x) * kLH * kD + tid * 64));
      if (PREP) {
        if (tid < 32) {  // the base's angles fl(pos · inv[i]), once per row instead of once per head
          float sn, cs;
          sincosf(__fmul_rn(float(p), inv[tid]), &sn, &cs);
          S.rcs[tid] = cs, S.rsn[tid] = sn;
        }
#pragma unroll
        for (int it = 0; it < 4; it++) {
          float ss = 0.f;
#pragma unroll
          for (int j = 0; j < 8; j++) ss += v[it][j] * v[it][j];
          ss = warp_sum(ss);
          if (lane == 0) S.red[it * 16 + warp] = ss;
        }
        __syncthreads();
#pragma unroll
        for (int it = 0; it < 4; it++) {
          const float rs =
              rsqrtf(__fdiv_rn(S.red[it * 16 + (warp & ~1)] + S.red[it * 16 + (warp | 1)], float(kD)) + kEps);
#pragma unroll
          for (int j = 0; j < 8; j++) v[it][j] *= rs;
          if (d >= kNope) rope8t(v[it], (d - kNope) / 2, S.rcs, S.rsn);
        }
      }
#pragma unroll
      for (int it = 0; it < 4; it++) st_bf16x8(&S.q[it * 8 + (tid >> 6)][d], v[it]);
    }
    __syncthreads();
    float acc[C::ACC][4];
    VFill<DS> vf;
#pragma unroll 1
    for (int ch = 0; ch < nch; ch++) {
      const int kc0 = ch * kCh, kc1 = min(nk, kc0 + kCh);
      if (tid < kLH) S.m[tid] = -INFINITY, S.l[tid] = 0.f;
#pragma unroll
      for (int i = 0; i < C::ACC; i++) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
#pragma unroll 1
      for (int kb = kc0; kb < kc1; kb += kBK) {
        const int kbe = min(kc1, kb + kBK), ntl = (kbe - kb + kKT - 1) / kKT;
        vf.load(kb, kbe, d0, ncomp, nwin, p, mt, rows, c);  // first V fill of the pass (whole pass or tile 0)
        __syncthreads();  // Q ready; previous pass done with s / p / v
        // ---- S = Q·Kᵀ: warp → keys kb + 8·warp …, both m-tiles ----------------------------------------------------
        {
          const int n0 = kb + 8 * warp;
          if (n0 < kb + kKT * ntl) {
            float s0[4] = {0.f, 0.f, 0.f, 0.f}, s1[4] = {0.f, 0.f, 0.f, 0.f};
            if (n0 < kbe) {
              const int key = n0 + g;
              KRaw kr;
              kr.load(key < kbe ? key_row(key, ncomp, nwin, p, mt, rows, c) : nullptr, t4);
              float sc2[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
              qk_chain<2>(S.q, 0, kr, lane, sc2);
#pragma unroll
              for (int i = 0; i < 4; i++) s0[i] = sc2[0][i], s1[i] = sc2[1][i];
            }
#pragma unroll
            for (int i = 0; i < 4; i++) {
              const int col = 8 * warp + 2 * t4 + (i & 1), h = g + (i >= 2 ? 8 : 0);
              const bool ok = kb + col < kbe;
              S.s[h][col] = ok ? s0[i] * scale : -INFINITY;
              S.s[16 + h][col] = ok ? s1[i] * scale : -INFINITY;
            }
          }
        }
        __syncthreads();
        if (C::VK == kBK) vf.store(S, kb, kbe, d0);
        __syncthreads();
        // ---- softmax statistics of the pass's tiles: warp → heads 2·warp, 2·warp+1; lane → (tile, 8 keys) ------------
        {
          const int h = 2 * warp + (lane >> 4), tt = (lane >> 2) & 3, q4 = lane & 3, hb = lane & 16;
          const bool tok = tt < ntl;
          float sv[8], tm = -INFINITY;
          {
            const float4* sp = reinterpret_cast<const float4*>(&S.s[h][32 * tt + 8 * q4]);
            const float4 x0 = sp[0], x1 = sp[1];
            sv[0] = x0.x, sv[1] = x0.y, sv[2] = x0.z, sv[3] = x0.w, sv[4] = x1.x, sv[5] = x1.y, sv[6] = x1.z,
            sv[7] = x1.w;
          }
#pragma unroll
          for (int i = 0; i < 8; i++) tm = fmaxf(tm, sv[i]);
          tm = fmaxf(tm, __shfl_xor_sync(0xffffffffu, tm, 1));
          tm = fmaxf(tm, __shfl_xor_sync(0xffffffffu, tm, 2));
          const float m_in = S.m[h], l_in = S.l[h];
          float mprev = m_in, mx = m_in;  // running max before / after this lane's tile
#pragma unroll
          for (int j = 0; j < 4; j++) {
            const float x = __shfl_sync(0xffffffffu, tm, hb + 4 * j);
            if (j < ntl && j <= tt) mprev = mx, mx = fmaxf(mx, x);
          }
          float e[8];
#pragma unroll
          for (int i = 0; i < 8; i++) e[i] = expf(sv[i] - mx);
          if (tok)
            *reinterpret_cast<uint4*>(&S.p[h][32 * tt + 8 * q4]) =
                make_uint4(pack_bf16x2(e[0], e[1]), pack_bf16x2(e[2], e[3]), pack_bf16x2(e[4], e[5]),
                           pack_bf16x2(e[6], e[7]));
          // the tile's sum, 0 + e_0 + … + e_31 in key order: each lane adds its 8, then hands on
          float sum = 0.f;
#pragma unroll
          for (int st = 0; st < 4; st++) {
            if (q4 == st)
#pragma unroll
              for (int i = 0; i < 8; i++) sum += e[i];
            sum = __shfl_sync(0xffffffffu, sum, (lane & ~3) | st);
          }
          const float f = expf(mprev - mx);
          if (q4 == 0 && tok) S.fac[h][tt] = f;
          float l = l_in, mlast = m_in;
#pragma unroll
          for (int j = 0; j < 4; j++) {
            const float fj = __shfl_sync(0xffffffffu, f, hb + 4 * j), sj = __shfl_sync(0xffffffffu, sum, hb + 4 * j);
            const float mj = __shfl_sync(0xffffffffu, mx, hb + 4 * j);
            if (j < ntl) l = l * fj + sj, mlast = mj;
          }
          __syncwarp();
          if ((lane & 15) == 0) S.m[h] = mlast, S.l[h] = l;
        }
        __syncthreads();
        // ---- O = O·fac + P·V over the pass's tiles ----------------------------------------------------------------
#pragma unroll 1
        for (int tt = 0; tt < ntl; tt++) {
          if (C::VK == kKT) {  // one V tile at a time: tile tt was loaded before; store it, prefetch tile tt + 1
            if (tt > 0) __syncthreads();  // everyone done with tile tt − 1
            vf.store(S, kb + kKT * tt, kbe, d0);
            if (tt + 1 < ntl) vf.load(kb + kKT * (tt + 1), kbe, d0, ncomp, nwin, p, mt, rows, c);
            __syncthreads();
          }
          const int vr = C::VK == kKT ? 0 : kKT * tt;
#pragma unroll
          for (int i = 0; i < C::ACC; i++) {
            const int id = warp * C::ACC + i, a = id & 1;
            const float f0 = S.fac[16 * a + g][tt], f1 = S.fac[16 * a + g + 8][tt];
            acc[i][0] *= f0, acc[i][1] *= f0, acc[i][2] *= f1, acc[i][3] *= f1;
          }
#pragma unroll
          for (int ks = 0; ks < 2; ks++) {
            uint32_t pa[2][4];
            if (C::ACC >= 2 || (warp & 1) == 0)
              ldsm4(pa[0][0], pa[0][1], pa[0][2], pa[0][3], &S.p[lane & 15][32 * tt + 16 * ks + (lane >> 4) * 8]);
            if (C::ACC >= 2 || (warp & 1) == 1)
              ldsm4(pa[1][0], pa[1][1], pa[1][2], pa[1][3],
                    &S.p[16 + (lane & 15)][32 * tt + 16 * ks + (lane >> 4) * 8]);
#pragma unroll
            for (int i = 0; i < C::ACC; i++) {
              const int id = warp * C::ACC + i, a = id & 1, b = id >> 1;
              uint32_t b0, b1;
              ldsm2t(b0, b1, &S.v[vr + 16 * ks + (lane & 15)][8 * b]);
              mma_bf16(acc[i], pa[a][0], pa[a][1], pa[a][2], pa[a][3], b0, b1);
            }
          }
        }
        __syncthreads();
      }
      if (ch + 1 < nch) {  // an earlier chunk: its partial (m, l, O) → this CTA's workspace slot
        float* w = wsl + (size_t)ch * (2 * kLH + kLH * DS);
        __syncthreads();
        if (tid < kLH) w[tid] = S.m[tid], w[kLH + tid] = S.l[tid];
#pragma unroll
        for (int i = 0; i < C::ACC; i++) {
          const int id = warp * C::ACC + i, a = id & 1, b = id >> 1, h0 = 16 * a + g, d = 8 * b + 2 * t4;
          float* wo = w + 2 * kLH;
          *reinterpret_cast<float2*>(&wo[h0 * DS + d]) = make_float2(acc[i][0], acc[i][1]);
          *reinterpret_cast<float2*>(&wo[(h0 + 8) * DS + d]) = make_float2(acc[i][2], acc[i][3]);
        }
      }
    }
    // ---- merge the chunks with the sink, in chunk order (k_merge) ---------------------------------------------------
    __syncthreads();
    if (tid < kLH) {
      const int h = tid;
      const size_t cs = 2 * kLH + kLH * DS;
      float Mx = sink[h];
      for (int cc = 0; cc + 1 < nch; cc++) Mx = fmaxf(Mx, wsl[cc * cs + h]);
      Mx = fmaxf(Mx, S.m[h]);
      float den = 0.f;
      for (int cc = 0; cc + 1 < nch; cc++) den += expf(wsl[cc * cs + h] - Mx) * wsl[cc * cs + kLH + h];
      den += expf(S.m[h] - Mx) * S.l[h];
      den += expf(sink[h] - Mx);
      S.M[h] = Mx, S.den[h] = den, S.flast[h] = expf(S.m[h] - Mx);
    }
    __syncthreads();
    bf16* ob = &S.q[0][0];  // merged output [32][kQS] (Q is no longer needed)
    {
      float x[C::ACC][4];
#pragma unroll
      for (int i = 0; i < C::ACC; i++) x[i][0] = x[i][1] = x[i][2] = x[i][3] = 0.f;
      for (int cc = 0; cc + 1 < nch; cc++) {  // earlier chunks from the workspace, in chunk order
        const float* w = wsl + (size_t)cc * (2 * kLH + kLH * DS);
        float2 v[C::ACC][2];
#pragma unroll
        for (int i = 0; i < C::ACC; i++) {
          const int id = warp * C::ACC + i, a = id & 1, b = id >> 1, d = 8 * b + 2 * t4, h = 16 * a + g;
          v[i][0] = *reinterpret_cast<const float2*>(&w[2 * kLH + h * DS + d]);
          v[i][1] = *reinterpret_cast<const float2*>(&w[2 * kLH + (h + 8) * DS + d]);
        }
        float f[2][2];
#pragma unroll
        for (int a = 0; a < 2; a++)
#pragma unroll
          for (int hh = 0; hh < 2; hh++) {
            const int h = 16 * a + g + 8 * hh;
            f[a][hh] = expf(w[h] - S.M[h]);
          }
#pragma unroll
        for (int i = 0; i < C::ACC; i++) {
          const int a = (warp * C::ACC + i) & 1;
          x[i][0] += f[a][0] * v[i][0].x, x[i][1] += f[a][0] * v[i][0].y;
          x[i][2] += f[a][1] * v[i][1].x, x[i][3] += f[a][1] * v[i][1].y;
        }
      }
#pragma unroll
      for (int i = 0; i < C::ACC; i++) {
        const int id = warp * C::ACC + i, a = id & 1, b = id >> 1, d = 8 * b + 2 * t4;
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          const int h = 16 * a + g + 8 * hh;
          const float f = S.flast[h];
          const float xx = x[i][2 * hh] + f * acc[i][2 * hh], yy = x[i][2 * hh + 1] + f * acc[i][2 * hh + 1];
          *reinterpret_cast<uint32_t*>(&ob[h * kQS + d]) = pack_bf16x2(__fdiv_rn(xx, S.den[h]), __fdiv_rn(yy, S.den[h]));
        }
      }
    }
    __syncthreads();
    // o rows (bf16) → global, 16 B per thread
    for (int i = tid; i < kLH * DS / 8; i += kTh) {
      const int h = i / (DS / 8), c8 = i % (DS / 8);
      *reinterpret_cast<uint4*>(o + ((size_t)r * kLH + h) * kD + d0 + 8 * c8) =
          *reinterpret_cast<const uint4*>(&ob[h * kQS + 8 * c8]);
    }
    if (VHA) {  // zp[r][rk][d] = Σ_h U[h][rk] · o[h][d] (k_vha_partial)
      constexpr int RPT = kVhaRank * DS / kTh;  // ranks per thread
      constexpr int RG = RPT < 4 ? RPT : 4;     // ranks per pass over the heads
      const int dd = tid % DS, rk0 = (tid / DS) * RPT;
#pragma unroll 1
      for (int j0 = 0; j0 < RPT; j0 += RG) {
        float s[RG];
#pragma unroll
        for (int j = 0; j < RG; j++) s[j] = 0.f;
#pragma unroll 8
        for (int h = 0; h < kLH; h++) {
          const float x = __bfloat162float(ob[h * kQS + dd]);
#pragma unroll
          for (int j = 0; j < RG; j++) s[j] += S.U[h * kVhaRank + rk0 + j0 + j] * x;
        }
#pragma unroll
        for (int j = 0; j < RG; j++) zp[((size_t)r * kVhaRank + rk0 + j0 + j) * kD + d0 + dd] = s[j];
      }
    }
    __syncthreads();
  }
}

template <int DS, bool PREP, bool VHA>
void launch(const bf16* q, const uint8_t* rows, const RowMeta* meta, HcaCache c, const float* sink, const float* inv,
            const float* U, int M, int max_keys, bf16* o, float* zp, float* ws, size_t ws_bytes, cudaStream_t s) {
  static int nsm = 0;
  if (!nsm) {
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
    CUDA_CHECK(cudaFuncSetAttribute(k_attn<DS, PREP, VHA>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    int(sizeof(Smem<DS>))));
  }
  const int nch = std::max(1, (max_keys + kCh - 1) / kCh);
  const size_t slot = size_t(nch - 1) * (2 * kLH + kLH * DS);  // floats per CTA
  const int ny = kD / DS;
  int nx = DS == kD ? std::min(M, nsm) : M;
  if (slot) nx = std::min<long>(nx, long(ws_bytes / 4 / (slot * ny)));
  if (nx < 1) qlog::fail("hca", "attention workspace %zu B < one CTA's %zu B", ws_bytes, slot * ny * 4);
  k_attn<DS, PREP, VHA><<<dim3(nx, ny), kTh, sizeof(Smem<DS>), s>>>(q, rows, meta, c, sink, inv, U, M, o, zp, ws,
                                                                     slot);
  CUDA_CHECK(cudaGetLastError());
}

template <bool PREP, bool VHA>
void attn(const bf16* q, const uint8_t* rows, const RowMeta* meta, HcaCache c, const float* sink, const float* inv,
          const float* U, int M, int max_keys, bf16* o, float* zp, float* ws, size_t ws_bytes, cudaStream_t s) {
  if (M <= 0) return;
  if (M <= 8) launch<64, PREP, VHA>(q, rows, meta, c, sink, inv, U, M, max_keys, o, zp, ws, ws_bytes, s);
  else launch<512, PREP, VHA>(q, rows, meta, c, sink, inv, U, M, max_keys, o, zp, ws, ws_bytes, s);
}

}  // namespace p041
}  // namespace e55
