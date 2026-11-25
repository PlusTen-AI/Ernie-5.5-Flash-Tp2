// p070_hca_ws.cuh — P070 hca_attn_ws: HCA attention for M > 10 rows (prefill, chunked prefill, drafts) with the
// score and P·V work on separate warps. Included by hca.cu after p041_hca_attn.cuh (uses its helpers).
//
// Bitwise contract — P041's (= the base's k_attn + k_merge): per row and 256-key chunk the same 32-key tiles, the same
// scores (the m16n8k16 chain over the 32 k-steps in order, × 512^-0.5; here as Sᵀ = K·Qᵀ, the operand swap P045 checked
// bit-identical), the same online softmax per tile (max, f = expf(m_old − mx), p = expf(s − mx), P = bf16(p), the tile
// sum Σ p in key order from 0, l = l·f + sum, o = o·f then the two k16 P·V MMAs), the same chunk merge with the sink
// (earlier chunks through a per-CTA workspace slot, in chunk order), q prep and VHA partial as P041 / P042. Schedule:
//   · persistent CTAs over rows; per row the K tiles are gathered and dequantized ONCE into shared memory (bf16) and
//     serve both Q·Kᵀ (A operand) and P·V (B operand) — P041 converts every K fragment in registers and again V;
//   · two groups of 4 score warps (group b: tiles j ≡ b mod 2; warp: heads 8w … 8w+7 with q's fragments in registers ×
//     the tile's 32 keys, two k-chains) handing the running max / sum to each other, so one group's softmax runs
//     beside the other's Q·Kᵀ; 8 P·V warps (dims 64v …) run tile j's P·V beside tile j+1's scores and dequantize tile
//     j+2 into the buffer tile j used (its loads one tile ahead); named barriers only inside a row;
//   · o and zp leave by bulk copies from shared memory (TMA), drained while the CTA runs its next row.
#pragma once

#ifndef P070_GROUPS
#define P070_GROUPS 1
#endif
namespace e55 {
namespace p070 {

constexpr int kLH = 32, kD = kHeadDim, kCh = 256, kKT = 32, TPC = kCh / kKT;
constexpr int kKS = kD + 8;  // K tile / staging row, bf16 (1040 B: conflict-free ldmatrix)
constexpr int kGrp = P070_GROUPS, kSW = 4 * kGrp, kVW = 8, kTh = 32 * (kSW + kVW);  // score groups, warps, threads
constexpr int kRB = kKvRowBytes;
enum { kBarP = 1, kBarK = 3, kBarV = 5, kBarM = 6, kBarL = 8, kBarA = 10, kBarB = 11 };

struct Smem {
  bf16 k[2][kKT][kKS];  // K tiles; k[1] stages q at a row's start, k[0] holds the merged output at its end
  bf16 p[2][kLH][kKT + 8];
  float e[2][kLH][kKT + 1];
  float fac[2][kLH];
  float m[kLH], l[kLH];
  float M[kLH], den[kLH], flast[kLH];
  float red[64];
  float rcs[32], rsn[32];
  __align__(16) float U[kLH * kVhaRank];
};

__device__ __forceinline__ void mma(float d[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}
__device__ __forceinline__ void ldsm4t(uint32_t (&r)[4], const void* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}
__device__ __forceinline__ void ldsm2(uint32_t& r0, uint32_t& r1, const void* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(a));
}
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_arrive(int id, int n) {
  asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "r"(n) : "memory");
}

// grid: persistent CTAs over rows (CTA x: rows x, x + gridDim.x, …); ws: per CTA slot of earlier chunks' partials
template <bool PREP, bool VHA>
__global__ void __launch_bounds__(kTh, 1)
    k_attn(const bf16* __restrict__ q, const uint8_t* __restrict__ rows, const RowMeta* __restrict__ meta, HcaCache c,
           const float* __restrict__ sink, const float* __restrict__ inv, const float* __restrict__ U, int M,
           bf16* __restrict__ o, float* __restrict__ zp, float* __restrict__ ws, size_t ws_slot) {
  extern __shared__ __align__(16) uint8_t smraw[];
  Smem& S = *reinterpret_cast<Smem*>(smraw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const bool sw = warp < kSW;
  const int vw = warp - kSW, vt = tid - 32 * kSW;  // P·V warp index, thread (0 … 255)
  const int vk = vt >> 3, vp = vt & 7;              // gather job: key vk of the tile, 72-byte piece vp of its row
  float* wsl = ws + ws_slot * blockIdx.x;
  constexpr int CS = 2 * kLH + kLH * kD;  // floats per chunk partial
  if (VHA)
    for (int i = tid; i < kLH * kVhaRank; i += kTh) S.U[i] = U[i];
  const float scale = rsqrtf(float(kD));

  for (int r = blockIdx.x; r < M; r += gridDim.x) {
    const RowMeta mt = meta[r];
    const int p = mt.pos, ncomp = (p + 1) / kRatio, nwin = min(p + 1, kWindow), nk = ncomp + nwin;
    const int nch = (nk + kCh - 1) / kCh, ntile = (nk + kKT - 1) / kKT;
    // ---- P·V warps' K tile gather: piece vp = 9 words (8 B) of the row's 72 — words 0–55 nope (8 e4m3 codes, scale
    // of their 64-dim group), 56–71 rope (4 bf16) — and the 7 scales; zero rows past the end
    uint2 rw[9], rsc;
    auto load = [&](int k0) {
      const int key = k0 + vk;
      if (key < nk) {
        const uint8_t* row = p041::key_row(key, ncomp, nwin, p, mt, rows, c);
        const uint2* src = reinterpret_cast<const uint2*>(row);
#pragma unroll
        for (int i = 0; i < 9; i++) rw[i] = src[9 * vp + i];
        rsc = src[72];
      }
    };
    auto store = [&](int k0, int buf) {
      const bool ok = k0 + vk < nk;
      bf16* dst = &S.k[buf][vk][0];
#pragma unroll
      for (int i = 0; i < 9; i++) {
        const int u = 9 * vp + i;
        if (u < kNope / 8) {  // 8 nope dims 8u …: the base's float(e4m3) · 2^e, bf16 (exact)
          uint4 v = make_uint4(0, 0, 0, 0);
          if (ok) {
            const int grp = u >> 3;
            const float s = ue8m0_f32(((grp < 4 ? rsc.x : rsc.y) >> (8 * (grp & 3))) & 0xFF);
            v = make_uint4(p041::e4m3x2_bf16x2(rw[i].x & 0xFFFF, s), p041::e4m3x2_bf16x2(rw[i].x >> 16, s),
                           p041::e4m3x2_bf16x2(rw[i].y & 0xFFFF, s), p041::e4m3x2_bf16x2(rw[i].y >> 16, s));
          }
          *reinterpret_cast<uint4*>(dst + 8 * u) = v;
        } else {  // 4 rope dims (bf16 as stored)
          *reinterpret_cast<uint2*>(dst + kNope + 4 * (u - kNope / 8)) = ok ? rw[i] : make_uint2(0, 0);
        }
      }
    };

    if (warp == 0) asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");  // the last row's o / zp sources
    __syncthreads();  // the previous row is done with shared memory
    // ---- Q (32 heads × 512) → k[1], prepped like k_q_prep when PREP (P041's code); tile 0 → k[0] -----------------------
    if (!sw) load(0);
    {  // 64 threads per head (thread i: dims 8i …), kTh / 64 heads per pass; P041 / k_q_prep's arithmetic and order
      constexpr int HP = kTh / 64, NP = (kLH + HP - 1) / HP;
      const bf16* qr = q + (size_t)r * kLH * kD;
      const int i = tid & 63, d = 8 * i;
      float v[NP][8];
#pragma unroll
      for (int it = 0; it < NP; it++) {
        const int hh = it * HP + (tid >> 6);
        if (hh < kLH) p041::ld_bf16x8(qr + hh * kD + d, v[it]);
      }
      if (r + (int)gridDim.x < M)  // the CTA's next row's q → L2
        for (int x = tid; x < kLH * kD * 2 / 64; x += kTh)
          asm volatile("prefetch.global.L2 [%0];" ::"l"(q + (size_t)(r + gridDim.x) * kLH * kD + x * 32));
      if (PREP) {
        if (tid < 32) {
          float sn, cs;
          sincosf(__fmul_rn(float(p), inv[tid]), &sn, &cs);
          S.rcs[tid] = cs, S.rsn[tid] = sn;
        }
#pragma unroll
        for (int it = 0; it < NP; it++) {
          const int hh = it * HP + (tid >> 6);
          float ss = 0.f;
#pragma unroll
          for (int j = 0; j < 8; j++) ss += v[it][j] * v[it][j];
          ss = warp_sum(ss);
          if (lane == 0 && hh < kLH) S.red[2 * hh + (warp & 1)] = ss;
        }
        __syncthreads();
#pragma unroll
        for (int it = 0; it < NP; it++) {
          const int hh = it * HP + (tid >> 6);
          if (hh >= kLH) continue;
          const float rs = rsqrtf(__fdiv_rn(S.red[2 * hh] + S.red[2 * hh + 1], float(kD)) + kEps);
#pragma unroll
          for (int j = 0; j < 8; j++) v[it][j] *= rs;
          if (d >= kNope) p041::rope8t(v[it], (d - kNope) / 2, S.rcs, S.rsn);
        }
      }
#pragma unroll
      for (int it = 0; it < NP; it++) {
        const int hh = it * HP + (tid >> 6);
        if (hh < kLH) p041::st_bf16x8(&S.k[1][hh][d], v[it]);
      }
    }
    if (!sw) store(0, 0);
    __syncthreads();
    uint32_t qb[kD / 16][2];
    if (sw) {
#pragma unroll
      for (int kk = 0; kk < kD / 16; kk++)
        ldsm2(qb[kk][0], qb[kk][1], &S.k[1][8 * (warp & 3) + (lane & 7)][16 * kk + ((lane >> 3) & 1) * 8]);
    } else if (ntile > 1) {
      load(kKT);
    }
    __syncthreads();  // k[1] free

    if (sw) {
      // ======== score warps ========
      const int grp = warp >> 2, hw = 8 * (warp & 3), h0 = hw + 2 * t;
      for (int j = grp; j < ntile; j += kGrp) {
        const int b = j & 1, k0 = j * kKT;
        const bool first = j % TPC == 0, lastt = j % TPC == TPC - 1 || j == ntile - 1, more = j + 1 < ntile;
        if (j > 0) bar_sync(kBarK + b, 32 * (kVW + 4));
        float sacc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        {
          uint32_t af[2][2][4];
          const bf16* kb0 = &S.k[b][lane & 15][(lane >> 4) * 8];
          const bf16* kb1 = &S.k[b][16 + (lane & 15)][(lane >> 4) * 8];
          ldsm4(af[0][0], kb0), ldsm4(af[0][1], kb1);
#pragma unroll
          for (int kk = 0; kk < kD / 16; kk++) {
            if (kk + 1 < kD / 16)
              ldsm4(af[(kk + 1) & 1][0], kb0 + 16 * (kk + 1)), ldsm4(af[(kk + 1) & 1][1], kb1 + 16 * (kk + 1));
            mma(sacc[0], af[kk & 1][0], qb[kk][0], qb[kk][1]);
            mma(sacc[1], af[kk & 1][1], qb[kk][0], qb[kk][1]);
          }
        }
        // softmax of the tile: sacc[m][i] = key 16m + g + 8(i >> 1), head h0 + (i & 1)
        float sv[2][4];
#pragma unroll
        for (int m = 0; m < 2; m++)
#pragma unroll
          for (int i = 0; i < 4; i++) sv[m][i] = k0 + 16 * m + g + (i >> 1) * 8 < nk ? sacc[m][i] * scale : -INFINITY;
        float ta = fmaxf(fmaxf(sv[0][0], sv[0][2]), fmaxf(sv[1][0], sv[1][2]));
        float tb = fmaxf(fmaxf(sv[0][1], sv[0][3]), fmaxf(sv[1][1], sv[1][3]));
#pragma unroll
        for (int off = 4; off < 32; off <<= 1)
          ta = fmaxf(ta, __shfl_xor_sync(0xffffffffu, ta, off)), tb = fmaxf(tb, __shfl_xor_sync(0xffffffffu, tb, off));
        if (kGrp == 2 && j > 0) bar_sync(kBarM + (b ^ 1), 32 * kSW);  // the running max after tile j−1 (other group)
        const float moa = first ? -INFINITY : S.m[h0], mob = first ? -INFINITY : S.m[h0 + 1];
        const float mxa = fmaxf(moa, ta), mxb = fmaxf(mob, tb);
        __syncwarp();
        if (g == 0) S.m[h0] = mxa, S.m[h0 + 1] = mxb;
        __syncwarp();
        if (kGrp == 2 && more) bar_arrive(kBarM + b, 32 * kSW);
#pragma unroll
        for (int m = 0; m < 2; m++)
#pragma unroll
          for (int i = 0; i < 4; i++) {
            const int key = 16 * m + g + (i >> 1) * 8, h = h0 + (i & 1);
            const float e = expf(sv[m][i] - ((i & 1) ? mxb : mxa));
            S.e[b][h][key] = e;
            S.p[b][h][key] = __float2bfloat16_rn(e);
          }
        if (g == 0) S.fac[b][h0] = expf(moa - mxa), S.fac[b][h0 + 1] = expf(mob - mxb);
        __syncwarp();
        float sum = 0.f;
        if (lane < 8) {  // the tile sum in key order (base)
          const int h = hw + lane;
#pragma unroll
          for (int jj = 0; jj < kKT; jj++) sum += S.e[b][h][jj];
        }
        if (kGrp == 2 && j > 0) bar_sync(kBarL + (b ^ 1), 32 * kSW);  // the running sum after tile j−1
        const bool early = lastt && j / TPC + 1 < nch;     // the end of a chunk before the row's last: partial → ws
        float* w = wsl + (size_t)(j / TPC) * CS;
        if (lane < 8) {
          const int h = hw + lane;
          const float l = (first ? 0.f : S.l[h]) * S.fac[b][h] + sum;
          S.l[h] = l;
          if (early) w[kLH + h] = l;
        }
        if (early && g == 0) w[h0] = mxa, w[h0 + 1] = mxb;
        __syncwarp();
        if (kGrp == 2 && more) bar_arrive(kBarL + b, 32 * kSW);
        bar_arrive(kBarP + b, 32 * (kVW + 4));
      }
      // ---- merge with the sink, in chunk order (k_merge / P041): the coefficients
      bar_sync(kBarA, kTh);  // the last tile's m, l and every partial written; k[0] free
      if (tid < kLH) {
        const int h = tid;
        float Mx = sink[h];
        for (int cc = 0; cc + 1 < nch; cc++) Mx = fmaxf(Mx, wsl[cc * CS + h]);
        Mx = fmaxf(Mx, S.m[h]);
        float den = 0.f;
        for (int cc = 0; cc + 1 < nch; cc++) den += expf(wsl[cc * CS + h] - Mx) * wsl[cc * CS + kLH + h];
        den += expf(S.m[h] - Mx) * S.l[h];
        den += expf(sink[h] - Mx);
        S.M[h] = Mx, S.den[h] = den, S.flast[h] = expf(S.m[h] - Mx);
      }
      bar_sync(kBarB, kTh);
    } else {
      // ======== P·V warps ========
      float acc[2][8][4];  // heads 16a + g (+8), dims 64·vw + 8n + 2t (+1)
#pragma unroll
      for (int a = 0; a < 2; a++)
#pragma unroll
        for (int n = 0; n < 8; n++) acc[a][n][0] = acc[a][n][1] = acc[a][n][2] = acc[a][n][3] = 0.f;
      if (ntile > 1) {
        store(kKT, 1);
        bar_arrive(kBarK + 1, 32 * (kVW + 4));
        if (ntile > 2) load(2 * kKT);
      }
      for (int j = 0; j < ntile; j++) {
        const int b = j & 1;
        bar_sync(kBarP + b, 32 * (kVW + 4));
#pragma unroll
        for (int a = 0; a < 2; a++) {
          const float f0 = S.fac[b][16 * a + g], f1 = S.fac[b][16 * a + g + 8];
#pragma unroll
          for (int n = 0; n < 8; n++) acc[a][n][0] *= f0, acc[a][n][1] *= f0, acc[a][n][2] *= f1, acc[a][n][3] *= f1;
        }
#pragma unroll
        for (int ks = 0; ks < kKT / 16; ks++) {
          uint32_t pa[2][4];
          ldsm4(pa[0], &S.p[b][lane & 15][16 * ks + (lane >> 4) * 8]);
          ldsm4(pa[1], &S.p[b][16 + (lane & 15)][16 * ks + (lane >> 4) * 8]);
#pragma unroll
          for (int bp = 0; bp < 4; bp++) {
            uint32_t vv[4];
            ldsm4t(vv, &S.k[b][16 * ks + (lane & 15)][64 * vw + 16 * bp + (lane >> 4) * 8]);
            if (ks == 1 && bp == 3 && j + 2 < ntile) bar_sync(kBarV, 32 * kVW);  // every P·V warp has read k[b]
            mma(acc[0][2 * bp], pa[0], vv[0], vv[1]);
            mma(acc[1][2 * bp], pa[1], vv[0], vv[1]);
            mma(acc[0][2 * bp + 1], pa[0], vv[2], vv[3]);
            mma(acc[1][2 * bp + 1], pa[1], vv[2], vv[3]);
          }
        }
        if (j + 2 < ntile) {  // tile j+2 → k[b], beside k-step 1's MMAs
          store((j + 2) * kKT, b);
          bar_arrive(kBarK + b, 32 * (kVW + 4));
          if (j + 3 < ntile) load((j + 3) * kKT);
        }
        if ((j % TPC == TPC - 1 || j == ntile - 1) && j / TPC + 1 < nch) {  // an earlier chunk's O → ws; fresh state
          float* wo = wsl + (size_t)(j / TPC) * CS + 2 * kLH;
#pragma unroll
          for (int a = 0; a < 2; a++)
#pragma unroll
            for (int n = 0; n < 8; n++) {
              const int hr = 16 * a + g, d = 64 * vw + 8 * n + 2 * t;
              *reinterpret_cast<float2*>(wo + hr * kD + d) = make_float2(acc[a][n][0], acc[a][n][1]);
              *reinterpret_cast<float2*>(wo + (hr + 8) * kD + d) = make_float2(acc[a][n][2], acc[a][n][3]);
              acc[a][n][0] = acc[a][n][1] = acc[a][n][2] = acc[a][n][3] = 0.f;
            }
        }
      }
      bar_sync(kBarA, kTh);
      bar_sync(kBarB, kTh);  // M, den, flast ready
      bf16* ob = &S.k[0][0][0];  // merged output [32][kKS]
      // per 16 × 8 block: earlier chunks from the workspace in chunk order, then the last chunk (registers); 4 blocks
      // at a time, so their workspace loads are in flight together
#pragma unroll
      for (int a = 0; a < 2; a++)
#pragma unroll
        for (int n0 = 0; n0 < 8; n0 += 4) {
          const int h = 16 * a + g;
          float x[4][4];
#pragma unroll
          for (int n = 0; n < 4; n++) x[n][0] = x[n][1] = x[n][2] = x[n][3] = 0.f;
          for (int cc = 0; cc + 1 < nch; cc++) {
            const float* w = wsl + (size_t)cc * CS;
            float2 v[4][2];
#pragma unroll
            for (int n = 0; n < 4; n++) {
              const int d = 64 * vw + 8 * (n0 + n) + 2 * t;
              v[n][0] = *reinterpret_cast<const float2*>(&w[2 * kLH + h * kD + d]);
              v[n][1] = *reinterpret_cast<const float2*>(&w[2 * kLH + (h + 8) * kD + d]);
            }
            const float f0 = expf(w[h] - S.M[h]), f1 = expf(w[h + 8] - S.M[h + 8]);
#pragma unroll
            for (int n = 0; n < 4; n++)
              x[n][0] += f0 * v[n][0].x, x[n][1] += f0 * v[n][0].y, x[n][2] += f1 * v[n][1].x, x[n][3] += f1 * v[n][1].y;
          }
          const float fa = S.flast[h], fb = S.flast[h + 8], da = S.den[h], db = S.den[h + 8];
#pragma unroll
          for (int n = 0; n < 4; n++) {
            const int d = 64 * vw + 8 * (n0 + n) + 2 * t;
            const float* ac = acc[a][n0 + n];
            *reinterpret_cast<uint32_t*>(&ob[h * kKS + d]) =
                pack_bf16x2(__fdiv_rn(x[n][0] + fa * ac[0], da), __fdiv_rn(x[n][1] + fa * ac[1], da));
            *reinterpret_cast<uint32_t*>(&ob[(h + 8) * kKS + d]) =
                pack_bf16x2(__fdiv_rn(x[n][2] + fb * ac[2], db), __fdiv_rn(x[n][3] + fb * ac[3], db));
          }
        }
    }
    __syncthreads();  // the merged output in k[0]
    const bf16* ob = &S.k[0][0][0];
    float* zb = reinterpret_cast<float*>(&S.k[1][0][0]);  // zp row staging [16][512] fp32 (k[1] is free)
    if (VHA) {  // zp[r][rk][d] = Σ_h U[h][rk] · o[h][d]: per (rank, dim) the k_vha_partial chain over h = 0 … 31
#pragma unroll 1
      for (int dd = tid; dd < kD; dd += kTh) {
        float sr[kVhaRank];
#pragma unroll
        for (int k = 0; k < kVhaRank; k++) sr[k] = 0.f;
#pragma unroll 4
        for (int h = 0; h < kLH; h++) {
          const float xv = __bfloat162float(ob[h * kKS + dd]);
#pragma unroll
          for (int k4 = 0; k4 < kVhaRank / 4; k4++) {
            const float4 u = *reinterpret_cast<const float4*>(&S.U[h * kVhaRank + 4 * k4]);
            sr[4 * k4] += u.x * xv, sr[4 * k4 + 1] += u.y * xv, sr[4 * k4 + 2] += u.z * xv, sr[4 * k4 + 3] += u.w * xv;
          }
        }
#pragma unroll
        for (int k = 0; k < kVhaRank; k++) zb[k * kD + dd] = sr[k];
      }
    }
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");  // merged o / staged zp → visible to the bulk copies
    __syncthreads();
    if (warp == 0) {  // o (32 head rows of 1 KB) and zp (16 rows of 2 KB): bulk copies, drained while the next row runs
      const uint32_t so = static_cast<uint32_t>(__cvta_generic_to_shared(ob + lane * kKS));
      asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n" ::"l"(o + ((size_t)r * kLH + lane) * kD),
                   "r"(so), "r"(kD * 2)
                   : "memory");
      if (VHA && lane < kVhaRank) {
        const uint32_t sz = static_cast<uint32_t>(__cvta_generic_to_shared(zb + lane * kD));
        asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n" ::"l"(zp + ((size_t)r * kVhaRank + lane) * kD),
                     "r"(sz), "r"(kD * 4)
                     : "memory");
      }
      asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
    }
  }
  if (warp == 0) asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
}

template <bool PREP, bool VHA>
void launch(const bf16* q, const uint8_t* rows, const RowMeta* meta, HcaCache c, const float* sink, const float* inv,
            const float* U, int M, int max_keys, bf16* o, float* zp, float* ws, size_t ws_bytes, cudaStream_t s) {
  static int nsm = 0;
  if (!nsm) {
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
  }
  static bool attr = false;
  if (!attr) {
    CUDA_CHECK(cudaFuncSetAttribute(k_attn<PREP, VHA>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(sizeof(Smem))));
    attr = true;
  }
  const int nch = std::max(1, (max_keys + kCh - 1) / kCh);
  const size_t slot = size_t(nch - 1) * (2 * kLH + kLH * kD);  // floats per CTA
  int nx = std::min(M, nsm);
  if (slot) nx = std::min<long>(nx, long(ws_bytes / 4 / slot));
  if (nx < 1) qlog::fail("hca", "attention workspace %zu B < one CTA's %zu B", ws_bytes, slot * 4);
  k_attn<PREP, VHA><<<nx, kTh, sizeof(Smem), s>>>(q, rows, meta, c, sink, inv, U, M, o, zp, ws, slot);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace p070
}  // namespace e55
