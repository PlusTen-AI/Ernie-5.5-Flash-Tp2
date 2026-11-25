// p031_f8_prefill.cuh — P031 f8_prefill: the FP8 block GEMM for many tokens (M > 32), selected by
// patch::on(f8_prefill); the base kernel (fp8.cu k_f8 + k_reduce) stays and runs when the patch is off.
// Included by fp8.cu inside its anonymous namespace (mma, cp_async from mma.cuh).
//
// Bitwise contract — every output element is the base kernel's: the same packed weights and 16-row tiles, the same MMA
// instruction with the same operands (swap-AB: weight tile = A, 8 tokens = B, the 128-block scale and the token's
// group scale as UE8M0), the k32 blocks of each K split chained in ascending order into one fp32 accumulator from 0,
// the fixed splits of the upload (L.split) summed as the base reduce does (v = p0, v += p1, … in split order — here in
// registers: one CTA runs a tile's splits one after the other, so no workspace and no reduce launch), the same bf16 /
// fp32 output rounding. Only the schedule differs from the base (4 warps × 16 rows, weights from global per warp):
//   · CTA tile 64 weight rows × 128 tokens, 4 warps as 2 × 2, warp tile 32 rows × 64 tokens (2 × 8 MMA tiles): each
//     A fragment is used 8 times and each B fragment twice from registers; 2 CTAs per SM (one's pipeline fill and
//     epilogue run under the other's MMAs);
//   · both operands through a 3-stage cp.async pipeline, 64 k bytes per stage (weights 4 tiles × 2 k32 blocks as
//     packed, tokens [128][64 B] XOR-swizzled for conflict-free ldmatrix), all block / group scales of the CTA staged
//     once at the start;
//   · grouped raster (GM token tiles sweep the weight tiles together: activations and weights reused from L2);
//   · epilogue through shared memory, 16-byte coalesced stores.
#pragma once

namespace p031 {

constexpr int BM = 128, KS = 64;  // tokens per CTA tile, k bytes per stage (2 k32 blocks)
constexpr int MI = 2, NJ = 8;     // warp tile: MI m16 weight tiles × NJ n8 token tiles (32 rows × 64 tokens)
constexpr int GM = 8;             // token tiles per raster group
constexpr int BST = BM * KS;      // tokens per stage
constexpr int kMaxKG = 64;        // k ≤ 8192

// WN warps along the weight rows (CTA tile BN = 32·WN rows) × 2 along the tokens, ST pipeline stages, MINB CTAs / SM
template <int WN_, int ST_, int MINB_>
struct Cfg {
  static constexpr int WN = WN_, ST = ST_, MINB = MINB_, BN = 32 * WN, NTH = 64 * WN;
  static constexpr int AST = (BN / 16) * 2 * 512;  // weights per stage: BN/16 tiles × 2 k32 × 512 B
  static constexpr int STAGE = AST + BST;
  static int smem(int KG, bool f32) {  // pipeline + scales, or the epilogue tile
    const int pipe = ST * STAGE + (BN / 16) * KG + BM * (KG + 4);
    const int epi = BM * (f32 ? (BN + 4) * 4 : (BN + 8) * 2);
    return pipe > epi ? pipe : epi;
  }
};

struct Args {
  const uint8_t* w;
  const uint8_t* ws;
  int n, k, split, rows_per_group, tiles;  // tiles = npad / 16
  const uint8_t* xq;
  const uint8_t* xs;
  int ld;
  long gx, gs;
  int M, NT, MT;  // NT weight-row tiles of 128, MT token tiles of 128
  void* y;
  int ldy;
  bool vec;  // y rows 16-byte aligned
};

__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(s));
}

template <class C, bool F32, bool SPLIT>
__global__ void __launch_bounds__(C::NTH, C::MINB) k_pre(Args a) {
  constexpr int BN = C::BN, NTH = C::NTH, ST = C::ST, AST = C::AST, STAGE = C::STAGE;
  extern __shared__ __align__(128) uint8_t sm[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wn = warp % C::WN, wm = warp / C::WN;
  // grouped raster: GM token tiles share the sweep over the weight tiles
  const int per = GM * a.NT, grp = blockIdx.x / per, fm = grp * GM, gm = min(a.MT - fm, GM);
  const int rem = blockIdx.x - grp * per, mt = fm + rem % gm, nt = rem / gm;
  const int T0 = nt * (BN / 16), r0 = nt * BN, t0 = mt * BM;
  const int KB = a.k >> 5, KG = a.k >> 7, KGP = KG + 4, nst = a.k / KS;
  const int kgrp = r0 / a.rows_per_group;
  const uint8_t* xq = a.xq + kgrp * a.gx;
  const uint8_t* xs = a.xs + kgrp * a.gs;
  uint8_t* ssa = sm + ST * STAGE;          // [8 tiles][KG] weight block scales
  uint8_t* ssb = ssa + (BN / 16) * KG;     // [128 tokens][KGP] token group scales

  auto load = [&](int st, int buf) {
    uint8_t* sa = sm + buf * STAGE;
    uint8_t* sb = sa + AST;
    const int kb = st * 2;  // first k32 block of the stage
#pragma unroll
    for (int p = tid; p < AST / 16; p += NTH) {  // tile p/64: 1 KB contiguous (2 k32 blocks)
      const int tl = p >> 6, T = T0 + tl;
      const bool ok = T < a.tiles;
      cp_async16(sa + p * 16, a.w + ((size_t)(ok ? T : 0) * KB + kb) * 512 + (p & 63) * 16, ok);
    }
#pragma unroll
    for (int p = tid; p < BST / 16; p += NTH) {  // token p/4, 16-B chunk p%4, swizzled
      const int r = p >> 2, c = p & 3, tok = t0 + r;
      const bool ok = tok < a.M;
      cp_async16(sb + r * KS + ((c ^ ((r >> 1) & 3)) << 4), xq + (size_t)(ok ? tok : 0) * a.ld + st * KS + c * 16, ok);
    }
  };

  // prologue: ST−1 stages in flight, then the scales (plain loads, under the copies)
#pragma unroll
  for (int s = 0; s < ST - 1; s++) {
    if (s < nst) load(s, s);
    cp_async_commit();
  }
  for (int c = tid; c < (BN / 16) * KG; c += NTH) {
    const int tl = c / KG, T = T0 + tl;
    ssa[c] = T < a.tiles ? a.ws[(size_t)T * KG + c % KG] : 0;
  }
  {
    const int sld = a.ld >> 7;
    for (int c = tid; c < BM * KG; c += NTH) {
      const int r = c / KG, j = c - r * KG, tok = t0 + r;
      ssb[r * KGP + j] = tok < a.M ? xs[(size_t)tok * sld + j] : 0;
    }
  }

  float acc[MI][NJ][4];
  [[maybe_unused]] float tot[MI][NJ][4];
#pragma unroll
  for (int i = 0; i < MI; i++)
#pragma unroll
    for (int j = 0; j < NJ; j++)
#pragma unroll
      for (int e = 0; e < 4; e++) {
        acc[i][j][e] = 0.f;
        if constexpr (SPLIT) tot[i][j][e] = 0.f;
      }

  const int nks = KB / a.split;  // k32 blocks per split chain
  int left = nks, chain = 0;
  // fragment addresses inside a stage
  const int aoff = (wn * MI) * 1024 + lane * 16;  // + i·1024 + kb·512
  // ldmatrix: lane → matrix lane>>3 (0: tile j lo, 1: tile j hi, 2: tile j+1 lo, 3: tile j+1 hi), row lane&7
  const int lrow = wm * 64 + ((lane >> 4) << 3) + (lane & 7), lhi = (lane >> 3) & 1;
  const uint8_t* ssb_l = ssb + (wm * 64 + g) * KGP;
  const uint8_t* ssa_w = ssa + wn * MI * KG;

  for (int st = 0; st < nst; st++) {
    cp_async_wait<ST - 2>();
    __syncthreads();  // stage st landed for everyone; buffer (st−1) % ST is free
    if (st + ST - 1 < nst) load(st + ST - 1, (st + ST - 1) % ST);
    cp_async_commit();
    const uint8_t* sa = sm + (st % ST) * STAGE;
    const uint8_t* sb = sa + AST;
#pragma unroll
    for (int kb = 0; kb < 2; kb++) {
      const int kg = (st * 2 + kb) >> 2;  // k128 group
      uint4 A[MI];
#pragma unroll
      for (int i = 0; i < MI; i++) A[i] = *reinterpret_cast<const uint4*>(sa + aoff + i * 1024 + kb * 512);
      uint32_t B[NJ][2];
#pragma unroll
      for (int jj = 0; jj < NJ; jj += 2) {
        const int r = lrow + jj * 8, c = kb * 2 + lhi;
        ldsm_x4(B[jj][0], B[jj][1], B[jj + 1][0], B[jj + 1][1], sb + r * KS + ((c ^ ((r >> 1) & 3)) << 4));
      }
      uint32_t sfa[MI], sfb[NJ];
#pragma unroll
      for (int i = 0; i < MI; i++) sfa[i] = ssa_w[i * KG + kg];
#pragma unroll
      for (int j = 0; j < NJ; j++) sfb[j] = ssb_l[j * 8 * KGP + kg];
#pragma unroll
      for (int j = 0; j < NJ; j++)
#pragma unroll
        for (int i = 0; i < MI; i++)
          mma_e4m3_e4m3(acc[i][j], A[i].x, A[i].y, A[i].z, A[i].w, B[j][0], B[j][1], sfa[i], sfb[j]);
      if constexpr (SPLIT) {
        if (--left == 0) {  // a split's chain ends: add it to the running sum in split order
#pragma unroll
          for (int i = 0; i < MI; i++)
#pragma unroll
            for (int j = 0; j < NJ; j++)
#pragma unroll
              for (int e = 0; e < 4; e++) {
                tot[i][j][e] = chain ? tot[i][j][e] + acc[i][j][e] : acc[i][j][e];
                acc[i][j][e] = 0.f;
              }
          left = nks, chain++;
        }
      }
    }
  }
  cp_async_wait<0>();
  __syncthreads();

  // epilogue: D fragment (row g / g+8, token 2t / 2t+1) → Ps[token][row] → coalesced rows of y
  using E = typename std::conditional<F32, float, bf16>::type;
  constexpr int P = F32 ? BN + 4 : BN + 8;
  E* Ps = reinterpret_cast<E*>(sm);
#pragma unroll
  for (int i = 0; i < MI; i++)
#pragma unroll
    for (int j = 0; j < NJ; j++) {
      const int rr = (wn * MI + i) * 16 + g, tt = wm * 64 + j * 8 + 2 * t;
#pragma unroll
      for (int e = 0; e < 4; e++) {
        const float v = SPLIT ? tot[i][j][e] : acc[i][j][e];
        const int r = rr + (e >> 1) * 8, m = tt + (e & 1);
        if constexpr (F32) Ps[m * P + r] = v;
        else Ps[m * P + r] = __float2bfloat16_rn(v);
      }
    }
  __syncthreads();
  constexpr int V = 16 / sizeof(E), CH = BN / V;  // elements per 16 B, chunks per token row
  E* y = static_cast<E*>(a.y);
  for (int c = tid; c < BM * CH; c += NTH) {
    const int m = c / CH, ch = c % CH, tok = t0 + m, r = r0 + ch * V;
    if (tok >= a.M || r >= a.n) continue;
    const E* src = Ps + m * P + ch * V;
    E* dst = y + (size_t)tok * a.ldy + r;
    if (a.vec) {
      *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
    } else {
#pragma unroll
      for (int e = 0; e < V; e++) dst[e] = src[e];
    }
  }
}

template <class C, bool F32, bool SPLIT>
void launch(const F8Lin& L, const Args& a, cudaStream_t s) {
  static bool attr = false;
  if (!attr) {
    CUDA_CHECK(cudaFuncSetAttribute(k_pre<C, F32, SPLIT>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    C::smem(kMaxKG, F32)));
    attr = true;
  }
  Args b = a;
  b.NT = cdiv(L.n, C::BN);
  k_pre<C, F32, SPLIT><<<b.NT * b.MT, C::NTH, C::smem(L.k / 128, F32), s>>>(b);
  CUDA_CHECK(cudaGetLastError());
}

// measured at M = 4096 (k 1024 / 2048 / 4096): 64 × 128 tiles, 3 stages, 2 CTAs per SM 110 / 122 / 124–129 TFLOPS;
// 128 × 128, 4 stages, 1 CTA 97 / 109 / 116–117; 128 × 128, 2 stages, 2 CTAs 93 / 110 / 117–124
using Pick = Cfg<2, 3, 2>;

// can this path run? k ≤ 8192 (staged scales), a group of rows covers whole row tiles; at M ≤ 64 a weight with fewer
// than 48 row tiles (qkv: 24) fills too few SMs — the base kernel is 2 µs faster there (measured), so it keeps it
inline bool ok(const F8Lin& L, int M) {
  if (M <= 32 || L.k > kMaxKG * 128 || L.k % KS) return false;
  if (M <= 64 && cdiv(L.n, Pick::BN) < 48) return false;
  return L.groups == 1 || (L.n / L.groups) % Pick::BN == 0;
}

template <bool F32>
void forward(const F8Lin& L, const F8Act& x, int M, void* y, int ldy, cudaStream_t s, long gx, long gs) {
  const size_t es = F32 ? 4 : 2;
  const bool vec = (ldy * es) % 16 == 0 && reinterpret_cast<uintptr_t>(y) % 16 == 0;
  Args a{L.w, L.s, L.n, L.k, L.split, L.n / L.groups, L.npad / 16, x.q, x.s, x.ld, gx, gs, M, 0, cdiv(M, BM), y,
         ldy, vec};
  if (L.split > 1) launch<Pick, F32, true>(L, a, s);
  else launch<Pick, F32, false>(L, a, s);
}

}  // namespace p031
