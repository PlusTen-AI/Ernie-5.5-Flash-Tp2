// p133_f8_seq.cuh — P133 f8_tile_seq: the FP8 block GEMM for prefill (M ≥ 256) on weights uploaded with a K split
// (2 or 4), selected by patch::on(f8_tile_seq) before P103 / P031: a 128-row weight tile per CTA whose warps run the
// splits' chains one after the other over K in ascending order, the running sum in registers (P031's scheme on P103's
// tile mechanics) — no thread-block cluster, no workspace, no partial tiles through L2. P060's epilogues (σ, + e, the
// MXFP8 copy) in its store as P104 does in P103's. Included by fp8.cu inside its anonymous namespace after p103 (mma,
// cp_async from mma.cuh; ldsm_x4 from p031; cp_async4 from p103; sigmoid_bf16 from p060).
//
// Bitwise contract — every output element is the base kernel's: the same packed weights and 16-row tiles, the same MMA
// instruction with the same operands (swap-AB: weight tile = A, 8 tokens = B, the 128-block scale and the token's
// group scale as UE8M0), the k32 blocks of each K split (L.split, fixed at upload) chained in ascending order into one
// fp32 accumulator from 0, the splits summed v = p0, v += p1, … in split order (here: tot = p0 when chain 0 ends,
// tot += p_z when chain z ends — the same additions), the same bf16 / fp32 output rounding; the epilogues are
// p060::k_epi's expressions on the same bf16 values (P104's code). Only the schedule differs:
//   · CTA tile 128 weight rows × BM tokens; warps 2 (rows) × WT (tokens), warp tile 64 × 32 (4 × 4 MMA tiles: each A
//     fragment used 4 times, each B fragment 4 times) — acc and the running sum (64 + 64 fp32) fit beside the
//     fragments, which a 64 × 64 warp tile's could not; configs (cfg_of, measured): C6 128 × 128 (8 warps, 1 CTA / SM,
//     P103's L2 bytes per MMA) with 128-k stages for the long chains of split 4, C3 128 × 64 (4 warps, 2 CTAs / SM)
//     with 64-k stages for the short chains of split 2 (K ≤ 2048); the others stay selectable for tests;
//   · both operands through an ST-stage cp.async pipeline of KS k bytes (P103's layouts: weights 8 tiles × KS / 32 k32
//     blocks as packed, tokens [BM][KS] XOR-swizzled for ldmatrix), scale words of 4 k128 groups staged per 512 k;
//   · programmatic launch (P033): weights' first stages before griddepcontrol.wait, the activation after it (P105's
//     late wait after the main loop for sibling GEMMs: no workspace, so any number of them chain); grouped raster.
// P134 f8_situ_epi (kEpiSituQ128, the shared expert's gate/up [gate n/2 ‖ up n/2]): a CTA computes the gate rows (warp
// row 0) and the up rows (warp row 1) of the same 64 columns; v = bf16(SiTU(bf16 gate, bf16 up)) with k_prep's
// expression, the 128-column amax per token over the CTA pair of a 2-CTA cluster (each CTA's 64-column amax through
// DSMEM; max is order-free), act_scale_e8 / quant_e4m3 as act_quant — the down's FP8 activation, no bf16 gate/up.
#pragma once

namespace p133 {
namespace cg = cooperative_groups;

constexpr int BN = 128, MI = 4, NJ = 4;  // weight rows; warp tile 4 m16 × 4 n8 tiles
constexpr int GM = 8;                     // token tiles per raster group
constexpr int PW = 72;                             // staged bf16 row of a warp's 64 weight rows (+8)

// WT warps along the tokens, ST stages of KS k bytes (64: 2 k32 blocks, 128: 4), MINB CTAs an SM
template <int WT_, int ST_, int MINB_, int KS_ = 64>
struct Cfg {
  static constexpr int WT = WT_, ST = ST_, MINB = MINB_, KS = KS_, KK = KS / 32;
  static constexpr int NW = 2 * WT, NTH = 32 * NW, BM = 32 * WT;  // warps, threads, tokens of the tile
  static constexpr int AST = BN * KS, BST = BM * KS, STAGE = AST + BST;
  static constexpr int SPB = 512 / KS;  // stages a 512-k block of scale words
  static constexpr int SCB = (BN / 16 + BM) * 4;  // a 512-k block of scale words: 8 tiles + BM tokens
  static constexpr int SMEM = ST * STAGE + SCB;
  static_assert(NW * 32 * PW * 2 <= ST * STAGE, "epilogue staging");
  static_assert((8 * KS) % NTH == 0 && (BM * KK * 2) % NTH == 0 && KS <= NTH && ST <= SPB, "copy split");
  __device__ static int swz(int r) { return KS == 64 ? (r >> 1) & 3 : r & 7; }  // the token rows' 16-B chunk XOR
};
// 0: 128 × 128, 4 stages, 1 CTA / SM · 1: 128 × 128, 6 stages · 2: 128 × 64, 3 stages, 2 CTAs / SM · 3: 128 × 64, 4 stages
// · 4: 128 × 128, 3 stages, persistent (one CTA an SM over the tiles, the epilogue staged in its own region)
// · 5: 128 × 64, 2 stages of 128 k, 2 CTAs / SM · 6: 128 × 128, 3 stages of 128 k, 1 CTA / SM
using C0 = Cfg<4, 4, 1>;
using C1 = Cfg<4, 6, 1>;
using C2 = Cfg<2, 3, 2>;
using C3 = Cfg<2, 4, 2>;
using C4 = Cfg<4, 3, 1>;
using C5 = Cfg<2, 2, 2, 128>;
using C6 = Cfg<4, 3, 1, 128>;
static_assert(C1::SMEM <= 101376 && C3::SMEM <= 50176 && C4::SMEM + C4::NW * 32 * PW * 2 <= 101376, "shared memory");
static_assert(C5::SMEM <= 50176 && C6::SMEM <= 101376, "shared memory");

// the gate (warp row 0) and up (row 1) warps of token quarter wt (P134): named barriers 1–4 by immediate
__device__ __forceinline__ void pair_bar(int wt) {
  if (wt == 0) asm volatile("bar.sync 1, 64;\n" ::: "memory");
  else if (wt == 1) asm volatile("bar.sync 2, 64;\n" ::: "memory");
  else if (wt == 2) asm volatile("bar.sync 3, 64;\n" ::: "memory");
  else asm volatile("bar.sync 4, 64;\n" ::: "memory");
}

struct Args {
  const uint8_t* w;
  const uint8_t* ws;
  int n, k, split, rows_per_group, tiles;  // tiles = npad / 16
  const uint8_t* xq;
  const uint8_t* xs;
  int ld;
  long gx, gs;
  int M, NT, MT;  // NT weight-row tiles of 128, MT token tiles of BM
  void* y;
  int ldy;
  int epi;        // F8Epi (kEpiNone, kEpiSigmoid, kEpiAdd, kEpiQ32)
  const bf16* e;  // kEpiAdd: + e
  int lde;
  uint8_t* q;     // kEpiQ32: e4m3 [M][ldq], scales [M][ldq / 32]
  uint8_t* qs;
  int ldq;
  bool vec, evec;  // y / e rows 16-byte aligned
  bool late;       // P105: griddepcontrol.wait after the main loop instead of before the activation
  bool situ;       // P134 kEpiSituQ128: rows [0, n/2) gate, [n/2, n) up; NT = blocks of 64 SiTU columns, CTA pairs
};

// PER (persistent): gridDim.x ≤ the tiles; a CTA takes tiles blockIdx.x, + gridDim.x, … and issues the next tile's
// pipeline fill (weights, scales, activation) right after its main loop, under the epilogue (staged in its own region)
template <class C, bool F32, bool PER>
__global__ void __launch_bounds__(C::NTH, C::MINB) k_seq(Args a) {
  constexpr int NTH = C::NTH, BM = C::BM, ST = C::ST, STAGE = C::STAGE, KS = C::KS, KK = C::KK, AST = C::AST;
  constexpr int SPB = C::SPB, CPR = KS / 16;                   // stages a scale block; 16-B chunks a token row
  constexpr int AN = 8 * KS / NTH, BNC = BM * CPR / NTH;       // 16-B copies a thread and stage: weights, tokens
  extern __shared__ __align__(128) uint8_t sm[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wr = warp & 1, wt = warp >> 1;  // the warp's 64 weight rows / 32 tokens of the tile
  const int KB = a.k >> 5, KG = a.k >> 7, sld = a.ld >> 7;
  const int nst = KB / KK, sch = (KB / a.split) / KK;  // stages; stages a chain
  const int ntile = a.NT * a.MT;
  int tile = blockIdx.x;
  // the loads' tile (the next one's from the end of a main loop on): the 16-row weight tiles of warp row 0 from Tg, of
  // warp row 1 from Tu (Tg + 4; P134: the up rows of the same columns), first token t0
  int Tg, Tu, t0;
  const uint8_t *xq, *xs;
  auto at_tile = [&](int tl, int& r0_, int& t0_) {  // r0_: first weight row (P134: first SiTU column)
    const int h = a.situ ? tl & 1 : 0, pr = a.situ ? tl >> 1 : tl, NTp = a.situ ? a.NT >> 1 : a.NT;
    const int per = GM * NTp, grp = pr / per, fm = grp * GM, gm = min(a.MT - fm, GM);
    const int rem = pr - grp * per, mt = fm + rem % gm, np = rem / gm;
    r0_ = a.situ ? (2 * np + h) * 64 : np * BN, t0_ = mt * BM;
  };
  auto set_tile = [&](int tl) {
    int r0_;
    at_tile(tl, r0_, t0);
    Tg = r0_ / 16, Tu = a.situ ? (a.n / 2 + r0_) / 16 : Tg + 4;
    const int kgrp = r0_ / a.rows_per_group;
    xq = a.xq + kgrp * a.gx, xs = a.xs + kgrp * a.gs;
  };
  auto gtile = [&](int tl) { return tl < 4 ? Tg + tl : Tu + tl - 4; };  // the CTA's weight tile tl (0–7)
  set_tile(tile);

  // copy assignments: weight chunk p = tid + NTH·i (tile p / KS, 16 B (p % KS) of its KS·16 bytes: KK k32 blocks as
  // packed), token chunk p = tid + NTH·i (token p / CPR, 16-B chunk p % CPR, stored swizzled)
  const int atl = tid / KS;  // + (NTH / KS)·i
  const size_t astep = (size_t)(NTH / KS) * KB * 512;
  const int brow = tid / CPR, bc = tid % CPR;  // + (NTH / CPR)·i
  const uint32_t bdst = brow * KS + ((bc ^ C::swz(brow)) << 4);
  auto load_a = [&](int st, int buf) {
    uint8_t* sa = sm + buf * STAGE;
#pragma unroll
    for (int i = 0; i < AN; i++) {
      const int T = gtile(atl + (NTH / KS) * i);
      const bool ok = T < a.tiles;
      cp_async16(sa + (tid + NTH * i) * 16, ok ? a.w + ((size_t)T * KB * 32 + st * KS + tid % KS) * 16 : a.w, ok);
    }
  };
  auto load_b = [&](int st, int buf) {
    uint8_t* sb = sm + buf * STAGE + AST;
    const uint8_t* bsrc = xq + (size_t)(t0 + brow) * a.ld + bc * 16;
#pragma unroll
    for (int i = 0; i < BNC; i++) {
      const bool ok = t0 + brow + (NTH / CPR) * i < a.M;
      cp_async16(sb + bdst + i * (NTH / CPR) * KS, ok ? bsrc + (size_t)i * (NTH / CPR) * a.ld + st * KS : a.xq, ok);
    }
  };
  // scales: 32-bit words of 4 k128 groups — a 512-k block's words of the 8 weight tiles and BM tokens in shared memory
  // (one buffer: each warp copies its words to registers at the block's first stage, the next block's words are
  // requested one stage later, landing ≥ 8 − ST + 1 stages before use)
  uint32_t* sca = reinterpret_cast<uint32_t*>(sm + ST * STAGE);  // [8 tiles]
  uint32_t* scb = sca + BN / 16;                                  // [BM tokens]
  auto load_sa = [&](int kw) {
    if (tid < BN / 16) {
      const int T = gtile(tid);
      const bool ok = T < a.tiles;
      p103::cp_async4(sca + tid, ok ? a.ws + (size_t)T * KG + kw * 4 : a.ws, ok);
    }
  };
  auto load_sb = [&](int kw) {
    if (tid < BM) {
      const bool ok = t0 + tid < a.M;
      p103::cp_async4(scb + tid, ok ? xs + (size_t)(t0 + tid) * sld + kw * 4 : a.xs, ok);
    }
  };
  uint32_t wa[MI], wb[NJ];

  // prologue: the weights' first stages and scales (constant) before griddepcontrol.wait, the activation after
#pragma unroll
  for (int s = 0; s < ST - 1; s++)
    if (s < nst) load_a(s, s);
  load_sa(0);
  if (!a.late) asm volatile("griddepcontrol.wait;\n" ::: "memory");
  load_sb(0);
#pragma unroll
  for (int s = 0; s < ST - 1; s++) {
    if (s < nst) load_b(s, s);
    cp_async_commit();
  }
  asm volatile("griddepcontrol.launch_dependents;\n" :::);

  const int aoff = (wr * MI) * (KS * 16) + lane * 16;  // + i·KS·16 + kk·512
  // ldmatrix: lane → matrix lane >> 3 (0: tile j lo, 1: tile j hi, 2: tile j+1 lo, 3: tile j+1 hi), row lane & 7
  const int lrow = wt * 32 + ((lane >> 4) << 3) + (lane & 7), lhi = (lane >> 3) & 1;
  bool first = true;
  for (;;) {
    float acc[MI][NJ][4], tot[MI][NJ][4];
#pragma unroll
    for (int i = 0; i < MI; i++)
#pragma unroll
      for (int j = 0; j < NJ; j++)
#pragma unroll
        for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f, tot[i][j][e] = 0.f;
    int left = sch, chain = 0;
    for (int st = 0; st < nst; st++) {
      cp_async_wait<ST - 2>();
      __syncthreads();  // stage st landed for everyone; buffer (st − 1) % ST is free
      if (st % SPB == 0) {  // a new 512-k block: its scale words to registers
#pragma unroll
        for (int i = 0; i < MI; i++) wa[i] = sca[4 * wr + i];
#pragma unroll
        for (int j = 0; j < NJ; j++) wb[j] = scb[32 * wt + 8 * j + g];
      } else if (st % SPB == 1 && st + SPB - 1 < nst) {  // every warp has its words: request the next block's
        load_sa(st / SPB + 1), load_sb(st / SPB + 1);
      }
      if (st + ST - 1 < nst) load_a(st + ST - 1, (st + ST - 1) % ST), load_b(st + ST - 1, (st + ST - 1) % ST);
      cp_async_commit();
      const uint8_t* sa = sm + (st % ST) * STAGE;
      const uint8_t* sb = sa + AST;
      const int sh = 8 * ((st * KS / 128) & 3);  // this stage's k128 group within the word (KS = 64: 2 stages a group)
      uint32_t sfa[MI], sfb[NJ];
#pragma unroll
      for (int i = 0; i < MI; i++) sfa[i] = (wa[i] >> sh) & 0xFFu;
#pragma unroll
      for (int j = 0; j < NJ; j++) sfb[j] = (wb[j] >> sh) & 0xFFu;
#pragma unroll
      for (int kk = 0; kk < KK; kk++) {
        uint4 A[MI];
#pragma unroll
        for (int i = 0; i < MI; i++) A[i] = *reinterpret_cast<const uint4*>(sa + aoff + i * (KS * 16) + kk * 512);
        uint32_t B[NJ][2];
#pragma unroll
        for (int jj = 0; jj < NJ; jj += 2) {
          const int r = lrow + jj * 8, c = kk * 2 + lhi;
          p031::ldsm_x4(B[jj][0], B[jj][1], B[jj + 1][0], B[jj + 1][1], sb + r * KS + ((c ^ C::swz(r)) << 4));
        }
#pragma unroll
        for (int j = 0; j < NJ; j++)
#pragma unroll
          for (int i = 0; i < MI; i++)
            mma_e4m3_e4m3(acc[i][j], A[i].x, A[i].y, A[i].z, A[i].w, B[j][0], B[j][1], sfa[i], sfb[j]);
      }
      if (--left == 0) {  // a split's chain ends: into the running sum in split order (v = p0, v += p1, …)
#pragma unroll
        for (int i = 0; i < MI; i++)
#pragma unroll
          for (int j = 0; j < NJ; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
              tot[i][j][e] = chain ? tot[i][j][e] + acc[i][j][e] : acc[i][j][e];
              acc[i][j][e] = 0.f;
            }
        left = sch, chain++;
      }
    }
    cp_async_wait<0>();
    __syncthreads();  // every warp is past the pipeline and the scale words: both free
    if (a.late && first) asm volatile("griddepcontrol.wait;\n" ::: "memory");  // P105 (see p103)
    first = false;
    int r0, tc;  // this tile's first weight row / token
    at_tile(tile, r0, tc);
    const int next = PER ? tile + int(gridDim.x) : ntile;
    if (next < ntile) {  // PER: the next tile's fill under this epilogue
      set_tile(next);
#pragma unroll
      for (int s = 0; s < ST - 1; s++)
        if (s < nst) load_a(s, s);
      load_sa(0), load_sb(0);
#pragma unroll
      for (int s = 0; s < ST - 1; s++) {
        if (s < nst) load_b(s, s);
        cp_async_commit();
      }
    }

    // ---- epilogue: this warp's 64 rows × 32 tokens ----
    const int rw = r0 + 64 * wr, tw = tc + 32 * wt;
    if constexpr (F32) {
#pragma unroll
      for (int i = 0; i < MI; i++)
#pragma unroll
        for (int j = 0; j < NJ; j++)
#pragma unroll
          for (int e = 0; e < 4; e++) {
            const int r = rw + i * 16 + g + (e >> 1) * 8, m = tw + j * 8 + 2 * t + (e & 1);
            if (r < a.n && m < a.M) static_cast<float*>(a.y)[(size_t)m * a.ldy + r] = tot[i][j][e];
          }
    } else {
      // [32 tokens][PW]: PER in its own region (the pipeline is filling), else over the drained pipeline
      bf16* Ps = reinterpret_cast<bf16*>(sm + (PER ? ST * STAGE + C::SCB : 0)) + warp * 32 * PW;
#pragma unroll
      for (int i = 0; i < MI; i++)
#pragma unroll
        for (int j = 0; j < NJ; j++) {
          const int rr = i * 16 + g, tt = j * 8 + 2 * t;
          Ps[tt * PW + rr] = __float2bfloat16_rn(tot[i][j][0]);
          Ps[(tt + 1) * PW + rr] = __float2bfloat16_rn(tot[i][j][1]);
          Ps[tt * PW + rr + 8] = __float2bfloat16_rn(tot[i][j][2]);
          Ps[(tt + 1) * PW + rr + 8] = __float2bfloat16_rn(tot[i][j][3]);
        }
      __syncwarp();
      if (a.situ) {  // P134 kEpiSituQ128: v = bf16(SiTU(gate, up)) (k_prep's), its FP8 copy over 128 columns
        // warp row 0 holds the gate, row 1 the up values of the same 64 columns: the pair splits the 32 tokens; each
        // CTA's 64-column amax per token, the cluster partner's (the other 64 columns of the group) through DSMEM
        float* am = reinterpret_cast<float*>(sm + C::SMEM + (PER ? C::NW * 32 * PW * 2 : 0));  // [BM tokens]
        pair_bar(wt);
        const bf16* Pg = reinterpret_cast<const bf16*>(sm + (PER ? ST * STAGE + C::SCB : 0)) + (2 * wt) * 32 * PW;
        const bf16* Pu = Pg + 32 * PW;
        const int c = lane & 7;
        float v[4][8];
#pragma unroll
        for (int it = 0; it < 4; it++) {
          const int tl = 16 * wr + 4 * it + (lane >> 3);
          const uint4 gv = *reinterpret_cast<const uint4*>(Pg + tl * PW + 8 * c);
          const uint4 uv = *reinterpret_cast<const uint4*>(Pu + tl * PW + 8 * c);
          const uint32_t gw[4] = {gv.x, gv.y, gv.z, gv.w}, uw[4] = {uv.x, uv.y, uv.z, uv.w};
          float mx = 0.f;
#pragma unroll
          for (int u = 0; u < 4; u++) {
            v[it][2 * u] = p060::bf16r(p060::situ_f(bf16_lo(gw[u]), bf16_lo(uw[u])));
            v[it][2 * u + 1] = p060::bf16r(p060::situ_f(bf16_hi(gw[u]), bf16_hi(uw[u])));
          }
#pragma unroll
          for (int u = 0; u < 8; u++) mx = fmaxf(mx, fabsf(v[it][u]));
          mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
          mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
          mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));
          if (c == 0) am[32 * wt + tl] = mx;
        }
        cg::cluster_group cl = cg::this_cluster();
        cl.sync();  // both CTAs' amax written
        const float* pam = cl.map_shared_rank(am, int(cl.block_rank()) ^ 1);
#pragma unroll
        for (int it = 0; it < 4; it++) {
          const int tl = 16 * wr + 4 * it + (lane >> 3), m = tw + tl;
          const uint32_t ex = act_scale_e8(fmaxf(am[32 * wt + tl], pam[32 * wt + tl]), 1e-10f);
          uint32_t qw[2] = {0, 0};
#pragma unroll
          for (int u = 0; u < 8; u++) qw[u >> 2] |= quant_e4m3(v[it][u], ex) << (8 * (u & 3));
          if (m < a.M) {
            *reinterpret_cast<uint2*>(a.q + (size_t)m * a.ldq + r0 + 8 * c) = make_uint2(qw[0], qw[1]);
            if (c == 0 && (r0 & 64) == 0) a.qs[(size_t)m * (a.ldq / 128) + r0 / 128] = uint8_t(ex);
          }
        }
        cl.sync();  // the partner has read this CTA's amax
        if (next >= ntile) return;
        tile = next;
        continue;
      }
      const int c = lane & 7, r = rw + 8 * c;  // lane: 8 consecutive rows (16 B) of token 4·it + (lane >> 3)
      const bool rok = r < a.n;
#pragma unroll 2
      for (int it = 0; it < 8; it++) {
        const int tl = 4 * it + (lane >> 3), m = tw + tl;
        const bool ok = m < a.M && rok;
        uint4 hv = *reinterpret_cast<const uint4*>(Ps + tl * PW + 8 * c);
        if (a.epi == kEpiSigmoid) {
          bf16* h = reinterpret_cast<bf16*>(&hv);
#pragma unroll
          for (int u = 0; u < 8; u++) h[u] = p060::sigmoid_bf16(h[u]);
        } else if (a.epi == kEpiAdd) {
          uint4 ev = make_uint4(0, 0, 0, 0);
          if (ok) {
            if (a.evec) {
              ev = *reinterpret_cast<const uint4*>(a.e + (size_t)m * a.lde + r);
            } else {
              bf16* ep = reinterpret_cast<bf16*>(&ev);
#pragma unroll
              for (int u = 0; u < 8; u++) ep[u] = a.e[(size_t)m * a.lde + r + u];
            }
          }
          bf16* h = reinterpret_cast<bf16*>(&hv);
          const bf16* ep = reinterpret_cast<const bf16*>(&ev);
#pragma unroll
          for (int u = 0; u < 8; u++) h[u] = __float2bfloat16_rn(__bfloat162float(h[u]) + __bfloat162float(ep[u]));
        } else if (a.epi == kEpiQ32) {  // h stays; its MXFP8 copy: amax over the 32 rows of lanes 4k … 4k + 3
          const uint32_t hw[4] = {hv.x, hv.y, hv.z, hv.w};
          float f[8], mx = 0.f;
#pragma unroll
          for (int u = 0; u < 4; u++) f[2 * u] = bf16_lo(hw[u]), f[2 * u + 1] = bf16_hi(hw[u]);
#pragma unroll
          for (int u = 0; u < 8; u++) mx = fmaxf(mx, fabsf(f[u]));
          mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
          mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
          const uint32_t ex = act_scale_e8(mx, 1e-10f);
          uint32_t qw[2] = {0, 0};
#pragma unroll
          for (int u = 0; u < 8; u++) qw[u >> 2] |= quant_e4m3(f[u], ex) << (8 * (u & 3));
          if (ok) {
            *reinterpret_cast<uint2*>(a.q + (size_t)m * a.ldq + r) = make_uint2(qw[0], qw[1]);
            if ((c & 3) == 0) a.qs[(size_t)m * (a.ldq / 32) + r / 32] = uint8_t(ex);
          }
        }
        if (!ok) continue;
        bf16* dst = static_cast<bf16*>(a.y) + (size_t)m * a.ldy + r;
        if (a.vec) {
          *reinterpret_cast<uint4*>(dst) = hv;
        } else {
          const bf16* h = reinterpret_cast<const bf16*>(&hv);
#pragma unroll
          for (int u = 0; u < 8; u++) dst[u] = h[u];
        }
      }
      __syncwarp();  // PER: every lane's staged reads done before the next tile's writes
    }
    if (next >= ntile) return;
    tile = next;
  }
}

inline int sm_count() {
  static int n = 0;
  if (!n) {
    int d = 0;
    CUDA_CHECK(cudaGetDevice(&d));
    CUDA_CHECK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, d));
  }
  return n;
}

template <class C, bool F32, bool PER>
void launch(Args a, cudaStream_t s) {
  constexpr int smem = C::SMEM + (PER ? C::NW * 32 * PW * 2 : 0) + C::BM * 4;  // + P134's amax
  static bool attr = false;
  if (!attr) {
    CUDA_CHECK(cudaFuncSetAttribute(k_seq<C, F32, PER>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    attr = true;
  }
  a.MT = cdiv(a.M, C::BM);
  const int nt = a.NT * a.MT;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(unsigned(PER ? std::min(nt, sm_count() * C::MINB) : nt));
  cfg.blockDim = dim3(C::NTH);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = s;
  cudaLaunchAttribute at[2];
  int na = 0;
  if (a.situ) {  // P134: the two CTAs of a 128-column group
    at[na].id = cudaLaunchAttributeClusterDimension;
    at[na].val.clusterDim.x = 2, at[na].val.clusterDim.y = 1, at[na].val.clusterDim.z = 1;
    na++;
  }
  if (e55::patch::on(e55::patch::f8_pdl)) {  // P033 (waits in the kernel)
    at[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[na].val.programmaticStreamSerializationAllowed = 1;
    na++;
  }
  cfg.attrs = at;
  cfg.numAttrs = na;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, k_seq<C, F32, PER>, a));
}

int g_cfg = -1;  // tests (f8_test_seq): −1 the measured choice (cfg_of), else the config 0–4 for every shape

// can this path run (L, M, activation)? M ≥ 256; a K split (2 or 4) whose chains are whole stages; k a multiple of 512
// (scale words); a group of rows covers whole 128-row tiles; 16-byte activation rows and 4-byte scale words
inline bool ok(const F8Lin& L, int M, const F8Act& x, long gx, long gs) {
  if (M < 256 || L.k % 512 || L.split < 2 || (L.k / 32 / L.split) % 4) return false;  // chains of whole stages
  if (L.groups > 1 && (L.n / L.groups) % BN) return false;
  if (x.ld % 512 || reinterpret_cast<uintptr_t>(x.q) % 16 || reinterpret_cast<uintptr_t>(x.s) % 4) return false;
  return gx % 16 == 0 && gs % 4 == 0;
}
// which config takes the shape (−1: none, P103 / P031 keep it) — measured at the 8192-row chunk's call sites
// (test_gemm.py --sites, isolated ms, current → this): split 4 → C6 (128 × 128, 128-k stages; q_down + kv M = 1024
// 0.109 → 0.100, fc1 + MXFP8 0.180 → 0.150, shared gate/up M = 2048 0.290 → 0.279 against P031 — C3 0.106 / 0.152 /
// 0.284); split 2 with chains ≤ 1024 → C3 (128 × 64, 2 CTAs / SM; gate σ ×2 M = 1024 0.255 → 0.205, fc2 + shared
// 0.225 → 0.212, shared down alone M = 2048 0.166 → 0.154 against P103's clusters — C6 0.209 / 0.179 / 0.167);
// split 2 with longer chains stays on P103 (wo_a ×2 M = 1024 0.515 → 0.551 / 0.621 on C6 / C3), and so does a weight
// with a partial 128-row tile on P031 (DSA q_a + kv_a, 1600 rows: M = 1024 0.139 → 0.143)
inline int cfg_of(const F8Lin& L, int M) {
  if (g_cfg >= 0) return g_cfg;
  (void)M;
  if (L.n % BN) return -1;
  if (L.split >= 4) return 6;
  return L.k / L.split <= 1024 ? 3 : -1;
}
inline bool epi_ok(const F8Lin& L, int epi) {
  if (epi == kEpiQ32) return L.n % 32 == 0;
  if (epi == kEpiSituQ128) return e55::patch::on(e55::patch::f8_situ_epi) && L.groups == 1 && L.n % 256 == 0;
  return epi == kEpiNone || epi == kEpiSigmoid || epi == kEpiAdd;
}

template <bool F32>
void forward(const F8Lin& L, const F8Act& x, int M, void* y, int ldy, cudaStream_t s, long gx, long gs,
             const F8Out* o = nullptr, bool late = false) {
  const size_t es = F32 ? 4 : 2;
  Args a{};
  a.w = L.w, a.ws = L.s, a.n = L.n, a.k = L.k, a.split = L.split, a.rows_per_group = L.n / L.groups;
  a.tiles = L.npad / 16, a.xq = x.q, a.xs = x.s, a.ld = x.ld, a.gx = gx, a.gs = gs, a.M = M;
  a.NT = cdiv(L.n, BN), a.y = y, a.ldy = ldy, a.late = late;
  a.vec = y && (ldy * es) % 16 == 0 && reinterpret_cast<uintptr_t>(y) % 16 == 0;
  if (o && o->epi != kEpiNone) {
    a.epi = o->epi, a.e = o->e, a.lde = o->lde, a.q = o->q, a.qs = o->qs, a.ldq = o->ldy;
    a.evec = o->e && reinterpret_cast<uintptr_t>(o->e) % 16 == 0 && (o->lde * 2) % 16 == 0;
  }
  if (a.epi == kEpiSituQ128) {  // P134: 64 SiTU columns (gate + up rows) a CTA, a 2-CTA cluster a 128-column group
    a.situ = true, a.NT = L.n / 2 / 64;
    if (F32) qlog::fail("gemm", "kEpiSituQ128 has no fp32 output");
    switch (cfg_of(L, M)) {  // (not the persistent config)
      case 0: return launch<C0, F32, false>(a, s);
      case 1: return launch<C1, F32, false>(a, s);
      case 2: return launch<C2, F32, false>(a, s);
      case 5: return launch<C5, F32, false>(a, s);
      case 6: return launch<C6, F32, false>(a, s);
      default: return launch<C3, F32, false>(a, s);
    }
  }
  switch (cfg_of(L, M)) {
    case 1: return launch<C1, F32, false>(a, s);
    case 2: return launch<C2, F32, false>(a, s);
    case 3: return launch<C3, F32, false>(a, s);
    case 4: return launch<C4, F32, true>(a, s);
    case 5: return launch<C5, F32, false>(a, s);
    case 6: return launch<C6, F32, false>(a, s);
    default: return launch<C0, F32, false>(a, s);
  }
}

}  // namespace p133
