// p103_f8_tile.cuh — P103 f8_tile: the FP8 block GEMM for prefill (M ≥ 256), selected by patch::on(f8_tile) before
// P031; P104 f8_tile_epi: P060's epilogues (σ, + e, the MXFP8 copy) and z ⊙ gate → the next GEMM's FP8 activation
// (kEpiMulQ128) in this kernel's store instead of a k_epi / k_prep launch. Included by fp8.cu inside its anonymous
// namespace after p031 (mma, cp_async from mma.cuh; ldsm_x4 from p031; sigmoid_bf16 / bf16r from p060).
//
// Bitwise contract — every output element is the base kernel's: the same packed weights and 16-row tiles, the same MMA
// instruction with the same operands (swap-AB: weight tile = A, 8 tokens = B, the 128-block scale and the token's
// group scale as UE8M0), the k32 blocks of each K split chained in ascending order into one fp32 accumulator from 0,
// the fixed splits of the upload (L.split) summed v = p0, v += p1, … in split order, the same bf16 / fp32 output
// rounding; the epilogues are p060::k_epi's / k_prep's expressions on the same bf16 values (σ, h + e, amax over the
// same 32 / 128 values, act_scale_e8, quant_e4m3). Only the schedule differs:
//   · CTA tile 128 weight rows × 128 tokens, 4 warps as 2 × 2 with a 64 × 64 warp tile (4 × 8 MMA tiles: each A
//     fragment used 8 times, each B fragment 4 times from registers), 2 CTAs per SM — half P031's shared / L2 bytes
//     per MMA (Pico's prefill tile, DeepSeekV4.1-Flash-Pico kernels/prefill.cu, MIT: the idea, not its code);
//   · a K split of 2 as a thread-block cluster: CTA z of the cluster runs chain z of the same tile; the other rank
//     passes its half of the partial tile through L2 (the caller's workspace, discarded after use) to the owner warps,
//     which add in split order (p0 + p1; fp32 addition is commutative, so p0 + p_1 on either rank is the base's sum) —
//     no 128-register running sum, and twice the CTAs to fill the GPU at small M. (The code takes any S; split 4 stays
//     on P031: its 3 partial quarter tiles a CTA through L2 cost more than the larger tile saves — qkv M = 1024
//     0.109 → 0.122 ms measured; DSMEM instead of L2 measured ~80 GB/s);
//   · both operands through a 3-stage cp.async pipeline, 64 k bytes a stage (weights 8 tiles × 2 k32 blocks as
//     packed, tokens [128][64 B] XOR-swizzled for ldmatrix); the scales as 32-bit words of four k128 groups, staged
//     per 512 k beside the pipeline and held in registers for those 8 stages;
//   · programmatic launch (P033 f8_pdl): the first weight stages are requested before griddepcontrol.wait, the
//     activation after it; grouped raster (8 token tiles sweep the weight tiles: activations reused from L2);
//     P105 f8_tile_chain (late): the wait after the main loop instead — for sibling GEMMs on one activation;
//   · epilogue per owner warp through shared memory, 16-byte coalesced stores, the P104 op on the way out.
#pragma once

namespace p103 {
namespace cg = cooperative_groups;

constexpr int BN = 128, BM = 128, KS = 64, ST = 3, NTH = 128;  // weight rows, tokens, k bytes a stage, stages, threads
constexpr int MI = 4, NJ = 8;                                   // warp tile: 4 m16 weight tiles × 8 n8 token tiles
constexpr int AST = BN * KS, BST = BM * KS, STAGE = AST + BST;  // 8 KB + 8 KB
constexpr int SCB = (BN / 16 + BM) * 4;                         // a 512-k block of scale words: 8 tiles + 128 tokens
constexpr int SMEM = ST * STAGE + SCB;                          // 48.5 KB: 2 CTAs per SM (≤ 50,176 B each)
constexpr int GM = 8;                                           // token tiles per raster group
constexpr int PW = 72;                                          // staged bf16 row of a warp's 64 weight rows (+8)
constexpr int kWarpPart = 32 * MI * NJ * 4;                     // floats of one warp's partial tile (16 KB)
static_assert(4 * 64 * PW * 2 + 2 * 2 * 64 * 4 <= SMEM, "epilogue staging");

struct Args {
  const uint8_t* w;
  const uint8_t* ws;
  int n, k, split, rows_per_group, tiles;  // tiles = npad / 16
  const uint8_t* xq;
  const uint8_t* xs;
  int ld;
  long gx, gs;
  int M, NT, MT;  // NT weight-row tiles of 128, MT token tiles of 128
  void* y;        // may be null with kEpiMulQ128 (only the FP8 copy is wanted)
  int ldy;
  int epi;        // F8Epi (P104); kEpiNone without it
  const bf16* e;  // kEpiAdd: + e; kEpiMulQ128: ⊙ e
  int lde;
  uint8_t* q;     // kEpiQ32 / kEpiMulQ128: e4m3 [M][ldq], scales [M][ldq / group]
  uint8_t* qs;
  int ldq;
  bool vec, evec;  // y / e rows 16-byte aligned
  float* ws32;     // split > 1: the partials' exchange, ≥ ws_bytes()
  bool late;       // P105 f8_tile_chain: griddepcontrol.wait after the main loop instead of before the activation
};

__device__ __forceinline__ void cp_async4(void* smem, const void* gmem, bool ok) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(s), "l"(gmem), "r"(ok ? 4 : 0));
}

// the token half's two warps (named barriers 1 / 2 by immediate: a barrier id in a register makes ptxas reserve all
// 16, one CTA per SM)
__device__ __forceinline__ void pair_sync(int wt) {
  if (wt == 0) asm volatile("bar.sync 1, 64;\n" ::: "memory");
  else asm volatile("bar.sync 2, 64;\n" ::: "memory");
}

template <bool F32, int S>
__global__ void __launch_bounds__(NTH, 2) k_tile(Args a) {
  extern __shared__ __align__(128) uint8_t sm[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wr = warp & 1, wt = warp >> 1;  // the warp's 64 weight rows / 64 tokens of the tile
  const int z = S > 1 ? int(blockIdx.x % S) : 0;
  const int tile = blockIdx.x / S;
  const int per = GM * a.NT, grp = tile / per, fm = grp * GM, gm = min(a.MT - fm, GM);
  const int rem = tile - grp * per, mt = fm + rem % gm, nt = rem / gm;
  const int T0 = nt * (BN / 16), r0 = nt * BN, t0 = mt * BM;
  const int KB = a.k >> 5, KG = a.k >> 7, sld = a.ld >> 7;
  const int nks = KB / S, kb0 = z * nks, nst = nks >> 1;  // this CTA's chain: k32 blocks kb0 … kb0 + nks − 1
  const int kgrp = r0 / a.rows_per_group;
  const uint8_t* xq = a.xq + kgrp * a.gx;
  const uint8_t* xs = a.xs + kgrp * a.gs;

  // copy assignments: weight chunk p = tid + 128·i (tile p >> 6, 16 B (p & 63) of its 1 KB), token chunk p = tid + 128·i
  // (token p >> 2, 16-B chunk p & 3, stored swizzled)
  const int atl = tid >> 6;  // + 2i
  const uint8_t* asrc = a.w + ((size_t)(T0 + atl) * KB + kb0) * 512 + (tid & 63) * 16;
  const size_t astep = (size_t)2 * KB * 512;
  const int brow = tid >> 2, bc = tid & 3;  // + 32i
  const uint8_t* bsrc = xq + (size_t)(t0 + brow) * a.ld + kb0 * 32 + bc * 16;
  const uint32_t bdst = brow * KS + ((bc ^ ((brow >> 1) & 3)) << 4);
  auto load_a = [&](int st, int buf) {
    uint8_t* sa = sm + buf * STAGE;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const bool ok = T0 + atl + 2 * i < a.tiles;
      cp_async16(sa + (tid + 128 * i) * 16, ok ? asrc + i * astep + st * 1024 : a.w, ok);
    }
  };
  auto load_b = [&](int st, int buf) {
    uint8_t* sb = sm + buf * STAGE + AST;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const bool ok = t0 + brow + 32 * i < a.M;
      cp_async16(sb + bdst + i * 32 * KS, ok ? bsrc + (size_t)i * 32 * a.ld + st * KS : a.xq, ok);
    }
  };
  // scales: 32-bit words of 4 k128 groups (chains start at multiples of 16 k32 blocks) — a 512-k block's words of the
  // 8 weight tiles and 128 tokens staged in shared memory (one buffer: each warp copies its words to registers at the
  // block's first stage, the next block's words are requested one stage later, landing ≥ 5 stages before use)
  const int kw0 = kb0 >> 4;
  uint32_t* sca = reinterpret_cast<uint32_t*>(sm + ST * STAGE);  // [8 tiles]
  uint32_t* scb = sca + BN / 16;                                  // [128 tokens]
  auto load_sa = [&](int kw) {
    if (tid < BN / 16) {
      const bool ok = T0 + tid < a.tiles;
      cp_async4(sca + tid, ok ? a.ws + (size_t)(T0 + tid) * KG + (kw0 + kw) * 4 : a.ws, ok);
    }
  };
  auto load_sb = [&](int kw) {
    const bool ok = t0 + tid < a.M;
    cp_async4(scb + tid, ok ? xs + (size_t)(t0 + tid) * sld + (kw0 + kw) * 4 : a.xs, ok);
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

  float acc[MI][NJ][4];
#pragma unroll
  for (int i = 0; i < MI; i++)
#pragma unroll
    for (int j = 0; j < NJ; j++) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

  const int aoff = (wr * MI) * 1024 + lane * 16;  // + i·1024 + kk·512
  // ldmatrix: lane → matrix lane >> 3 (0: tile j lo, 1: tile j hi, 2: tile j+1 lo, 3: tile j+1 hi), row lane & 7
  const int lrow = wt * 64 + ((lane >> 4) << 3) + (lane & 7), lhi = (lane >> 3) & 1;
  for (int st = 0; st < nst; st++) {
    cp_async_wait<ST - 2>();
    __syncthreads();  // stage st landed for everyone; buffer (st − 1) % ST is free
    if ((st & 7) == 0) {  // a new 512-k block: its scale words to registers
#pragma unroll
      for (int i = 0; i < MI; i++) wa[i] = sca[4 * wr + i];
#pragma unroll
      for (int j = 0; j < NJ; j++) wb[j] = scb[64 * wt + 8 * j + g];
    } else if ((st & 7) == 1 && st + 7 < nst) {  // every warp has its words: request the next block's
      load_sa((st >> 3) + 1), load_sb((st >> 3) + 1);
    }
    if (st + ST - 1 < nst) load_a(st + ST - 1, (st + ST - 1) % ST), load_b(st + ST - 1, (st + ST - 1) % ST);
    cp_async_commit();
    const uint8_t* sa = sm + (st % ST) * STAGE;
    const uint8_t* sb = sa + AST;
    const int sh = 8 * ((st >> 1) & 3);  // this stage's k128 group within the word
    uint32_t sfa[MI], sfb[NJ];
#pragma unroll
    for (int i = 0; i < MI; i++) sfa[i] = (wa[i] >> sh) & 0xFFu;
#pragma unroll
    for (int j = 0; j < NJ; j++) sfb[j] = (wb[j] >> sh) & 0xFFu;
#pragma unroll
    for (int kk = 0; kk < 2; kk++) {
      uint4 A[MI];
#pragma unroll
      for (int i = 0; i < MI; i++) A[i] = *reinterpret_cast<const uint4*>(sa + aoff + i * 1024 + kk * 512);
      uint32_t B[NJ][2];
#pragma unroll
      for (int jj = 0; jj < NJ; jj += 2) {
        const int r = lrow + jj * 8, c = kk * 2 + lhi;
        p031::ldsm_x4(B[jj][0], B[jj][1], B[jj + 1][0], B[jj + 1][1], sb + r * KS + ((c ^ ((r >> 1) & 3)) << 4));
      }
#pragma unroll
      for (int j = 0; j < NJ; j++)
#pragma unroll
        for (int i = 0; i < MI; i++)
          mma_e4m3_e4m3(acc[i][j], A[i].x, A[i].y, A[i].z, A[i].w, B[j][0], B[j][1], sfa[i], sfb[j]);
    }
  }
  cp_async_wait<0>();
  __syncthreads();
  // P105: a sibling GEMM on the previous one's activation (nothing of it read or written) ran its main loop under the
  // previous GEMM's last wave; it waits here, before its first write, so it cannot complete before its predecessor
  if (a.late) asm volatile("griddepcontrol.wait;\n" ::: "memory");

  if constexpr (S > 1) {  // the cluster's chains → each owner warp: p0 + p1 + … in split order
    // owner of warp w's subtile: S = 2 — rank w >> 1 (token half), S = 4 — rank w. The other ranks' warps w write their
    // partials to the workspace (L2: [tile][warp w][slot][16 KB], slot = rank − (rank > owner)), one cluster barrier
    // (release / acquire at cluster scope), the owner reads them back and drops the lines from L2 (discard: no DRAM
    // write-back).
    const int orank = S == 2 ? wt : warp;
    const bool own = orank == z;
    float4* part = reinterpret_cast<float4*>(a.ws32) + ((size_t)tile * 4 + warp) * (S - 1) * (kWarpPart / 4) + lane;
    if (!own) {
      float4* d = part + (z < orank ? z : z - 1) * (kWarpPart / 4);
#pragma unroll
      for (int i = 0; i < MI; i++)
#pragma unroll
        for (int j = 0; j < NJ; j++)
          __stcg(d + (i * NJ + j) * 32, make_float4(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3]));
    }
    cg::this_cluster().sync();  // partials in L2
    if (!own) return;
#pragma unroll
    for (int i = 0; i < MI; i++)
#pragma unroll
      for (int j = 0; j < NJ; j++) {
        const int o = (i * NJ + j) * 32;
        float* c = acc[i][j];
        float4 v[S - 1];
#pragma unroll
        for (int u = 0; u < S - 1; u++) v[u] = __ldcg(part + u * (kWarpPart / 4) + o);
        if (z > 0) {  // pre = p0 + … + p_{z−1} (slots 0 … z − 1), then pre + p_z
          float4 pre = v[0];
#pragma unroll
          for (int u = 1; u < S - 1; u++)
            if (u < z) pre.x += v[u].x, pre.y += v[u].y, pre.z += v[u].z, pre.w += v[u].w;
          c[0] = pre.x + c[0], c[1] = pre.y + c[1], c[2] = pre.z + c[2], c[3] = pre.w + c[3];
        }
#pragma unroll
        for (int u = 0; u < S - 1; u++)  // ranks after z: slots z … S − 2
          if (u >= z) c[0] += v[u].x, c[1] += v[u].y, c[2] += v[u].z, c[3] += v[u].w;
      }
    // the lines are dead: drop them from L2 without a write-back (128-B lines, 4 per lane per slot; read by other
    // lanes of the warp: __syncwarp orders those reads first)
    __syncwarp();
#pragma unroll
    for (int u = 0; u < S - 1; u++) {
      const char* base = reinterpret_cast<const char*>(part - lane + u * (kWarpPart / 4));
#pragma unroll
      for (int l = 0; l < 4; l++)
        asm volatile("discard.global.L2 [%0], 128;\n" ::"l"(base + (lane * 4 + l) * 128) : "memory");
    }
  }

  // ---- epilogue: this warp's 64 rows × 64 tokens ----
  const int rw = r0 + 64 * wr, tw = t0 + 64 * wt;
  if constexpr (F32) {
#pragma unroll
    for (int i = 0; i < MI; i++)
#pragma unroll
      for (int j = 0; j < NJ; j++)
#pragma unroll
        for (int e = 0; e < 4; e++) {
          const int r = rw + i * 16 + g + (e >> 1) * 8, m = tw + j * 8 + 2 * t + (e & 1);
          if (r < a.n && m < a.M) static_cast<float*>(a.y)[(size_t)m * a.ldy + r] = acc[i][j][e];
        }
    return;
  } else {
    bf16* Ps = reinterpret_cast<bf16*>(sm) + warp * 64 * PW;  // [64 tokens][PW]
#pragma unroll
    for (int i = 0; i < MI; i++)
#pragma unroll
      for (int j = 0; j < NJ; j++) {
        const int rr = i * 16 + g, tt = j * 8 + 2 * t;
        Ps[tt * PW + rr] = __float2bfloat16_rn(acc[i][j][0]);
        Ps[(tt + 1) * PW + rr] = __float2bfloat16_rn(acc[i][j][1]);
        Ps[tt * PW + rr + 8] = __float2bfloat16_rn(acc[i][j][2]);
        Ps[(tt + 1) * PW + rr + 8] = __float2bfloat16_rn(acc[i][j][3]);
      }
    __syncwarp();
    const int c = lane & 7, r = rw + 8 * c;  // lane: 8 consecutive rows (16 B) of token 4·it + (lane >> 3)
    const bool rok = r < a.n;
    float* am = reinterpret_cast<float*>(sm + 4 * 64 * PW * 2) + (wt * 2) * 64;  // kEpiMulQ128: [wr][64 tokens] amax
    if (a.epi == kEpiMulQ128) {  // pass 1: this warp's 64-row amax of bf16(h ⊙ e) per token, swapped with the partner
#pragma unroll 4
      for (int it = 0; it < 16; it++) {
        const int tl = 4 * it + (lane >> 3), m = tw + tl;
        const uint4 hv = *reinterpret_cast<const uint4*>(Ps + tl * PW + 8 * c);
        float mx = 0.f;
        if (m < a.M && rok) {
          const uint4 ev = *reinterpret_cast<const uint4*>(a.e + (size_t)m * a.lde + r);  // (evec checked on the host)
          const uint32_t hw[4] = {hv.x, hv.y, hv.z, hv.w}, ew[4] = {ev.x, ev.y, ev.z, ev.w};
#pragma unroll
          for (int u = 0; u < 4; u++)
            mx = fmaxf(mx, fmaxf(fabsf(p060::bf16r(bf16_lo(hw[u]) * bf16_lo(ew[u]))),
                                 fabsf(p060::bf16r(bf16_hi(hw[u]) * bf16_hi(ew[u])))));
        }
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));
        if (c == 0) am[wr * 64 + tl] = mx;
      }
      pair_sync(wt);
    }
#pragma unroll 2
    for (int it = 0; it < 16; it++) {
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
      } else if (a.epi == kEpiMulQ128) {  // v = bf16(h ⊙ e), quantized over the 128 rows of the warp pair
        float f[8];
        if (ok) {
          const uint4 ev = *reinterpret_cast<const uint4*>(a.e + (size_t)m * a.lde + r);
          const uint32_t hw[4] = {hv.x, hv.y, hv.z, hv.w}, ew[4] = {ev.x, ev.y, ev.z, ev.w};
#pragma unroll
          for (int u = 0; u < 4; u++)
            f[2 * u] = p060::bf16r(bf16_lo(hw[u]) * bf16_lo(ew[u])),
            f[2 * u + 1] = p060::bf16r(bf16_hi(hw[u]) * bf16_hi(ew[u]));
          const uint32_t ex = act_scale_e8(fmaxf(am[tl], am[64 + tl]), 1e-10f);
          uint32_t qw[2] = {0, 0};
#pragma unroll
          for (int u = 0; u < 8; u++) qw[u >> 2] |= quant_e4m3(f[u], ex) << (8 * (u & 3));
          *reinterpret_cast<uint2*>(a.q + (size_t)m * a.ldq + r) = make_uint2(qw[0], qw[1]);
          if (wr == 0 && c == 0) a.qs[(size_t)m * (a.ldq / 128) + r / 128] = uint8_t(ex);
        }
        if (!a.y) continue;
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
  }
}

template <bool F32, int S>
void launch(const Args& a, cudaStream_t s) {
  static bool attr = false;
  if (!attr) {
    CUDA_CHECK(cudaFuncSetAttribute(k_tile<F32, S>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    attr = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(unsigned(a.NT * a.MT * S));
  cfg.blockDim = dim3(NTH);
  cfg.dynamicSmemBytes = SMEM;
  cfg.stream = s;
  cudaLaunchAttribute at[2];
  int na = 0;
  if (S > 1) {
    at[na].id = cudaLaunchAttributeClusterDimension;
    at[na].val.clusterDim.x = S, at[na].val.clusterDim.y = 1, at[na].val.clusterDim.z = 1;
    na++;
  }
  if (e55::patch::on(e55::patch::f8_pdl)) {  // P033: may start while the previous kernel runs (waits in the kernel)
    at[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[na].val.programmaticStreamSerializationAllowed = 1;
    na++;
  }
  cfg.attrs = at;
  cfg.numAttrs = na;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, k_tile<F32, S>, a));
}

// workspace bytes for the split partials: (split − 1) × tiles × 64 KB — within f8_ws(L, M) (= split × M × n × 4) when
// M and n fill their 128-tiles closely enough (checked)
inline size_t ws_bytes(const F8Lin& L, int M) {
  return L.split > 1 ? (size_t)(L.split - 1) * cdiv(L.n, BN) * cdiv(M, BM) * (BN * BM * 4) : 0;
}
// can this path run (L, M, activation, workspace)? M ≥ 256; split 1 or 2 with chains of whole 512-k words; a group
// of rows covers whole 128-row tiles; 16-byte activation rows and 4-byte scale words; split > 1: the workspace
inline bool ok(const F8Lin& L, int M, const F8Act& x, long gx, long gs, const float* ws) {
  if (M < 256 || L.k % 512 || L.split > 2 || (L.k / 32 / L.split) % 16) return false;
  if (L.split > 1 && (!ws || reinterpret_cast<uintptr_t>(ws) % 128 || ws_bytes(L, M) > f8_ws(L, M))) return false;
  if (L.groups > 1 && (L.n / L.groups) % BN) return false;
  if (x.ld % 512 || reinterpret_cast<uintptr_t>(x.q) % 16 || reinterpret_cast<uintptr_t>(x.s) % 4) return false;
  return gx % 16 == 0 && gs % 4 == 0;
}
// a plain GEMM (no P104 epilogue / P105 chain to keep) takes P103 unless split 2 with K ≤ 1024 below 2048 rows: 8 stages
// a chain leave the L2 exchange too large a share (shared down 4096 × 1024: M = 512 / 1024 0.057 / 0.100 ms on P031,
// 0.059 / 0.104 on P103; M = 2048 / 4096 0.215 / 0.426 → 0.206 / 0.405)
inline bool prefer(const F8Lin& L, int M) { return !(L.split == 2 && L.k <= 1024 && M < 2048); }
// with P104's epilogue op (bf16 out): kEpiQ32 needs n % 32, kEpiMulQ128 n % 128 (the 128-row group in one CTA), the
// ⊙ operand's rows 16-byte aligned. (fp8.cu takes kEpiMulQ128 only at M ≥ 2048: wo_a → o_proj measured 0.326 / 0.561
// → 0.346 / 0.585 ms at M = 512 / 1024 against P103 + P060's k_prep, 1.135 / 2.235 → 1.111 / 2.135 at 2048 / 4096.)
inline bool epi_ok(const F8Lin& L, int epi, const bf16* e, int lde) {
  if (epi == kEpiQ32) return L.n % 32 == 0;
  if (epi == kEpiMulQ128)
    return L.n % 128 == 0 && reinterpret_cast<uintptr_t>(e) % 16 == 0 && (lde * 2) % 16 == 0;
  return epi == kEpiNone || epi == kEpiSigmoid || epi == kEpiAdd;
}

template <bool F32>
void forward(const F8Lin& L, const F8Act& x, int M, void* y, int ldy, float* ws, cudaStream_t s, long gx, long gs,
             const F8Out* o = nullptr, bool late = false) {
  const size_t es = F32 ? 4 : 2;
  Args a{};
  a.w = L.w, a.ws = L.s, a.n = L.n, a.k = L.k, a.split = L.split, a.rows_per_group = L.n / L.groups;
  a.tiles = L.npad / 16, a.xq = x.q, a.xs = x.s, a.ld = x.ld, a.gx = gx, a.gs = gs, a.M = M;
  a.NT = cdiv(L.n, BN), a.MT = cdiv(M, BM), a.y = y, a.ldy = ldy, a.ws32 = ws, a.late = late;
  a.vec = y && (ldy * es) % 16 == 0 && reinterpret_cast<uintptr_t>(y) % 16 == 0;
  if (o && o->epi != kEpiNone) {
    a.epi = o->epi, a.e = o->e, a.lde = o->lde, a.q = o->q, a.qs = o->qs, a.ldq = o->ldy;
    a.evec = o->e && reinterpret_cast<uintptr_t>(o->e) % 16 == 0 && (o->lde * 2) % 16 == 0;
  }
  if (L.split == 2) launch<F32, 2>(a, s);
  else launch<F32, 1>(a, s);
}

}  // namespace p103
