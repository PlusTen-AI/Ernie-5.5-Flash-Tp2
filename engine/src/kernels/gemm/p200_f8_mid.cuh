// p200_f8_mid.cuh — P200 f8_mid: P030's decode FP8 block GEMM for 33 … 128 rows (verify steps of 4 / 8 / 16
// sequences: 32 / 64 / 128 rows, their drafts 28 / 56 / 112), selected by patch::on(f8_mid) and fp8.cu mid_takes (the measured rule). Below it P030 (M ≤ 32),
// today at these rows the base kernel (k_f8<64 / 128> + k_reduce) or P031's prefill tiles.
// Included by fp8.cu inside its anonymous namespace after p030_f8_decode.cuh (uses its Args / Part / put_out /
// ldg_stream / kD / g_pfn, and mma, cp_async, act_scale_e8, quant_e4m3 from mma.cuh).
//
// Bitwise contract — every output element is P030's (= the base kernel's): the same packed weights and 16-row tiles,
// the same MMA instruction with the same operands (weight tile = A, 8 tokens = B, the tile's 128-block scale and the
// token's group scale as UE8M0), the k32 blocks of one K split chained in ascending order into one fp32 accumulator
// from 0, the fixed splits of the upload (L.split), the split sum v = p0 + p1 + … in rank order z = 0..S−1 through
// distributed shared memory (P030's cluster loop, unchanged), the same bf16 / fp32 output rounding and P060 epilogues
// (put_out: σ / + e; kEpiQ32 in the cluster loop). A row's result does not depend on M or on the other rows. Only the
// schedule differs from P030:
//   · token tiles of MT = 32 / 64 (/ 128) rows, H = ⌈M / MT⌉ CTAs a 64-row weight block, adjacent in blockIdx.y: each
//     weight fragment streamed to registers (P030's kD = 8 k32 blocks in flight a warp, L1 bypassed) feeds MT / 8
//     MMAs, and the block's other tiles read it from L2, so the weights come from DRAM about once a call (mt_for);
//   · the activation does not fit in shared memory whole (P030 stages MT × K/S bytes): it comes through a 2-stage
//     cp.async ring of 128 k a stage (one scale group: the stage's scales are read once into registers); the
//     activation and weight scales of the CTA's k range are staged once at the start (as P030);
//   · P033 programmatic launch, P034 several weights in one grid (P030's cluster units), P064 L2 prefetch after
//     griddepcontrol.wait — as P030. No P032 fused quantization and no P060 Aux unit (M > 8: the caller's p060::prep).
#pragma once

namespace p200 {
namespace cg = cooperative_groups;
using p030::Args;
using p030::kD;
using p030::Part;

constexpr int kKS = 128;         // k bytes per ring stage: 4 k32 blocks, one 128-group scale
constexpr int kNS = 2;           // ring stages
constexpr int kRB = kKS + 16;    // staged row bytes: fragment reads conflict-free (bank 4g + t)
constexpr int kMaxKgl = 32;      // k range of a CTA ≤ 4096 (scales staged once: MT × kgl bytes)
constexpr int kMinM = 33, kMaxM = 128;

// shared memory: ring [kNS][MT][kRB] (the cluster's partial tile [MT][68] fp32 aliases it) | activation scales
// [MT][kgl] | weight block scales [4][kgl]
__host__ __device__ constexpr int ring_bytes(int MT) {
  const int ring = kNS * MT * kRB, part = MT * 68 * 4;
  return ((ring > part ? ring : part) + 15) / 16 * 16;
}
__host__ __device__ constexpr int smem_bytes(int MT, int kgl) { return ring_bytes(MT) + (MT + 4) * kgl; }

template <int MT, bool F32>
__global__ void __launch_bounds__(128) k_mid(Args a) {
  extern __shared__ __align__(16) uint8_t smem[];
  constexpr int NT = MT / 8;
  static_assert(kD == 8 && kKS == 128, "the register ring covers two stages");
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int H = gridDim.y / a.units, u = blockIdx.y / H;  // H token tiles of a weight block: adjacent y (L2 reuse)
  const int t0 = (blockIdx.y - u * H) * MT, Mv = min(MT, a.M - t0);  // this CTA's token rows t0 … t0 + Mv − 1
  Part P = a.p[0];
#pragma unroll
  for (int i = 1; i < p030::kParts; i++)
    if (i < a.np && u >= a.p[i].ubeg) P = a.p[i];
  const int S = P.split, z = blockIdx.x, zl = z % S, b = z / S;  // K split, block of the unit (P030's units)
  const int blk = (u - P.ubeg) * (a.smax / S) + b;                // this CTA's 64-row block
  if (blk >= P.blocks) {  // past the weight's last block: idle, but in its cluster's barriers
    if (S > 1) cg::this_cluster().sync(), cg::this_cluster().sync();
    return;
  }
  const int T = blk * 4 + warp;  // this warp's 16-row weight tile
  const int r0 = blk * 64;
  const int klen = a.k / S, kbeg = zl * klen;  // this CTA's k range
  const int nkb = klen >> 5, kgl = klen >> 7;  // k32 blocks, 128-groups = ring stages (even: klen % 256 == 0)
  const int KB = a.k >> 5, KG = a.k >> 7;
  uint8_t* ssx = smem + ring_bytes(MT);  // [MT][kgl] activation scales
  uint8_t* ssw = ssx + MT * kgl;         // [4][kgl] weight block scales

  // 1. the weight stream starts before anything else (P030)
  const uint4* wt = P.w + ((size_t)T * KB + (kbeg >> 5)) * 32 + lane;
  uint4 wr[kD];
#pragma unroll
  for (int i = 0; i < kD; i++) wr[i] = p030::ldg_stream(wt + i * 32);

  // 2. block scales of the 4 tiles; wait for the previous kernel (P033), let the next one start; P064 prefetch; the
  //    first ring stage and all activation scales of the k range (rows past M: zeros)
  for (int c = threadIdx.x; c < 4 * kgl; c += 128)
    ssw[c] = P.ws[(size_t)(blk * 4 + c / kgl) * KG + (kbeg >> 7) + c % kgl];
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;\n" :::);
  if (a.pfn && lane == 0 && nkb > kD) {
    const uint32_t bytes = uint32_t(min(nkb - kD, a.pfn)) * 512u;
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(wt - lane + (size_t)kD * 32), "r"(bytes) : "memory");
  }
  const int grp = r0 / P.rows_per_group;
  const uint8_t* xq = a.xq + grp * a.gx + (size_t)t0 * a.ld + kbeg;
  const uint8_t* xs = a.xs + grp * a.gs + (size_t)t0 * (a.ld >> 7);
  auto load = [&](int st) {  // ring stage st: MT rows × 128 k bytes → buffer st % 2
    uint8_t* dst = smem + (st & 1) * (MT * kRB);
    for (int c = threadIdx.x; c < MT * (kKS / 16); c += 128) {
      const int row = c >> 3, ch = c & 7;
      const bool ok = row < Mv;
      cp_async16(dst + row * kRB + ch * 16, xq + (size_t)(ok ? row : 0) * a.ld + st * kKS + ch * 16, ok);
    }
    cp_async_commit();
  };
  load(0);
  {
    const int sld = a.ld >> 7;
    for (int c = threadIdx.x; c < MT * kgl; c += 128) {
      const int row = c / kgl, j = c - row * kgl;
      ssx[c] = row < Mv ? xs[(size_t)row * sld + (kbeg >> 7) + j] : 0;
    }
  }

  // 3. the chain: k32 blocks of this split in ascending order, kD weight blocks in flight (two ring stages)
  float acc[NT][4];
#pragma unroll
  for (int i = 0; i < NT; i++) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
  const uint8_t* bs = ssx + g * kgl;      // token g's scales (+ nt · 8 · kgl)
  const uint8_t* wsc = ssw + warp * kgl;  // this tile's block scales
  for (int st0 = 0; st0 < kgl; st0 += 2) {
#pragma unroll
    for (int h = 0; h < 2; h++) {
      const int st = st0 + h;
      cp_async_wait<0>();
      __syncthreads();  // stage st landed (and the scales, first time); the other buffer's readers are done
      if (st + 1 < kgl) load(st + 1);
      const uint8_t* bx = smem + (st & 1) * (MT * kRB) + g * kRB + 4 * t;
      const uint32_t sfa = wsc[st];
      uint32_t sfb[NT];
#pragma unroll
      for (int nt = 0; nt < NT; nt++) sfb[nt] = bs[nt * 8 * kgl + st];
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int kb = st * 4 + i;  // k32 block of the split; it sits in wr[(kb) % kD] = wr[h · 4 + i]
        const uint4 wa = wr[h * 4 + i];
        if (kb + kD < nkb) wr[h * 4 + i] = p030::ldg_stream(wt + (size_t)(kb + kD) * 32);
#pragma unroll
        for (int nt = 0; nt < NT; nt++) {
          const uint8_t* p = bx + nt * 8 * kRB + i * 32;
          const uint32_t b0 = *reinterpret_cast<const uint32_t*>(p);
          const uint32_t b1 = *reinterpret_cast<const uint32_t*>(p + 16);
          mma_e4m3_e4m3(acc[nt], wa.x, wa.y, wa.z, wa.w, b0, b1, sfa, sfb[nt]);
        }
      }
    }
  }

  // 4. epilogue (P030's with t0 = 0): D[row g / g+8][token 2t / 2t+1]
  if (S == 1) {
    const int ra = 16 * T + g, rb = ra + 8;
#pragma unroll
    for (int nt = 0; nt < NT; nt++) {
      const int tk = nt * 8 + 2 * t;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int r = i < 2 ? ra : rb, m = tk + (i & 1);
        if (r >= P.n || m >= Mv) continue;
        p030::put_out<F32>(P, t0 + m, r, acc[nt][i]);
      }
    }
    return;
  }
  // cluster: partial tile Pt[token][row of the 64] in shared memory, summed over the block's ranks in split order
  __syncthreads();  // Pt aliases the ring
  float* Pt = reinterpret_cast<float*>(smem);
  {
    const int ra = warp * 16 + g;
#pragma unroll
    for (int nt = 0; nt < NT; nt++) {
      const int m = nt * 8 + 2 * t;
      Pt[m * 68 + ra] = acc[nt][0];
      Pt[(m + 1) * 68 + ra] = acc[nt][1];
      Pt[m * 68 + ra + 8] = acc[nt][2];
      Pt[(m + 1) * 68 + ra + 8] = acc[nt][3];
    }
  }
  cg::cluster_group cl = cg::this_cluster();
  cl.sync();
  const int mv = Mv, r0k = b * S;  // the block's ranks: r0k … r0k + S − 1
  for (int e = zl * 128 + threadIdx.x; e < mv * 64; e += S * 128) {  // a warp: 32 consecutive rows of one token
    const int m = t0 + (e >> 6), rr = e & 63, r = r0 + rr;
    if (P.epi != kEpiQ32 && r >= P.n) continue;  // (kEpiQ32: n % 64 == 0, checked on the host)
    const int o = (e >> 6) * 68 + rr;
    float v = cl.map_shared_rank(Pt, r0k)[o];
    for (int s = 1; s < S; s++) v += cl.map_shared_rank(Pt, r0k + s)[o];
    p030::put_out<F32>(P, m, r, v);
    if (!F32 && P.epi == kEpiQ32) {  // act_quant group 32 of the bf16 output: amax over the warp's 32 rows
      const float hv = p060::bf16r(v);
      const uint32_t ex = act_scale_e8(warp_max(fabsf(hv)), 1e-10f);
      P.q[(size_t)m * P.ldy + r] = uint8_t(quant_e4m3(hv, ex));
      if (lane == 0) P.qs[(size_t)m * (P.ldy / 32) + r / 32] = uint8_t(ex);
    }
  }
  cl.sync();  // keep every rank's shared memory alive until all have read it
}

template <int MT, bool F32>
void launch(const Args& a, int units, int H, int smem, cudaStream_t s) {
  static bool attr = false;
  if (!attr) {
    CUDA_CHECK(cudaFuncSetAttribute(k_mid<MT, F32>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_bytes(MT, kMaxKgl)));
    attr = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(a.smax, units * H, 1);
  cfg.blockDim = dim3(128);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = s;
  cudaLaunchAttribute at[2];
  int na = 0;
  if (a.smax > 1) {
    at[na].id = cudaLaunchAttributeClusterDimension;
    at[na].val.clusterDim.x = a.smax, at[na].val.clusterDim.y = 1, at[na].val.clusterDim.z = 1;
    na++;
  }
  if (e55::patch::on(e55::patch::f8_pdl)) {  // P033: may start while the previous kernel runs (waits in the kernel)
    at[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[na].val.programmaticStreamSerializationAllowed = 1;
    na++;
  }
  cfg.attrs = at;
  cfg.numAttrs = na;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, k_mid<MT, F32>, a));
}

int g_mt = 0;  // tests (f8_test_mid): this token tile (32 / 64 / 128) wherever P200 can run (0: the measured rule)

// the token tile MT for weight L at M rows: H = ⌈M / MT⌉ CTAs a 64-row weight block, adjacent in blockIdx.y, so the
// weights come from DRAM once and the other tiles' reads hit L2. Measured on B (cold, 2026-11-28): MT 128 is never the
// fastest (40 KB of shared memory: 2 CTAs an SM, 128+ CTAs a tile → 2 waves); MT 32 is at 33 … 64 rows, MT 64 at
// 65 … 128 except on long split-1 chains (dense gate/up 14336 × 4096: MT 32, 272 vs 280 µs at 128 rows)
inline int mt_for(const F8Lin& L, int M) {
  if (g_mt) return g_mt;
  if (M <= 64) return 32;
  return L.split == 1 && L.k >= 4096 ? 32 : 64;
}

// can this path run (L, M)? (whether it should: fp8.cu mid_takes, the measured rule) 33 … 128 rows, cluster ≤ 8 ranks (P030's), a CTA's k range ≤ 4096 (staged scales)
// P220 f8_mid32: from 17 rows (fp8.cu tries P030 first, so P200 takes only the shapes P030 cannot stage at MT 32)
constexpr int kMinM32 = 17;
inline bool ok(const F8Lin& L, int M) {
  const int lo = e55::patch::on(e55::patch::f8_mid32) ? kMinM32 : kMinM;
  if (M < lo || M > kMaxM || L.split > 8) return false;
  return (L.k / L.split) / 128 <= kMaxKgl;
}

// one launch for weights Ls[0..n) on the same quantized activation (n ≤ kParts, same k; every split divides the
// largest); outs (P060, may be null): each weight's epilogue (σ / + e in put_out, kEpiQ32 on the cluster path)
template <bool F32>
void forward_parts(const F8Lin* const* Ls, void* const* ys, const int* ldys, int n, const F8Act& x, int M,
                   cudaStream_t s, long gx, long gs, const F8Out* outs = nullptr) {
  if (x.x || x.pro) qlog::fail("gemm", "P200 takes a quantized activation (M %d)", M);
  Args a{};
  a.np = n, a.k = Ls[0]->k, a.xq = x.q, a.xs = x.s, a.ld = x.ld, a.gx = gx, a.gs = gs, a.M = M, a.smax = 1;
  for (int i = 0; i < n; i++) a.smax = std::max(a.smax, Ls[i]->split);
  int MT = 0;  // (several weights: the largest tile any of them takes)
  for (int i = 0; i < n; i++) MT = std::max(MT, mt_for(*Ls[i], M));
  int units = 0, smem = 0;
  for (int i = 0; i < n; i++) {
    const F8Lin& L = *Ls[i];
    Part& p = a.p[i];
    p.w = reinterpret_cast<const uint4*>(L.w), p.ws = L.s, p.y = ys[i], p.ldy = ldys[i];
    p.n = L.n, p.split = L.split, p.rows_per_group = L.n / L.groups, p.blocks = L.npad / 64, p.ubeg = units;
    if (outs) p.epi = outs[i].epi, p.e = outs[i].e, p.lde = outs[i].lde, p.q = outs[i].q, p.qs = outs[i].qs;
    units += cdiv(p.blocks, a.smax / L.split);
    smem = std::max(smem, smem_bytes(MT, L.k / L.split / 128));
  }
  a.units = units;
  a.pfn = e55::patch::on(e55::patch::f8_pf_next) ? p030::g_pfn : 0;
  const int H = cdiv(M, MT);
  if (MT == 32) return launch<32, F32>(a, units, H, smem, s);
  if (MT == 64) return launch<64, F32>(a, units, H, smem, s);
  return launch<128, F32>(a, units, H, smem, s);
}

template <bool F32>
void forward(const F8Lin& L, const F8Act& x, int M, void* y, int ldy, cudaStream_t s, long gx, long gs) {
  const F8Lin* Ls[1] = {&L};
  void* ys[1] = {y};
  const int ldys[1] = {ldy};
  forward_parts<F32>(Ls, ys, ldys, 1, x, M, s, gx, gs);
}

// P034 with P200: may these weights run as one launch? (each would take this path alone; same k; ungrouped;
// splits divide the largest)
inline bool multi_ok(const F8Lin* const* Ls, int n, int M) {
  if (n < 2 || n > p030::kParts) return false;
  int smax = 1;
  for (int i = 0; i < n; i++) {
    const F8Lin& L = *Ls[i];
    if (L.k != Ls[0]->k || L.groups != 1 || !ok(L, M)) return false;
    smax = std::max(smax, L.split);
  }
  for (int i = 0; i < n; i++)
    if (smax % Ls[i]->split) return false;
  return true;
}

}  // namespace p200
