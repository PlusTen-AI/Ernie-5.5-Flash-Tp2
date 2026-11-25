// p030_f8_decode.cuh — P030 f8_decode: the FP8 block GEMM for few tokens (M ≤ 32), selected by patch::on(f8_decode);
// the base kernel (fp8.cu k_f8 + k_reduce) stays and runs when the patch is off. Same file: P032 f8_fuse_act (F8Act::x
// set: the activation is quantized here, per CTA, instead of by a separate act_quant launch), P033 f8_pdl (launch
// attribute), P034 f8_multi (up to 3 weights on one activation in one launch: forward_parts).
// Included by fp8.cu inside its anonymous namespace (mma, cp_async, act_scale_e8, quant_e4m3 from mma.cuh).
//
// Bitwise contract — every output element is the base kernel's: the same weights layout and 16-row tiles, the same
// MMA instruction with the same operands (k32 blocks of one split chained in ascending order into one fp32 accumulator
// from 0), the same fixed K splits (L.split, chosen at upload), the same split sum v = p0 + p1 + … in order z = 0..S−1,
// the same bf16 / fp32 output rounding. Fused activation quantization is act_quant's arithmetic (amax of the 128-group,
// act_scale_e8, quant_e4m3), so the operands are the same bytes. Only scheduling differs:
//   · weights stream straight to registers, D k32 blocks (D × 512 B per warp) in flight from the first instruction
//     (they do not depend on the previous kernel), L1 bypassed; each warp owns one 16-row weight tile;
//   · the activation (MT tokens × the CTA's k range) and all its scales are staged in shared memory once, the weight
//     tiles' block scales too — no barrier inside the k loop;
//   · split > 1: the S split CTAs of a 64-row block form a thread-block cluster and the partial tiles are summed in
//     rank order through distributed shared memory — no workspace, no k_reduce launch (measured: as fast as or faster
//     than one warp running all S chains of its tile, which leaves too few warps streaming on the small shapes);
//   · P033 f8_pdl: launched as a programmatic dependent — the weight stream starts under the previous kernel's tail;
//   · P034 f8_multi: several weights in one grid (cluster units per weight, see Part) — the second weight's stream
//     starts with the first's instead of after a launch gap;
//   · P064 f8_pf_next: after griddepcontrol.wait, each warp's next weight blocks are prefetched into L2 (bulk), so the
//     activation prologue does not leave DRAM idle.
//   · P060 f8_fuse_ops (FUSE only): the activation is computed in the prologue from the previous kernel's output
//     (Pro: RMSNorm / z ⊙ gate / SiTU, each with its misc kernel's arithmetic and bf16 rounding, then act_quant's), the
//     bf16 output gets an epilogue (Part::epi: sigmoid / + e / MXFP8 copy, with sigmoid_inplace's / add_bf16's /
//     act_quant's arithmetic), and one extra cluster unit may run a side RMSNorm (Aux) — each saves a launch and a round
//     trip; every value is the separate kernels' (rms_scale reproduces k_rms's reduction tree exactly).
#pragma once
// (fp8.cu includes <cooperative_groups.h> at file scope)

namespace p030 {
namespace cg = cooperative_groups;

constexpr int kMaxSmem = 99 * 1024 - 1024;  // dynamic opt-in; the rest of the 99 KB for static shared (P060 rs_s)

constexpr int kParts = 3;  // P034 f8_multi: up to 3 weights on one activation per launch

// one weight of a launch: its 64-row blocks are grouped in cluster units of smax / split blocks (smax = the launch's
// largest split = cluster size); a unit's CTAs: rank z → block z / split of the unit, K split z % split
struct Part {
  const uint4* w;
  const uint8_t* ws;
  void* y;
  int n, ldy, split, rows_per_group;
  int blocks;  // npad / 64
  int ubeg;    // first cluster unit (blockIdx.y) of this weight
  int epi;     // P060: kEpi* on the bf16 output
  const bf16* e;
  int lde;
  uint8_t* q;   // kEpiQ32: e4m3 [M][ldy], scales [M][ldy / 32]
  uint8_t* qs;
};

struct Args {
  Part p[kParts];
  int np, k, smax;
  const uint8_t* xq;
  const uint8_t* xs;
  int ld;
  long gx, gs;
  const bf16* x;  // P032: bf16 activation (quantized here) instead of xq / xs; row stride ldx
  int ldx;
  int M;
  const bf16* pw;  // P060 prologue operand (F8Act::pw, pld)
  int pld;
  int units;  // GEMM units (blockIdx.y < units); blockIdx.y == units: the Aux unit
  F8Aux aux;  // P060: side RMSNorm by the unit after the GEMM units (x == nullptr: none)
  int pfn;    // P064: k32 blocks per warp prefetched into L2 after griddepcontrol.wait (0: off)
};

__device__ __forceinline__ uint4 ldg_stream(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}

constexpr int kD = 8;  // k32 weight blocks in flight per warp (16: measured no faster)
int g_pfn = kD;        // P064: blocks prefetched after the wait (tests: f8_test_pfn)

// bf16 output element (m, r) of part P with its epilogue (P060)
template <bool F32>
__device__ __forceinline__ void put_out(const Part& P, int m, int r, float v) {
  if (F32) {
    reinterpret_cast<float*>(P.y)[(size_t)m * P.ldy + r] = v;
    return;
  }
  bf16 h = __float2bfloat16_rn(v);
  if (P.epi == kEpiSigmoid) h = p060::sigmoid_bf16(h);
  else if (P.epi == kEpiAdd) h = __float2bfloat16_rn(__bfloat162float(h) + __bfloat162float(P.e[(size_t)m * P.lde + r]));
  reinterpret_cast<bf16*>(P.y)[(size_t)m * P.ldy + r] = h;
}

// shared memory: staged activation [MT][klen + 16] + scales [MT][klen/128] + weight scales [4][klen/128]; split > 1
// reuses it for the partial tile [MT][68] fp32
__host__ __device__ constexpr int smem_bytes(int MT, int klen) {
  const int stage = MT * (klen + 16) + MT * (klen / 128) + 4 * (klen / 128);
  const int part = MT * 68 * 4;
  return ((stage > part ? stage : part) + 15) / 16 * 16;
}

// MT tokens per CTA (n8 tiles: MT / 8); FUSE: quantize a.x (group 128) here, PRO: computed by F8Pro op PRO first
template <int MT, bool F32, bool FUSE, int PRO>
__global__ void __launch_bounds__(128) k_dec(Args a) {
  extern __shared__ __align__(16) uint8_t smem[];
  constexpr int NT = MT / 8;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int u = blockIdx.y;
  if (u >= a.units) {  // P060 Aux unit: no cluster barriers here (none of its CTAs takes part in one)
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;\n" :::);
    if (blockIdx.z) return;
    p060::aux_rows(a.aux, a.M, blockIdx.x * 4 + warp, gridDim.x * 4, lane);
    return;
  }
  Part P = a.p[0];
#pragma unroll
  for (int i = 1; i < kParts; i++)
    if (i < a.np && u >= a.p[i].ubeg) P = a.p[i];
  const int S = P.split, z = blockIdx.x, zl = z % S, b = z / S;  // K split, block of the unit
  const int blk = (u - P.ubeg) * (a.smax / S) + b;                // this CTA's 64-row block
  if (blk >= P.blocks) {  // past the weight's last block: idle, but in its cluster's barriers
    if (S > 1) cg::this_cluster().sync(), cg::this_cluster().sync();
    return;
  }
  const int T = blk * 4 + warp;             // this warp's 16-row weight tile
  const int r0 = blk * 64;
  const int t0 = blockIdx.z * MT;
  const int klen = a.k / S, kbeg = zl * klen;  // this CTA's k range
  const int nkb = klen >> 5, kgl = klen >> 7;
  const int KB = a.k >> 5, KG = a.k >> 7;
  const int RB = klen + 16;                 // staged row bytes (conflict-free fragment reads)
  uint8_t* sx = smem;                       // [MT][RB]
  uint8_t* ssx = smem + MT * RB;            // [MT][kgl] activation scales
  uint8_t* ssw = ssx + MT * kgl;            // [4][kgl] weight block scales

  // 1. the weight stream starts before anything else
  const uint4* wt = P.w + ((size_t)T * KB + (kbeg >> 5)) * 32 + lane;
  uint4 wr[kD];
#pragma unroll
  for (int i = 0; i < kD; i++) wr[i] = ldg_stream(wt + i * 32);

  // 2. block scales of the 4 tiles; then (P033 f8_pdl) wait for the previous kernel, let the next one start; then the
  //    activation and its scales (both griddepcontrol instructions are no-ops without programmatic launch)
  for (int c = threadIdx.x; c < 4 * kgl; c += 128)
    ssw[c] = P.ws[(size_t)(blk * 4 + c / kgl) * KG + (kbeg >> 7) + c % kgl];
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;\n" :::);
  // P064 f8_pf_next: the first kD blocks have often landed already (requested under the previous kernel); while the
  // activation is staged / computed below, the next a.pfn blocks of this warp go to L2, so DRAM keeps streaming
  if (a.pfn && lane == 0 && nkb > kD) {
    const uint32_t bytes = uint32_t(min(nkb - kD, a.pfn)) * 512u;
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(wt - lane + (size_t)kD * 32), "r"(bytes) : "memory");
  }
  const int grp = r0 / P.rows_per_group;
  if constexpr (FUSE) {
    // one thread per 8 consecutive elements, 16 threads per 128-group (act_quant's arithmetic); valid rows only
    // (4 loads in flight per thread), rows past M zero
    const int per = klen >> 3;  // multiple of 32: a warp never straddles rows, 16 lanes = one group
    const int mv = min(MT, a.M - t0), tot = mv * per;
    [[maybe_unused]] __shared__ float rs_s[MT], red_s[8 * 8];  // P060 kProRms: each row's rs
    if constexpr (PRO == kProRms) {
      static_assert(MT <= 8, "rms_rows: 8 rows");
      p060::rms_rows_n(a.x + (size_t)t0 * a.ldx, a.ldx, a.k, mv, red_s, rs_s);
    }
    for (int c0 = threadIdx.x; c0 < tot; c0 += 4 * 128) {
      uint4 u[4];
      [[maybe_unused]] uint4 u2[4];  // P060: the second operand (norm weight / gate / up half)
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const int c = c0 + j * 128, row = c / per, col = (c - row * per) * 8;
        const bool in = c < tot;
        const bf16* xp = a.x + (size_t)(t0 + row) * a.ldx + kbeg + col;
        u[j] = in ? *reinterpret_cast<const uint4*>(xp) : make_uint4(0, 0, 0, 0);
        if constexpr (PRO == kProRms)
          u2[j] = in ? *reinterpret_cast<const uint4*>(a.pw + kbeg + col) : make_uint4(0, 0, 0, 0);
        else if constexpr (PRO == kProMul)
          u2[j] = in ? *reinterpret_cast<const uint4*>(a.pw + (size_t)(t0 + row) * a.pld + kbeg + col)
                     : make_uint4(0, 0, 0, 0);
        else if constexpr (PRO == kProSitu)
          u2[j] = in ? *reinterpret_cast<const uint4*>(xp + a.k) : make_uint4(0, 0, 0, 0);
      }
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const int c = c0 + j * 128;
        if (c >= tot) break;  // warp-uniform (tot is a multiple of 32)
        const int row = c / per, col = (c - row * per) * 8;
        float v[8] = {bf16_lo(u[j].x), bf16_hi(u[j].x), bf16_lo(u[j].y), bf16_hi(u[j].y),
                      bf16_lo(u[j].z), bf16_hi(u[j].z), bf16_lo(u[j].w), bf16_hi(u[j].w)};
        if constexpr (PRO != kProNone) {
          const float w[8] = {bf16_lo(u2[j].x), bf16_hi(u2[j].x), bf16_lo(u2[j].y), bf16_hi(u2[j].y),
                              bf16_lo(u2[j].z), bf16_hi(u2[j].z), bf16_lo(u2[j].w), bf16_hi(u2[j].w)};
          [[maybe_unused]] const float rs = PRO == kProRms ? rs_s[row] : 0.f;
#pragma unroll
          for (int i = 0; i < 8; i++) {
            if constexpr (PRO == kProRms) v[i] = p060::bf16r(v[i] * rs * w[i]);  // k_rms: x · rs · w
            else if constexpr (PRO == kProMul) v[i] = p060::bf16r(v[i] * w[i]);  // k_mul: a · b
            else v[i] = p060::bf16r(p060::situ_f(v[i], w[i]));                  // k_situ
          }
        }
        float m = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) m = fmaxf(m, fabsf(v[i]));
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        const uint32_t e = act_scale_e8(m, 1e-10f);
        uint32_t q0 = 0, q1 = 0;
#pragma unroll
        for (int i = 0; i < 4; i++) q0 |= quant_e4m3(v[i], e) << (8 * i), q1 |= quant_e4m3(v[4 + i], e) << (8 * i);
        *reinterpret_cast<uint2*>(sx + row * RB + col) = make_uint2(q0, q1);
        if ((col & 127) == 0) ssx[row * kgl + (col >> 7)] = uint8_t(e);
      }
    }
    for (int c = mv * (RB >> 4) + threadIdx.x; c < MT * (RB >> 4); c += 128)
      reinterpret_cast<uint4*>(sx)[c] = make_uint4(0, 0, 0, 0);
    for (int c = mv * kgl + threadIdx.x; c < MT * kgl; c += 128) ssx[c] = 0;
  } else {
    const uint8_t* xq = a.xq + grp * a.gx;
    const uint8_t* xs = a.xs + grp * a.gs;
    const int per = klen >> 4;
    for (int c = threadIdx.x; c < MT * per; c += 128) {
      const int row = c / per, ch = c - row * per, tok = t0 + row;
      const bool ok = tok < a.M;
      cp_async16(sx + row * RB + ch * 16, xq + (size_t)(ok ? tok : 0) * a.ld + kbeg + ch * 16, ok);
    }
    cp_async_commit();
    const int sld = a.ld >> 7;
    for (int c = threadIdx.x; c < MT * kgl; c += 128) {
      const int row = c / kgl, j = c - row * kgl, tok = t0 + row;
      ssx[c] = tok < a.M ? xs[(size_t)tok * sld + (kbeg >> 7) + j] : 0;
    }
    cp_async_wait<0>();
  }
  __syncthreads();

  // 3. the chain: k32 blocks of this split in ascending order, kD weight blocks in flight
  float acc[NT][4];
#pragma unroll
  for (int i = 0; i < NT; i++) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
  const uint8_t* bx = sx + g * RB + 4 * t;
  const uint8_t* bs = ssx + g * kgl;
  const uint8_t* ws = ssw + warp * kgl;
  for (int kb0 = 0; kb0 < nkb; kb0 += kD) {
#pragma unroll
    for (int i = 0; i < kD; i++) {
      const int kb = kb0 + i;
      const uint4 wa = wr[i];
      if (kb + kD < nkb) wr[i] = ldg_stream(wt + (size_t)(kb + kD) * 32);
      const uint32_t sfa = ws[kb >> 2];
#pragma unroll
      for (int nt = 0; nt < NT; nt++) {
        const uint8_t* p = bx + nt * 8 * RB + kb * 32;
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(p);
        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(p + 16);
        const uint32_t sfb = bs[nt * 8 * kgl + (kb >> 2)];
        mma_e4m3_e4m3(acc[nt], wa.x, wa.y, wa.z, wa.w, b0, b1, sfa, sfb);
      }
    }
  }

  // 4. epilogue: D[row g / g+8][token 2t / 2t+1]
  if (S == 1) {
    const int ra = 16 * T + g, rb = ra + 8;
#pragma unroll
    for (int nt = 0; nt < NT; nt++) {
      const int tk = t0 + nt * 8 + 2 * t;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int r = i < 2 ? ra : rb, m = tk + (i & 1);
        if (r >= P.n || m >= a.M) continue;
        put_out<F32>(P, m, r, acc[nt][i]);
      }
    }
    return;
  }
  // cluster: partial tile Pt[token][row of the 64] in shared memory, summed over the block's ranks in split order
  __syncthreads();  // Pt aliases the staged activations
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
  const int mv = min(MT, a.M - t0), r0k = b * S;  // the block's ranks: r0k … r0k + S − 1
  for (int e = zl * 128 + threadIdx.x; e < mv * 64; e += S * 128) {  // a warp: 32 consecutive rows of one token
    const int m = e >> 6, rr = e & 63, r = r0 + rr;
    if (P.epi != kEpiQ32 && r >= P.n) continue;  // (kEpiQ32: n % 64 == 0, checked on the host)
    const int o = m * 68 + rr;
    float v = cl.map_shared_rank(Pt, r0k)[o];
    for (int s = 1; s < S; s++) v += cl.map_shared_rank(Pt, r0k + s)[o];
    put_out<F32>(P, t0 + m, r, v);
    if (!F32 && P.epi == kEpiQ32) {  // act_quant group 32 of the bf16 output: amax over the warp's 32 rows
      const float h = p060::bf16r(v);
      const uint32_t ex = act_scale_e8(warp_max(fabsf(h)), 1e-10f);
      P.q[(size_t)(t0 + m) * P.ldy + r] = uint8_t(quant_e4m3(h, ex));
      if (lane == 0) P.qs[(size_t)(t0 + m) * (P.ldy / 32) + r / 32] = uint8_t(ex);
    }
  }
  cl.sync();  // keep every rank's shared memory alive until all have read it
}

template <int MT, bool F32, bool FUSE, int PRO>
void launch(const Args& a, int units, int smem, cudaStream_t s) {
  static bool attr = false;
  if (!attr) {
    CUDA_CHECK(cudaFuncSetAttribute(k_dec<MT, F32, FUSE, PRO>, cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxSmem));
    attr = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(a.smax, units + (a.aux.x ? 1 : 0), cdiv(a.M, MT));
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
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, k_dec<MT, F32, FUSE, PRO>, a));
}

inline int mt_for(int M) { return M <= 8 ? 8 : M <= 16 ? 16 : 32; }

// can this path run (L, M)? cluster ≤ 8 ranks; the staged chunk (MT tokens × K/S) within 33 KB (≥ 3 CTAs per SM) —
// beyond it (M > 8 with K/S ≥ 2048) the base kernel is as fast or faster (measured: fewer CTAs per SM, or more token
// tiles each re-reading the weights from DRAM)
inline bool ok(const F8Lin& L, int M) {
  if (M > 32 || L.split > 8) return false;
  return smem_bytes(mt_for(M), L.k / L.split) <= 33 * 1024;
}
// P032: fused quantization only at M ≤ 8, ungrouped
inline bool fuse_ok(const F8Lin& L, int M) { return M <= 8 && L.groups == 1 && ok(L, M); }

// one launch for weights Ls[0..n) on the same activation (n ≤ kParts, same k; every split divides the largest);
// outs (P060, may be null): each weight's epilogue; aux (P060, may be null): the side RMSNorm
template <bool F32>
void forward_parts(const F8Lin* const* Ls, void* const* ys, const int* ldys, int n, const F8Act& x, int M,
                   cudaStream_t s, long gx, long gs, const F8Out* outs = nullptr, const F8Aux* aux = nullptr) {
  Args a{};
  a.np = n, a.k = Ls[0]->k, a.xq = x.q, a.xs = x.s, a.ld = x.ld, a.gx = gx, a.gs = gs, a.x = x.x, a.ldx = x.ldx;
  a.M = M, a.smax = 1, a.pw = x.pw, a.pld = x.pld;
  for (int i = 0; i < n; i++) a.smax = std::max(a.smax, Ls[i]->split);
  const int MT = x.x ? 8 : mt_for(M);
  int units = 0, smem = 0;
  for (int i = 0; i < n; i++) {
    const F8Lin& L = *Ls[i];
    Part& p = a.p[i];
    p.w = reinterpret_cast<const uint4*>(L.w), p.ws = L.s, p.y = ys[i], p.ldy = ldys[i];
    p.n = L.n, p.split = L.split, p.rows_per_group = L.n / L.groups, p.blocks = L.npad / 64, p.ubeg = units;
    if (outs) p.epi = outs[i].epi, p.e = outs[i].e, p.lde = outs[i].lde, p.q = outs[i].q, p.qs = outs[i].qs;
    units += cdiv(p.blocks, a.smax / L.split);
    smem = std::max(smem, smem_bytes(MT, L.k / L.split));
  }
  a.units = units;
  if (aux) a.aux = *aux;
  a.pfn = e55::patch::on(e55::patch::f8_pf_next) ? g_pfn : 0;
  if (x.x) {
    switch (x.pro) {
      case kProRms: return launch<8, F32, true, kProRms>(a, units, smem, s);
      case kProMul: return launch<8, F32, true, kProMul>(a, units, smem, s);
      case kProSitu: return launch<8, F32, true, kProSitu>(a, units, smem, s);
      default: return launch<8, F32, true, kProNone>(a, units, smem, s);
    }
  }
  switch (MT) {
    case 8: return launch<8, F32, false, kProNone>(a, units, smem, s);
    case 16: return launch<16, F32, false, kProNone>(a, units, smem, s);
    default: return launch<32, F32, false, kProNone>(a, units, smem, s);
  }
}

template <bool F32>
void forward(const F8Lin& L, const F8Act& x, int M, void* y, int ldy, cudaStream_t s, long gx, long gs,
             const F8Out* out = nullptr) {
  const F8Lin* Ls[1] = {&L};
  void* ys[1] = {y};
  const int ldys[1] = {ldy};
  forward_parts<F32>(Ls, ys, ldys, 1, x, M, s, gx, gs, out);
}

// P034: may these weights run as one launch? (each would take this path alone; same k; splits divide the largest)
inline bool multi_ok(const F8Lin* const* Ls, int n, const F8Act& x, int M) {
  if (n < 2 || n > kParts) return false;
  int smax = 1;
  for (int i = 0; i < n; i++) {
    const F8Lin& L = *Ls[i];
    if (L.k != Ls[0]->k || L.groups != 1 || !(x.x ? fuse_ok(L, M) : ok(L, M))) return false;
    smax = std::max(smax, L.split);
  }
  for (int i = 0; i < n; i++)
    if (smax % Ls[i]->split) return false;
  return true;
}

}  // namespace p030
