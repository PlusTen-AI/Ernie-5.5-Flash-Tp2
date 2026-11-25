// p114_lin_pair.cuh — the decode linear (M ≤ 32) for small grids: P114 lin_pair, P115 lin_small and P165 lin_small2. Included by
// linear.cu inside its anonymous namespace (Fmt, cp16, cp16z, cp_commit, cp_wait, mma, p013). The base lin_kernel stays
// and runs whenever both switches are off or a call is not covered here.
//
//   · P114 lin_pair: the P078 column halves of the router (r0h, r1h) and of the HCA compressor (cvh, cgh) — two views
//     of one Lin, same K and split — as one launch (lin_forward_f32_pair): the parts' row blocks side by side in one
//     grid, each storing to its own output. The second half streams with the first instead of behind its launch (P061
//     lets it start early, but its griddepcontrol.wait holds its pipeline until the first has ended).
//   · P115 lin_small: a grid of few 16-row tiles and ≤ 2 K splits (the router / compressor halves: 16 tiles × 2 splits,
//     a pair 32 × 2) runs as CTAs of one warp instead of four, each with 8 stages (7 in flight) — 4× the CTAs (up to
//     all SMs streaming) and 3.5× the bytes in flight, most requested before griddepcontrol.wait, i.e. under the
//     previous kernel. Measured on the router pairs: one-warp CTAs beat 2- and 4-warp ones and 96 KB budgets; asking for
//     the whole chain into L2 up front (bulk prefetch) did not help — the lines land out of order and the pipeline
//     then waits on L2. The indexer k (8 splits) stays on lin_kernel (one-warp CTAs were slower there).
//   · P165 lin_small2: a grid whose one-warp CTAs would not fit one wave but two-warp CTAs do (the whole router /
//     compressor at 17–32 rows, 64 tiles × 2 splits: 128 one-warp CTAs for 96 slots) as two-warp CTAs (64) instead of
//     lin_kernel's 4-warp ones (32 CTAs: 16 SMs idle).
//   · all: the weights are requested first, the P013 row bases / exception list (dependent loads) after them.
//
// Bitwise: per warp the base's work — one 16-row tile of one K split, the same shared-memory staging of weights and x
// pieces, the same unpacking (P013) and MMAs in the same order into one fp32 accumulator; the split partials summed in
// rank order v = P0 + P1 + … through the cluster's shared memory; the same stores. Only the grid, the number of warps a
// CTA and the request order differ.
#pragma once

namespace p114 {

constexpr int kParts = 2;

struct Part {
  const uint8_t* w;    // kernel-layout weights (12-bit copy or bf16) of this part's first tile
  const uint8_t* pb;   // P013 row bases (this part's row 0)
  const uint32_t* xo;  // P013 per tile exception offsets (this part's tile 0)
  const uint32_t* xe;  // P013 exceptions
  void* y;             // [M, ldy] fp32 | bf16
  int n, ldy;
  int bbeg;            // first row block (blockIdx.x) of this part
};

struct Args {
  Part p[kParts];
  int np;
  const bf16* x;  // [M, K]
  int M, K, KB, kb_split;
};

// CTA geometry: W warps (16 weight rows each), CK k32 blocks a stage, the stage count from the shared-memory budget
template <int FMT, int NT, int CK, int W> struct Geo {
  static constexpr int BUDGET = 48 * 1024;            // 2 CTAs an SM
  static constexpr int WARPS = W, THREADS = 32 * W, ROWS = 16 * W;
  static constexpr int PT = CK * Fmt<FMT>::TB / 16;   // 16-B pieces per tile and stage
  static constexpr int WST = W * CK * Fmt<FMT>::TB;   // weights per stage
  static constexpr int XST = NT * CK * 512;           // x pieces per stage
  static constexpr int STAGE = WST + XST;
  static constexpr int ST = std::min(8, std::max(2, BUDGET / STAGE));
  static constexpr int PSTRIDE = ROWS + 4;            // epilogue partial tile [NT·8][ROWS + 4] fp32
  static constexpr int SMEM = std::max(ST * STAGE, NT * 8 * PSTRIDE * 4);
};

template <int FMT, int NT, bool F32, class C, int CK>
__global__ void __launch_bounds__(C::THREADS) k_lin(Args a) {
  constexpr int TB = Fmt<FMT>::TB;
  extern __shared__ __align__(16) uint8_t sm[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int S = gridDim.y, kb0 = blockIdx.y * a.kb_split, nch = a.kb_split / CK;
  Part P = a.p[0];
  if (a.np > 1 && int(blockIdx.x) >= a.p[1].bbeg) P = a.p[1];
  const int blk = int(blockIdx.x) - P.bbeg;
  const size_t tile0 = size_t(blk) * C::WARPS;

  auto load_w = [&](int c, int buf) {
    uint8_t* st = sm + buf * C::STAGE;
    const int kb = kb0 + c * CK;
#pragma unroll
    for (int p = tid; p < C::WST / 16; p += C::THREADS)  // tile p / PT: CK·TB contiguous bytes
      cp16(st + p * 16, P.w + ((tile0 + p / C::PT) * a.KB + kb) * TB + (p % C::PT) * 16);
  };
  // activation pieces of a stage, piece p = (kbl·NT + nt)·32 + lane': cp.async (zero-filled past M), or (SYNC) plain
  // loads / stores of the same bytes
  auto load_x = [&](int c, int buf, auto sync) {
    uint8_t* st = sm + buf * C::STAGE;
    const int kb = kb0 + c * CK;
#pragma unroll
    for (int p = tid; p < NT * CK * 32; p += C::THREADS) {
      const int ln = p & 31, nt = (p >> 5) % NT, kbl = (p >> 5) / NT, m = nt * 8 + (ln >> 2);
      const bf16* src = a.x + size_t(min(m, a.M - 1)) * a.K + (kb + kbl) * 32 + (ln & 3) * 8;
      if constexpr (decltype(sync)::value)
        *reinterpret_cast<uint4*>(st + C::WST + p * 16) = m < a.M ? *reinterpret_cast<const uint4*>(src) : make_uint4(0, 0, 0, 0);
      else
        cp16z(st + C::WST + p * 16, src, m < a.M);
    }
  };

  // 1. the weights of the first ST − 1 stages (they do not depend on the previous kernel)
#pragma unroll
  for (int i = 0; i < C::ST - 1; i++) {
    if (i < nch) load_w(i, i);
    cp_commit();
  }

  // 2. P013: row bases and this warp's exceptions (dependent loads, behind the weight requests)
  float acc[NT][4];
#pragma unroll
  for (int i = 0; i < NT; i++) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
  [[maybe_unused]] uint32_t bb0 = 0, bb1 = 0;
  [[maybe_unused]] p013::Exc ex;
  if constexpr (FMT == kPack12) {
    const int T = int(tile0) + warp;
    bb0 = P.pb[16 * T + g] * 0x01010101u, bb1 = P.pb[16 * T + g + 8] * 0x01010101u;
    ex.init(P.xo, P.xe, T, kb0, lane);
  }

  // 3. griddepcontrol.wait (a no-op without a programmatic launch), then the activation the previous kernel wrote; the
  //    warp's next ST − 1 chunks into L2 meanwhile (P061)
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;\n" :::);
  if (lane == 0 && nch > C::ST - 1) {
    const int c0 = C::ST - 1, nc = min(nch - c0, C::ST - 1);
    const uint8_t* p = P.w + ((tile0 + warp) * a.KB + kb0 + c0 * CK) * TB;
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(p), "r"(uint32_t(nc * CK * TB)) : "memory");
  }
#pragma unroll
  for (int i = 0; i < C::ST - 1; i++)
    if (i < nch) load_x(i, i, std::true_type{});

  // 4. the chain — lin_kernel's loop
  for (int c = 0; c < nch; c++) {
    cp_wait<C::ST - 2>();
    __syncthreads();
    if (c + C::ST - 1 < nch) {
      load_w(c + C::ST - 1, (c + C::ST - 1) % C::ST);
      load_x(c + C::ST - 1, (c + C::ST - 1) % C::ST, std::false_type{});
    }
    cp_commit();
    const uint8_t* st = sm + (c % C::ST) * C::STAGE;
#pragma unroll
    for (int kbl = 0; kbl < CK; kbl++) {
      uint32_t A0[4], A1[4];
      if constexpr (FMT == kPack12) {  // the bf16 branch's u (row g) = w[0..3], v (row g+8) = w[4..7], exactly
        const uint8_t* b = st + (warp * CK + kbl) * TB;
        const uint4 lo = *reinterpret_cast<const uint4*>(b + lane * 16);
        const uint2 nb = *reinterpret_cast<const uint2*>(b + 512 + lane * 8);
        uint32_t w[8];
        p013::unpack8(lo.x, lo.y, nb.x, bb0, w);
        p013::unpack8(lo.z, lo.w, nb.y, bb1, w + 4);
        const int kb = kb0 + c * CK + kbl;
        if (kb == int(ex.nk)) ex.apply(P.xe, kb, lane, w);
        A0[0] = w[0], A0[1] = w[4], A0[2] = w[1], A0[3] = w[5];
        A1[0] = w[2], A1[1] = w[6], A1[2] = w[3], A1[3] = w[7];
      } else {
        const uint8_t* b = st + (warp * CK + kbl) * TB;
        const uint4 u = *reinterpret_cast<const uint4*>(b + lane * 16), v = *reinterpret_cast<const uint4*>(b + 512 + lane * 16);
        A0[0] = u.x, A0[1] = v.x, A0[2] = u.y, A0[3] = v.y;
        A1[0] = u.z, A1[1] = v.z, A1[2] = u.w, A1[3] = v.w;
      }
      uint4 xv[NT];
#pragma unroll
      for (int nt = 0; nt < NT; nt++)
        xv[nt] = *reinterpret_cast<const uint4*>(st + C::WST + ((kbl * NT + nt) * 32 + lane) * 16);
#pragma unroll
      for (int nt = 0; nt < NT; nt++) mma(acc[nt], A0, xv[nt].x, xv[nt].y);  // k16 step j = 0 for all n tiles
#pragma unroll
      for (int nt = 0; nt < NT; nt++) mma(acc[nt], A1, xv[nt].z, xv[nt].w);  // then j = 1
    }
  }
  cp_wait<0>();
  __syncthreads();

  // 5. split sum in rank order through the cluster's shared memory, stores — lin_kernel's epilogue
  float* Pt = reinterpret_cast<float*>(sm);
  const int r = warp * 16 + g;
#pragma unroll
  for (int nt = 0; nt < NT; nt++) {
    const int m = nt * 8 + 2 * t;
    Pt[m * C::PSTRIDE + r] = acc[nt][0];
    Pt[(m + 1) * C::PSTRIDE + r] = acc[nt][1];
    Pt[m * C::PSTRIDE + r + 8] = acc[nt][2];
    Pt[(m + 1) * C::PSTRIDE + r + 8] = acc[nt][3];
  }
  cg::cluster_group cl = cg::this_cluster();
  if (S > 1) cl.sync();
  else __syncthreads();
  const int n0 = blk * C::ROWS, rank = S > 1 ? int(cl.block_rank()) : 0;
  for (int i = rank * C::THREADS + tid; i < a.M * C::ROWS; i += S * C::THREADS) {  // ranks share the store work
    const int m = i / C::ROWS, rr = i % C::ROWS, o = m * C::PSTRIDE + rr;
    float v;
    if (S > 1) {
      v = cl.map_shared_rank(Pt, 0)[o];
      for (int s = 1; s < S; s++) v += cl.map_shared_rank(Pt, s)[o];  // fixed order s = 0..S-1
    } else {
      v = Pt[o];
    }
    if (n0 + rr < P.n) {
      if constexpr (F32) static_cast<float*>(P.y)[size_t(m) * P.ldy + n0 + rr] = v;
      else static_cast<bf16*>(P.y)[size_t(m) * P.ldy + n0 + rr] = __float2bfloat16_rn(v);
    }
  }
  if (S > 1) cl.sync();  // keep shared memory alive until every rank has read it
}

int g_sms = 48;
int g_warps = 0;  // tests (lin_p114_warps): force the warps a CTA (0: from the grid)

template <int FMT, int NT, bool F32, int CK, int W>
void launch(Args a, const Lin* const* Ls, int n, cudaStream_t s) {
  using C = Geo<FMT, NT, CK, W>;
  int blocks = 0;
  for (int i = 0; i < n; i++) a.p[i].bbeg = blocks, blocks += Ls[i]->npad / C::ROWS;
  const int split = Ls[0]->split;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(blocks, split);
  cfg.blockDim = dim3(C::THREADS);
  cfg.dynamicSmemBytes = C::SMEM;
  cfg.stream = s;
  cudaLaunchAttribute at[2];
  int na = 0;
  if (split > 1) {
    at[na].id = cudaLaunchAttributeClusterDimension;
    at[na].val.clusterDim.x = 1, at[na].val.clusterDim.y = split, at[na].val.clusterDim.z = 1;
    na++;
  }
  if (patch::on(patch::lin_pdl)) {  // P061: a programmatic dependent, as the base at M ≤ 32
    at[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[na].val.programmaticStreamSerializationAllowed = 1;
    na++;
  }
  cfg.attrs = at;
  cfg.numAttrs = na;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, k_lin<FMT, NT, F32, C, CK>, a));
}

// shared-memory opt-in for every instantiation, once, at upload (init_kernels)
template <int FMT, int NT, bool F32, int CK>
void set_smem1() {
  CUDA_CHECK(cudaFuncSetAttribute(k_lin<FMT, NT, F32, Geo<FMT, NT, CK, 1>, CK>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, Geo<FMT, NT, CK, 1>::SMEM));
  CUDA_CHECK(cudaFuncSetAttribute(k_lin<FMT, NT, F32, Geo<FMT, NT, CK, 2>, CK>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, Geo<FMT, NT, CK, 2>::SMEM));
  CUDA_CHECK(cudaFuncSetAttribute(k_lin<FMT, NT, F32, Geo<FMT, NT, CK, 4>, CK>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, Geo<FMT, NT, CK, 4>::SMEM));
}
template <int FMT, int CK, int... I>
void set_smem(std::integer_sequence<int, I...>) {
  (set_smem1<FMT, I + 1, true, CK>(), ...);
  (set_smem1<FMT, I + 1, false, CK>(), ...);
}
void init() {
  constexpr auto nts = std::make_integer_sequence<int, 4>{};
  set_smem<kPack12, 4>(nts), set_smem<kPack12, 2>(nts), set_smem<WMat::BF16, 2>(nts);
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, dev));
}

// the decode format forward() reads L in at M ≤ 64: P013's 12-bit copy, or the bf16 copy; −1: neither (MXFP8, NVFP4)
inline int fmt_of(const Lin& L) {
  if (L.fmt != WMat::BF16) return -1;
  if (L.pk && (!L.w || patch::on(patch::lin_pack12))) return kPack12;
  return L.w ? int(WMat::BF16) : -1;
}
inline int ck_of(const Lin& L, int fmt) { return fmt == kPack12 && (L.k / 32 / L.split) % 4 == 0 ? 4 : 2; }
inline int tiles_of(const Lin* const* Ls, int n) {
  int t = 0;
  for (int i = 0; i < n; i++) t += Ls[i]->npad / 16;
  return t;
}
// warps a CTA: P115 — one when every warp chain gets its own CTA in one wave (2 an SM) and the clusters are ≤ 2 CTAs;
// P165 lin_small2 — two when one-warp CTAs would need a second wave but two-warp ones fit one (the whole router /
// compressor at 17–32 rows: 64 tiles × 2 splits); else lin_kernel's 4 (measured: indexer k, 12 tiles × 8 splits, one-warp
// CTAs 12.4 → 14.4 µs; two warps a CTA no better on the router pairs)
inline int warps_of(const Lin* const* Ls, int n) {
  if (g_warps) return g_warps;
  if (!patch::on(patch::lin_small) || Ls[0]->split > 2) return 4;
  const int chains = tiles_of(Ls, n) * Ls[0]->split;  // warp chains (16-row tile × K split)
  if (chains <= 2 * g_sms) return 1;
  if (chains <= 4 * g_sms && patch::on(patch::lin_small2)) return 2;  // P165
  return 4;
}

// may these Lins (same x) run here as one launch at M rows? same format route, K, split; ≤ 32 rows; one wave of
// lin_kernel's CTAs (2 an SM)
inline bool takes(const Lin* const* Ls, int n, int M) {
  if (M < 1 || M > 32 || n < 1 || n > kParts) return false;
  const int f = fmt_of(*Ls[0]);
  if (f < 0) return false;
  for (int i = 0; i < n; i++) {
    const Lin& L = *Ls[i];
    if (fmt_of(L) != f || L.k != Ls[0]->k || L.split != Ls[0]->split || ck_of(L, f) != ck_of(*Ls[0], f)) return false;
  }
  return tiles_of(Ls, n) / 4 * Ls[0]->split <= 2 * g_sms;
}

template <bool F32>
void forward_parts(const Lin* const* Ls, void* const* ys, int n, const bf16* x, int M, cudaStream_t s) {
  const Lin& L0 = *Ls[0];
  const int f = fmt_of(L0), ck = ck_of(L0, f), w = warps_of(Ls, n);
  Args a{};
  a.np = n, a.x = x, a.M = M, a.K = L0.k, a.KB = L0.k / 32, a.kb_split = L0.k / 32 / L0.split;
  for (int i = 0; i < n; i++) {
    const Lin& L = *Ls[i];
    Part& p = a.p[i];
    p.w = f == kPack12 ? L.pk : static_cast<const uint8_t*>(L.w);
    p.pb = L.pk_base, p.xo = L.pk_xo, p.xe = L.pk_x, p.y = ys[i], p.n = L.n, p.ldy = L.n;
  }
  auto go = [&](auto nt) {
    constexpr int NT = decltype(nt)::value;
    auto by_w = [&](auto fmt, auto ckc) {
      constexpr int FM = decltype(fmt)::value, CK = decltype(ckc)::value;
      if (w == 1) launch<FM, NT, F32, CK, 1>(a, Ls, n, s);
      else if (w == 2) launch<FM, NT, F32, CK, 2>(a, Ls, n, s);
      else launch<FM, NT, F32, CK, 4>(a, Ls, n, s);
    };
    if (f == kPack12 && ck == 4) by_w(std::integral_constant<int, kPack12>{}, std::integral_constant<int, 4>{});
    else if (f == kPack12) by_w(std::integral_constant<int, kPack12>{}, std::integral_constant<int, 2>{});
    else by_w(std::integral_constant<int, WMat::BF16>{}, std::integral_constant<int, 2>{});
  };
  switch ((M + 7) / 8) {
    case 1: return go(std::integral_constant<int, 1>{});
    case 2: return go(std::integral_constant<int, 2>{});
    case 3: return go(std::integral_constant<int, 3>{});
    default: return go(std::integral_constant<int, 4>{});
  }
}

}  // namespace p114
