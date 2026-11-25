// p098_lin12_prefill.cuh — P098 lin12_prefill: the large-M path of lin_forward (M > 64) over P013's 12-bit copy, so a
// bf16 Lin needs no bf16 copy (P097 lm12_only: lm_head; P098: the router and the HCA compressor, P062's copies).
// Included by linear.cu inside its anonymous namespace (mma, cp_commit / cp_wait, p013::, big::cpz).
//
// Bitwise contract (as linear_prefill.cuh / p004_gemm_tune.cuh): every MMA is the decode kernel's instruction with
// identical operands — the 12-bit copy unpacked in registers to the exact bf16 bits (p013::unpack8's integer operations
// split into its two k16 halves, then the tile's exception entries put back, as lin_kernel does), swap-AB (weight rows on
// the MMA M side), the same k → slot permutation, the same chain (a split's k32 blocks ascending, k16 step 0 then 1, into
// one fp32 accumulator from 0) — and K-split shapes (router / compressor: S = 2) sum their splits in the decode
// cluster's order v = P0, v += P1, …: all k32 blocks run in ascending order through one pipeline; when split s ends,
// each thread writes its accumulators to its own outputs (s = 0: y = P0; s > 0: y = y + Ps in fp32, read back by the
// same thread) and restarts them from 0. So S > 1 needs fp32 outputs (the partial sums live in y).
// Schedule (p004's): a CTA = 4 warps × 32 weight rows (2 m16 tiles each) × all BM activation rows; one k32 block per
// stage (weights 6 KB, activations BM × 64 B), all stages in flight; the next block's raw words are read at the start
// of the current one, its step-0 half unpacked after the current step 0, its step-1 half after step 1, B fragment j
// reloaded right after its last use. Each warp walks its two tiles' exception lists with p013::Exc (one cursor per
// tile: the blocks come in ascending order across the splits).
#pragma once

namespace p098 {

using big::cpz;

struct Args {
  const uint8_t* w;          // 12-bit copy: 768 B per (16-row tile, k32 block)
  const uint8_t* pb;         // row bases [npad]
  const uint32_t *xo, *xe;   // per tile first exception [tiles + 1], entries
  const bf16* x;             // [M, K]
  void* y;                   // [M, N]
  int M, N, K, KB, kb_split, tiles;  // tiles = npad / 16
  int MT, NT;                        // CTA tiles along activation rows / weight rows
};

template <int BM_, int ST_, int MINB_>
struct Cfg {
  static constexpr int BN = 128, BM = BM_, ST = ST_, MINB = MINB_, WN = 4;
  static constexpr int TB = p013::TB, MI = 2, NJ = BM / 8, NTH = WN * 32, TILES = BN / 16;
  static constexpr int WST = TILES * TB, XST = BM * 64, STAGE = WST + XST, SMEM = ST * STAGE;
};
// Small: 128 × 64 activation rows, 3 stages (30 KB), 3 CTAs / SM (168 registers) — the one forward() takes; Row:
// 128 × 128, 3 stages (42 KB), 2 CTAs / SM (255 registers + 8 B stack), slower on GB10 (tests only)
using Row = Cfg<128, 3, 2>;
using Small = Cfg<64, 3, 3>;

// values 0–3 (or 4–7) of one row: low bytes lo, nibbles x (byte j: value j, masked 0x0F0F0F0F), bb = base · 0x01010101 →
// bf16 pairs (v0, v1), (v2, v3) — p013::unpack8's operations for one half
__device__ __forceinline__ void half4(uint32_t lo, uint32_t x, uint32_t bb, uint32_t& p0, uint32_t& p1) {
  const uint32_t h = (x & 0x08080808u) * 15u + x + bb;
  p0 = __byte_perm(lo, h, 0x5140), p1 = __byte_perm(lo, h, 0x7362);
}

template <class C, bool F32>
__global__ void __launch_bounds__(C::NTH, C::MINB) kernel(Args a) {
  constexpr int TB = C::TB, MI = C::MI, NJ = C::NJ, ST = C::ST;
  extern __shared__ __align__(128) uint8_t sm[];
  const int tid = threadIdx.x, wn = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  // grouped raster: GM activation tiles share the sweep over all weight tiles (their activations stay in L2)
  constexpr int GM = 8;
  const int per = GM * a.NT, grp = blockIdx.x / per, fm = grp * GM, gm = min(a.MT - fm, GM);
  const int rem = blockIdx.x - grp * per, mt = fm + rem % gm, nt = rem / gm;
  const int T0 = nt * C::TILES, m0 = mt * C::BM, KB = a.KB;

  auto load = [&](int kb, int buf) {  // k32 block kb → stage buffer buf: weights [tile][768 B], x [row][64 B]
    uint8_t* st = sm + buf * C::STAGE;
    constexpr int PT = TB / 16;
#pragma unroll
    for (int p = tid; p < C::WST / 16; p += C::NTH) {
      const int T = T0 + p / PT;
      const bool ok = T < a.tiles;
      cpz(st + p * 16, a.w + (size_t(ok ? T : 0) * KB + kb) * TB + (p % PT) * 16, ok);
    }
#pragma unroll
    for (int p = tid; p < C::BM * 4; p += C::NTH) {
      const int r = p >> 2, ch = p & 3, m = m0 + r;
      const bf16* src = a.x + size_t(min(m, a.M - 1)) * a.K + kb * 32 + ch * 8;
      cpz(st + C::WST + p * 16, src, m < a.M);
    }
  };

  // this warp's tiles: row bases of rows g, g+8 and the exception cursors (tiles past the weight: none)
  uint32_t bb[MI][2];
  p013::Exc ex[MI];
#pragma unroll
  for (int i = 0; i < MI; i++) {
    const int T = T0 + wn * MI + i;
    if (T < a.tiles) {
      bb[i][0] = a.pb[16 * T + g] * 0x01010101u, bb[i][1] = a.pb[16 * T + g + 8] * 0x01010101u;
      ex[i].init(a.xo, a.xe, T, 0, lane);
    } else {
      bb[i][0] = bb[i][1] = 0;
      ex[i].e = p013::kEnd, ex[i].nk = p013::kMaxKb, ex[i].w0 = ex[i].end = ex[i].i = 0;
    }
  }

  const int aoff = (wn * MI) * TB;                 // + i·TB: tile wn·MI + i (lo at lane·16, nibbles at 512 + lane·8)
  const int boff = C::WST + (g * 4 + t) * 16;      // + j·512: activation row 8j+g, k 8t..8t+7

  float acc[MI][NJ][4];
#pragma unroll
  for (int i = 0; i < MI; i++)
#pragma unroll
    for (int j = 0; j < NJ; j++) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

  uint32_t A0[MI][4], A1[MI][4];  // lin_kernel's kPack12 operands: A0 = {w0, w4, w1, w5}, A1 = {w2, w6, w3, w7}
  uint4 B[NJ];                    // B fragments: .x .y step 0, .z .w step 1
  uint4 RL[MI];                   // raw low bytes of the next block (values 0–7 row g: .x .y; row g+8: .z .w)
  uint2 RN[MI];                   // raw nibbles of the next block (row g: .x; row g+8: .y)

  auto lds_raw = [&](const uint8_t* st) {
#pragma unroll
    for (int i = 0; i < MI; i++) {
      RL[i] = *reinterpret_cast<const uint4*>(st + aoff + i * TB + lane * 16);
      RN[i] = *reinterpret_cast<const uint2*>(st + aoff + i * TB + 512 + lane * 8);
    }
  };
  auto deq0 = [&] {  // values 0–3 of rows g / g+8 → w0 w1 / w4 w5
#pragma unroll
    for (int i = 0; i < MI; i++) {
      half4(RL[i].x, RN[i].x & 0x0F0F0F0Fu, bb[i][0], A0[i][0], A0[i][2]);
      half4(RL[i].z, RN[i].y & 0x0F0F0F0Fu, bb[i][1], A0[i][1], A0[i][3]);
    }
  };
  auto deq1 = [&] {  // values 4–7 → w2 w3 / w6 w7
#pragma unroll
    for (int i = 0; i < MI; i++) {
      half4(RL[i].y, (RN[i].x >> 4) & 0x0F0F0F0Fu, bb[i][0], A1[i][0], A1[i][2]);
      half4(RL[i].w, (RN[i].y >> 4) & 0x0F0F0F0Fu, bb[i][1], A1[i][1], A1[i][3]);
    }
  };
  auto fix = [&](int kb) {  // block kb's exception entries (warp-uniform test), as lin_kernel's ex.apply on w[8]
#pragma unroll
    for (int i = 0; i < MI; i++)
      if (kb == int(ex[i].nk)) {
        uint32_t w[8] = {A0[i][0], A0[i][2], A1[i][0], A1[i][2], A0[i][1], A0[i][3], A1[i][1], A1[i][3]};
        ex[i].apply(a.xe, kb, lane, w);
        A0[i][0] = w[0], A0[i][2] = w[1], A1[i][0] = w[2], A1[i][2] = w[3];
        A0[i][1] = w[4], A0[i][3] = w[5], A1[i][1] = w[6], A1[i][3] = w[7];
      }
  };
  // split pass s ends: y = acc (s = 0) or y + acc (fp32, this thread's own earlier store), then acc from 0 again.
  // C fragment: c0 (w-row g, act-row 2t) c1 (g, 2t+1) c2 (g+8, 2t) c3 (g+8, 2t+1); a store instruction writes 8
  // consecutive outputs (one 32-B sector in fp32) of each of 4 activation rows.
  using E = typename std::conditional<F32, float, bf16>::type;
  auto store = [&](int s) {
    E* y = static_cast<E*>(a.y);
#pragma unroll
    for (int i = 0; i < MI; i++)
#pragma unroll
      for (int j = 0; j < NJ; j++)
#pragma unroll
        for (int e = 0; e < 4; e++) {
          const int n = nt * C::BN + (wn * MI + i) * 16 + g + (e & 2) * 4, m = m0 + j * 8 + 2 * t + (e & 1);
          if (m < a.M && n < a.N) {
            E* p = y + size_t(m) * a.N + n;
            if constexpr (F32) *p = s ? *p + acc[i][j][e] : acc[i][j][e];
            else *p = __float2bfloat16_rn(acc[i][j][e]);  // S = 1 only
          }
          acc[i][j][e] = 0.f;
        }
  };

  // prologue: all ST stages in flight, then the fragments of block 0
#pragma unroll
  for (int s = 0; s < ST; s++) {
    if (s < KB) load(s, s);
    cp_commit();
  }
  cp_wait<ST - 1>();
  __syncthreads();
  lds_raw(sm);
  deq0();
  deq1();
  fix(0);
#pragma unroll
  for (int j = 0; j < NJ; j++) B[j] = *reinterpret_cast<const uint4*>(sm + boff + j * 512);

  int buf = 0;  // stage buffer of block kb
  for (int kb = 0; kb < KB; kb++) {
    // block kb's fragments are in registers: once every warp is here its buffer is free for block kb + ST
    cp_wait<ST - 2>();  // block kb + 1 has landed
    __syncthreads();
    if (kb + ST < KB) load(kb + ST, buf);
    cp_commit();
    buf = buf + 1 == ST ? 0 : buf + 1;
    const uint8_t* nx = sm + buf * C::STAGE;  // block kb + 1 (after the last block: stale data, never used)
    lds_raw(nx);
#pragma unroll
    for (int j = 0; j < NJ; j++)
#pragma unroll
      for (int i = 0; i < MI; i++) mma(acc[i][j], A0[i], B[j].x, B[j].y);  // k16 step 0
    deq0();
#pragma unroll
    for (int j = 0; j < NJ; j++) {
#pragma unroll
      for (int i = 0; i < MI; i++) mma(acc[i][j], A1[i], B[j].z, B[j].w);  // then step 1
      B[j] = *reinterpret_cast<const uint4*>(nx + boff + j * 512);
    }
    deq1();
    if (kb + 1 < KB) {
      fix(kb + 1);
      if ((kb + 1) % a.kb_split == 0) store(kb / a.kb_split);  // split boundary (S > 1)
    }
  }
  cp_wait<0>();
  store(KB / a.kb_split - 1);
}

int g_force = -1;  // tests: -1 = pick(), 0 = Row, 1 = Small

void init() {
  CUDA_CHECK(cudaFuncSetAttribute(kernel<Row, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, Row::SMEM));
  CUDA_CHECK(cudaFuncSetAttribute(kernel<Row, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, Row::SMEM));
  CUDA_CHECK(cudaFuncSetAttribute(kernel<Small, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, Small::SMEM));
  CUDA_CHECK(cudaFuncSetAttribute(kernel<Small, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, Small::SMEM));
}

// a 12-bit copy, and fp32 outputs when the K splits meet in y
template <bool F32>
inline bool takes(const Lin& L) { return L.pk && (F32 || L.split == 1); }

// CTAs of the Small tile (the M ≤ 128 test of forward(): fewer than one per SM → 64-row decode passes instead)
inline int ctas(const Lin& L, int M) { return cdiv(M, Small::BM) * cdiv(L.npad, Small::BN); }

// Row or Small — scheduling only, the tile never changes a value. Router / compressor (1024 × 4096, 2 K splits), ms per
// call at M = 1024 / 4096: Small 0.156 / 0.542 · Row 0.184 / 0.572 (bf16 copy, linear_prefill.cuh: 0.211 / 0.756).
inline int pick(const Lin& L, int M) {
  (void)L, (void)M;
  return g_force >= 0 ? g_force : 1;
}

template <class C, bool F32>
void launch(const Lin& L, const bf16* x, void* y, int M, cudaStream_t s) {
  Args a{L.pk, L.pk_base, L.pk_xo, L.pk_x, x, y, M, L.n, L.k, L.k / 32, L.k / 32 / L.split, L.npad / 16,
         cdiv(M, C::BM), cdiv(L.npad, C::BN)};
  kernel<C, F32><<<a.MT * a.NT, C::NTH, C::SMEM, s>>>(a);
  CUDA_CHECK(cudaGetLastError());
}

template <bool F32>
void forward(const Lin& L, const bf16* x, void* y, int M, cudaStream_t s) {
  if (pick(L, M) == 0) launch<Row, F32>(L, x, y, M, s);
  else launch<Small, F32>(L, x, y, M, s);
}

}  // namespace p098
