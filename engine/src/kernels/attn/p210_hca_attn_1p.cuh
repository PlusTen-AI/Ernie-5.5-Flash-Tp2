// p210_hca_attn_1p.cuh — P210 hca_attn_1p: prefill hca_attn_fused under the prefill rule (2026-11-28): its own
// accumulation order, NOT P088 / P150 / P172's bits. Chosen in hca_attn_fused by the step's kind (step_prefill(),
// core/step_kind.h) for EVERY prefill row (any M, any key count: a row's arithmetic must not depend on the call it
// is in), never in decode / verify / draft steps. Included by hca.cu after p172 (uses p041::key_row, p088::ldsm4t,
// p150::dq8 / ld2 / pol_last).
//
// Arithmetic of a row (one fixed chain whatever M, the chunk, the other rows, the CTA):
//   q prep   q_h · rsqrt(Σ x² / 512 + 1e-6) then RoPE on dims 448… (the base's products, angles fl(pos · inv[i]),
//            sincosf), bf16 RN — the reference's rounding point; Σ x² in this kernel's order (a thread's 32 dims in
//            k-step order, the 4 lanes of a head (xor 1, xor 2), then the 4 dims quarters ((q0 + q1) + (q2 + q3)))
//   keys     the row's keys in order (compressed 0 … ncomp − 1, then the window), cut into tiles of 16 from key 0;
//            K = V = the dequantized row (exact: e4m3 · 2^e → bf16, rope as stored)
//   score    s = ((S0 + S1) + (S2 + S3)) · 512^-0.5, S_q = the m16n8k16 chain over dims 128q … 128q + 127 (8 k16
//            steps from +0), fp32
//   softmax  ONE online softmax over all tiles from m = −inf, l = 0, o = 0 (no 256-key chunks, no partials, no
//            merge): mn = max(m, max_tile s); p = expf(s − mn); l = fma(l, expf(m − mn), Σ p) (Σ p: lane sums
//            ((p0 + p1) + (p2 + p3)) of 4 consecutive keys, then xor 1, xor 2); o = o · expf(m − mn) + bf16(p) · V
//            (one k16 MMA step a tile); m = mn
//   final    M = max(m, sink); den = fl(l · e^(m−M)) + e^(sink−M); o_h = bf16(fl(e^(m−M) · o) / den) (the base's
//            one-chunk formula; the quotient correctly rounded: Markstein's correction RN(q + (a − q·den)·RN(1/den)),
//            q = RN(a·RN(1/den)) — __fdiv_rn's value for a numerator in [2^-90, 2^90], den ∈ [1, nk + 1]; a warp with
//            any other numerator (zero, tiny, huge, NaN) takes __fdiv_rn itself); zp[rank][d] = Σ_h U[h][rank] · o_h[d],
//            h = 0 … 31 in order, fma from 0 (the base's VHA chain)
// Deterministic, no atomics; a row's result depends on its own keys, q and the sink only.
//
// Scheduling (no effect on the bits) — HY4 P520's one-pass core adapted to the 584-byte FP8 rows:
//   · persistent CTAs (one per SM), rows r = blockIdx.x + k · gridDim.x; 384 threads = 8 consumer warps (two
//     warpgroups, setmaxnreg 184) + 4 producer warps (one warpgroup, setmaxnreg 136): 2 · 184 + 136 = 3 · 168, the
//     launch allocation of an SM quarter's three warps;
//   · producer warps: the CTA's tile stream across its rows (each row's tiles, then an end marker), 8 threads a key
//     (64-byte runs), K rows gathered from L2 (evict_last) two items ahead in registers, dequantized exactly into a
//     ring of 4 shared K buffers of 16 keys (bf16, 1040-byte rows) with full / empty mbarriers. The dequant is P150's
//     dq8 integer fast path as straight-line code (dq8's branch made every word its own reconverging block: a serial
//     chain, 2.9k cycles a put — the fill bound the loop) with a warp-uniform fallback to dq8 for a word it does not
//     cover; at 104 registers the stages spilled right behind their loads (load latency exposed), so 136;
//   · consumers: two groups of 4 warps = 16 heads each, running free (no turns: each group's barriers are its own;
//     the other group's warps fill an SM quarter's MMA pipe while one group syncs or runs its softmax): QK warp =
//     (dims quarter, the tile's 16 keys, the group's 16 heads), q as B fragments in registers (32), K fragments two
//     k-steps ahead; P·V warp = (the group's 16 heads, dims quarter), o in registers (64); the softmax 4 lanes a head;
//   · row start: q loaded raw into the fragments (the q of the rows one and two ahead bulk-prefetched into L2) and
//     prepped in registers; row end: every warp's final scale at once, then 4 passes (dims quarters) through a
//     shared staging tile: o stored (32-byte runs), the VHA chains (2 dims × 4 ranks a thread), zp stored.
// Shared memory 81.6 KB (the K ring 65 KB); global: no workspace.
// Measured (one GPU, 2048 rows × 32 heads, median of 9, current = P088 / P172 by key count): ctx 0 / 131,072 /
// 524,288: 1.068 / 4.481 / 16.40 → 0.957 / 3.297 / 10.81 ms (1.79× the bf16 MMA floor at 524,288; the tile loop is
// consumer-bound there: the fill waits ~6 % of it).
#pragma once

namespace e55 {
namespace p210 {

constexpr int kLH = 32, kD = kHeadDim, kKT = 16, kNB = 4;
constexpr int kKS = kD + 8;                                   // K row in shared memory (bf16): 1040 B, conflict-free
constexpr int kCW = 8, kPW = 4, kCT = 32 * kCW, kTh = 32 * (kCW + kPW);  // consumer / producer warps: 384 threads
constexpr int kRC = 184, kRP = 136;  // registers a thread (setmaxnreg): 2 · kRC + kRP = 3 · 168 (the launch allocation: 3 warps an SM quarter)
static_assert(2 * kRC + kRP == 3 * 168, "setmaxnreg pool");
constexpr int kVS = 64 + 4;                                   // VHA staging row (bf16 pairs): 272 B
// named barriers (0: the one __syncthreads at the start): group 0, group 1, all consumers
enum { kBarG0 = 1, kBarG1 = 2, kBarC = 3 };

struct Smem {
  bf16 k[kNB][kKT][kKS];  // the K ring: tile i of the stream in k[i % 4]
  union {
    float s[2][4][16][kKT + 4];  // [group][dims quarter][head][key]: the score partials of the tile
    uint32_t st[kLH][kVS];       // row end: o of one dims quarter (32 heads × 64 bf16 pairs) for the VHA
  } u;
  bf16 p[2][16][kKT + 8];  // [group][head][key]: bf16(p) of the tile
  float f[2][16];          // the tile's expf(m − mn) per head
  float fa[2][16], den[2][16];  // row end: e^(m − M), den per head
  float red[2][4][16];     // q prep: Σ x² per group, dims quarter, head
  float rcs[32], rsn[32];  // q prep: RoPE cos / sin at the row's position
  __align__(16) float U[kLH * kVhaRank];
  int n[kNB];              // keys of the tile in buffer b (0: the row's end marker)
  uint64_t full[kNB], empty[kNB];
};

__device__ __forceinline__ uint32_t sa(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mb_init(uint64_t* b, int cnt) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(sa(b)), "r"(cnt) : "memory");
}
__device__ __forceinline__ void mb_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(sa(b)) : "memory");
}
__device__ __forceinline__ void mb_wait(uint64_t* b, uint32_t parity) {
  asm volatile(
      "{\n.reg .pred P1;\nLAB_WAIT:\nmbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n@P1 bra DONE;\nbra LAB_WAIT;\n"
      "DONE:\n}\n" ::"r"(sa(b)),
      "r"(parity)
      : "memory");
}
using p088::bar_sync;
using p088::ldsm4t;

struct Row {
  RowMeta mt;
  int p, ncomp, nwin, nk, ntile;
  __device__ __forceinline__ void init(const RowMeta* meta, int r) {
    mt = meta[r];
    p = mt.pos, ncomp = (p + 1) / kRatio, nwin = min(p + 1, kWindow), nk = ncomp + nwin;
    ntile = (nk + kKT - 1) / kKT;
  }
};

// a producer thread's share of one stream item: key pk of the tile, piece pp (of 8) — nope words pp + 8i (i < 7: the
// word is in 64-dim group i), rope words 56 + pp + 8i (i < 2), the scale word 72 (8-byte words of the 584-byte row:
// 64 contiguous bytes a key and i across the 8 pieces)
constexpr int kPP = 8, kNW = 7, kNR = 2;
struct Stage {
  uint2 nw[kNW], rw[kNR], sc;
  int n;    // keys in the tile; 0: the row's end marker; −1: the stream's end
  bool ok;  // this thread's key is one of the row's
};

// the producer warps' walk over the CTA's stream: each of its rows' tiles, then the row's end marker
struct Producer {
  int pt, pk, pp;  // thread 0 … 127: key pk of the tile, piece pp
  int lr, lj;      // the stream's next item: tile lj of row lr (lj == ntile: the row's end marker)
  Row LR;
  uint64_t pol;
  __device__ __forceinline__ void init(int t, const RowMeta* meta, int M) {
    pt = t, pk = t / kPP, pp = t % kPP, lr = blockIdx.x, lj = 0, pol = p150::pol_last();
    if (lr < M) LR.init(meta, lr);
  }
  // the next item's raw words into s (loads in flight until put)
  __device__ __forceinline__ void load(Stage& s, const RowMeta* meta, const uint8_t* rows, const HcaCache& c, int M) {
    s.ok = false;
    if (lr >= M) {
      s.n = -1;
      return;
    }
    if (lj == LR.ntile) {
      s.n = 0;
      lr += gridDim.x, lj = 0;
      if (lr < M) LR.init(meta, lr);
      return;
    }
    const int k0 = lj * kKT;
    s.n = min(kKT, LR.nk - k0);
    s.ok = pk < s.n;
    if (s.ok) {
      const uint2* src = reinterpret_cast<const uint2*>(p041::key_row(k0 + pk, LR.ncomp, LR.nwin, LR.p, LR.mt, rows, c));
#pragma unroll
      for (int i = 0; i < kNW; i++) s.nw[i] = p150::ld2(src + pp + kPP * i, pol);
#pragma unroll
      for (int i = 0; i < kNR; i++) s.rw[i] = p150::ld2(src + kNope / 8 + pp + kPP * i, pol);
      s.sc = p150::ld2(src + 72, pol);
    }
    lj++;
  }
  // item ti → buffer ti % 4 once the consumers released its previous use; false at the stream's end
  __device__ __forceinline__ bool put(const Stage& s, int ti, Smem& S) const {
    if (s.n < 0) return false;
    const int b = ti % kNB;
    if (ti >= kNB) mb_wait(&S.empty[b], ((ti / kNB) - 1) & 1);
    if (s.n > 0) {
      bf16* dst = &S.k[b][pk][0];
      // straight-line integer dequant (P150's dq8 fast path without its branch: independent chains); a word the fast
      // path does not cover (zero / subnormal / NaN code, scale out of range) marks the warp, which then rewrites its
      // words with dq8 itself (rare) — the stored values are dq8's either way
      uint32_t bad = 0;
#pragma unroll
      for (int i = 0; i < kNW; i++) {
        const uint32_t e = ((i < 4 ? s.sc.x : s.sc.y) >> (8 * (i & 3))) & 0xFF;
        const uint2 w = s.nw[i];
        const uint32_t ea = ((w.x & 0x78787878u) + 0x78787878u) & ((w.y & 0x78787878u) + 0x78787878u);
        const uint32_t na = ((w.x & 0x7f7f7f7fu) + 0x01010101u) | ((w.y & 0x7f7f7f7fu) + 0x01010101u);
        bad |= (~(ea & ~na) & 0x80808080u) | (e - 7u >= 233u ? 1u : 0u);
        const uint32_t B = (e - 7u) * 0x00800080u;
        const uint4 v = make_uint4(p150::dqi2(w.x, 0x1404, B), p150::dqi2(w.x, 0x3424, B), p150::dqi2(w.y, 0x1404, B),
                                   p150::dqi2(w.y, 0x3424, B));
        *reinterpret_cast<uint4*>(dst + 8 * (pp + kPP * i)) = s.ok ? v : make_uint4(0, 0, 0, 0);
      }
      if (__any_sync(0xffffffffu, s.ok && bad)) {
#pragma unroll
        for (int i = 0; i < kNW; i++) {
          const uint32_t e = ((i < 4 ? s.sc.x : s.sc.y) >> (8 * (i & 3))) & 0xFF;
          if (s.ok) *reinterpret_cast<uint4*>(dst + 8 * (pp + kPP * i)) = p150::dq8(s.nw[i], e);
        }
      }
#pragma unroll
      for (int i = 0; i < kNR; i++)  // rows past the row's last key: zero (P = 0 there; no stale values in the MMAs)
        *reinterpret_cast<uint2*>(dst + kNope + 4 * (pp + kPP * i)) = s.ok ? s.rw[i] : make_uint2(0, 0);
    }
    if (pt == 0) S.n[b] = s.n;
    mb_arrive(&S.full[b]);
    return true;
  }
};

// grid: persistent CTAs (≤ SMs) over rows r = blockIdx.x + k · gridDim.x; 384 threads
__global__ void __launch_bounds__(kTh, 1)
    k_attn(const bf16* __restrict__ q, const uint8_t* __restrict__ rows, const RowMeta* __restrict__ meta, HcaCache c,
           const float* __restrict__ sink, const float* __restrict__ inv, const float* __restrict__ U, int M,
           bf16* __restrict__ o, float* __restrict__ zp) {
  extern __shared__ __align__(16) uint8_t smraw[];
  Smem& S = *reinterpret_cast<Smem*>(smraw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  for (int i = tid; i < kLH * kVhaRank; i += kTh) S.U[i] = U[i];
  if (tid == 0) {
#pragma unroll
    for (int b = 0; b < kNB; b++) mb_init(&S.full[b], 32 * kPW), mb_init(&S.empty[b], kCW);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();

  if (warp >= kCW) {  // ================= producer warps =================
    Producer P;
    P.init(tid - kCT, meta, M);
    Stage A, B;
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(kRP));
    P.load(A, meta, rows, c, M), P.load(B, meta, rows, c, M);  // two items' raw rows in flight
    for (int ti = 0;; ti += 2) {
      if (!P.put(A, ti, S)) break;
      P.load(A, meta, rows, c, M);
      if (!P.put(B, ti + 1, S)) break;
      P.load(B, meta, rows, c, M);
    }
    return;
  }

  // ================= consumer warps =================
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(kRC));
  const int grp = warp >> 2, wg = warp & 3, hb = 16 * grp;  // group: heads hb …; dims quarter 128·wg …
  const int bg = kBarG0 + grp;                                       // my group's barrier
  const int sh = 4 * wg + ((lane >> 2) & 3), kg = lane & 3;         // softmax lanes 0…15: head hb + sh, keys 4kg …
  const float scale = rsqrtf(float(kD));
  int ti = 0;  // the CTA's stream item
  for (int r = blockIdx.x; r < M; r += gridDim.x) {
    Row R;
    R.init(meta, r);
    // ---- q: raw fragments (B of Sᵀ = K·Qᵀ): qb[nt][kk] = head hb + 8nt + g, dims 128wg + 16kk + 2t (+1) | + 8 (+9)
    uint32_t qb[2][8][2];
    {
      const bf16* q0 = q + ((size_t)r * kLH + hb + g) * kD + 128 * wg + 2 * t;
#pragma unroll
      for (int nt = 0; nt < 2; nt++)
#pragma unroll
        for (int kk = 0; kk < 8; kk++)
          qb[nt][kk][0] = *reinterpret_cast<const uint32_t*>(q0 + 8 * nt * kD + 16 * kk),
          qb[nt][kk][1] = *reinterpret_cast<const uint32_t*>(q0 + 8 * nt * kD + 16 * kk + 8);
    }
    if (tid == 0)  // q of the rows two (first row: one and two) ahead → L2 (bulk prefetches)
      for (int k = r == (int)blockIdx.x ? 1 : 2; k <= 2; k++)
        if (r + k * (int)gridDim.x < M)
          asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(q + (size_t)(r + k * gridDim.x) * kLH * kD),
                       "r"(kLH * kD * 2)
                       : "memory");
    // ---- q prep: Σ x² (thread → lanes → quarters), × rsqrt, RoPE, bf16 (the reference's rounding point)
#pragma unroll
    for (int nt = 0; nt < 2; nt++) {
      float ss = 0.f;
#pragma unroll
      for (int kk = 0; kk < 8; kk++)
#pragma unroll
        for (int i = 0; i < 2; i++) {
          const float x0 = bf16_lo(qb[nt][kk][i]), x1 = bf16_hi(qb[nt][kk][i]);
          ss = fmaf(x0, x0, ss), ss = fmaf(x1, x1, ss);
        }
      ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, 1));
      ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, 2));
      if (t == 0) S.red[grp][wg][8 * nt + g] = ss;
    }
    if (warp == 3) {  // the base's angles fl(pos · inv[i]), pair i = lane
      float sn, cs;
      sincosf(__fmul_rn(float(R.p), inv[lane]), &sn, &cs);
      S.rcs[lane] = cs, S.rsn[lane] = sn;
    }
    bar_sync(kBarC, kCT);
#pragma unroll
    for (int nt = 0; nt < 2; nt++) {
      const int h = 8 * nt + g;
      const float sum = __fadd_rn(__fadd_rn(S.red[grp][0][h], S.red[grp][1][h]),
                                  __fadd_rn(S.red[grp][2][h], S.red[grp][3][h]));
      const float rs = rsqrtf(__fdiv_rn(sum, float(kD)) + kEps);
#pragma unroll
      for (int kk = 0; kk < 8; kk++)
#pragma unroll
        for (int i = 0; i < 2; i++) {
          float e = bf16_lo(qb[nt][kk][i]) * rs, od = bf16_hi(qb[nt][kk][i]) * rs;
          if (wg == 3 && kk >= 4) {  // dims 448 + 2·pi, +1 (interleaved pairs)
            const int pi = 8 * (kk - 4) + 4 * i + t;
            const float cs = S.rcs[pi], sn = S.rsn[pi];
            const float e2 = __fsub_rn(__fmul_rn(e, cs), __fmul_rn(od, sn));
            od = __fadd_rn(__fmul_rn(od, cs), __fmul_rn(e, sn));
            e = e2;
          }
          qb[nt][kk][i] = pack_bf16x2(e, od);
        }
    }

    // ---- the row's tiles
    float oa[16][4];  // P·V: heads hb + g (+8), dims 128wg + 8j + 2t (+1)
#pragma unroll
    for (int j = 0; j < 16; j++) oa[j][0] = oa[j][1] = oa[j][2] = oa[j][3] = 0.f;
    float m = -INFINITY, l = 0.f;  // softmax lanes: running max / sum of head hb + sh (in its 4 lanes)
    for (;; ti++) {
      const int b = ti % kNB;
      mb_wait(&S.full[b], (ti / kNB) & 1);
      const int n = S.n[b];
      if (n == 0) {  // the row's end marker
        __syncwarp();
        if (lane == 0) mb_arrive(&S.empty[b]);
        ti++;
        break;
      }
      {  // Sᵀ partial of dims quarter wg: the tile's 16 keys × the group's 16 heads
        float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        const bf16* kr = &S.k[b][lane & 15][128 * wg + (lane >> 4) * 8];
        uint32_t a[3][4];
        ldsm_x4(a[0][0], a[0][1], a[0][2], a[0][3], kr);
        ldsm_x4(a[1][0], a[1][1], a[1][2], a[1][3], kr + 16);
#pragma unroll
        for (int kk = 0; kk < 8; kk++) {
          if (kk + 2 < 8) {
            uint32_t* an = a[(kk + 2) % 3];
            ldsm_x4(an[0], an[1], an[2], an[3], kr + 16 * (kk + 2));
          }
          const uint32_t* ac = a[kk % 3];
          mma_bf16(acc[0], ac[0], ac[1], ac[2], ac[3], qb[0][kk][0], qb[0][kk][1]);
          mma_bf16(acc[1], ac[0], ac[1], ac[2], ac[3], qb[1][kk][0], qb[1][kk][1]);
        }
#pragma unroll
        for (int nt = 0; nt < 2; nt++) {  // acc[nt]: keys g / g + 8, heads 8nt + 2t (+1)
          const int hh = 8 * nt + 2 * t;
          S.u.s[grp][wg][hh][g] = acc[nt][0], S.u.s[grp][wg][hh + 1][g] = acc[nt][1];
          S.u.s[grp][wg][hh][g + 8] = acc[nt][2], S.u.s[grp][wg][hh + 1][g + 8] = acc[nt][3];
        }
      }
      bar_sync(bg, 128);  // the group's partials complete
      if (lane < 16) {    // online softmax: head hb + sh, keys 4kg … 4kg + 3
        float sc[4], p[4];
        const float4 x0 = *reinterpret_cast<const float4*>(&S.u.s[grp][0][sh][4 * kg]);
        const float4 x1 = *reinterpret_cast<const float4*>(&S.u.s[grp][1][sh][4 * kg]);
        const float4 x2 = *reinterpret_cast<const float4*>(&S.u.s[grp][2][sh][4 * kg]);
        const float4 x3 = *reinterpret_cast<const float4*>(&S.u.s[grp][3][sh][4 * kg]);
        const float a0[4] = {x0.x, x0.y, x0.z, x0.w}, a1[4] = {x1.x, x1.y, x1.z, x1.w};
        const float a2[4] = {x2.x, x2.y, x2.z, x2.w}, a3[4] = {x3.x, x3.y, x3.z, x3.w};
#pragma unroll
        for (int i = 0; i < 4; i++)
          sc[i] = 4 * kg + i < n ? __fmul_rn(__fadd_rn(__fadd_rn(a0[i], a1[i]), __fadd_rn(a2[i], a3[i])), scale)
                                 : -INFINITY;
        float mt = fmaxf(fmaxf(sc[0], sc[1]), fmaxf(sc[2], sc[3]));
        mt = fmaxf(mt, __shfl_xor_sync(0x0000ffffu, mt, 1));
        mt = fmaxf(mt, __shfl_xor_sync(0x0000ffffu, mt, 2));
        const float mn = fmaxf(m, mt);
#pragma unroll
        for (int i = 0; i < 4; i++) p[i] = expf(__fsub_rn(sc[i], mn));
        const float fac = expf(__fsub_rn(m, mn));  // the row's first tile: m = −inf → 0
        float sum = __fadd_rn(__fadd_rn(p[0], p[1]), __fadd_rn(p[2], p[3]));
        sum = __fadd_rn(sum, __shfl_xor_sync(0x0000ffffu, sum, 1));
        sum = __fadd_rn(sum, __shfl_xor_sync(0x0000ffffu, sum, 2));
        l = fmaf(l, fac, sum);
        m = mn;
        *reinterpret_cast<uint2*>(&S.p[grp][sh][4 * kg]) = make_uint2(pack_bf16x2(p[0], p[1]), pack_bf16x2(p[2], p[3]));
        if (kg == 0) S.f[grp][sh] = fac;
      }
      bar_sync(bg, 128);  // the group's P complete (P·V)
      {
        const float f0 = S.f[grp][g], f1 = S.f[grp][g + 8];
#pragma unroll
        for (int j = 0; j < 16; j++)
          oa[j][0] = __fmul_rn(oa[j][0], f0), oa[j][1] = __fmul_rn(oa[j][1], f0),
          oa[j][2] = __fmul_rn(oa[j][2], f1), oa[j][3] = __fmul_rn(oa[j][3], f1);
        uint32_t pa[4];
        ldsm_x4(pa[0], pa[1], pa[2], pa[3], &S.p[grp][lane & 15][(lane >> 4) * 8]);
        const bf16* vr = &S.k[b][lane & 15][128 * wg + (lane >> 4) * 8];
        uint32_t v[3][4];
        ldsm4t(v[0], vr);
        ldsm4t(v[1], vr + 16);
#pragma unroll
        for (int jj = 0; jj < 8; jj++) {
          if (jj + 2 < 8) ldsm4t(v[(jj + 2) % 3], vr + 16 * (jj + 2));
          const uint32_t* vc = v[jj % 3];
          mma_bf16(oa[2 * jj], pa[0], pa[1], pa[2], pa[3], vc[0], vc[1]);
          mma_bf16(oa[2 * jj + 1], pa[0], pa[1], pa[2], pa[3], vc[2], vc[3]);
        }
      }
      __syncwarp();
      if (lane == 0) mb_arrive(&S.empty[b]);  // this warp is done with buffer b
    }

    // ---- row end: M = max(m, sink), den = fl(l·e^(m−M)) + e^(sink−M), o = fl(e^(m−M)·o) / den
    if (lane < 16 && kg == 0) {
      const float sk = sink[hb + sh], Mx = fmaxf(m, sk), fa = expf(__fsub_rn(m, Mx));
      S.fa[grp][sh] = fa;
      S.den[grp][sh] = __fadd_rn(__fmul_rn(l, fa), expf(__fsub_rn(sk, Mx)));
    }
    bar_sync(kBarC, kCT);  // both groups' coefficients; every warp past its last P·V (the score partials free)
    uint32_t ob[16][2];  // every warp: its o, final (bf16 pairs)
    {
      const float fa0 = S.fa[grp][g], fa1 = S.fa[grp][g + 8], d0 = S.den[grp][g], d1 = S.den[grp][g + 8];
      // o / den correctly rounded: Markstein's correction (y = RN(1 / den), q = RN(a·y), r = a − q·den exact by fma,
      // RN(q + r·y) = RN(a / den) for a in the normal range, den ∈ [1, nk + 1]) — __fdiv_rn's value; a thread with a
      // numerator outside [2^-90, 2^90] (zero, tiny, huge, NaN) sends its warp through __fdiv_rn itself
      const float y0 = __frcp_rn(d0), y1 = __frcp_rn(d1);
      bool slow = false;
#pragma unroll
      for (int j = 0; j < 16; j++) {
        float x[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
          const float an = __fmul_rn(i < 2 ? fa0 : fa1, oa[j][i]), d = i < 2 ? d0 : d1, y = i < 2 ? y0 : y1;
          slow |= !(fabsf(an) >= 0x1p-90f && fabsf(an) <= 0x1p90f);
          const float q0 = __fmul_rn(an, y);
          x[i] = __fmaf_rn(__fmaf_rn(-q0, d, an), y, q0);
        }
        ob[j][0] = pack_bf16x2(x[0], x[1]), ob[j][1] = pack_bf16x2(x[2], x[3]);
      }
      if (__any_sync(0xffffffffu, slow)) {
#pragma unroll
        for (int j = 0; j < 16; j++) {
          ob[j][0] = pack_bf16x2(__fdiv_rn(__fmul_rn(fa0, oa[j][0]), d0), __fdiv_rn(__fmul_rn(fa0, oa[j][1]), d0));
          ob[j][1] = pack_bf16x2(__fdiv_rn(__fmul_rn(fa1, oa[j][2]), d1), __fdiv_rn(__fmul_rn(fa1, oa[j][3]), d1));
        }
      }
    }
    const int ct = tid;    // consumer thread 0 … 255
#pragma unroll
    for (int qq = 0; qq < 4; qq++) {
      if (wg == qq) {  // the dims quarter's two warps: o (bf16) → staging [head][dim of the quarter]
#pragma unroll
        for (int j = 0; j < 16; j++) S.u.st[hb + g][4 * j + t] = ob[j][0], S.u.st[hb + g + 8][4 * j + t] = ob[j][1];
      }
      bar_sync(kBarC, kCT);
      {  // o: head ct / 8, dims 16·(ct % 8) … of the quarter (32-byte runs)
        const int h = ct >> 3, c16 = ct & 7;
        const uint4 x0 = *reinterpret_cast<const uint4*>(&S.u.st[h][8 * c16]);
        const uint4 x1 = *reinterpret_cast<const uint4*>(&S.u.st[h][8 * c16 + 4]);
        uint4* dst = reinterpret_cast<uint4*>(o + ((size_t)r * kLH + h) * kD + 128 * qq + 16 * c16);
        __stcs(dst, x0), __stcs(dst + 1, x1);
      }
      {  // zp: dims 2dp, 2dp + 1 of the quarter, ranks 4rq … 4rq + 3 — Σ_h U[h][rank] · o_h, h ascending, from 0
        const int dp = ct & 63, rq = ct >> 6;
        float z[4][2] = {{0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};
#pragma unroll 8
        for (int h = 0; h < kLH; h++) {
          const uint32_t w = S.u.st[h][dp];
          const float x0 = bf16_lo(w), x1 = bf16_hi(w);
          const float4 u = *reinterpret_cast<const float4*>(&S.U[h * kVhaRank + 4 * rq]);
          z[0][0] = fmaf(u.x, x0, z[0][0]), z[0][1] = fmaf(u.x, x1, z[0][1]);
          z[1][0] = fmaf(u.y, x0, z[1][0]), z[1][1] = fmaf(u.y, x1, z[1][1]);
          z[2][0] = fmaf(u.z, x0, z[2][0]), z[2][1] = fmaf(u.z, x1, z[2][1]);
          z[3][0] = fmaf(u.w, x0, z[3][0]), z[3][1] = fmaf(u.w, x1, z[3][1]);
        }
        float* zr = zp + ((size_t)r * kVhaRank + 4 * rq) * kD + 128 * qq + 2 * dp;
#pragma unroll
        for (int k = 0; k < 4; k++) __stcs(reinterpret_cast<float2*>(zr + k * kD), make_float2(z[k][0], z[k][1]));
      }
      bar_sync(kBarC, kCT);  // staging free (the next quarter; the next row's score partials)
    }
  }
}

inline void launch(const bf16* q, const uint8_t* rows, const RowMeta* meta, HcaCache c, const float* sink,
                   const float* inv, const float* U, int M, bf16* o, float* zp, cudaStream_t s) {
  static int nsm = 0;
  if (!nsm) {
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
    CUDA_CHECK(cudaFuncSetAttribute(k_attn, cudaFuncAttributeMaxDynamicSharedMemorySize, int(sizeof(Smem))));
  }
  k_attn<<<std::min(M, nsm), kTh, sizeof(Smem), s>>>(q, rows, meta, c, sink, inv, U, M, o, zp);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace p210
}  // namespace e55
