// p091_select_pf.cuh — P091 dsa_select_pf: prefill dsa_select (M ≥ 32, rows with < 65536 candidates — any number
// with P142 dsa_select_wide; P046's row-group launch), selected by patch::on(dsa_select_pf). Included by dsa.cu
// inside its anonymous namespace after p046_select.cuh (uses its helpers).
//
// Bitwise contract — P046's (= the base's): every logit is the base's float and the selection is the same list.
//   · logits: per (row, key, 16-head tile m) the e2m1 × e2m1 products of the same operands and UE8M0 scales summed in
//     fp32 from 0 — k64 MMA where P046's exponent-spread guard proves every partial sum exact (then the result is the
//     exact sum, whichever operand holds the keys), else P046's k32 chain in the base orientation; per key the base's
//     leaves relu(acc_h)·w_h + relu(acc_h+8)·w_h+8 (same expression), the base's tree over the 8 head pairs
//     ((a0+a1)+(a2+a3)) + ((a4+a5)+(a6+a7)), then ((t0 + t1) + t2) + t3 over the four tiles;
//   · selection: the 2048 largest by the order-preserving key of the float bits, equal keys at the threshold from the
//     lowest positions, listed in position order, then the window — as P046 / the base.
// Only the schedule differs from P046:
//   · keys on the MMA's M side (ldmatrix from a padded ring), the row's 64 query heads on N in registers, columns
//     ordered so a lane holds both leaves of the tree's first level (in-register), the other two levels a
//     reduce-scatter over lanes xor 1, 2 (2 shuffles a tile instead of P046's 5); one 16-head tile at a time, tile i's
//     leaves / tree beside tile i+1's MMAs, the 64 keys × 4 tiles of a ring stage one straight sequence; the exactness
//     guard once per stage (a stage with a wide spread takes P046's k32 path);
//   · 128 registers, 48 KB of shared memory: two CTAs (16 rows) per SM, so one CTA's selection runs beside the other's
//     MMAs (P046: one CTA per SM, its MMAs, butterflies, key stream and selection one after another); per-CTA prefix
//     counts in 16-bit halves of shared words, level 1 of the selection straight from the row's global counts;
//   · selection in two passes over the logits instead of three: after level 2 (the 22-bit prefix of the threshold),
//     one output pass lists the keys above that prefix and keeps the prefix's members aside (≤ 256, else P046's third
//     pass); the `need` largest of them (equal keys: lowest positions) are then merged into the list.
#pragma once

namespace p091 {

using p046::discard_l2;
using p046::exp_spread;
using p046::okey;

constexpr int kBins = 2048;
constexpr int RPC = 8;   // rows per CTA, one warp each
constexpr int SK = 64;   // keys per ring stage
constexpr int NS = 3;    // ring stages
constexpr int KS = 80;   // shared bytes per key: 64 B codes, 4 B scales, pad (16-byte rows for ldmatrix, no conflicts)
constexpr int kPCap = 256;  // selection: most members of the threshold's 22-bit prefix kept aside (else a third pass)

struct Smem {
  uint8_t ring[NS][SK * KS];       // 15 KB
  uint32_t hist[RPC][kBins / 2];   // 16-bit counts, bin 2i in the low half of word i: 4 KB a row
  int misc[4];
};

// k64 e2m1 × e2m1 (P046's instruction) with the scale halves chosen by byte selectors: sa / sb hold the four UE8M0
// bytes of a 128-vector, SEL 0 = bytes 0, 1 (k 0–63), SEL 2 = bytes 2, 3 (k 64–127). Not volatile (independent chains
// may be interleaved by the compiler; each chain keeps its order).
template <int SEL>
__device__ __forceinline__ void mma_k64(float* d, const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sa,
                                        uint32_t sb) {
  asm("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%12, 0}, %11, {%12, 0};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "r"(sb), "n"(SEL));
}
__device__ __forceinline__ void cp_async4z(void* smem, const void* gmem, int n) {  // n = 4: copy, 0: zero fill
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(s), "l"(gmem), "r"(n));
}
__device__ __forceinline__ int cnt16(const uint32_t* h, int b) { return (h[b >> 1] >> ((b & 1) << 4)) & 0xFFFF; }
__device__ __forceinline__ void add16(uint32_t* h, int b) { atomicAdd(h + (b >> 1), 1u << ((b & 1) << 4)); }

// one warp: exclusive prefixes (lane order) and totals of U per-lane counts, each ≤ 4 (≤ 128 over the warp): four
// 8-bit fields a word, one scan of U / 4 words
template <int U>
__device__ __forceinline__ void scan_cnt(const int (&v)[U], int (&pre)[U], int (&tot)[U]) {
  constexpr int W = U / 4;
  const int lane = threadIdx.x & 31;
  uint32_t w[W], x[W];
#pragma unroll
  for (int i = 0; i < W; i++)
    w[i] = x[i] = uint32_t(v[4 * i]) | uint32_t(v[4 * i + 1]) << 8 | uint32_t(v[4 * i + 2]) << 16 | uint32_t(v[4 * i + 3]) << 24;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1)
#pragma unroll
    for (int i = 0; i < W; i++) {
      const uint32_t y = __shfl_up_sync(0xffffffffu, x[i], o);
      if (lane >= o) x[i] += y;
    }
#pragma unroll
  for (int i = 0; i < W; i++) {
    const uint32_t t = __shfl_sync(0xffffffffu, x[i], 31), e = x[i] - w[i];
#pragma unroll
    for (int b = 0; b < 4; b++) pre[4 * i + b] = (e >> (8 * b)) & 0xFF, tot[4 * i + b] = (t >> (8 * b)) & 0xFF;
  }
}

// one warp: in counts c(b), b < n (bins, from the top down), the bin where the running count reaches need; need −=
// the count above it. Lane i holds bins n−1−per·i … n−per·(i+1).
template <class F>
__device__ __forceinline__ int warp_find(int n, int& need, F c) {
  const int lane = threadIdx.x & 31, per = n / 32, hi = n - 1 - per * lane;
  int s = 0;
  for (int j = 0; j < per; j++) s += c(hi - j);
  int x = s;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= o) x += y;
  }
  const int pre = x - s;
  int found = -1, rem = 0;
  if (pre < need && pre + s >= need) {
    int acc = pre;
    for (int j = 0; j < per; j++) {
      const int h = c(hi - j);
      if (acc + h >= need) {
        found = hi - j, rem = need - acc;
        break;
      }
      acc += h;
    }
  }
  const unsigned b = __ballot_sync(0xffffffffu, found >= 0);
  const int src = __ffs(b) - 1;
  found = __shfl_sync(0xffffffffu, found, src), rem = __shfl_sync(0xffffffffu, rem, src);
  need = rem;
  return found;
}

// one warp selects its row after level 1 (b1: the threshold's 11-bit prefix, need: how many of its members are
// taken): lg[0 … ncand) logits, h the row's 4 KB of shared memory (level-2 / -3 counts); out: list row.
// WIDE (P142 dsa_select_wide: bin b1 has ≥ 65536 members, so 16-bit counts could wrap): the level-2 counts 32-bit,
// 1024 bins at a time (the upper half of bits 20 … 10 first, then — if the need is not reached there — the lower
// half: one more pass over the logits), level 3's 1024 counts 32-bit; the same exact counts, the same bins → the same
// list. Not WIDE: P091's code as it was.
template <int U, bool WIDE>  // float4 loads in flight per lane
__device__ __forceinline__ void select_rest(const float* lg, int ncand, int p, uint32_t b1, int need, uint32_t* h,
                                            int* out) {
  const p046::Grp<1> G{int(threadIdx.x & 31), 32};
  const int step = 4 * U * 32;
  uint32_t b2 = 0;
  int npend = 0;  // keys with the 22-bit prefix; `need` of them are taken
  if constexpr (!WIDE) {
    // level 2: bits 20 … 10 of the members of bin b1
    for (int i = G.rank; i < kBins / 2; i += 32) h[i] = 0;
    __syncwarp();
    for (int i0 = 0; i0 < ncand; i0 += step) {
      uint32_t u[U][4];
      p046::load_keys<1, U>(G, lg, ncand, i0, u);
#pragma unroll
      for (int k = 0; k < U; k++)
#pragma unroll
        for (int j = 0; j < 4; j++)
          if ((u[k][j] >> 21) == b1) add16(h, (u[k][j] >> 10) & 0x7FF);
    }
    __syncwarp();
    b2 = warp_find(kBins, need, [&](int b) { return cnt16(h, b); });
    __syncwarp();
    npend = cnt16(h, b2);
  } else {
    for (int half = 1; half >= 0; half--) {  // bins (half << 10) + 0 … 1023 of level 2
      for (int i = G.rank; i < 1024; i += 32) h[i] = 0;
      __syncwarp();
      const uint32_t b1h = (b1 << 1) | uint32_t(half);
      for (int i0 = 0; i0 < ncand; i0 += step) {
        uint32_t u[U][4];
        p046::load_keys<1, U>(G, lg, ncand, i0, u);
#pragma unroll
        for (int k = 0; k < U; k++)
#pragma unroll
          for (int j = 0; j < 4; j++)
            if ((u[k][j] >> 20) == b1h) atomicAdd(h + ((u[k][j] >> 10) & 0x3FF), 1u);
      }
      __syncwarp();
      int sum = 0;
      for (int i = G.rank; i < 1024; i += 32) sum += int(h[i]);
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
      if (half && sum < need) {  // all of the upper half taken: the threshold is in the lower half
        need -= sum;
        __syncwarp();
        continue;
      }
      b2 = (uint32_t(half) << 10) | uint32_t(warp_find(1024, need, [&](int b) { return int(h[b]); }));
      __syncwarp();
      npend = int(h[b2 & 1023]);
      __syncwarp();
      break;
    }
  }
  const uint32_t pre22 = (b1 << 11) | b2;
  if (npend <= kPCap) {
    // ---- one output pass: keys above the prefix listed in position order (D), its members kept aside (key and
    // position, in order); then the `need` largest members (equal keys: lowest positions) merged into D
    uint32_t* pk = h;
    int* pp = reinterpret_cast<int*>(h + kPCap);
    int* T = reinterpret_cast<int*>(h + 2 * kPCap);
    __syncwarp();  // h's counts read by every lane
    int nd = 0, npd = 0;
    for (int i0 = 0; i0 < ncand; i0 += step) {
      uint32_t u[U][4];
      p046::load_keys<1, U>(G, lg, ncand, i0, u);
      int nt[U], np[U];
      bool anyp = false;
#pragma unroll
      for (int k = 0; k < U; k++) {
        const int i = i0 + 4 * (G.rank + k * 32);
        nt[k] = np[k] = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const uint32_t q = u[k][j] >> 10;
          nt[k] += i + j < ncand && q > pre22;
          np[k] += i + j < ncand && q == pre22;
        }
        anyp |= np[k] > 0;
      }
      int pre[U], ttot[U];
      scan_cnt<U>(nt, pre, ttot);
#pragma unroll
      for (int k = 0; k < U; k++) {
        const int i = i0 + 4 * (G.rank + k * 32);
        int pos = nd + pre[k];
#pragma unroll
        for (int j = 0; j < 4; j++)
          if (i + j < ncand && (u[k][j] >> 10) > pre22) out[pos++] = i + j;
        nd += ttot[k];
      }
      if (__any_sync(0xffffffffu, anyp)) {
        int ppre[U], ptot[U];
        scan_cnt<U>(np, ppre, ptot);
#pragma unroll
        for (int k = 0; k < U; k++) {
          const int i = i0 + 4 * (G.rank + k * 32);
          int pos = npd + ppre[k];
#pragma unroll
          for (int j = 0; j < 4; j++)
            if (i + j < ncand && (u[k][j] >> 10) == pre22) pk[pos] = u[k][j], pp[pos++] = i + j;
          npd += ptot[k];
        }
      }
    }
    __syncwarp();
    // the taken members, in position order: rank (larger keys, then equal keys at lower positions) below need
    int nT = 0;
    for (int e0 = 0; e0 < npend; e0 += 32) {
      const int e = e0 + G.rank;
      bool tk = false;
      if (e < npend) {
        const uint32_t ue = pk[e];
        int rank = 0;
        for (int j = 0; j < npend; j++) rank += pk[j] > ue || (pk[j] == ue && j < e);
        tk = rank < need;
      }
      const unsigned bt = __ballot_sync(0xffffffffu, tk);
      if (tk) T[nT + __popc(bt & ((1u << G.rank) - 1))] = pp[e];
      nT += __popc(bt);
    }
    __syncwarp();
    // merge: D element i moves to i + #(T below it); T[j] lands at j + #(D below it), written by the D element
    // above it (or after the last one). Chunks of 32 from the top: every move is upward, reads precede writes.
    auto below = [&](int v) {  // #(T < v)
      int lo = 0, hi = nT;
      while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (T[mid] < v) lo = mid + 1;
        else hi = mid;
      }
      return lo;
    };
    const int slast = nd > 0 ? below(out[nd - 1]) : 0;
    for (int j = slast + G.rank; j < nT; j += 32) out[j + nd] = T[j];
    for (int c0 = (nd - 1) & ~31; c0 >= 0; c0 -= 32) {
      const int i = c0 + G.rank;
      int d = 0, sd = 0, sp = 0;
      if (i < nd) {
        d = out[i];
        sd = below(d);
        sp = i > 0 ? below(out[i - 1]) : 0;
      }
      __syncwarp();
      if (i < nd) {
        out[i + sd] = d;
        for (int j = sp; j < sd; j++) out[j + i] = T[j];
      }
      __syncwarp();
    }
    for (int i = G.rank; i < kWindow; i += 32) out[kIdxTop + i] = p - kWindow + 1 + i;
    return;
  }
  // level 3: bits 9 … 0 (WIDE: 32-bit counts)
  for (int i = G.rank; i < (WIDE ? 1024 : 512); i += 32) h[i] = 0;
  __syncwarp();
  for (int i0 = 0; i0 < ncand; i0 += step) {
    uint32_t u[U][4];
    p046::load_keys<1, U>(G, lg, ncand, i0, u);
#pragma unroll
    for (int k = 0; k < U; k++)
#pragma unroll
      for (int j = 0; j < 4; j++)
        if ((u[k][j] >> 10) == pre22) {
          if constexpr (WIDE) atomicAdd(h + (u[k][j] & 0x3FF), 1u);
          else add16(h, u[k][j] & 0x3FF);
        }
  }
  __syncwarp();
  const uint32_t b3 = warp_find(1024, need, [&](int b) { return WIDE ? int(h[b]) : cnt16(h, b); });
  const uint32_t thr = (pre22 << 10) | b3;
  const int need_eq = need;
  // output in position order (P046's pass)
  int base = 0, eqbase = 0;
  for (int i0 = 0; i0 < ncand; i0 += step) {
    uint32_t u[U][4];
    p046::load_keys<1, U>(G, lg, ncand, i0, u);
    int ne[U], er[U], etot[U];
    bool anyeq = false;
#pragma unroll
    for (int k = 0; k < U; k++) {
      ne[k] = 0;
#pragma unroll
      for (int j = 0; j < 4; j++) ne[k] += u[k][j] == thr;
      anyeq |= ne[k] > 0;
      er[k] = 0, etot[k] = 0;
    }
    if (__any_sync(0xffffffffu, anyeq)) scan_cnt<U>(ne, er, etot);
    int nt[U];
    bool take[U][4];
#pragma unroll
    for (int k = 0; k < U; k++) {
      int e = eqbase + er[k];  // equals before my first in sub-round k
      nt[k] = 0;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const bool eq = u[k][j] == thr;
        take[k][j] = u[k][j] > thr || (eq && e < need_eq);
        e += eq;
        nt[k] += take[k][j];
      }
      eqbase += etot[k];
    }
    int pre[U], ttot[U];
    scan_cnt<U>(nt, pre, ttot);
#pragma unroll
    for (int k = 0; k < U; k++) {
      const int i = i0 + 4 * (G.rank + k * 32);
      int pos = base + pre[k];
#pragma unroll
      for (int j = 0; j < 4; j++)
        if (take[k][j]) out[pos++] = i + j;
      base += ttot[k];
    }
  }
  for (int i = G.rank; i < kWindow; i += 32) out[kIdxTop + i] = p - kWindow + 1 + i;
}

template <int U>  // (out of line: the 16-bit path's code stays as it was)
__device__ __noinline__ void select_rest_wide(const float* lg, int ncand, int p, uint32_t b1, int need, uint32_t* h,
                                              int* out) {
  select_rest<U, true>(lg, ncand, p, b1, need, h, out);
}

// one warp selects its row: lg[0 … ncand) logits, hg the row's complete 11-bit prefix counts (global, 32-bit), h the
// row's 4 KB of shared memory; out: list row. Level 1 from hg, then select_rest (WIDE only when the threshold's bin
// has ≥ 65536 members — never with fewer than 65536 candidates)
template <int U>
__device__ __noinline__ void select_row(const float* lg, int ncand, int p, const int* hg, uint32_t* h, int* out) {
  int need = kIdxTop;
  const uint32_t b1 = warp_find(kBins, need, [&](int b) { return __ldcg(hg + b); });
  if (__ldcg(hg + b1) <= 0xFFFF) select_rest<U, false>(lg, ncand, p, b1, need, h, out);
  else select_rest_wide<U>(lg, ncand, p, b1, need, h, out);
}

// grid: ⌈n / 8⌉ row groups × ks_n key splits of kr keys (split fastest). Warp w: row r0 + 8·grp + w over the CTA's
// keys. lg [n][ldl] fp32; hist [n][2048] (zero on entry, zeroed again by the selecting CTA); ctr per group (zero, reset).
__global__ void __launch_bounds__(256, 2)
    k_select(const uint8_t* __restrict__ qf4, const uint8_t* __restrict__ qs, const float* __restrict__ wts,
             const RowMeta* __restrict__ meta, DsaCache c, int row0, int n, int ks_n, int kr, int ldl, float* lg,
             int* hist, int* ctr, int* list, int* cnt) {
  extern __shared__ __align__(16) uint8_t smraw[];
  Smem& S = *reinterpret_cast<Smem*>(smraw);
  constexpr int NT = 32 * RPC;
  const int grp = blockIdx.x / ks_n, ks = blockIdx.x % ks_n;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int ri = grp * RPC + warp, r = row0 + ri;
  const bool rowok = ri < n;
  const RowMeta mt = meta[rowok ? r : row0];
  const int ncand = rowok ? mt.pos - kWindow + 1 : 0;
  const bool work = rowok && ncand > kIdxTop;
  int kmax = 0;
  for (int i = 0; i < RPC; i++) {
    const int rr = grp * RPC + i;
    if (rr < n) {
      const int nc = meta[row0 + rr].pos - kWindow + 1;
      if (nc > kIdxTop) kmax = max(kmax, nc);
    }
  }
  const int kc0 = ks * kr, kc1 = min(kmax, kc0 + kr);
  for (int i = tid; i < RPC * kBins / 2; i += NT) (&S.hist[0][0])[i] = 0;
  // ---- the row's query as B fragments: n-tile j = 2m + hf holds heads 16m + 2(c >> 1) + hf + 8(c & 1) as columns c;
  // so lane t's columns 2t, 2t+1 of n-tiles (m, 0), (m, 1) are the head pairs g' = 2t, 2t+1 of the base's butterfly
  // (leaf g' = relu(acc of head 16m + g')·w + relu(acc of head 16m + g' + 8)·w): its first level stays in the lane
  uint32_t QB[8][2][2], SB[8];
  float WA[8], WB[8];
  int rq = 0;
  const uint8_t* qrow = qf4 + (size_t)r * 64 * 64;
  if (work) {
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const int hb = 16 * (j >> 1) + (j & 1), H = hb + 2 * (g >> 1) + 8 * (g & 1);
#pragma unroll
      for (int kq = 0; kq < 2; kq++) {
        QB[j][kq][0] = *reinterpret_cast<const uint32_t*>(qrow + H * 64 + 32 * kq + 4 * t);
        QB[j][kq][1] = *reinterpret_cast<const uint32_t*>(qrow + H * 64 + 32 * kq + 16 + 4 * t);
      }
      SB[j] = *reinterpret_cast<const uint32_t*>(qs + ((size_t)r * 64 + H) * 4);
      WA[j] = wts[(size_t)r * 64 + hb + 2 * t], WB[j] = wts[(size_t)r * 64 + hb + 2 * t + 8];
    }
    rq = max(exp_spread(*reinterpret_cast<const uint32_t*>(qs + ((size_t)r * 64 + lane) * 4)),
             exp_spread(*reinterpret_cast<const uint32_t*>(qs + ((size_t)r * 64 + 32 + lane) * 4)));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) rq = max(rq, __shfl_xor_sync(0xffffffffu, rq, o));
  }
  __syncthreads();
  uint32_t* hrow = S.hist[warp];
  float* lrow = lg + (size_t)ri * ldl;
  // ldmatrix row address of this lane inside a 16-key block (matrices: keys 0–7 / 8–15 × bytes 0–15 / 16–31)
  const int lrow16 = (lane & 7) + 8 * ((lane >> 3) & 1), lcol = 16 * (lane >> 4);
  const bool odd = t & 1;
  // ---- rounds over the distinct slots of the group's working rows (prefill lanes: usually one)
  uint32_t done = 0;
  for (;;) {
    int slot = -1;
    for (int i = 0; i < RPC; i++) {
      const int rr = grp * RPC + i;
      if (rr < n && !(done >> i & 1) && meta[row0 + rr].pos - kWindow + 1 > kIdxTop) {
        slot = meta[row0 + rr].slot;
        break;
      }
    }
    if (slot < 0) break;
    for (int i = 0; i < RPC; i++) {
      const int rr = grp * RPC + i;
      if (rr < n && meta[row0 + rr].slot == slot) done |= 1u << i;
    }
    const bool mine = work && mt.slot == slot;
    const uint8_t* kb = c.idx + (size_t)slot * c.ctx * kIR;
    const int nst = kc1 > kc0 ? (kc1 - kc0 + SK - 1) / SK : 0;
    auto issue = [&](int st) {  // stage st → ring[st % NS] (rows of 80 B); keys past kc1 zero-filled
      const int t0 = kc0 + st * SK, nb = min(SK, kc1 - t0) * kIR;
      uint8_t* dst = S.ring[st % NS];
      const uint8_t* src = kb + (size_t)t0 * kIR;
      for (int w = tid; w < SK * (kIR / 4); w += NT) {
        const int key = w / (kIR / 4), wd = w - key * (kIR / 4);
        cp_async4z(dst + key * KS + 4 * wd, src + 4 * w, 4 * w < nb ? 4 : 0);
      }
    };
#pragma unroll
    for (int st = 0; st < NS - 1; st++) {
      if (st < nst) issue(st);
      cp_async_commit();
    }
    for (int st = 0; st < nst; st++) {
      cp_async_wait<NS - 2>();
      __syncthreads();
      if (st + NS - 1 < nst) issue(st + NS - 1);
      cp_async_commit();
      const uint8_t* ring = S.ring[st % NS];
      const int kst = kc0 + st * SK;
      if (!mine || kst >= ncand) continue;
      // exactness guard over the stage's 64 keys (lane: keys g + 8(t & 1) of each block)
      uint32_t SAv[SK / 16];
      int rk = 0;
#pragma unroll
      for (int blk = 0; blk < SK / 16; blk++) {
        SAv[blk] = *reinterpret_cast<const uint32_t*>(ring + (16 * blk + g + 8 * (t & 1)) * KS + 64);
        rk = max(rk, exp_spread(SAv[blk]));
      }
      if (__any_sync(0xffffffffu, rq + rk > 9)) {
        // ---- a wide exponent spread somewhere in the stage: P046's k32 chain in the base orientation (heads on M,
        // keys on N) and its butterfly, groups of 8 keys
#pragma unroll 1
        for (int q = 0; q < 2 * SK / 16; q++) {
          const int n0 = kst + 8 * q;
          if (n0 >= ncand) break;
          float acc[4][4];
#pragma unroll
          for (int m = 0; m < 4; m++) acc[m][0] = acc[m][1] = acc[m][2] = acc[m][3] = 0.f;
          const uint8_t* kk8 = ring + (8 * q + g) * KS;
          const uint32_t sbk = *reinterpret_cast<const uint32_t*>(kk8 + 64);
#pragma unroll 1
          for (int kk = 0; kk < 4; kk++) {
            const uint32_t b0 = e2m1x4_expand(*reinterpret_cast<const uint16_t*>(kk8 + 16 * kk + 2 * t));
            const uint32_t b1 = e2m1x4_expand(*reinterpret_cast<const uint16_t*>(kk8 + 16 * kk + 8 + 2 * t));
            const uint32_t sfb = (sbk >> (8 * kk)) & 0xFF;
#pragma unroll
            for (int m = 0; m < 4; m++) {
              const int h0 = 16 * m + g;
              mma_e2m1_e2m1(acc[m], e2m1x4_expand(*reinterpret_cast<const uint16_t*>(qrow + h0 * 64 + 16 * kk + 2 * t)),
                            e2m1x4_expand(*reinterpret_cast<const uint16_t*>(qrow + (h0 + 8) * 64 + 16 * kk + 2 * t)),
                            e2m1x4_expand(*reinterpret_cast<const uint16_t*>(qrow + h0 * 64 + 16 * kk + 8 + 2 * t)),
                            e2m1x4_expand(*reinterpret_cast<const uint16_t*>(qrow + (h0 + 8) * 64 + 16 * kk + 8 + 2 * t)),
                            b0, b1, qs[((size_t)r * 64 + h0 + 8 * (lane & 1)) * 4 + kk], sfb);
            }
          }
          float v[4][2];
#pragma unroll
          for (int m = 0; m < 4; m++) {
            const float w0 = wts[(size_t)r * 64 + 16 * m + g], w1 = wts[(size_t)r * 64 + 16 * m + g + 8];
            v[m][0] = fmaxf(acc[m][0], 0.f) * w0 + fmaxf(acc[m][2], 0.f) * w1;  // key 2t
            v[m][1] = fmaxf(acc[m][1], 0.f) * w0 + fmaxf(acc[m][3], 0.f) * w1;  // key 2t+1
          }
          const int jk = g & 1, k1 = (g >> 1) & 1, k2 = (g >> 2) & 1;
          float r1[4];
#pragma unroll
          for (int m = 0; m < 4; m++) {
            const float snd = jk ? v[m][0] : v[m][1];
            r1[m] = (jk ? v[m][1] : v[m][0]) + __shfl_xor_sync(0xffffffffu, snd, 4);
          }
          float r2[2];
#pragma unroll
          for (int i = 0; i < 2; i++) {
            const float snd = k1 ? r1[2 * i] : r1[2 * i + 1];
            r2[i] = (k1 ? r1[2 * i + 1] : r1[2 * i]) + __shfl_xor_sync(0xffffffffu, snd, 8);
          }
          const float snd = k2 ? r2[0] : r2[1];
          const float r3 = (k2 ? r2[1] : r2[0]) + __shfl_xor_sync(0xffffffffu, snd, 16);
          const float x1 = __shfl_down_sync(0xffffffffu, r3, 8), x2 = __shfl_down_sync(0xffffffffu, r3, 16),
                      x3 = __shfl_down_sync(0xffffffffu, r3, 24);
          if (lane < 8) {
            const int key = n0 + 2 * t + jk;
            if (key < ncand) {
              const float lgv = ((r3 + x1) + x2) + x3;
              lrow[key] = lgv;
              add16(hrow, okey(lgv) >> 21);
            }
          }
        }
        continue;
      }
      // ---- k64, keys × heads: the stage's 4 blocks × 4 head tiles as one straight sequence of steps; step i's MMAs
      // are issued before step i−1's epilogue (leaves, butterfly, running sum over the tiles)
      float acc[2][2][4];  // [step parity][half hf][fragment]
      uint32_t A[2][4];
      float sblk = 0.f;
#pragma unroll
      for (int i = 0; i <= 4 * (SK / 16); i++) {
        if (i < 4 * (SK / 16)) {
          const int blk = i >> 2, m = i & 3;
          if (m == 0) {
#pragma unroll
            for (int kq = 0; kq < 2; kq++)
              ldsm_x4(A[kq][0], A[kq][1], A[kq][2], A[kq][3], ring + (16 * blk + lrow16) * KS + 32 * kq + lcol);
          }
          float(&a)[2][4] = acc[i & 1];
#pragma unroll
          for (int hf = 0; hf < 2; hf++) a[hf][0] = a[hf][1] = a[hf][2] = a[hf][3] = 0.f;
#pragma unroll
          for (int hf = 0; hf < 2; hf++) mma_k64<0>(a[hf], A[0], QB[2 * m + hf][0][0], QB[2 * m + hf][0][1], SAv[blk], SB[2 * m + hf]);
#pragma unroll
          for (int hf = 0; hf < 2; hf++) mma_k64<2>(a[hf], A[1], QB[2 * m + hf][1][0], QB[2 * m + hf][1][1], SAv[blk], SB[2 * m + hf]);
        }
        if (i > 0) {
          const int blk = (i - 1) >> 2, mm = (i - 1) & 3;
          const float(&a)[2][4] = acc[(i - 1) & 1];
          float v[2][2];  // [hf][key g / g+8]
#pragma unroll
          for (int hf = 0; hf < 2; hf++) {
            const float w0 = WA[2 * mm + hf], w1 = WB[2 * mm + hf];
            v[hf][0] = fmaxf(a[hf][0], 0.f) * w0 + fmaxf(a[hf][1], 0.f) * w1;
            v[hf][1] = fmaxf(a[hf][2], 0.f) * w0 + fmaxf(a[hf][3], 0.f) * w1;
          }
          // level 1 (g' = 2t, 2t+1): in the lane; level 2 (lanes xor 1): even t keeps key g, odd t key g + 8;
          // level 3 (lanes xor 2): the tile's sum for this lane's key
          const float u0 = v[0][0] + v[1][0], u1 = v[0][1] + v[1][1];
          const float snd = odd ? u0 : u1;
          const float r2 = (odd ? u1 : u0) + __shfl_xor_sync(0xffffffffu, snd, 1);
          const float tm = r2 + __shfl_xor_sync(0xffffffffu, r2, 2);
          sblk = mm == 0 ? tm : sblk + tm;
          if (mm == 3) {
            if (t < 2) {
              const int key = kst + 16 * blk + g + 8 * t;
              if (key < ncand) {
                lrow[key] = sblk;
                add16(hrow, okey(sblk) >> 21);
              }
            }
          }
        }
      }
    }
    cp_async_wait<0>();
    __syncthreads();
  }
  // ---- counts → the rows' global counts; the group's last CTA selects
  for (int i = tid; i < RPC * kBins / 2; i += NT) {
    const uint32_t v = (&S.hist[0][0])[i];
    if (v) {
      int* hg = hist + (size_t)(grp * RPC) * kBins + 2 * i;
      if (v & 0xFFFF) atomicAdd(hg, int(v & 0xFFFF));
      if (v >> 16) atomicAdd(hg + 1, int(v >> 16));
    }
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) S.misc[0] = atomicAdd(ctr + grp, 1) == ks_n - 1;
  __syncthreads();
  if (!S.misc[0]) return;
  __threadfence();
  if (tid == 0) ctr[grp] = 0;
  if (!rowok) return;
  int* out = list + (size_t)r * kMaxKeys;
  int* hg = hist + (size_t)ri * kBins;
  if (ncand <= kIdxTop) {
    for (int i = lane; i <= mt.pos; i += 32) out[i] = i;
    if (lane == 0) cnt[r] = mt.pos + 1;
  } else {
    select_row<8>(lrow, ncand, mt.pos, hg, hrow, out);
    if (lane == 0) cnt[r] = kIdxTop + kWindow;
    const char* lb = reinterpret_cast<const char*>(lrow);
    for (int i = lane; i < (ncand * 4 + 127) / 128; i += 32) discard_l2(lb + (size_t)i * 128);
    for (int i = lane; i < kBins; i += 32) hg[i] = 0;
  }
}

// per-row 11-bit counts [nh][2048], per-group counters [nh] and (P153) per-row bounds [nh] for ≥ `rows` rows: zero
// when allocated, left zero by their users (P091's and P153's selecting CTAs) — one set for both patches
inline void bufs(size_t rows, int*& hist, int*& ctr, int*& bnd) {
  static int* h = nullptr;
  static int* c = nullptr;
  static size_t nh = 0;
  if (rows > nh) {
    if (h) {
      CUDA_CHECK(cudaFree(h));
      CUDA_CHECK(cudaFree(c));
    }
    nh = std::max<size_t>(rows, 64);
    CUDA_CHECK(cudaMalloc(&h, nh * kBins * 4));
    CUDA_CHECK(cudaMalloc(&c, nh * 2 * 4));
    CUDA_CHECK(cudaMemset(h, 0, nh * kBins * 4));
    CUDA_CHECK(cudaMemset(c, 0, nh * 2 * 4));
  }
  hist = h, ctr = c, bnd = c + nh;
}

// false: not handled here (decode-size M, or — P142 dsa_select_wide off — rows with ≥ 65536 candidates)
inline bool run(const uint8_t* qf4, const uint8_t* qs, const float* wts, const RowMeta* meta, DsaCache c, int M,
                int max_pos, int* list, int* cnt, float* ws, size_t ws_bytes, cudaStream_t s) {
  const int maxc = max_pos - kWindow + 1;
  if (M < 32 || (maxc > 65535 && !e55::patch::on(e55::patch::dsa_select_wide))) return false;
  const int ldl = std::max(64, maxc + 63) / 64 * 64;
  const int pass = int(std::min<size_t>(M, ws_bytes / ((size_t)ldl * 4)));
  if (pass < 1) qlog::fail("dsa", "indexer workspace too small");
  // key splits of ≤ 4096 keys, equal sizes (multiples of the 64-key stage)
  const int ks_n = std::max(1, (maxc + 4095) / 4096);
  const int kr = ((maxc + ks_n - 1) / ks_n + SK - 1) / SK * SK;
  int *hist, *ctr, *bnd;
  bufs(pass, hist, ctr, bnd);
  static bool attr = false;
  if (!attr) {
    CUDA_CHECK(cudaFuncSetAttribute(k_select, cudaFuncAttributeMaxDynamicSharedMemorySize, int(sizeof(Smem))));
    attr = true;
  }
  for (int r0 = 0; r0 < M; r0 += pass) {
    const int n = std::min(pass, M - r0);
    k_select<<<((n + RPC - 1) / RPC) * ks_n, 32 * RPC, sizeof(Smem), s>>>(qf4, qs, wts, meta, c, r0, n, ks_n, kr, ldl, ws,
                                                                         hist, ctr, list, cnt);
    CUDA_CHECK(cudaGetLastError());
  }
  return true;
}

}  // namespace p091
