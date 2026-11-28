// p046_select.cuh — P046 dsa_select_fast: the indexer's logits and top-2048 of dsa_select (base: dsa.cu k_idx_logits +
// k_topk), selected by patch::on(dsa_select_fast). Included by dsa.cu inside its anonymous namespace.
//
// Bitwise contract — the selected key lists and counts are the base's: every logit is the base's float (per 16-head
// tile the e2m1 × e2m1 products of the same operands and UE8M0 scales summed into fp32 from 0; per key
// relu(acc_h)·w_h + relu(acc_h+8)·w_h+8 per lane, the same xor-4/8/16 butterfly tree over the 8 lane groups, then
// ((t0 + t1) + t2) + t3 over the four 16-head tiles), and the selection is the same set with the same ties rule — the
// 2048 largest by the order-preserving key of the float bits, equal keys at the threshold taken from the lowest
// positions — listed in position order, then the window. Only the way there differs:
//   · k64 MMA (kind::mxf4nvf4, twice the k32 rate, the packed codes used as stored) where it is exact: a block's sum of
//     32 e2m1 products is a multiple of 2^(E−2) below 2^(E+11) (E = its two scale exponents), so when the exponent
//     spreads of the row's query heads and of the key add up to ≤ 9 every partial sum of the base's k32 chain is an
//     exact fp32 and both instructions return the same exact floats (checked: 3.4e10 real-shape accumulators and 2.7e9
//     random ones identical under the guard; wider spreads do differ, so those key groups take the base k32 chain);
//   · one warp holds a row's whole query (64 heads × 128 e2m1 as MMA fragments, scales, weights) in registers and
//     streams keys through a cp.async ring shared by the CTA's 8 rows (prefill) or 8 key ranges of one row (decode);
//     the 16-head butterflies run as a reduce-scatter (the same additions, 10 shuffles per 8 keys instead of 24);
//   · exact threshold pre-filter (ERNIE README "Lightweight indexer"): while the logits are produced, an exact count
//     of them per 11-bit key prefix (2048 bins); the bin holding the 2048-th largest then bounds the selection, so only
//     that bin's members are refined (two more prefix histograms over them) — instead of 4 radix passes and a
//     compaction over every logit of every row;
//   · a row group's last CTA (integer counter) selects right after its logits were written (read back from L2, then
//     discarded there) — no second launch over logits that went through DRAM.
#pragma once

namespace p046 {

constexpr int kBins = 2048;
// P221 dsa_sel_v32: decode / verify steps (not prefill) of 17 … 64 rows: 1024-key splits, rpc 1 up to 32 rows (a CTA
// a row × key split), rpc 8 above — instead of P091 / rpc 8 × 4096 keys (≥ 32 rows) or rpc 1 × 512 (17–31). Measured
// (B, cold, ~9.6K candidates, 3 / 4 / 8 sequences × 8 rows): 0.100 / 0.211 / 0.215 → 0.082 / 0.101 / 0.162 ms. The same
// logits and lists (rows a CTA and key splits only schedule the work; integer histograms)
constexpr int kV32Min = 17, kV32Max = 64, kV32Kr = 1024;
int g_rpc = 0, g_kr = 0;  // tests (dsa_select_cfg): force rows a CTA (1 / 8) and keys a split (0: the rule)
inline bool v32(int M) {
  return e55::patch::on(e55::patch::dsa_sel_v32) && !e55::step_prefill() && M >= kV32Min && M <= kV32Max;
}

template <int RPC>
struct Smem {
  static constexpr int WPR = 8 / RPC;           // warps per row
  static constexpr int NS = RPC == 8 ? 4 : 2;   // ring stages of 64·WPR keys
  uint8_t ring[NS][64 * WPR * kIR];
  int hist[RPC][kBins];
  int red[8][64];
  int misc[8];
};

__device__ __forceinline__ uint32_t okey(float f) {
  const uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ void cp_async4(void* smem, const void* gmem) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(s), "l"(gmem));
}
// e2m1 × e2m1, k64, two UE8M0 per row / column (bytes 0, 1: k 0–31, 32–63) — kind::mxf4nvf4 at twice the k32 rate.
// Layout (DeepSeekV4.1-Flash-Pico tests/mma_fp4_probe.cu, MIT; lane 4g+t, nibble i = element 8t+i, low first):
// A a0 = [g][8t…], a1 = [g+8][8t…], a2 = [g][32+8t…], a3 = [g+8][32+8t…]; B b0 = [col g][8t…], b1 = [g][32+8t…];
// scales: A row g + 8·(t & 1), B column g.
__device__ __forceinline__ void mma_e2m1_k64(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                             uint32_t b0, uint32_t b1, uint32_t sa, uint32_t sb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
}
// spread (max − min) of the four UE8M0 exponents of a 128-vector
__device__ __forceinline__ int exp_spread(uint32_t s) {
  const int a = s & 0xFF, b = (s >> 8) & 0xFF, c = (s >> 16) & 0xFF, d = s >> 24;
  return max(max(a, b), max(c, d)) - min(min(a, b), min(c, d));
}
// cp.async of 16 bytes of which the first n (0 … 16) are read, the rest zero-filled
__device__ __forceinline__ void cp_async16n(void* smem, const void* gmem, int n) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(n));
}
__device__ __forceinline__ void discard_l2(const void* p) {
  asm volatile("discard.global.L2 [%0], 128;\n" ::"l"(p) : "memory");
}

// ---- a group of W warps (1: one warp, 8: the CTA) working on one row
template <int W>
struct Grp {
  int rank, size;  // thread rank in the group, W·32
  __device__ __forceinline__ void sync() const {
    if constexpr (W == 1) __syncwarp();
    else __syncthreads();
  }
};

template <int W>
__device__ __forceinline__ bool grp_any(bool v) {
  if constexpr (W == 1) return __any_sync(0xffffffffu, v);
  else return __syncthreads_or(v);
}
// U independent exclusive prefixes (rank order) at once, and their totals; scratch: ≥ 8·U ints
template <int W, int U>
__device__ __forceinline__ void grp_scan_n(const Grp<W>& G, const int (&v)[U], int (&pre)[U], int (&total)[U],
                                           int* scratch) {
  const int lane = threadIdx.x & 31;
  int x[U];
#pragma unroll
  for (int k = 0; k < U; k++) x[k] = v[k];
#pragma unroll
  for (int o = 1; o < 32; o <<= 1)
#pragma unroll
    for (int k = 0; k < U; k++) {
      const int y = __shfl_up_sync(0xffffffffu, x[k], o);
      if (lane >= o) x[k] += y;
    }
  if constexpr (W == 1) {
#pragma unroll
    for (int k = 0; k < U; k++) total[k] = __shfl_sync(0xffffffffu, x[k], 31), pre[k] = x[k] - v[k];
  } else {
    const int warp = threadIdx.x >> 5;
    if (lane == 31)
#pragma unroll
      for (int k = 0; k < U; k++) scratch[8 * k + warp] = x[k];
    G.sync();
#pragma unroll
    for (int k = 0; k < U; k++) {
      int p = 0, tt = 0;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const int sv = scratch[8 * k + i];
        p += i < warp ? sv : 0;
        tt += sv;
      }
      pre[k] = p + x[k] - v[k], total[k] = tt;
    }
    G.sync();
  }
}
// exclusive prefix of v over the group's threads in rank order, and the total; scratch: ≥ 8 ints of shared memory
template <int W>
__device__ __forceinline__ int grp_scan(const Grp<W>& G, int v, int& total, int* scratch) {
  const int lane = threadIdx.x & 31;
  int x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= o) x += y;
  }
  if constexpr (W == 1) {
    total = __shfl_sync(0xffffffffu, x, 31);
    return x - v;
  } else {
    const int warp = threadIdx.x >> 5;
    if (lane == 31) scratch[warp] = x;
    G.sync();
    int pre = 0, tot = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int s = scratch[i];
      pre += i < warp ? s : 0;
      tot += s;
    }
    G.sync();
    total = tot;
    return pre + x - v;
  }
}

// in hist[n] (n = 2048 or 1024), from the top bin down: the bin b where the running count reaches `need`; returns b,
// and need − (count above b)
template <int W>
__device__ __forceinline__ int grp_find(const Grp<W>& G, const int* hist, int n, int& need, int* scratch) {
  const int per = n / G.size;  // bins per thread, from the top: thread i holds bins n−1−per·i … n−per·(i+1)
  const int hi = n - 1 - per * G.rank;
  int s = 0;
  for (int j = 0; j < per; j++) s += hist[hi - j];
  int tot;
  const int pre = grp_scan<W>(G, s, tot, scratch);
  int found = -1, rem = 0;
  if (pre < need && pre + s >= need) {
    int acc = pre;
    for (int j = 0; j < per; j++) {
      const int h = hist[hi - j];
      if (acc + h >= need) {
        found = hi - j, rem = need - acc;
        break;
      }
      acc += h;
    }
  }
  // exactly one thread found it (the counts reach need: total ≥ need)
  if constexpr (W == 1) {
    const unsigned b = __ballot_sync(0xffffffffu, found >= 0);
    const int src = __ffs(b) - 1;
    found = __shfl_sync(0xffffffffu, found, src), rem = __shfl_sync(0xffffffffu, rem, src);
  } else {
    if (found >= 0) scratch[8] = found, scratch[9] = rem;
    G.sync();
    found = scratch[8], rem = scratch[9];
    G.sync();
  }
  need = rem;
  return found;
}

// okeys of 4·U positions of a row (4 consecutive per thread per round, U rounds of the group in flight);
// positions ≥ ncand give 0 (below every logit's key)
template <int W, int U>
__device__ __forceinline__ void load_keys(const Grp<W>& G, const float* lg, int ncand, int i0, uint32_t (&u)[U][4]) {
  float4 v[U];
#pragma unroll
  for (int k = 0; k < U; k++) {
    const int i = i0 + 4 * (G.rank + k * G.size);
    if (i + 3 < ncand) v[k] = __ldcg(reinterpret_cast<const float4*>(lg + i));
    else if (i < ncand) {
      v[k].x = __ldcg(lg + i), v[k].y = i + 1 < ncand ? __ldcg(lg + i + 1) : 0.f;
      v[k].z = i + 2 < ncand ? __ldcg(lg + i + 2) : 0.f, v[k].w = 0.f;
    }
  }
#pragma unroll
  for (int k = 0; k < U; k++) {
    const int i = i0 + 4 * (G.rank + k * G.size);
    u[k][0] = i < ncand ? okey(v[k].x) : 0u, u[k][1] = i + 1 < ncand ? okey(v[k].y) : 0u;
    u[k][2] = i + 2 < ncand ? okey(v[k].z) : 0u, u[k][3] = i + 3 < ncand ? okey(v[k].w) : 0u;
  }
}

// the selection of one row from its logits lg[0 … ncand) and the exact count per 11-bit prefix (hist1, complete):
// list[0 … 2048) = the top-2048 positions in order, then the window; h2: kBins ints of shared scratch (may be hist1)
template <int W>
__device__ void select_row(const Grp<W>& G, const float* lg, int ncand, int p, const int* hist1, int* h2, int* scratch,
                           int* out) {
  constexpr int U = 8;  // float4 loads per thread in flight
  const int step = 4 * U * G.size;
  int need = kIdxTop;
  const uint32_t b1 = grp_find<W>(G, hist1, kBins, need, scratch);
  G.sync();  // hist1 may share h2's memory
  // level 2: bits 20 … 10 of the members of bin b1
  for (int i = G.rank; i < kBins; i += G.size) h2[i] = 0;
  G.sync();
  for (int i0 = 0; i0 < ncand; i0 += step) {
    uint32_t u[U][4];
    load_keys<W, U>(G, lg, ncand, i0, u);
#pragma unroll
    for (int k = 0; k < U; k++)
#pragma unroll
      for (int j = 0; j < 4; j++)
        if ((u[k][j] >> 21) == b1) atomicAdd(&h2[(u[k][j] >> 10) & 0x7FF], 1);
  }
  G.sync();
  const uint32_t b2 = grp_find<W>(G, h2, kBins, need, scratch);
  G.sync();
  const uint32_t pre22 = (b1 << 11) | b2;
  // level 3: bits 9 … 0
  for (int i = G.rank; i < 1024; i += G.size) h2[i] = 0;
  G.sync();
  for (int i0 = 0; i0 < ncand; i0 += step) {
    uint32_t u[U][4];
    load_keys<W, U>(G, lg, ncand, i0, u);
#pragma unroll
    for (int k = 0; k < U; k++)
#pragma unroll
      for (int j = 0; j < 4; j++)
        if ((u[k][j] >> 10) == pre22) atomicAdd(&h2[u[k][j] & 0x3FF], 1);
  }
  G.sync();
  const uint32_t b3 = grp_find<W>(G, h2, 1024, need, scratch);
  const uint32_t thr = (pre22 << 10) | b3;
  const int need_eq = need;  // keys equal to thr taken, lowest positions first
  // output in position order (keys of positions ≥ ncand are 0 < thr: never taken); the U sub-rounds' prefix sums run
  // interleaved, the equal-key ranks only in rounds that hold a key equal to thr
  int base = 0, eqbase = 0;
  for (int i0 = 0; i0 < ncand; i0 += step) {
    uint32_t u[U][4];
    load_keys<W, U>(G, lg, ncand, i0, u);
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
    if (grp_any<W>(anyeq)) grp_scan_n<W, U>(G, ne, er, etot, scratch);
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
    grp_scan_n<W, U>(G, nt, pre, ttot, scratch);
#pragma unroll
    for (int k = 0; k < U; k++) {
      const int i = i0 + 4 * (G.rank + k * G.size);
      int pos = base + pre[k];
#pragma unroll
      for (int j = 0; j < 4; j++)
        if (take[k][j]) out[pos++] = i + j;
      base += ttot[k];
    }
  }
  for (int i = G.rank; i < kWindow; i += G.size) out[kIdxTop + i] = p - kWindow + 1 + i;
}

// grid: ⌈n / RPC⌉ row groups × ks key splits (split fastest). Warp w: row r0 + grp·RPC + w % RPC, keys of split ks,
// sub-range w / RPC of WPR. lg: [n rows][ldl] fp32; hist: [n rows][2048] (zero on entry, zeroed again by the
// selecting CTA); ctr: per row group (zero, reset).
template <int RPC>
__global__ void __launch_bounds__(256, 1)
    k_select(const uint8_t* __restrict__ qf4, const uint8_t* __restrict__ qs, const float* __restrict__ wts,
             const RowMeta* __restrict__ meta, DsaCache c, int row0, int n, int ks_n, int kr, int ldl, float* lg,
             int* hist, int* ctr, int* list, int* cnt) {
  constexpr int WPR = 8 / RPC, kStages = Smem<RPC>::NS;
  extern __shared__ __align__(16) uint8_t smraw[];
  Smem<RPC>& S = *reinterpret_cast<Smem<RPC>*>(smraw);
  const int grp = blockIdx.x / ks_n, ks = blockIdx.x % ks_n;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int rl = warp % RPC, sub = warp / RPC;  // row in group, key sub-range of the row
  const int ri = grp * RPC + rl, r = row0 + ri;
  const bool rowok = ri < n;
  const RowMeta mt = meta[rowok ? r : row0];
  const int ncand = rowok ? mt.pos - kWindow + 1 : 0;
  const bool work = rowok && ncand > kIdxTop;
  // ---- the CTA's key range and slots; rows of one group usually share a slot (else one round per slot)
  int kmax = 0;
  for (int i = 0; i < RPC; i++) {
    const int rr = grp * RPC + i;
    if (rr < n) {
      const int nc = meta[row0 + rr].pos - kWindow + 1;
      if (nc > kIdxTop) kmax = max(kmax, nc);
    }
  }
  const int kc0 = ks * kr, kc1 = min(kmax, kc0 + kr);
  for (int i = tid; i < RPC * kBins; i += 256) (&S.hist[0][0])[i] = 0;
  // ---- query fragments of this warp's row (64 heads: 4 m16 tiles × 4 k32 steps)
  // k64 fragments of the query (packed e2m1 straight from qf4) and the largest exponent spread of its 64 heads
  uint32_t A[4][2][4], SA[4][2];
  float W0[4], W1[4];
  int rq = 0;
  const uint8_t* qrow = qf4 + (size_t)r * 64 * 64;
  if (work) {
#pragma unroll
    for (int m = 0; m < 4; m++) {
      const int h0 = 16 * m + g;
#pragma unroll
      for (int kq = 0; kq < 2; kq++) {
        A[m][kq][0] = *reinterpret_cast<const uint32_t*>(qrow + h0 * 64 + 32 * kq + 4 * t);
        A[m][kq][1] = *reinterpret_cast<const uint32_t*>(qrow + (h0 + 8) * 64 + 32 * kq + 4 * t);
        A[m][kq][2] = *reinterpret_cast<const uint32_t*>(qrow + h0 * 64 + 32 * kq + 16 + 4 * t);
        A[m][kq][3] = *reinterpret_cast<const uint32_t*>(qrow + (h0 + 8) * 64 + 32 * kq + 16 + 4 * t);
        SA[m][kq] = *reinterpret_cast<const uint16_t*>(qs + ((size_t)r * 64 + h0 + 8 * (t & 1)) * 4 + 2 * kq);
      }
      W0[m] = wts[(size_t)r * 64 + h0], W1[m] = wts[(size_t)r * 64 + h0 + 8];
    }
    rq = max(exp_spread(*reinterpret_cast<const uint32_t*>(qs + ((size_t)r * 64 + lane) * 4)),
             exp_spread(*reinterpret_cast<const uint32_t*>(qs + ((size_t)r * 64 + 32 + lane) * 4)));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) rq = max(rq, __shfl_xor_sync(0xffffffffu, rq, o));
  }
  __syncthreads();
  // ---- rounds over the distinct slots of the group's working rows
  uint32_t done = 0;  // bit i: row i of the group handled
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
    constexpr int SK = 64 * WPR;  // keys per stage
    const int nst = kc1 > kc0 ? (kc1 - kc0 + SK - 1) / SK : 0;
    auto issue = [&](int st) {  // stage st → ring[st % kStages]; keys past kc1 zero-filled
      const int t0 = kc0 + st * SK, nb = min(SK, kc1 - t0) * kIR;
      uint8_t* dst = S.ring[st % kStages];
      const uint8_t* src = kb + (size_t)t0 * kIR;
      if (((uintptr_t)src & 15) == 0) {
        for (int i = tid; i < SK * kIR / 16; i += 256) cp_async16n(dst + 16 * i, src + 16 * i, max(0, min(16, nb - 16 * i)));
      } else {  // a slot whose rows start 4-byte aligned only (4 ∤ ctx)
        for (int i = tid; i < SK * kIR / 4; i += 256) {
          if (4 * i < nb) cp_async4(dst + 4 * i, src + 4 * i);
          else *reinterpret_cast<uint32_t*>(dst + 4 * i) = 0;
        }
      }
    };
#pragma unroll
    for (int st = 0; st < kStages - 1; st++) {
      if (st < nst) issue(st);
      cp_async_commit();
    }
    for (int st = 0; st < nst; st++) {
      cp_async_wait<kStages - 2>();
      __syncthreads();
      if (st + kStages - 1 < nst) issue(st + kStages - 1);
      cp_async_commit();
      const uint8_t* ring = S.ring[st % kStages] + sub * 64 * kIR;
      const int kb0 = kc0 + st * SK + sub * 64;  // this warp's 64 keys
      if (mine && kb0 < ncand) {
        for (int np = 0; np < 4; np++) {  // two groups of 8 keys at a time (8 independent MMA chains)
          const int n0 = kb0 + 16 * np;
          if (n0 >= ncand) break;
          float acc[2][4][4];
#pragma unroll
          for (int q = 0; q < 2; q++)
#pragma unroll
            for (int m = 0; m < 4; m++) acc[q][m][0] = acc[q][m][1] = acc[q][m][2] = acc[q][m][3] = 0.f;
          const uint8_t* kr8[2] = {ring + (16 * np + g) * kIR, ring + (16 * np + 8 + g) * kIR};
          uint32_t sb[2];
#pragma unroll
          for (int q = 0; q < 2; q++) sb[q] = *reinterpret_cast<const uint32_t*>(kr8[q] + 64);
          // exact k64: with every exponent spread rq + rk ≤ 9, each partial sum of the k32 chain is exact in fp32
          // (block sums are multiples of 2^(E−2) below 2^(E+11)), so k64 and the k32 chain give the same exact
          // floats; a wider spread takes the base k32 chain
          if (!__any_sync(0xffffffffu, rq + max(exp_spread(sb[0]), exp_spread(sb[1])) > 9)) {
#pragma unroll
            for (int kq = 0; kq < 2; kq++)
#pragma unroll
              for (int q = 0; q < 2; q++) {
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(kr8[q] + 32 * kq + 4 * t);
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(kr8[q] + 32 * kq + 16 + 4 * t);
                const uint32_t sfb = (sb[q] >> (16 * kq)) & 0xFFFF;
#pragma unroll
                for (int m = 0; m < 4; m++)
                  mma_e2m1_k64(acc[q][m], A[m][kq][0], A[m][kq][1], A[m][kq][2], A[m][kq][3], b0, b1, SA[m][kq], sfb);
              }
          } else {
#pragma unroll 1
            for (int kk = 0; kk < 4; kk++) {
#pragma unroll
              for (int q = 0; q < 2; q++) {
                const uint32_t b0 = e2m1x4_expand(*reinterpret_cast<const uint16_t*>(kr8[q] + 16 * kk + 2 * t));
                const uint32_t b1 = e2m1x4_expand(*reinterpret_cast<const uint16_t*>(kr8[q] + 16 * kk + 8 + 2 * t));
                const uint32_t sfb = (sb[q] >> (8 * kk)) & 0xFF;
#pragma unroll
                for (int m = 0; m < 4; m++) {
                  const int h0 = 16 * m + g;
                  mma_e2m1_e2m1(acc[q][m],
                                e2m1x4_expand(*reinterpret_cast<const uint16_t*>(qrow + h0 * 64 + 16 * kk + 2 * t)),
                                e2m1x4_expand(*reinterpret_cast<const uint16_t*>(qrow + (h0 + 8) * 64 + 16 * kk + 2 * t)),
                                e2m1x4_expand(*reinterpret_cast<const uint16_t*>(qrow + h0 * 64 + 16 * kk + 8 + 2 * t)),
                                e2m1x4_expand(*reinterpret_cast<const uint16_t*>(qrow + (h0 + 8) * 64 + 16 * kk + 8 + 2 * t)),
                                b0, b1, qs[((size_t)r * 64 + h0 + 8 * (lane & 1)) * 4 + kk], sfb);
                }
              }
            }
          }
#pragma unroll
          for (int q = 0; q < 2; q++) {
            // per 16-head tile: relu·w pairs (base expression), then the butterfly over g as a reduce-scatter
            float v[4][2];
#pragma unroll
            for (int m = 0; m < 4; m++) {
              v[m][0] = fmaxf(acc[q][m][0], 0.f) * W0[m] + fmaxf(acc[q][m][2], 0.f) * W1[m];  // key 2t
              v[m][1] = fmaxf(acc[q][m][1], 0.f) * W0[m] + fmaxf(acc[q][m][3], 0.f) * W1[m];  // key 2t+1
            }
            const int jk = g & 1, k1 = (g >> 1) & 1, k2 = (g >> 2) & 1;
            float r1[4];
#pragma unroll
            for (int m = 0; m < 4; m++) {
              const float snd = jk ? v[m][0] : v[m][1];
              const float rcv = __shfl_xor_sync(0xffffffffu, snd, 4);
              r1[m] = (jk ? v[m][1] : v[m][0]) + rcv;
            }
            float r2[2];
#pragma unroll
            for (int i = 0; i < 2; i++) {
              const float snd = k1 ? r1[2 * i] : r1[2 * i + 1];
              const float rcv = __shfl_xor_sync(0xffffffffu, snd, 8);
              r2[i] = (k1 ? r1[2 * i + 1] : r1[2 * i]) + rcv;
            }
            const float snd = k2 ? r2[0] : r2[1];
            const float rcv = __shfl_xor_sync(0xffffffffu, snd, 16);
            const float r3 = (k2 ? r2[1] : r2[0]) + rcv;  // tile 2·k2 + k1, key 2t + jk
            const float x1 = __shfl_down_sync(0xffffffffu, r3, 8), x2 = __shfl_down_sync(0xffffffffu, r3, 16),
                        x3 = __shfl_down_sync(0xffffffffu, r3, 24);
            if (lane < 8) {
              const int key = n0 + 8 * q + 2 * t + jk;
              if (key < ncand) {
                const float lgv = ((r3 + x1) + x2) + x3;
                lg[(size_t)ri * ldl + key] = lgv;
                atomicAdd(&S.hist[rl][okey(lgv) >> 21], 1);
              }
            }
          }
        }
      }
    }
    cp_async_wait<0>();
    __syncthreads();
  }
  // ---- counts → the rows' global histograms; the group's last CTA selects
  for (int i = tid; i < RPC * kBins; i += 256) {
    const int v = (&S.hist[0][0])[i];
    if (v) atomicAdd(hist + (size_t)(grp * RPC) * kBins + i, v);
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) S.misc[0] = atomicAdd(ctr + grp, 1) == ks_n - 1;
  __syncthreads();
  if (!S.misc[0]) return;
  __threadfence();
  if (tid == 0) ctr[grp] = 0;
  // the group's rows: WPR warps per row
  if (!rowok) return;
  Grp<WPR> G{WPR == 1 ? lane : tid, 32 * WPR};
  int* out = list + (size_t)r * kMaxKeys;
  int* hg = hist + (size_t)ri * kBins;
  if (ncand <= kIdxTop) {
    for (int i = G.rank; i <= mt.pos; i += G.size) out[i] = i;
    if (G.rank == 0) cnt[r] = mt.pos + 1;
  } else {
    int* h1 = S.hist[rl];  // this row's region: its complete histogram, then the level-2 / 3 scratch
    {  // the complete counts, 16-byte loads all in flight
      constexpr int NV = kBins / 4 / (32 * WPR);
      int4 v[NV];
#pragma unroll
      for (int k = 0; k < NV; k++) v[k] = __ldcg(reinterpret_cast<const int4*>(hg) + G.rank + k * G.size);
#pragma unroll
      for (int k = 0; k < NV; k++) reinterpret_cast<int4*>(h1)[G.rank + k * G.size] = v[k];
    }
    G.sync();
    int* h2 = h1;  // select_row reuses the region after reading the counts
    select_row<WPR>(G, lg + (size_t)ri * ldl, ncand, mt.pos, h1, h2, WPR == 1 ? S.red[warp] : &S.red[0][0], out);
    if (G.rank == 0) cnt[r] = kIdxTop + kWindow;
    // discard this row's logits from L2 (no write-back of consumed data)
    const char* lb = reinterpret_cast<const char*>(lg + (size_t)ri * ldl);
    for (int i = G.rank; i < (ncand * 4 + 127) / 128; i += G.size) discard_l2(lb + (size_t)i * 128);
  }
  for (int i = G.rank; i < kBins; i += G.size) hg[i] = 0;
}

}  // namespace p046

namespace p046 {
// false: not handled here (the base runs)
inline bool run(const uint8_t* qf4, const uint8_t* qs, const float* wts, const RowMeta* meta, DsaCache c, int M,
                int max_pos, int* list, int* cnt, float* ws, size_t ws_bytes, cudaStream_t s) {
  const int maxc = max_pos - kWindow + 1;
  const int ldl = std::max(64, maxc + 63) / 64 * 64;
  const int pass = int(std::min<size_t>(M, ws_bytes / ((size_t)ldl * 4)));
  if (pass < 1) qlog::fail("dsa", "indexer workspace too small");
  const bool v = v32(M);
  const int rpc = g_rpc ? g_rpc : v ? (M <= 32 ? 1 : 8) : M >= 32 ? 8 : 1;
  const int kr = g_kr ? g_kr : v ? kV32Kr : rpc == 8 ? 4096 : 512;
  const int ks_n = std::max(1, (maxc + kr - 1) / kr);
  // per-row histograms and per-group counters: zero when allocated, zeroed again by their users
  static int* hist = nullptr;
  static int* ctr = nullptr;
  static size_t nh = 0;
  if ((size_t)pass > nh) {
    if (hist) {
      CUDA_CHECK(cudaFree(hist));
      CUDA_CHECK(cudaFree(ctr));
    }
    nh = std::max<size_t>(pass, 64);
    CUDA_CHECK(cudaMalloc(&hist, nh * kBins * 4));
    CUDA_CHECK(cudaMalloc(&ctr, nh * 4));
    CUDA_CHECK(cudaMemset(hist, 0, nh * kBins * 4));
    CUDA_CHECK(cudaMemset(ctr, 0, nh * 4));
  }
  static bool attr = false;
  if (!attr) {
    CUDA_CHECK(cudaFuncSetAttribute(k_select<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(sizeof(Smem<8>))));
    CUDA_CHECK(cudaFuncSetAttribute(k_select<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(sizeof(Smem<1>))));
    attr = true;
  }
  for (int r0 = 0; r0 < M; r0 += pass) {
    const int n = std::min(pass, M - r0);
    if (rpc == 8)
      k_select<8><<<((n + 7) / 8) * ks_n, 256, sizeof(Smem<8>), s>>>(qf4, qs, wts, meta, c, r0, n, ks_n, kr, ldl, ws, hist,
                                                                     ctr, list, cnt);
    else
      k_select<1><<<n * ks_n, 256, sizeof(Smem<1>), s>>>(qf4, qs, wts, meta, c, r0, n, ks_n, kr, ldl, ws, hist, ctr, list,
                                                         cnt);
    CUDA_CHECK(cudaGetLastError());
  }
  return true;
}
}  // namespace p046

