// p153_select_long.cuh — P153 dsa_select_long: prefill dsa_select at long context (M ≥ 32, a row with ≥ kMinCand
// indexer candidates), selected by patch::on(dsa_select_long). Included by dsa.cu inside its anonymous namespace after
// p091_select_pf.cuh (uses its and P046's helpers, and P091's count buffers).
//
// Bitwise contract — P091's (= P046's = the base's): every logit is P091's float (its MMA loop, exactness guard, k32
// fallback, leaves, tree and tile sums, unchanged), and the selection is the same list (the 2048 largest by the
// order-preserving key of the float bits, equal keys at the threshold from the lowest positions, in position order,
// then the window). What differs is what is stored and read (P091 writes every logit as fp32 and reads them 2–3 times:
// at 1M 4.3 GB written + 8.5 GB read a 1024-row call, and each 8-row group read the 71 MB key cache from DRAM):
//   · exact lower bound: the threshold's 11-bit bin b1 (level 1 of the selection) is ≥ any bin b with ≥ 2048 of the
//     row's keys at or above it, counted over any subset of its keys. Each CTA reads its row's bound (the largest such
//     b found so far from the counts of the row's finished CTAs — a subset; re-read every 8 stages), and keys below it
//     are neither counted nor kept: they are below the threshold. Every key of bin ≥ b1 is counted (its bin is ≥ every
//     bound), so level 1 over these counts finds the same b1 and the same need, and levels 2 / 3 and the output only
//     ever look at keys of bin ≥ b1 — all of them kept. Which keys are kept beyond those depends on timing; the list
//     does not;
//   · storage: per (row, key split) the kept keys' order keys in position order, compactly (up to kC, their offsets in
//     the split as uint16), and once a split has more than kC kept keys its remaining keys densely at their offsets
//     (all of them — exact keys of the row): 24 / 7.4 / 4.5 % of P091's logit bytes at 131K / 524K / 1M;
//   · grid: key splits major inside super-groups of ≤ 32 row groups (P091: one row group's splits at a time), so the
//     CTAs in flight share key tiles in L2 and a row's bound tightens split by split;
//   · selection (each group's last CTA, as P091): a row's kept keys as one stream through a per-row split table in
//     shared memory (the ring's bytes); one pass collects the keys of bin ≥ b1 (2048 − need + count(b1) of them) in
//     position order into the workspace's spare bytes, and P091's levels 2 / 3, kept-aside members, merge and tie rule
//     run on that set — or, when it does not fit, on the stream itself.
#pragma once

namespace p153 {
using p046::discard_l2;
using p046::exp_spread;
using p046::okey;
using p091::add16;
using p091::cnt16;
using p091::cp_async4z;
using p091::mma_k64;
using p091::scan_cnt;
using p091::warp_find;

constexpr int kBins = 2048;
constexpr int RPC = 8;    // rows per CTA, one warp each
constexpr int SK = 64;    // keys per ring stage
constexpr int NS = 3;     // ring stages
constexpr int KS = 80;    // shared bytes per key
constexpr int kPCap = 256;
constexpr int kC = 256;           // compact kept keys per (row, split); the rest of a split beyond them dense
constexpr int kMaxSplits = 239;   // splits a row: the selection's split table (2 words a split + 2) in the ring's bytes
constexpr int kMinCand = 131072;  // P153 takes calls with a row of at least this many candidates (P091 below: faster at
                                  // 16384 / 65536: 1.51 / 5.25 ms vs 1.81 / 5.75)

struct Smem {
  union {
    uint8_t ring[NS][SK * KS];          // MMA phase: 15 KB
    uint32_t tab[RPC][2 * (kMaxSplits + 1)];  // selection: per row (V_j, nc | kb << 16), 2 words a split
  };
  uint32_t hist[RPC][kBins / 2];
  int misc[4];
};
static_assert(sizeof(uint32_t) * RPC * 2 * (kMaxSplits + 1) <= NS * SK * KS, "split tables in the ring's bytes");

// ballot bits of lanes 4g + t (t = 0, 1: key g + 8t of a 16-key block) → mask in key order
__device__ __forceinline__ uint32_t key_order16(uint32_t b) {
  auto c = [](uint32_t x) {  // bits 0, 4, … 28 → 0 … 7
    x &= 0x11111111u;
    x = (x | (x >> 3)) & 0x03030303u;
    x = (x | (x >> 6)) & 0x000F000Fu;
    return (x | (x >> 12)) & 0xFFu;
  };
  return c(b) | (c(b >> 1) << 8);
}
// ballot bits of lanes 4jk + t < 8 (key 2t + jk of an 8-key group, P091's k32 butterfly) → mask in key order
__device__ __forceinline__ uint32_t key_order8(uint32_t b) {
  auto s = [](uint32_t x) {  // bit i → 2i, 4 bits
    x &= 0xFu;
    x = (x | (x << 2)) & 0x33u;
    return (x | (x << 1)) & 0x55u;
  };
  return s(b) | (s(b >> 4) << 1);
}

// the highest bin b with ≥ kIdxTop of the counted keys at or above it (0: fewer counted) — one warp; lane i holds
// bins 2047 − 64i … 1984 − 64i (warp_find's layout)
__device__ __noinline__ int row_bound(const int* hg) {
  const int lane = threadIdx.x & 31, lo = kBins - 64 * (lane + 1);
  int c[64];
#pragma unroll
  for (int k = 0; k < 16; k++) {
    const int4 v = __ldcg(reinterpret_cast<const int4*>(hg + lo) + k);
    c[4 * k] = v.x, c[4 * k + 1] = v.y, c[4 * k + 2] = v.z, c[4 * k + 3] = v.w;
  }
  int s = 0;
#pragma unroll
  for (int k = 0; k < 64; k++) s += c[k];
  int x = s;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= o) x += y;
  }
  if (__shfl_sync(0xffffffffu, x, 31) < kIdxTop) return 0;
  int found = -1;
  if (x - s < kIdxTop && x >= kIdxTop) {
    int acc = x - s;
#pragma unroll
    for (int k = 63; k >= 0; k--) {
      acc += c[k];
      if (found < 0 && acc >= kIdxTop) found = lo + k;
    }
  }
  const unsigned b = __ballot_sync(0xffffffffu, found >= 0);
  return __shfl_sync(0xffffffffu, found, __ffs(b) - 1);
}

__device__ __forceinline__ uint32_t lds32(uint32_t a) {
  uint32_t v;
  asm("ld.shared.u32 %0, [%1];\n" : "=r"(v) : "r"(a));
  return v;
}

// a row's kept keys as one stream in position order: split j holds virtual entries [V_j, V_j+1): its compact keys
// (nc, padded to 4), then its dense part [kb, len_j) (padded to 4: the last split only); padding loads as 0 (below
// every key). Passed by value (registers).
struct Stream {
  const uint32_t* lg;     // the row's ws regions: split j at lg + j·kr
  const uint16_t* offs;   // [split][kC] offsets of the compact keys
  uint32_t tab;           // shared address: word 2j = V_j (j ≤ nspl; V_nspl = total), 2j + 1 = nc | kb << 16
  int kr, nspl, lastlen, vtot;
};

// order keys (and with POS positions) of 4·U entries a lane: entries i0 + 4(lane + 32k) … + 3; jc: the lane's split
// cursor (monotone)
template <int U, bool POS>
__device__ __forceinline__ void vload(const Stream S, int i0, int& jc, uint32_t (&u)[U][4], int (&ps)[U][4]) {
  const int lane = threadIdx.x & 31;
  uint4 v[U];
  uint2 o[U];
  int pb[U], nv[U];
  bool cp[U];
#pragma unroll
  for (int k = 0; k < U; k++) {
    const int vi = i0 + 4 * (lane + 32 * k);
    nv[k] = 0, cp[k] = false, pb[k] = 0;
    v[k] = make_uint4(0, 0, 0, 0), o[k] = make_uint2(0, 0);
    if (vi < S.vtot) {
      while (int(lds32(S.tab + 8 * (jc + 1))) <= vi) jc++;
      const int local = vi - int(lds32(S.tab + 8 * jc));
      const uint32_t w = lds32(S.tab + 8 * jc + 4);
      const int nc = int(w & 0xFFFF), nc4 = (nc + 3) & ~3;
      const uint32_t* base = S.lg + jc * S.kr;
      if (local < nc4) {
        cp[k] = true, nv[k] = nc - local, pb[k] = jc * S.kr;
        v[k] = __ldcg(reinterpret_cast<const uint4*>(base + local));
        if constexpr (POS) o[k] = __ldcg(reinterpret_cast<const uint2*>(S.offs + jc * kC + local));
      } else {
        const int d = int(w >> 16) + local - nc4;
        nv[k] = jc == S.nspl - 1 ? S.lastlen - d : 4, pb[k] = jc * S.kr + d;
        v[k] = __ldcg(reinterpret_cast<const uint4*>(base + d));
      }
    }
  }
#pragma unroll
  for (int k = 0; k < U; k++) {
    u[k][0] = nv[k] > 0 ? v[k].x : 0u, u[k][1] = nv[k] > 1 ? v[k].y : 0u;
    u[k][2] = nv[k] > 2 ? v[k].z : 0u, u[k][3] = nv[k] > 3 ? v[k].w : 0u;
    if constexpr (POS) {
      if (cp[k]) {
        ps[k][0] = pb[k] + int(o[k].x & 0xFFFF), ps[k][1] = pb[k] + int(o[k].x >> 16);
        ps[k][2] = pb[k] + int(o[k].y & 0xFFFF), ps[k][3] = pb[k] + int(o[k].y >> 16);
      } else {
        ps[k][0] = pb[k], ps[k][1] = pb[k] + 1, ps[k][2] = pb[k] + 2, ps[k][3] = pb[k] + 3;
      }
    }
  }
}

// the kept keys of bin ≥ b1 of a row, collected in position order (order keys, positions), in spare ws bytes
struct Scratch {
  const uint32_t* key;
  const int* pos;
  int vtot;
};
template <int U, bool POS>
__device__ __forceinline__ void vload(const Scratch S, int i0, int&, uint32_t (&u)[U][4], int (&ps)[U][4]) {
  const int lane = threadIdx.x & 31;
  uint4 v[U], q[U];
#pragma unroll
  for (int k = 0; k < U; k++) {
    const int i = i0 + 4 * (lane + 32 * k);
    v[k] = q[k] = make_uint4(0, 0, 0, 0);
    if (i < S.vtot) {
      v[k] = __ldcg(reinterpret_cast<const uint4*>(S.key + i));
      if constexpr (POS) q[k] = __ldcg(reinterpret_cast<const uint4*>(S.pos + i));
    }
  }
#pragma unroll
  for (int k = 0; k < U; k++) {
    const int nv = S.vtot - (i0 + 4 * (lane + 32 * k));
    u[k][0] = nv > 0 ? v[k].x : 0u, u[k][1] = nv > 1 ? v[k].y : 0u;
    u[k][2] = nv > 2 ? v[k].z : 0u, u[k][3] = nv > 3 ? v[k].w : 0u;
    if constexpr (POS) ps[k][0] = int(q[k].x), ps[k][1] = int(q[k].y), ps[k][2] = int(q[k].z), ps[k][3] = int(q[k].w);
  }
}

// P091's select_rest over the stream: one warp selects its row after level 1 (b1, need); h: the row's 4 KB of shared
// counts; out: list row. The same counts, bins, kept-aside members, merge and tie rule — on the kept keys, which hold
// every key of bin ≥ b1 (entries loading as 0 are below every key: never counted, listed or kept aside).
template <int U, bool WIDE, class SRC>
__device__ __forceinline__ void select_rest(const SRC S, int p, uint32_t b1, int need, uint32_t* h, int* out) {
  const int lane = threadIdx.x & 31;
  const int step = 4 * U * 32;
  uint32_t b2 = 0;
  int npend = 0;
  if constexpr (!WIDE) {
    for (int i = lane; i < kBins / 2; i += 32) h[i] = 0;
    __syncwarp();
    int jc = 0;
    for (int i0 = 0; i0 < S.vtot; i0 += step) {
      uint32_t u[U][4];
      int ps[U][4];
      vload<U, false>(S, i0, jc, u, ps);
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
    for (int half = 1; half >= 0; half--) {
      for (int i = lane; i < 1024; i += 32) h[i] = 0;
      __syncwarp();
      const uint32_t b1h = (b1 << 1) | uint32_t(half);
      int jc = 0;
      for (int i0 = 0; i0 < S.vtot; i0 += step) {
        uint32_t u[U][4];
        int ps[U][4];
        vload<U, false>(S, i0, jc, u, ps);
#pragma unroll
        for (int k = 0; k < U; k++)
#pragma unroll
          for (int j = 0; j < 4; j++)
            if ((u[k][j] >> 20) == b1h) atomicAdd(h + ((u[k][j] >> 10) & 0x3FF), 1u);
      }
      __syncwarp();
      int sum = 0;
      for (int i = lane; i < 1024; i += 32) sum += int(h[i]);
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
      if (half && sum < need) {
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
    uint32_t* pk = h;
    int* pp = reinterpret_cast<int*>(h + kPCap);
    int* T = reinterpret_cast<int*>(h + 2 * kPCap);
    __syncwarp();
    int nd = 0, npd = 0, jc = 0;
    for (int i0 = 0; i0 < S.vtot; i0 += step) {
      uint32_t u[U][4];
      int ps[U][4];
      vload<U, true>(S, i0, jc, u, ps);
      int nt[U], np[U];
      bool anyp = false;
#pragma unroll
      for (int k = 0; k < U; k++) {
        nt[k] = np[k] = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const uint32_t q = u[k][j] >> 10;
          nt[k] += q > pre22;
          np[k] += q == pre22;
        }
        anyp |= np[k] > 0;
      }
      int pre[U], ttot[U];
      scan_cnt<U>(nt, pre, ttot);
#pragma unroll
      for (int k = 0; k < U; k++) {
        int pos = nd + pre[k];
#pragma unroll
        for (int j = 0; j < 4; j++)
          if ((u[k][j] >> 10) > pre22) out[pos++] = ps[k][j];
        nd += ttot[k];
      }
      if (__any_sync(0xffffffffu, anyp)) {
        int ppre[U], ptot[U];
        scan_cnt<U>(np, ppre, ptot);
#pragma unroll
        for (int k = 0; k < U; k++) {
          int pos = npd + ppre[k];
#pragma unroll
          for (int j = 0; j < 4; j++)
            if ((u[k][j] >> 10) == pre22) pk[pos] = u[k][j], pp[pos++] = ps[k][j];
          npd += ptot[k];
        }
      }
    }
    __syncwarp();
    int nT = 0;
    for (int e0 = 0; e0 < npend; e0 += 32) {
      const int e = e0 + lane;
      bool tk = false;
      if (e < npend) {
        const uint32_t ue = pk[e];
        int rank = 0;
        for (int j = 0; j < npend; j++) rank += pk[j] > ue || (pk[j] == ue && j < e);
        tk = rank < need;
      }
      const unsigned bt = __ballot_sync(0xffffffffu, tk);
      if (tk) T[nT + __popc(bt & ((1u << lane) - 1))] = pp[e];
      nT += __popc(bt);
    }
    __syncwarp();
    auto below = [&](int v) {
      int lo = 0, hi = nT;
      while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (T[mid] < v) lo = mid + 1;
        else hi = mid;
      }
      return lo;
    };
    const int slast = nd > 0 ? below(out[nd - 1]) : 0;
    for (int j = slast + lane; j < nT; j += 32) out[j + nd] = T[j];
    for (int c0 = (nd - 1) & ~31; c0 >= 0; c0 -= 32) {
      const int i = c0 + lane;
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
    for (int i = lane; i < kWindow; i += 32) out[kIdxTop + i] = p - kWindow + 1 + i;
    return;
  }
  for (int i = lane; i < (WIDE ? 1024 : 512); i += 32) h[i] = 0;
  __syncwarp();
  {
    int jc = 0;
    for (int i0 = 0; i0 < S.vtot; i0 += step) {
      uint32_t u[U][4];
      int ps[U][4];
      vload<U, false>(S, i0, jc, u, ps);
#pragma unroll
      for (int k = 0; k < U; k++)
#pragma unroll
        for (int j = 0; j < 4; j++)
          if ((u[k][j] >> 10) == pre22) {
            if constexpr (WIDE) atomicAdd(h + (u[k][j] & 0x3FF), 1u);
            else add16(h, u[k][j] & 0x3FF);
          }
    }
  }
  __syncwarp();
  const uint32_t b3 = warp_find(1024, need, [&](int b) { return WIDE ? int(h[b]) : cnt16(h, b); });
  const uint32_t thr = (pre22 << 10) | b3;
  const int need_eq = need;
  int base = 0, eqbase = 0, jc = 0;
  for (int i0 = 0; i0 < S.vtot; i0 += step) {
    uint32_t u[U][4];
    int ps[U][4];
    vload<U, true>(S, i0, jc, u, ps);
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
      int e = eqbase + er[k];
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
      int pos = base + pre[k];
#pragma unroll
      for (int j = 0; j < 4; j++)
        if (take[k][j]) out[pos++] = ps[k][j];
      base += ttot[k];
    }
  }
  for (int i = lane; i < kWindow; i += 32) out[kIdxTop + i] = p - kWindow + 1 + i;
}

template <int U>
__device__ __noinline__ void select_rest_wide(const uint32_t* lg, const uint16_t* offs, uint32_t tab, int kr, int nspl,
                                              int lastlen, int vtot, int p, uint32_t b1, int need, uint32_t* h,
                                              int* out) {
  select_rest<U, true>(Stream{lg, offs, tab, kr, nspl, lastlen, vtot}, p, b1, need, h, out);
}
template <int U>
__device__ __noinline__ void select_rest_scr(const uint32_t* key, const int* pos, int n, int p, uint32_t b1, int need,
                                             uint32_t* h, int* out) {
  select_rest<U, false>(Scratch{key, pos, n}, p, b1, need, h, out);
}

// level 1 of the selection from the row's global counts (warp_find's layout and result): the bin b1 where the count
// from the top reaches kIdxTop, need = how many of its members are taken, c1 = its count
__device__ __forceinline__ uint32_t level1(const int* hg, int& need, int& c1) {
  const int lane = threadIdx.x & 31, lo = kBins - 64 * (lane + 1);
  int c[64];
#pragma unroll
  for (int k = 0; k < 16; k++) {
    const int4 v = __ldcg(reinterpret_cast<const int4*>(hg + lo) + k);
    c[4 * k] = v.x, c[4 * k + 1] = v.y, c[4 * k + 2] = v.z, c[4 * k + 3] = v.w;
  }
  int s = 0;
#pragma unroll
  for (int k = 0; k < 64; k++) s += c[k];
  int x = s;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= o) x += y;
  }
  int found = -1, rem = 0, cb = 0;
  if (x - s < kIdxTop && x >= kIdxTop) {
    int acc = x - s;
#pragma unroll
    for (int k = 63; k >= 0; k--) {
      if (found < 0 && acc + c[k] >= kIdxTop) found = lo + k, rem = kIdxTop - acc, cb = c[k];
      acc += c[k];
    }
  }
  const unsigned b = __ballot_sync(0xffffffffu, found >= 0);
  const int src = __ffs(b) - 1;
  need = __shfl_sync(0xffffffffu, rem, src), c1 = __shfl_sync(0xffffffffu, cb, src);
  return uint32_t(__shfl_sync(0xffffffffu, found, src));
}

// one warp selects its row: the split table (shared, tab), level 1 from the row's counts hg; then — when the keys of
// bin ≥ b1 fit the row's scratch (scap) — one pass over the stream collecting them in position order and select_rest
// over that set, else select_rest over the stream (two or three passes)
template <int U>
__device__ __noinline__ void select_row(const uint32_t* lg, const uint16_t* offs, const uint32_t* hdr, int ncand,
                                        int p, int kr, const int* hg, uint32_t* tab, uint32_t* h, int* out,
                                        uint32_t* scr, int scap) {
  const int lane = threadIdx.x & 31;
  const int nspl = (ncand + kr - 1) / kr, lastlen = ncand - (nspl - 1) * kr;
  int vb = 0;
  for (int j0 = 0; j0 < nspl; j0 += 32) {
    const int j = j0 + lane;
    int tot = 0;
    uint32_t w = 0;
    if (j < nspl) {
      const uint32_t hd = __ldcg(hdr + j);
      const int nc = int(hd & 0xFFFF), len = j == nspl - 1 ? lastlen : kr, kb = min(int(hd >> 16), len);
      w = uint32_t(nc) | (uint32_t(kb) << 16);
      tot = ((nc + 3) & ~3) + ((len - kb + 3) & ~3);
    }
    int x = tot;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int y = __shfl_up_sync(0xffffffffu, x, o);
      if (lane >= o) x += y;
    }
    if (j < nspl) tab[2 * j] = uint32_t(vb + x - tot), tab[2 * j + 1] = w;
    vb += __shfl_sync(0xffffffffu, x, 31);
  }
  if (lane == 0) tab[2 * nspl] = uint32_t(vb), tab[2 * nspl + 1] = 0;
  __syncwarp();
  const uint32_t ts = uint32_t(__cvta_generic_to_shared(tab));
  const Stream S{lg, offs, ts, kr, nspl, lastlen, vb};
  int need, c1;
  const uint32_t b1 = level1(hg, need, c1);
  const int E = kIdxTop - need + c1;  // the row's keys of bin ≥ b1 (all kept)
  if (E <= scap) {
    uint32_t* sk = scr;
    int* sp = reinterpret_cast<int*>(scr + scap);
    int ne = 0, jc = 0;
    for (int i0 = 0; i0 < S.vtot; i0 += 4 * U * 32) {
      uint32_t u[U][4];
      int ps[U][4];
      vload<U, true>(S, i0, jc, u, ps);
      int nh[U];
      bool any = false;
#pragma unroll
      for (int k = 0; k < U; k++) {
        nh[k] = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) nh[k] += (u[k][j] >> 21) >= b1;
        any |= nh[k] > 0;
      }
      if (!__any_sync(0xffffffffu, any)) continue;
      int pre[U], tot[U];
      scan_cnt<U>(nh, pre, tot);
#pragma unroll
      for (int k = 0; k < U; k++) {
        int e = ne + pre[k];
#pragma unroll
        for (int j = 0; j < 4; j++)
          if ((u[k][j] >> 21) >= b1) sk[e] = u[k][j], sp[e++] = ps[k][j];
        ne += tot[k];
      }
    }
    __syncwarp();
    select_rest_scr<U>(sk, sp, ne, p, b1, need, h, out);
  } else if (c1 <= 0xFFFF) {
    select_rest<U, false>(S, p, b1, need, h, out);
  } else {
    select_rest_wide<U>(lg, offs, ts, kr, nspl, lastlen, vb, p, b1, need, h, out);
  }
}

// grid: ⌈n / 8⌉ row groups × ks_n key splits, as super-groups of sg row groups (the last one may be smaller), splits
// major inside each (block b: super-group b / (sg·ks_n), then split, then group). Warp w: row r0 + 8·grp + w over the
// CTA's keys. lg [n][ldl] (kept order keys); offs [n][ks_n][kC]; hdr [n][ks_n] (nc | kb << 16); hist [n][2048], bnd
// [n] (zero on entry, zeroed again by the selecting CTA); ctr per group (zero, reset).
__global__ void __launch_bounds__(256, 2)
    k_select(const uint8_t* __restrict__ qf4, const uint8_t* __restrict__ qs, const float* __restrict__ wts,
             const RowMeta* __restrict__ meta, DsaCache c, int row0, int n, int ks_n, int kr, int ldl, int sg,
             uint32_t* lg, uint16_t* offs, uint32_t* hdr, int* hist, int* bnd, int* ctr, int* list, int* cnt,
             uint32_t* scr, int scap) {
  extern __shared__ __align__(16) uint8_t smraw[];
  Smem& S = *reinterpret_cast<Smem*>(smraw);
  constexpr int NT = 32 * RPC;
  const int ngrp = (n + RPC - 1) / RPC;
  const int sgi = blockIdx.x / (sg * ks_n), g0 = sgi * sg, sgn = min(sg, ngrp - g0);
  const int bi = blockIdx.x - sgi * sg * ks_n;
  const int ks = bi / sgn, grp = g0 + bi % sgn;
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
  // ---- the row's query as B fragments (P091)
  uint32_t QB[8][2][2], SB[8];
  float WA[8], WB[8];
  int rq = 0;
  const uint8_t* qrow = qf4 + (size_t)r * 64 * 64;
  uint32_t blo = work ? uint32_t(__ldcg(bnd + ri)) : 0u;  // this row's exact lower bound bin
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
  uint32_t* R = lg + (size_t)ri * ldl + (size_t)ks * kr;     // this (row, split)'s region
  uint16_t* O = offs + ((size_t)ri * ks_n + ks) * kC;
  int nsv = 0, kbd = kr;  // compact keys kept; dense from split offset kbd (kr: none)
  const int lrow16 = (lane & 7) + 8 * ((lane >> 3) & 1), lcol = 16 * (lane >> 4);
  const bool odd = t & 1;
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
    auto issue = [&](int st) {
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
      if ((st & 7) == 7) blo = max(blo, uint32_t(__ldcg(bnd + ri)));  // finished CTAs' bound updates
      uint32_t SAv[SK / 16];
      int rk = 0;
#pragma unroll
      for (int blk = 0; blk < SK / 16; blk++) {
        SAv[blk] = *reinterpret_cast<const uint32_t*>(ring + (16 * blk + g + 8 * (t & 1)) * KS + 64);
        rk = max(rk, exp_spread(SAv[blk]));
      }
      if (__any_sync(0xffffffffu, rq + rk > 9)) {
        // ---- P091's k32 chain (a wide exponent spread somewhere in the stage), groups of 8 keys
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
          // lanes < 8: key n0 + 2t + jk
          const int kl = n0 - kc0 + 2 * t + jk;
          const bool valid = lane < 8 && n0 + 2 * t + jk < ncand;
          const uint32_t u = okey(((r3 + x1) + x2) + x3);
          const bool sv = valid && (u >> 21) >= blo;
          const uint32_t bal = __ballot_sync(0xffffffffu, sv);
          if (bal) {  // kept keys in the group (warp-uniform)
            if (sv) add16(hrow, u >> 21);
            const uint32_t m8 = key_order8(bal);
            const int s = __popc(m8);
            if (kbd == kr && nsv + s > kC) kbd = n0 - kc0;
            if (kbd == kr) {
              if (sv) {
                const int e = nsv + __popc(m8 & ((1u << (2 * t + jk)) - 1));
                R[e] = u, O[e] = uint16_t(kl);
              }
              nsv += s;
            }
          }
          if (kbd != kr && valid) R[kl] = u;
        }
        continue;
      }
      // ---- k64, keys × heads (P091's sequence)
      const bool full = kst + SK <= ncand;  // the whole stage is the row's (warp-uniform)
      float acc[2][2][4];
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
          float v[2][2];
#pragma unroll
          for (int hf = 0; hf < 2; hf++) {
            const float w0 = WA[2 * mm + hf], w1 = WB[2 * mm + hf];
            v[hf][0] = fmaxf(a[hf][0], 0.f) * w0 + fmaxf(a[hf][1], 0.f) * w1;
            v[hf][1] = fmaxf(a[hf][2], 0.f) * w0 + fmaxf(a[hf][3], 0.f) * w1;
          }
          const float u0 = v[0][0] + v[1][0], u1 = v[0][1] + v[1][1];
          const float snd = odd ? u0 : u1;
          const float r2 = (odd ? u1 : u0) + __shfl_xor_sync(0xffffffffu, snd, 1);
          const float tm = r2 + __shfl_xor_sync(0xffffffffu, r2, 2);
          sblk = mm == 0 ? tm : sblk + tm;
          if (mm == 3) {
            // lanes t < 2: key kst + 16·blk + g + 8t
            const int kl = kst - kc0 + 16 * blk + g + 8 * t;
            const bool valid = t < 2 && (full || kc0 + kl < ncand);
            const uint32_t u = okey(sblk);
            const bool sv = valid && (u >> 21) >= blo;
            const uint32_t bal = __ballot_sync(0xffffffffu, sv);
            if (bal) {  // kept keys in the block (warp-uniform)
              if (sv) add16(hrow, u >> 21);
              const uint32_t m16 = key_order16(bal);
              const int s = __popc(m16);
              if (kbd == kr && nsv + s > kC) kbd = kst - kc0 + 16 * blk;
              if (kbd == kr) {
                if (sv) {
                  const int e = nsv + __popc(m16 & ((1u << (g + 8 * t)) - 1));
                  R[e] = u, O[e] = uint16_t(kl);
                }
                nsv += s;
              }
            }
            if (kbd != kr && valid) R[kl] = u;
          }
        }
      }
    }
    cp_async_wait<0>();
    __syncthreads();
  }
  // ---- counts → the rows' global counts, the split's header, the row's bound; the group's last CTA selects
  for (int i = tid; i < RPC * kBins / 2; i += NT) {
    const uint32_t v = (&S.hist[0][0])[i];
    if (v) {
      int* hg = hist + (size_t)(grp * RPC) * kBins + 2 * i;
      if (v & 0xFFFF) atomicAdd(hg, int(v & 0xFFFF));
      if (v >> 16) atomicAdd(hg + 1, int(v >> 16));
    }
  }
  if (work && lane == 0) hdr[(size_t)ri * ks_n + ks] = uint32_t(nsv) | (uint32_t(kbd) << 16);
  __threadfence();
  __syncthreads();
  if (work && ks + 1 < ks_n) {
    const int b = row_bound(hist + (size_t)ri * kBins);
    if (lane == 0 && uint32_t(b) > blo) atomicMax(bnd + ri, b);
  }
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
    select_row<8>(lg + (size_t)ri * ldl, offs + (size_t)ri * ks_n * kC, hdr + (size_t)ri * ks_n, ncand, mt.pos, kr,
                  hg, S.tab[warp], hrow, out, scr + (size_t)ri * 2 * scap, scap);
    if (lane == 0) cnt[r] = kIdxTop + kWindow, bnd[ri] = 0;
    for (int i = lane; i < kBins; i += 32) hg[i] = 0;
  }
}

// false: not handled here (M < 32, no row of kMinCand candidates)
inline bool run(const uint8_t* qf4, const uint8_t* qs, const float* wts, const RowMeta* meta, DsaCache c, int M,
                int max_pos, int* list, int* cnt, float* ws, size_t ws_bytes, cudaStream_t s) {
  const int maxc = max_pos - kWindow + 1;
  if (M < 32 || maxc < kMinCand) return false;
  const int ldl = std::max(64, maxc + 63) / 64 * 64;
  int pass = int(std::min<size_t>(M, ws_bytes / ((size_t)ldl * 4)));
  if (pass < 1) qlog::fail("dsa", "indexer workspace too small");
  const int npass = (M + pass - 1) / pass;
  pass = (M + npass - 1) / npass;  // balanced passes (P091's row count a launch at most)
  // key splits: ≈ 4096 keys (P091's), at most kMaxSplits, multiples of the 64-key stage
  const int ks0 = std::min((maxc + 4095) / 4096, kMaxSplits);
  const int kr = ((maxc + ks0 - 1) / ks0 + SK - 1) / SK * SK;
  const int ks_n = (maxc + kr - 1) / kr;
  int *hist, *ctr, *bnd;
  p091::bufs(pass, hist, ctr, bnd);  // P091's per-row counts and per-group counters (zero, kept zero), and bounds
  static uint8_t* aux = nullptr;  // [pass][ks_n][kC] uint16 offsets, then [pass][ks_n] headers
  static size_t na = 0;
  if ((size_t)pass * ks_n > na) {
    if (aux) CUDA_CHECK(cudaFree(aux));
    na = (size_t)pass * ks_n;
    CUDA_CHECK(cudaMalloc(&aux, na * (kC * 2 + 4)));
  }
  static bool attr = false;
  if (!attr) {
    CUDA_CHECK(cudaFuncSetAttribute(k_select, cudaFuncAttributeMaxDynamicSharedMemorySize, int(sizeof(Smem))));
    attr = true;
  }
  uint16_t* offs = reinterpret_cast<uint16_t*>(aux);
  uint32_t* hdr = reinterpret_cast<uint32_t*>(aux + na * kC * 2);
  // the selection's scratch: the workspace's bytes past the pass's rows, ≤ 16384 keys a row (0: none — the stream)
  const size_t spare = ws_bytes / 4 - (size_t)pass * ldl;
  int scap = int(std::min<size_t>(16384, spare / (2 * (size_t)pass)) / 4 * 4);
  if (scap < kIdxTop + 256) scap = 0;
  uint32_t* scr = reinterpret_cast<uint32_t*>(ws) + (size_t)pass * ldl;
  for (int r0 = 0; r0 < M; r0 += pass) {
    const int n = std::min(pass, M - r0), ngrp = (n + RPC - 1) / RPC;
    const int nsg = (ngrp + 31) / 32, sg = (ngrp + nsg - 1) / nsg;  // super-groups: equal parts of ≤ 32 row groups
    k_select<<<ngrp * ks_n, 32 * RPC, sizeof(Smem), s>>>(qf4, qs, wts, meta, c, r0, n, ks_n, kr, ldl, sg,
                                                        reinterpret_cast<uint32_t*>(ws), offs, hdr, hist, bnd, ctr,
                                                        list, cnt, scr, scap);
    CUDA_CHECK(cudaGetLastError());
  }
  return true;
}

}  // namespace p153
