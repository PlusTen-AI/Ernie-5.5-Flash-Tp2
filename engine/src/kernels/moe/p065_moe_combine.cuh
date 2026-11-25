// p065_moe_combine.cuh — P065 moe_combine: the routed experts of a prefill call (M > 64 rows) without the per-pair
// fp32 y round trip and with the routing plan in one launch; included by moe.cu after P037 (its layouts and its up
// kernel k_mm37<true> unchanged).
//   plan : k_plan65a / b — two small launches build what k_plan ×2 + k_scan + k_tiles37 built (cnt / off / act / nact /
//          prow / pidx, an expert's pairs in (row, slot) order, and P037's tile list), plus pair → row·10 + slot.
//   down : k_dn65 — P037's down tile (256 latent rows × ≤ 64 pairs of one expert, the same MMA chain per output) with
//          the combine in its epilogue. part[row] = Σ_slot w·y in slot order is a chain per (row, 64 latent dims):
//          a pair's y is applied by the lane that finds the chain at its slot (the running fp32 sum kept in an
//          L2-resident buffer), otherwise parked (written to the base's y buffer) and applied by whichever owner
//          reaches that slot — lock-free (one atomic state word per chain: next slot ‖ parked mask), nobody waits;
//          the 10th application writes the output. CTAs take the tiles latent-block-major (2 256-row blocks a
//          pass, an expert's tiles in id order inside a pass), so a pass's running sums stay in L2.
// Bitwise the base: every y is P037's MMA chain; every row's sum is the base k_combine's FFMA chain
// a = fma(w0, y0, 0), a = fma(wk, yk, a) for k = 1 … 9 in slot order, rounded once (bf16) at the end.
namespace p065 {

using p037::kNP;
using p037::kSmem;
constexpr int kYS = 68;  // staged y row stride (floats): conflict-free stores from the MMA layout, aligned float4 reads

__device__ __forceinline__ uint32_t ld_acq(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint32_t atom_or(uint32_t* p, uint32_t v) {
  uint32_t o;
  asm volatile("atom.acq_rel.gpu.global.or.b32 %0, [%1], %2;" : "=r"(o) : "l"(p), "r"(v) : "memory");
  return o;
}
__device__ __forceinline__ uint32_t atom_cas(uint32_t* p, uint32_t c, uint32_t v) {
  uint32_t o;
  asm volatile("atom.acq_rel.gpu.global.cas.b32 %0, [%1], %2, %3;" : "=r"(o) : "l"(p), "r"(c), "r"(v) : "memory");
  return o;
}
__device__ __forceinline__ void st_rlx(uint32_t* p, uint32_t v) {
  asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
// L2 eviction policy for the running sums (keep them while the weights stream through)
__device__ __forceinline__ uint64_t pol_last() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ float4 ldcg4p(const float* p, uint64_t pol) {
  float4 v;
  asm volatile("ld.global.cg.L2::cache_hint.v4.f32 {%0,%1,%2,%3}, [%4], %5;"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p), "l"(pol));
  return v;
}
__device__ __forceinline__ void stcg4p(float* p, float4 v, uint64_t pol) {
  asm volatile("st.global.cg.L2::cache_hint.v4.f32 [%0], {%1,%2,%3,%4}, %5;" ::"l"(p), "f"(v.x), "f"(v.y), "f"(v.z),
               "f"(v.w), "l"(pol) : "memory");
}
__device__ __forceinline__ float4 ldcg4(const float* p) {
  float4 v;
  asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ void stcg4(float* p, float4 v) {
  asm volatile("st.global.cg.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}

// ------------------------------------------------------------------------------------------------------------ plan
// Two small launches over kPC chunks of the P ids (pair i = row·10 + slot). a: each CTA counts its chunk per expert.
// b: each CTA derives its chunk's base inside every expert (the counts of the chunks before it) and the expert
// offsets (all chunks' counts, exclusive scan), then places its chunk — warp w a sub-chunk, h[w][e] its count / base —
// so an expert's pairs come out in pair-index ((row, slot)) order, exactly the base's plan. CTA 0 also writes cnt /
// off / act / nact and P037's tile list.
constexpr int kPC = 64;

__global__ void __launch_bounds__(256) k_plan65a(const int* __restrict__ ids, int P, int* hist) {
  __shared__ int c[kExperts];
  for (int e = threadIdx.x; e < kExperts; e += 256) c[e] = 0;
  __syncthreads();
  const int chunk = (P + kPC - 1) / kPC, b0 = blockIdx.x * chunk, b1 = min(P, b0 + chunk);
  for (int i = b0 + threadIdx.x; i < b1; i += 256) atomicAdd(&c[ids[i]], 1);
  __syncthreads();
  for (int e = threadIdx.x; e < kExperts; e += 256) hist[blockIdx.x * kExperts + e] = c[e];
}

__global__ void __launch_bounds__(kExperts) k_plan65b(const int* __restrict__ ids, int P, const int* __restrict__ hist,
                                                      int* cnt, int* off, int* act, int* nact, int* prow, int* pidx,
                                                      int* pfl, int2* tiles, int* ntiles) {
  constexpr int NW = kExperts / 32;  // 16 warps
  __shared__ uint16_t h[NW][kExperts];  // 16 KB: a sub-chunk's counts, then its bases inside the chunk's
  __shared__ int s_off[kExperts], s_wc[NW], s_wn[NW], s_wa[NW];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, cb = blockIdx.x;
  // this chunk's base inside expert tid, the expert's total
  int base = 0, c = 0;
  for (int k = 0; k < kPC; k++) {
    const int v = hist[k * kExperts + tid];
    base += k < cb ? v : 0, c += v;
  }
  const int nt = (c + kNP - 1) / kNP, ia0 = c > 0;
  int ic = c, in = nt, ia = ia0;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int x = __shfl_up_sync(0xffffffffu, ic, o), y = __shfl_up_sync(0xffffffffu, in, o),
              z = __shfl_up_sync(0xffffffffu, ia, o);
    if (lane >= o) ic += x, in += y, ia += z;
  }
  if (lane == 31) s_wc[warp] = ic, s_wn[warp] = in, s_wa[warp] = ia;
  for (int i = tid; i < NW * kExperts / 2; i += kExperts) reinterpret_cast<uint32_t*>(&h[0][0])[i] = 0;
  __syncthreads();
  {
    int bc = 0, bn = 0, ba = 0;
    for (int w = 0; w < warp; w++) bc += s_wc[w], bn += s_wn[w], ba += s_wa[w];
    const int oc = bc + ic - c, on = bn + in - nt, oa = ba + ia - ia0;
    s_off[tid] = oc + base;
    if (cb == 0) {
      cnt[tid] = c, off[tid] = oc;
      if (ia0) act[oa] = tid;
      for (int j = 0; j < nt; j++) tiles[on + j] = make_int2(tid, j * kNP);
      if (tid == kExperts - 1) off[kExperts] = oc + c, *ntiles = on + nt, *nact = oa + ia0;
    }
  }
  // place the chunk: warp w owns the sub-chunk [s0, s1)
  const int chunk = (P + kPC - 1) / kPC, b0 = cb * chunk, b1 = min(P, b0 + chunk);
  const int sub = (b1 - b0 + NW - 1) / NW, s0 = min(b1, b0 + warp * sub), s1 = min(b1, s0 + sub);
  constexpr int U = 4;
  auto fetch = [&](int i0, int (&v)[U]) {
#pragma unroll
    for (int u = 0; u < U; u++) {
      const int i = i0 + 32 * u + lane;
      v[u] = i < s1 ? ids[i] : -1 - lane;  // idle lanes: distinct negatives, alone in their match group
    }
  };
  for (int i0 = s0; i0 < s1; i0 += 32 * U) {
    int v[U];
    fetch(i0, v);
#pragma unroll
    for (int u = 0; u < U; u++) {
      const unsigned mask = __match_any_sync(0xffffffffu, v[u]);
      if (v[u] >= 0 && lane == __ffs(mask) - 1) h[warp][v[u]] += __popc(mask);
      __syncwarp();
    }
  }
  __syncthreads();
  {
    int run = s_off[tid];
    for (int w = 0; w < NW; w++) {
      const int x = h[w][tid];
      h[w][tid] = uint16_t(run - s_off[tid]);  // base inside this chunk's part of the expert (≤ chunk)
      run += x;
    }
  }
  __syncthreads();
  for (int i0 = s0; i0 < s1; i0 += 32 * U) {
    int v[U];
    fetch(i0, v);
#pragma unroll
    for (int u = 0; u < U; u++) {
      const int i = i0 + 32 * u + lane;
      const unsigned mask = __match_any_sync(0xffffffffu, v[u]);
      if (v[u] >= 0) {
        const int p = s_off[v[u]] + h[warp][v[u]] + __popc(mask & ((1u << lane) - 1u));
        prow[p] = i / kTopK, pfl[p] = i, pidx[i] = p;
      }
      __syncwarp();
      if (v[u] >= 0 && lane == __ffs(mask) - 1) h[warp][v[u]] += __popc(mask);
      __syncwarp();
    }
  }
}

// ------------------------------------------------------------------------------------------------------------ chain
// One lane, pair p (y: its 64 values of latent dims col … col+63, in shared memory). Chain state (row, by, rq):
// bits 0–3 the next slot to apply, bit 8 + j slot j parked. Applying = a = fma(w, y, a) into the running sum.
__device__ __forceinline__ void put8(float* o, const float* a) {
  *reinterpret_cast<float4*>(o) = make_float4(a[0], a[1], a[2], a[3]);
  *reinterpret_cast<float4*>(o + 4) = make_float4(a[4], a[5], a[6], a[7]);
}
__device__ __forceinline__ void put8(bf16* o, const float* a) {
  *reinterpret_cast<uint4*>(o) =
      make_uint4(pack_bf16x2(a[0], a[1]), pack_bf16x2(a[2], a[3]), pack_bf16x2(a[4], a[5]), pack_bf16x2(a[6], a[7]));
}

template <class OutT>
__device__ __forceinline__ void chain(int p, int col, uint32_t* sp, const float* y, const int* __restrict__ pfl,
                                      const int* __restrict__ pidx, const float* __restrict__ wts, float* run,
                                      float* park, OutT* part, int ld) {
  const int fl = pfl[p], row = fl / kTopK, k = fl % kTopK;
  uint32_t s = ld_acq(sp);
  bool own = (s & 15u) == uint32_t(k);
  if (!own) {  // park, then tell the chain; if it reached k meanwhile (its owner let go at k), it is ours after all
    float* pk = park + (size_t)p * kLatent + col;
#pragma unroll
    for (int m = 0; m < 16; m++) stcg4(pk + 4 * m, *reinterpret_cast<const float4*>(y + 4 * m));
    own = (atom_or(sp, 1u << (8 + k)) & 15u) == uint32_t(k);
  }
  if (!own) return;
  const uint64_t pl = pol_last();
  float a[64];
  float* rp = run + (size_t)row * kLatent + col;
  {
    const float wk = wts[fl];
    if (k == 0) {
#pragma unroll
      for (int m = 0; m < 64; m++) a[m] = __fmaf_rn(wk, y[m], 0.f);
    } else {
#pragma unroll
      for (int m = 0; m < 16; m++) {
        const float4 v = ldcg4p(rp + 4 * m, pl);
        a[4 * m] = __fmaf_rn(wk, y[4 * m], v.x), a[4 * m + 1] = __fmaf_rn(wk, y[4 * m + 1], v.y);
        a[4 * m + 2] = __fmaf_rn(wk, y[4 * m + 2], v.z), a[4 * m + 3] = __fmaf_rn(wk, y[4 * m + 3], v.w);
      }
    }
  }
  int j = k + 1;
  for (;;) {
    s = ld_acq(sp);
    while (j < kTopK && (s >> (8 + j) & 1u)) {  // parked successors, in slot order
      const float wj = wts[row * kTopK + j];
      const float* pk = park + (size_t)pidx[row * kTopK + j] * kLatent + col;
#pragma unroll
      for (int m = 0; m < 16; m++) {
        const float4 v = ldcg4(pk + 4 * m);
        a[4 * m] = __fmaf_rn(wj, v.x, a[4 * m]), a[4 * m + 1] = __fmaf_rn(wj, v.y, a[4 * m + 1]);
        a[4 * m + 2] = __fmaf_rn(wj, v.z, a[4 * m + 2]), a[4 * m + 3] = __fmaf_rn(wj, v.w, a[4 * m + 3]);
      }
      j++;
    }
    if (j == kTopK) {  // the row's sum is complete: write it, reset the chain for the next call
      OutT* o = part + (size_t)row * ld + col;
#pragma unroll
      for (int m = 0; m < 8; m++) put8(o + 8 * m, a + 8 * m);
      st_rlx(sp, 0u);
      return;
    }
#pragma unroll
    for (int m = 0; m < 16; m++) stcg4p(rp + 4 * m, make_float4(a[4 * m], a[4 * m + 1], a[4 * m + 2], a[4 * m + 3]), pl);
    if (atom_cas(sp, s, (s & ~15u) | uint32_t(j)) == s) return;  // let go at j (fails only if a slot got parked)
  }
}

// ------------------------------------------------------------------------------------------------------------ down
// grid: tiles × 8 latent blocks, CTA t → pass t / (tiles·BPP), inside it tile-major (expert id order), BPP blocks.
template <int BPP, class OutT>
__global__ void __launch_bounds__(256, 2) k_dn65(const uint2* __restrict__ W, const uint8_t* __restrict__ WS,
                                                 const uint8_t* __restrict__ xq, const uint8_t* __restrict__ xs,
                                                 const int* __restrict__ off, const int2* __restrict__ tiles,
                                                 const int* __restrict__ ntiles, const int* __restrict__ pfl,
                                                 const int* __restrict__ pidx, const float* __restrict__ wts,
                                                 float* run, float* park, uint32_t* cst, OutT* part, int ld) {
  constexpr int kKC = p037::kKC, kST = p037::kST, kXS = p037::kXS, kSS = p037::kSS, kWB = p037::kWB,
                kWSB = p037::kWSB, kStageB = p037::kStageB;
  constexpr int KB = kDnKB, XROW = kInterR, SROW = XROW / 32, NS = KB / kKC;
  const int ntl = *ntiles, t = blockIdx.x;
  if (t >= ntl * 8) return;
  const int per = ntl * BPP, pass = t / per, rr = t - pass * per;
  const int2 tile = tiles[rr / BPP];
  const int by = pass * BPP + rr % BPP;
  const int e = tile.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tq = lane & 3;
  const int p0 = off[e], np = off[e + 1] - p0;
  const int rq = warp >> 1, ph = warp & 1;
  extern __shared__ __align__(16) uint8_t smem[];
  uint8_t* sxs = smem + kST * kStageB;
  const uint2* wbase = W + (size_t)e * kDnWords + (size_t)(16 * by) * KB * 32;
  const uint8_t* wsbase = WS + (size_t)e * kDnScales + (size_t)(16 * by) * KB * 16;
  const int q0 = tile.y, nq = min(kNP, np - q0);
  const int my_q = tid >> 2, my_c = tid & 3;
  const bool my_ok = my_q < nq;
  const uint8_t* my_x = xq + (size_t)(my_ok ? p0 + q0 + my_q : 0) * XROW;
  auto load = [&](int st, int buf) {
    uint8_t* sw = smem + buf * kStageB;
    const int kb0 = st * kKC;
#pragma unroll
    for (int i = tid; i < kWB / 16; i += 256) {
      const int tl = i / (kKC * 16), pc = i % (kKC * 16);
      cp_async16(sw + i * 16, reinterpret_cast<const uint8_t*>(wbase + ((size_t)tl * KB + kb0) * 32) + pc * 16, true);
    }
    if (tid < kWSB / 16) {
      const int tl = tid / kKC, pc = tid % kKC;
      cp_async16(sw + kWB + tid * 16, wsbase + ((size_t)tl * KB + kb0 + pc) * 16, true);
    }
    if (kKC * 32 / 16 == 4 || my_c < kKC * 2)
      cp_async16(sw + kWB + kWSB + my_q * kXS + my_c * 16, my_x + kb0 * 32 + my_c * 16, my_ok);
  };
  for (int i = tid; i < kNP * (SROW / 4); i += 256) {
    const int q = i / (SROW / 4), c4 = i % (SROW / 4);
    const bool ok = q < nq;
    p037::cp_async4(sxs + q * kSS + c4 * 4, xs + (size_t)(ok ? p0 + q0 + q : 0) * SROW + c4 * 4, ok);
  }
#pragma unroll
  for (int s = 0; s < kST - 1; s++) {
    load(s, s);
    cp_async_commit();
  }
  float acc[4][4][4];
#pragma unroll
  for (int a = 0; a < 4; a++)
#pragma unroll
    for (int b = 0; b < 4; b++) acc[a][b][0] = acc[a][b][1] = acc[a][b][2] = acc[a][b][3] = 0.f;
  const int ntv = min(4, max(0, (nq - 32 * ph + 7) / 8));
#pragma unroll 1
  for (int st = 0; st < NS; st++) {
    cp_async_wait<kST - 2>();
    __syncthreads();
    if (st + kST - 1 < NS) load(st + kST - 1, (st + kST - 1) % kST);
    cp_async_commit();
    const uint8_t* sw = smem + (st % kST) * kStageB;
    const uint8_t* sx = sw + kWB + kWSB;
#pragma unroll
    for (int kbl = 0; kbl < kKC; kbl++) {
      const int kb = st * kKC + kbl;
      uint32_t A[4][4], sa[4];
#pragma unroll
      for (int mt = 0; mt < 4; mt++) {
        const int tl = 4 * rq + mt;
        const uint2 w = *reinterpret_cast<const uint2*>(sw + ((tl * kKC + kbl) * 32 + lane) * 8);
        A[mt][0] = e2m1x4_expand(w.x & 0xFFFF), A[mt][1] = e2m1x4_expand(w.x >> 16);
        A[mt][2] = e2m1x4_expand(w.y & 0xFFFF), A[mt][3] = e2m1x4_expand(w.y >> 16);
        sa[mt] = sw[kWB + (tl * kKC + kbl) * 16 + g + 8 * (lane & 1)];
      }
#pragma unroll
      for (int nt = 0; nt < 4; nt++) {
        if (nt < ntv) {
          const int q = 32 * ph + 8 * nt + g;
          const uint8_t* p = sx + q * kXS + kbl * 32 + 4 * tq;
          const uint32_t b0 = *reinterpret_cast<const uint32_t*>(p), b1 = *reinterpret_cast<const uint32_t*>(p + 16);
          const uint32_t sb = sxs[q * kSS + kb];
#pragma unroll
          for (int mt = 0; mt < 4; mt++)
            mma_e2m1_e4m3(acc[mt][nt], A[mt][0], A[mt][1], A[mt][2], A[mt][3], b0, b1, sa[mt], sb);
        }
      }
    }
  }
  cp_async_wait<0>();
  __syncthreads();  // the stages are free: stage y per warp (16 pairs × 64 dims at a time) and run the chains
  float* ysw = reinterpret_cast<float*>(smem) + warp * 16 * kYS;
  const int col = 256 * by + 64 * rq;
#pragma unroll
  for (int hh = 0; hh < 2; hh++) {
    __syncwarp();
#pragma unroll
    for (int n2 = 0; n2 < 2; n2++)
#pragma unroll
      for (int mt = 0; mt < 4; mt++)
#pragma unroll
        for (int i = 0; i < 4; i++)
          ysw[(8 * n2 + 2 * tq + (i & 1)) * kYS + 16 * mt + g + (i < 2 ? 0 : 8)] = acc[mt][2 * hh + n2][i];
    __syncwarp();
    const int q = 32 * ph + 16 * hh + lane;
    if (lane < 16 && q < nq) {
      const int p = p0 + q0 + q;
      const int row = pfl[p] / kTopK;
      chain<OutT>(p, col, cst + ((size_t)row * 8 + by) * 4 + rq, ysw + lane * kYS, pfl, pidx, wts, run, park, part,
                  ld);
    }
  }
}

template <class OutT>
void run(const MoeExperts* E, MoeScratch* S, const uint8_t* xq, const uint8_t* xs, const int* ids, const float* w,
         int M, OutT* part, int ld, cudaStream_t s) {
  constexpr int BPP = 2;  // 256-row latent blocks a pass: 4 passes (h read 4×), running sums of a pass 8.4 MB at 4096 rows
  static bool attr = [] {
    CUDA_CHECK(cudaFuncSetAttribute(p037::k_mm37<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
    CUDA_CHECK(cudaFuncSetAttribute(k_dn65<BPP, OutT>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
    return true;
  }();
  (void)attr;
  const int P = M * kTopK;
  const int bound = cdiv(P, kNP) + std::min(P, kExperts);
  k_plan65a<<<kPC, 256, 0, s>>>(ids, P, S->phist);
  k_plan65b<<<kPC, kExperts, 0, s>>>(ids, P, S->phist, S->cnt, S->off, S->act, S->nact, S->prow, S->pidx, S->pfl,
                                     S->tiles, S->ntiles);
  CUDA_CHECK(cudaGetLastError());
  p037::k_mm37<true><<<dim3(8, bound), 256, kSmem, s>>>(E->up, E->ups, xq, xs, S->off, S->tiles, S->ntiles, S->prow,
                                                         S->hq, S->hs, nullptr);
  CUDA_CHECK(cudaGetLastError());
  k_dn65<BPP, OutT><<<bound * 8, 256, kSmem, s>>>(E->dn, E->dns, S->hq, S->hs, S->off, S->tiles, S->ntiles, S->pfl,
                                                   S->pidx, w, S->run, S->y, S->cst, part, ld);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace p065
