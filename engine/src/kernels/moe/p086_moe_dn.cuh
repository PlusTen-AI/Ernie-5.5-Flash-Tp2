// p086_moe_dn.cuh — P086 moe_pf_dn: the down projection + combine of a prefill call (M > 64 rows), included by moe.cu
// after P065 (its plan, chain states and park buffer; P037's up kernel and down tile unchanged).
//   P065 applies a pair's 64-dim y segment to its row's running sum by ONE lane (16 lanes a warp, each with 16
//   uncoalesced float4 loads / stores). Measured on a real layer with real routing, that epilogue costs +1.7 ms
//   (4096 rows) / +5.2 ms (8192) over P065's GEMM alone — data movement: the same time with relaxed atomics, and with
//   the atomics removed altogether.
//   Here a warp applies its 16 staged pairs together: lane j < 16 owns pair j's chain state (P065's protocol: acquire,
//   park bit, release CAS), the data moves coalesced (lanes 0–15 one pair's 256 bytes, lanes 16–31 the next one's),
//   every round trip batched over the 16 pairs (__syncwarp orders the lanes' data around lane j's acquire / release,
//   as CUTLASS's semaphore does with __syncthreads), and the sums stay in place of y in shared memory (a register copy
//   made ptxas spill the accumulators inside the main loop).
//   Passes: GB10's L2 keeps only ~6 MB of dirty lines at full read-modify-write speed (measured 1.4 TB/s up to 6 MB,
//   0.70 at 8 MB, 0.30 at 16 MB; ~8 MB with evict_last), so P065's 8.4 MB (4096 rows) / 16.8 MB (8192) of running
//   sums a pass already miss — fewer passes cost little and save h re-reads: 2 passes up to 4096 rows, 1 above.
// Bitwise P065 / the base: every y is P037's MMA chain; every row's sum is k_combine's FFMA chain
// a = fma(w0, y0, 0), a = fma(wk, yk, a) for k = 1 … 9 in slot order, rounded once (bf16) at the end.
namespace p086 {

using p037::kNP;
constexpr int kYS = 68;  // staged y row stride (floats), as P065

__device__ __forceinline__ void put4(float* o, float4 a) { *reinterpret_cast<float4*>(o) = a; }
__device__ __forceinline__ void put4(bf16* o, float4 a) {
  *reinterpret_cast<uint2*>(o) = make_uint2(pack_bf16x2(a.x, a.y), pack_bf16x2(a.z, a.w));
}
__device__ __forceinline__ float4 fma4(float w, float4 y, float4 a) {
  return make_float4(__fmaf_rn(w, y.x, a.x), __fmaf_rn(w, y.y, a.y), __fmaf_rn(w, y.z, a.z), __fmaf_rn(w, y.w, a.w));
}

// One down tile: latent rows [256·by, +256) × pairs [q0, q0 + 64) of expert e (P065's tile: 8 warps = 4 row quarters
// × 2 pair halves), its y applied to the row chains. smem: p037::kSmem bytes (free on entry).
template <class OutT>
__device__ __forceinline__ void dn_tile(int e, int q0, int by, uint8_t* smem, const uint2* __restrict__ W,
                                        const uint8_t* __restrict__ WS, const uint8_t* __restrict__ xq,
                                        const uint8_t* __restrict__ xs, const int* __restrict__ off,
                                        const int* __restrict__ pfl, const int* __restrict__ pidx,
                                        const float* __restrict__ wts, float* run, float* park, uint32_t* cst,
                                        OutT* part, int ld) {
  constexpr int kKC = p037::kKC, kST = p037::kST, kXS = p037::kXS, kSS = p037::kSS, kWB = p037::kWB,
                kWSB = p037::kWSB, kStageB = p037::kStageB;
  constexpr int KB = kDnKB, XROW = kInterR, SROW = XROW / 32, NS = KB / kKC;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tq = lane & 3;
  const int p0 = off[e], np = off[e + 1] - p0;
  const int rq = warp >> 1, ph = warp & 1;
  uint8_t* sxs = smem + kST * kStageB;
  const uint2* wbase = W + (size_t)e * kDnWords + (size_t)(16 * by) * KB * 32;
  const uint8_t* wsbase = WS + (size_t)e * kDnScales + (size_t)(16 * by) * KB * 16;
  const int nq = min(kNP, np - q0);
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
  if (tid < kNP * (SROW / 16)) {  // h's scales: 16-B cp.async pieces (P065: 4-B pieces)
    const int q = tid / (SROW / 16), c16 = tid % (SROW / 16);
    const bool ok = q < nq;
    cp_async16(sxs + q * kSS + c16 * 16, xs + (size_t)(ok ? p0 + q0 + q : 0) * SROW + c16 * 16, ok);
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
  __syncthreads();  // the stages are free: stage y per warp (16 pairs × 64 dims at a time), then the chains
  float* ysw = reinterpret_cast<float*>(smem) + warp * 16 * kYS;
  const int col = 256 * by + 64 * rq;
  const int half = lane >> 4, c4 = lane & 15;
  const uint64_t pl = p065::pol_last();
#pragma unroll 1
  for (int hh = 0; hh < 2; hh++) {
    if (32 * ph + 16 * hh >= nq) break;  // warp-uniform
    __syncwarp();
#pragma unroll
    for (int n2 = 0; n2 < 2; n2++)
#pragma unroll
      for (int mt = 0; mt < 4; mt++)
#pragma unroll
        for (int i = 0; i < 4; i++)  // (static acc indices: a dynamic one would put acc in local memory)
          ysw[(8 * n2 + 2 * tq + (i & 1)) * kYS + 16 * mt + g + (i < 2 ? 0 : 8)] =
              hh ? acc[mt][2 + n2][i] : acc[mt][n2][i];
    __syncwarp();
    // lane j < 16: pair j's chain (row, slot, weight, state)
    const int qj = 32 * ph + 16 * hh + lane;
    const bool valid = lane < 16 && qj < nq;
    int pj = 0, rowj = 0, kj = 0;
    float wj = 0.f;
    uint32_t* spj = cst;
    uint32_t sj = 0;
    if (valid) {
      pj = p0 + q0 + qj;
      const int fl = pfl[pj];
      rowj = fl / kTopK, kj = fl - rowj * kTopK, wj = wts[fl];
      spj = cst + ((size_t)rowj * 8 + by) * 4 + rq;
      sj = p065::ld_acq(spj);
    }
    bool own = valid && (sj & 15u) == uint32_t(kj);
    // the data lanes' view: pair jj = 2i + half, float4 c4 of its 64 dims, the sum kept in place of y in ysw (few
    // live registers: a register copy made ptxas spill acc across the main loop)
    float4* y4 = reinterpret_cast<float4*>(ysw) + c4;
    const unsigned pm = __ballot_sync(0xffffffffu, valid && !own);
    if (pm) {  // park, then tell the chain; if it reached the slot meanwhile, the pair is ours after all
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const int pii = __shfl_sync(0xffffffffu, pj, 2 * i + half);
        if (pm >> (2 * i + half) & 1u)
          p065::stcg4(park + (size_t)pii * kLatent + col + 4 * c4, y4[(2 * i + half) * (kYS / 4)]);
      }
      __syncwarp();
      if (valid && !own) {
        const uint32_t o = p065::atom_or(spj, 1u << (8 + kj));
        own = (o & 15u) == uint32_t(kj);
        sj = o | (1u << (8 + kj));
      }
    }
    const unsigned om = __ballot_sync(0xffffffffu, own);
    if (!om) continue;
    __syncwarp();  // the state acquires before the running-sum loads
#pragma unroll
    for (int i0 = 0; i0 < 8; i0 += 4) {  // 4 pairs a lane at a time (register pressure)
      float4 v[4];
#pragma unroll
      for (int u = 0; u < 4; u++) {
        const int jj = 2 * (i0 + u) + half;
        const int rii = __shfl_sync(0xffffffffu, rowj, jj), kii = __shfl_sync(0xffffffffu, kj, jj);
        v[u] = make_float4(0.f, 0.f, 0.f, 0.f);
        if ((om >> jj & 1u) && kii != 0) v[u] = p065::ldcg4p(run + (size_t)rii * kLatent + col + 4 * c4, pl);
      }
#pragma unroll
      for (int u = 0; u < 4; u++) {
        const int jj = 2 * (i0 + u) + half;
        const float wii = __shfl_sync(0xffffffffu, wj, jj);
        float4& y = y4[jj * (kYS / 4)];
        if (om >> jj & 1u) y = fma4(wii, y, v[u]);
      }
    }
    int nx = kj + 1;
    for (;;) {
      for (;;) {  // parked successors, in slot order
        const bool need = own && nx < kTopK && (sj >> (8 + nx) & 1u);
        const unsigned nm = __ballot_sync(0xffffffffu, need);
        if (!nm) break;
        int sp = 0;
        float sw = 0.f;
        if (need) sp = pidx[rowj * kTopK + nx], sw = wts[rowj * kTopK + nx];
#pragma unroll
        for (int i0 = 0; i0 < 8; i0 += 4) {
          float4 v[4];
#pragma unroll
          for (int u = 0; u < 4; u++) {
            const int jj = 2 * (i0 + u) + half;
            const int spi = __shfl_sync(0xffffffffu, sp, jj);
            if (nm >> jj & 1u) v[u] = p065::ldcg4(park + (size_t)spi * kLatent + col + 4 * c4);
          }
#pragma unroll
          for (int u = 0; u < 4; u++) {
            const int jj = 2 * (i0 + u) + half;
            const float swi = __shfl_sync(0xffffffffu, sw, jj);
            float4& y = y4[jj * (kYS / 4)];
            if (nm >> jj & 1u) y = fma4(swi, v[u], y);
          }
        }
        if (need) nx++;
      }
      const bool fin = own && nx == kTopK;
      const unsigned fm = __ballot_sync(0xffffffffu, fin);
      if (fm) {  // complete sums: the output, and the chain reset for the next call
#pragma unroll
        for (int i = 0; i < 8; i++) {
          const int rii = __shfl_sync(0xffffffffu, rowj, 2 * i + half);
          if (fm >> (2 * i + half) & 1u) put4(part + (size_t)rii * ld + col + 4 * c4, y4[(2 * i + half) * (kYS / 4)]);
        }
        if (fin) p065::st_rlx(spj, 0u), own = false;
      }
      const unsigned rm = __ballot_sync(0xffffffffu, own);
      if (!rm) break;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const int rii = __shfl_sync(0xffffffffu, rowj, 2 * i + half);
        if (rm >> (2 * i + half) & 1u)
          p065::stcg4p(run + (size_t)rii * kLatent + col + 4 * c4, y4[(2 * i + half) * (kYS / 4)], pl);
      }
      __syncwarp();  // the running sums stored before lane j lets go
      if (own) {
        const uint32_t o = p065::atom_cas(spj, sj, (sj & ~15u) | uint32_t(nx));
        if (o == sj) own = false;
        else sj = o;  // a successor got parked meanwhile: apply it, try again
      }
      if (!__ballot_sync(0xffffffffu, own)) break;
      __syncwarp();  // the CAS results (acquire) before the parked loads
    }
  }
}

// grid: tiles × 8 latent blocks, CTA t → pass t / (tiles·BPP), inside it tile-major (expert id order), BPP blocks
template <int BPP, class OutT>
__global__ void __launch_bounds__(256, 2) k_dn86(const uint2* __restrict__ W, const uint8_t* __restrict__ WS,
                                                 const uint8_t* __restrict__ xq, const uint8_t* __restrict__ xs,
                                                 const int* __restrict__ off, const int2* __restrict__ tiles,
                                                 const int* __restrict__ ntiles, const int* __restrict__ pfl,
                                                 const int* __restrict__ pidx, const float* __restrict__ wts,
                                                 float* run, float* park, uint32_t* cst, OutT* part, int ld) {
  extern __shared__ __align__(16) uint8_t smem[];
  const int ntl = *ntiles, t = blockIdx.x;
  if (t >= ntl * 8) return;
  const int per = ntl * BPP, pass = t / per, rr = t - pass * per;
  const int2 tile = tiles[rr / BPP];
  dn_tile<OutT>(tile.x, tile.y, pass * BPP + rr % BPP, smem, W, WS, xq, xs, off, pfl, pidx, wts, run, park, cst, part,
                ld);
}

template <int BPP, class OutT>
void launch_dn(const MoeExperts* E, MoeScratch* S, const float* w, int bound, OutT* part, int ld, cudaStream_t s) {
  static bool attr = [] {
    CUDA_CHECK(cudaFuncSetAttribute(k_dn86<BPP, OutT>, cudaFuncAttributeMaxDynamicSharedMemorySize, p037::kSmem));
    return true;
  }();
  (void)attr;
  k_dn86<BPP, OutT><<<bound * 8, 256, p037::kSmem, s>>>(E->dn, E->dns, S->hq, S->hs, S->off, S->tiles, S->ntiles,
                                                         S->pfl, S->pidx, w, S->run, S->y, S->cst, part, ld);
  CUDA_CHECK(cudaGetLastError());
}

// a prefill call (M > 64): P065's plan, P037's up kernel, then the down + combine above
template <class OutT>
void run(const MoeExperts* E, MoeScratch* S, const uint8_t* xq, const uint8_t* xs, const int* ids, const float* w,
         int M, OutT* part, int ld, cudaStream_t s) {
  static bool attr = [] {
    CUDA_CHECK(cudaFuncSetAttribute(p037::k_mm37<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, p037::kSmem));
    return true;
  }();
  (void)attr;
  const int P = M * kTopK;
  const int bound = cdiv(P, kNP) + std::min(P, kExperts);
  p065::k_plan65a<<<p065::kPC, 256, 0, s>>>(ids, P, S->phist);
  p065::k_plan65b<<<p065::kPC, kExperts, 0, s>>>(ids, P, S->phist, S->cnt, S->off, S->act, S->nact, S->prow, S->pidx,
                                                 S->pfl, S->tiles, S->ntiles);
  CUDA_CHECK(cudaGetLastError());
  p037::k_mm37<true><<<dim3(8, bound), 256, p037::kSmem, s>>>(E->up, E->ups, xq, xs, S->off, S->tiles, S->ntiles,
                                                              S->prow, S->hq, S->hs, nullptr);
  CUDA_CHECK(cudaGetLastError());
  if (M <= 4096) launch_dn<4, OutT>(E, S, w, bound, part, ld, s);  // 2 passes of 4 256-row latent blocks
  else launch_dn<8, OutT>(E, S, w, bound, part, ld, s);             // 1 pass
}

}  // namespace p086
