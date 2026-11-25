// p035_moe_decode.cuh — P035 moe_decode: the routed experts of a decode / verify call (M ≤ 8 rows) in two launches
// instead of six (k_plan ×2, k_scan, k_up, k_down, k_combine), included by moe.cu (uses its layouts and helpers).
//   k_up35  : CTA (c, u) = union slot u's expert × inter dims [32c, 32c+32) (the base's k_up tile). Every CTA builds
//             the union of the call's experts (ascending id) from ids itself; the expert's pairs (row, slot) in row
//             order, their latent rows staged in shared memory once; each warp streams its 16-row weight tile with
//             the next k stage's words and scales already in flight (registers), no barriers in the K loop.
//             h → hq / hs at the flat pair index row·10 + slot.
//   k_dn35  : CTA = R 16-row latent tiles × every selected expert, launched as a programmatic dependent of k_up35:
//             it builds the union and starts streaming its first w2 words while k_up35 finishes (griddepcontrol.wait
//             before reading h). Units (expert, tile) go round-robin to the warps; y stays in shared memory and the
//             CTA combines its rows itself: part = Σ_slot w · y in slot order (the base's k_combine expression).
// Bitwise the base: every (pair, row) output is the same MMA chain — same 16-row weight tiles and A fragments, k32
// steps 0 … K/32−1 ascending into one fp32 accumulator from 0 — and the same SiTU / MXFP8 / combine expressions; only
// which CTA computes it and where the activation operand sits change (MMA columns are independent).
namespace p035 {

constexpr int kMaxM = 8, kMaxP = kMaxM * kTopK;  // rows, pairs (above 8 rows the base is as fast: measured at 16)

__device__ __forceinline__ uint2 ld_na64(const void* p) {
  uint2 r;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}
__device__ __forceinline__ uint32_t ld_na8(const void* p) {
  uint32_t r;
  asm volatile("ld.global.nc.L1::no_allocate.u8 %0, [%1];" : "=r"(r) : "l"(p));
  return r;
}
__device__ __forceinline__ uint32_t ld_nc32(const void* p) {
  uint32_t r;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(r) : "l"(p));
  return r;
}
__device__ __forceinline__ uint32_t ld_nc8(const void* p) {
  uint32_t r;
  asm volatile("ld.global.nc.u8 %0, [%1];" : "=r"(r) : "l"(p));
  return r;
}
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

// The union of the experts picked by the call's P = M·10 (row, slot) pairs: bitmap + prefix → slot of an expert.
struct Sel {
  uint32_t mask[16];
  int pre[16];
  int n;
};
// all threads; ids of pairs tid, tid + blockDim (P ≤ 160 ≤ 2·blockDim) are returned in id[0..1] (−1 past P)
__device__ __forceinline__ void sel_build(Sel& S, const int* __restrict__ ids, int P, int (&id)[2]) {
  const int tid = threadIdx.x;
  if (tid < 16) S.mask[tid] = 0;
#pragma unroll
  for (int j = 0; j < 2; j++) {
    const int i = tid + j * blockDim.x;
    id[j] = i < P ? ids[i] : -1;
  }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 2; j++)
    if (id[j] >= 0) atomicOr(&S.mask[id[j] >> 5], 1u << (id[j] & 31));
  __syncthreads();
  if (tid < 32) {
    const int c = tid < 16 ? __popc(S.mask[tid]) : 0;
    int incl = c;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, incl, o);
      if (tid >= o) incl += v;
    }
    if (tid < 16) S.pre[tid] = incl - c;
    if (tid == 31) S.n = incl;
  }
  __syncthreads();
}
__device__ __forceinline__ int sel_slot(const Sel& S, int e) {
  return S.pre[e >> 5] + __popc(S.mask[e >> 5] & ((1u << (e & 31)) - 1u));
}

// ------------------------------------------------------------------------------------------------------------ up
// grid (32, P): CTA (c, u); 4 warps = the base's tiles (gate 2, up 2). NT n8 tiles of pairs (≤ 8·NT pairs).
template <int NT, int kS>
__global__ void __launch_bounds__(128) k_up35(const uint2* __restrict__ W, const uint8_t* __restrict__ WS,
                                              const uint8_t* __restrict__ xq, const uint8_t* __restrict__ xs,
                                              const int* __restrict__ ids, int M, uint8_t* hq, uint8_t* hs) {
  constexpr int MT = 8 * NT, RB = kLatent + 16;  // activation row stride (bank-conflict free B fragments)
  __shared__ Sel S;
  __shared__ int s_e, s_pair[kMaxM], s_slot[kMaxM];
  __shared__ uint32_t s_rows;
  extern __shared__ __align__(16) uint8_t smem[];
  const int cap = min(M, MT);                                         // pairs of one expert ≤ M
  float* gu = reinterpret_cast<float*>(smem);                         // [2][32][MT]
  uint8_t* sx = smem + 2 * 32 * MT * 4;                               // [cap][RB] e4m3
  uint8_t* ssx = sx + cap * RB;                                       // [cap][64] UE8M0
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int c = blockIdx.x, u = blockIdx.y, P = M * kTopK;
  int id[2];
  if (tid == 0) s_rows = 0, s_e = -1;
  sel_build(S, ids, P, id);
  if (u >= S.n) return;
  pdl_launch();
  if (tid < 16) {
    const int cnt = __popc(S.mask[tid]), k = u - S.pre[tid];
    if (k >= 0 && k < cnt) s_e = tid * 32 + int(__fns(S.mask[tid], 0, k + 1));
  }
  __syncthreads();
  const int e = s_e;
  // weights of this warp's tile, stage 0 in flight first
  const uint2* wt = W + (size_t)e * kUpWords + ((size_t)(c * 4 + warp) * kUpKB) * 32 + lane;
  const uint8_t* wst = WS + (size_t)e * kUpScales + ((size_t)(c * 4 + warp) * kUpKB) * 16 + g + 8 * (lane & 1);
  uint2 wf[2][kS];
  uint32_t sf[2][kS];
  auto ldw = [&](int st, uint2 (&b)[kS], uint32_t (&s)[kS]) {
#pragma unroll
    for (int i = 0; i < kS; i++) {
      b[i] = ld_na64(wt + (size_t)(st * kS + i) * 32);
      s[i] = ld_na8(wst + (size_t)(st * kS + i) * 16);
    }
  };
  ldw(0, wf[0], sf[0]);
  // the expert's pairs, row order
#pragma unroll
  for (int j = 0; j < 2; j++)
    if (id[j] == e) {
      const int i = tid + j * blockDim.x;
      atomicOr(&s_rows, 1u << (i / kTopK));
      s_slot[i / kTopK] = i % kTopK;
    }
  __syncthreads();
  const uint32_t rows = s_rows;
  if (tid < M && (rows >> tid & 1u)) s_pair[__popc(rows & ((1u << tid) - 1u))] = tid * kTopK + s_slot[tid];
  const int np = __popc(rows);
  __syncthreads();
  for (int i = tid; i < np * (kLatent / 16); i += 128) {
    const int j = i / (kLatent / 16), ch = i % (kLatent / 16), row = s_pair[j] / kTopK;
    cp_async16(sx + j * RB + ch * 16, xq + (size_t)row * kLatent + ch * 16, true);
  }
  for (int i = tid; i < np * 4; i += 128) {
    const int j = i / 4, ch = i % 4, row = s_pair[j] / kTopK;
    cp_async16(ssx + j * 64 + ch * 16, xs + (size_t)row * (kLatent / 32) + ch * 16, true);
  }
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();

  float acc[NT][4];
#pragma unroll
  for (int i = 0; i < NT; i++) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
  bool live[NT];
#pragma unroll
  for (int nt = 0; nt < NT; nt++) live[nt] = nt * 8 + g < np;
  constexpr int NS = kUpKB / kS;
#pragma unroll
  for (int st = 0; st < NS; st++) {
    if (st + 1 < NS) ldw(st + 1, wf[(st + 1) & 1], sf[(st + 1) & 1]);
#pragma unroll
    for (int i = 0; i < kS; i++) {
      const int kb = st * kS + i;
      const uint2 w = wf[st & 1][i];
      const uint32_t a0 = e2m1x4_expand(w.x & 0xFFFF), a1 = e2m1x4_expand(w.x >> 16);
      const uint32_t a2 = e2m1x4_expand(w.y & 0xFFFF), a3 = e2m1x4_expand(w.y >> 16);
#pragma unroll
      for (int nt = 0; nt < NT; nt++) {
        uint32_t b0 = 0, b1 = 0, sb = 0;
        if (live[nt]) {
          const uint8_t* p = sx + (nt * 8 + g) * RB + kb * 32 + 4 * t;
          b0 = *reinterpret_cast<const uint32_t*>(p), b1 = *reinterpret_cast<const uint32_t*>(p + 16);
          sb = ssx[(nt * 8 + g) * 64 + kb];
        }
        mma_e2m1_e4m3(acc[nt], a0, a1, a2, a3, b0, b1, sf[st & 1][i], sb);
      }
    }
  }
  // epilogue: the base's (gate warps 0, 1 / up warps 2, 3 → shared, SiTU, MXFP8 over these 32 dims per pair)
  float* dst = gu + (warp >> 1) * 32 * MT;
  const int ra = 16 * (warp & 1) + g;
#pragma unroll
  for (int nt = 0; nt < NT; nt++) {
    const int tk = nt * 8 + 2 * t;
    dst[ra * MT + tk] = acc[nt][0], dst[ra * MT + tk + 1] = acc[nt][1];
    dst[(ra + 8) * MT + tk] = acc[nt][2], dst[(ra + 8) * MT + tk + 1] = acc[nt][3];
  }
  __syncthreads();
  for (int r = warp; r < np; r += 4) {
    const float h = situ(gu[lane * MT + r], gu[32 * MT + lane * MT + r]);
    const uint32_t ex = act_scale_e8(warp_max(fabsf(h)), 1e-10f);
    const size_t p = (size_t)s_pair[r];
    hq[p * kInterR + 32 * c + lane] = uint8_t(quant_e4m3(h, ex));
    if (lane == 0) hs[p * (kInterR / 32) + c] = uint8_t(ex);
  }
}

// ------------------------------------------------------------------------------------------------------------ down
// grid 128 / R: CTA = latent tiles R·b … R·b+R−1 (16 rows each) × every selected expert; WN warps.
template <int NT, int WN, int R, int kS, class OutT>
__global__ void __launch_bounds__(32 * WN) k_dn35(const uint2* __restrict__ W, const uint8_t* __restrict__ WS,
                                                  const uint8_t* hq, const uint8_t* hs, const int* __restrict__ ids,
                                                  const float* __restrict__ wts, int M, OutT* part, int ld) {
  __shared__ Sel S;
  __shared__ int s_id[kMaxP], s_np[kMaxP], s_pair[kMaxP][kMaxM];
  __shared__ uint32_t s_rows[kMaxP];
  extern __shared__ __align__(16) float ys[];  // [P][16·R]
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int P = M * kTopK;
  int id[2];
  for (int i = tid; i < kMaxP; i += 32 * WN) s_rows[i] = 0;
  sel_build(S, ids, P, id);
  int slot[2];
#pragma unroll
  for (int j = 0; j < 2; j++)
    if (id[j] >= 0) {
      const int i = tid + j * blockDim.x;
      slot[j] = sel_slot(S, id[j]);
      atomicOr(&s_rows[slot[j]], 1u << (i / kTopK));
      s_id[slot[j]] = id[j];
    }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 2; j++)
    if (id[j] >= 0) {
      const int i = tid + j * blockDim.x, m = i / kTopK;
      const uint32_t rw = s_rows[slot[j]];
      s_pair[slot[j]][__popc(rw & ((1u << m) - 1u))] = i;
      if (m == 31 - __clz(rw)) s_np[slot[j]] = __popc(rw);  // the expert's last row writes its count
    }
  __syncthreads();
  const int nu = S.n * R;  // units (slot u = q / R, tile R·b + q % R)

  uint2 wf[2][kS];
  uint32_t sf[2][kS], bx[2][kS][NT][3];
  auto wptr = [&](int q, const uint2*& wt, const uint8_t*& wst) {
    const int e = s_id[q / R], T = blockIdx.x * R + q % R;
    wt = W + (size_t)e * kDnWords + ((size_t)T * kDnKB) * 32 + lane;
    wst = WS + (size_t)e * kDnScales + ((size_t)T * kDnKB) * 16 + g + 8 * (lane & 1);
  };
  auto ldw = [&](int q, int st, uint2 (&b)[kS], uint32_t (&s)[kS]) {
    const uint2* wt;
    const uint8_t* wst;
    wptr(q, wt, wst);
#pragma unroll
    for (int i = 0; i < kS; i++) {
      b[i] = ld_na64(wt + (size_t)(st * kS + i) * 32);
      s[i] = ld_na8(wst + (size_t)(st * kS + i) * 16);
    }
  };
  auto ldx = [&](int q, int st, uint32_t (&x)[kS][NT][3]) {
    const int u = q / R, np = s_np[u];
#pragma unroll
    for (int nt = 0; nt < NT; nt++) {
      const int j = nt * 8 + g;
      if (j < np) {
        const size_t p = (size_t)s_pair[u][j];
        const uint8_t* hp = hq + p * kInterR + st * kS * 32 + 4 * t;
        const uint8_t* sp = hs + p * (kInterR / 32) + st * kS;
#pragma unroll
        for (int i = 0; i < kS; i++) {
          x[i][nt][0] = ld_nc32(hp + i * 32), x[i][nt][1] = ld_nc32(hp + i * 32 + 16);
          x[i][nt][2] = ld_nc8(sp + i);
        }
      } else {
#pragma unroll
        for (int i = 0; i < kS; i++) x[i][nt][0] = x[i][nt][1] = x[i][nt][2] = 0;
      }
    }
  };
  constexpr int NS = kDnKB / kS;  // 4
  int q = warp;
  if (q < nu) ldw(q, 0, wf[0], sf[0]);
  pdl_wait();  // k_up35 complete: hq / hs written
  if (q < nu) ldx(q, 0, bx[0]);
  for (; q < nu; q += WN) {
    float acc[NT][4];
#pragma unroll
    for (int i = 0; i < NT; i++) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
#pragma unroll
    for (int st = 0; st < NS; st++) {
      const int nb = (st + 1) & 1;
      if (st + 1 < NS) {
        ldw(q, st + 1, wf[nb], sf[nb]);
        ldx(q, st + 1, bx[nb]);
      } else if (q + WN < nu) {
        ldw(q + WN, 0, wf[nb], sf[nb]);
        ldx(q + WN, 0, bx[nb]);
      }
#pragma unroll
      for (int i = 0; i < kS; i++) {
        const uint2 w = wf[st & 1][i];
        const uint32_t a0 = e2m1x4_expand(w.x & 0xFFFF), a1 = e2m1x4_expand(w.x >> 16);
        const uint32_t a2 = e2m1x4_expand(w.y & 0xFFFF), a3 = e2m1x4_expand(w.y >> 16);
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
          mma_e2m1_e4m3(acc[nt], a0, a1, a2, a3, bx[st & 1][i][nt][0], bx[st & 1][i][nt][1], sf[st & 1][i],
                        bx[st & 1][i][nt][2]);
      }
    }
    // y of this unit → shared [pair][16·R]: D rows g / g+8 (latent), columns 2t, 2t+1 (pairs)
    const int u = q / R, np = s_np[u], col = 16 * (q % R) + g;
#pragma unroll
    for (int nt = 0; nt < NT; nt++)
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int j = nt * 8 + 2 * t + (i & 1);
        if (j < np) ys[(size_t)s_pair[u][j] * (16 * R) + col + (i < 2 ? 0 : 8)] = acc[nt][i];
      }
  }
  __syncthreads();
  // combine this CTA's latent rows: the base's k_combine expression, slot order
  for (int i = tid; i < M * 16 * R; i += 32 * WN) {
    const int m = i / (16 * R), r = i % (16 * R);
    float a = 0.f;
#pragma unroll
    for (int k = 0; k < kTopK; k++) a += wts[m * kTopK + k] * ys[(size_t)(m * kTopK + k) * (16 * R) + r];
    put_out(part + (size_t)m * ld + blockIdx.x * 16 * R + r, a);
  }
}

template <int NT, int WN, int R, int SU, int SD, class OutT>
void launch(const MoeExperts* E, MoeScratch* S, const uint8_t* xq, const uint8_t* xs, const int* ids, const float* w,
            int M, OutT* part, int ld, cudaStream_t s) {
  static_assert(64 * WN >= 8 * NT * kTopK, "sel_build: two pairs per thread");
  constexpr int MT = 8 * NT;
  const int P = M * kTopK;
  const size_t sm_up = 2 * 32 * MT * 4 + std::min(M, MT) * (kLatent + 16 + 64);
  const size_t sm_dn = size_t(P) * 16 * R * 4;
  k_up35<NT, SU><<<dim3(32, std::min(P, kExperts)), 128, sm_up, s>>>(E->up, E->ups, xq, xs, ids, M, S->hq, S->hs);
  CUDA_CHECK(cudaGetLastError());
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(kDnTiles / R);
  cfg.blockDim = dim3(32 * WN);
  cfg.dynamicSmemBytes = sm_dn;
  cfg.stream = s;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, k_dn35<NT, WN, R, SD, OutT>, E->dn, E->dns, (const uint8_t*)S->hq, (const uint8_t*)S->hs,
                                ids, w, M, part, ld));
}

// M ≤ kMaxM
template <class OutT>
void run(const MoeExperts* E, MoeScratch* S, const uint8_t* xq, const uint8_t* xs, const int* ids, const float* w,
         int M, OutT* part, int ld, cudaStream_t s) {
  launch<1, 4, 1, 8, 8>(E, S, xq, xs, ids, w, M, part, ld, s);
}

}  // namespace p035
