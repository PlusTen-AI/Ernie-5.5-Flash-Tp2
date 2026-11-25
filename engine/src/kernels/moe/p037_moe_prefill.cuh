// p037_moe_prefill.cuh — P037 moe_prefill: the routed experts of a prefill call (M > 64 rows), included by moe.cu
// (base plan k_plan / k_scan and k_combine kept; k_up / k_down replaced).
//   CTA = (256 weight rows, one expert) × the expert's pairs in tiles of 64, 8 warps; both operands through a 3-stage
//   cp.async pipeline in shared memory (2 k32 steps a stage), so weights stream while the tensor cores run and two
//   CTAs fit an SM. Base: 64 rows × 64 pairs a CTA, weights from global into registers one 16-step stage at a time,
//   86 KB of shared memory → one CTA of 4 warps per SM.
//   up   : rows = 4 of the base's 32-dim blocks (gate 2 + up 2 tiles each); warp (block b, pair half) holds a block's
//          gate and up tiles, so SiTU and the MXFP8 group (amax over the block's 32 dims) are formed in registers.
//   down : rows = 16 latent tiles; warp (4 tiles, pair half) → y[pair][2048] fp32.
// Bitwise the base: per (16-row tile, pair) the same A fragments, B fragments and scales into the same m16n8k32
// MMA, k32 steps ascending into one fp32 accumulator from 0; the same SiTU / amax / quantization expressions.
namespace p037 {

constexpr int kKC = 2, kST = 3, kNP = 64;  // k32 steps a stage, stages, pairs a tile
constexpr int kXS = kKC * 32 + 16;         // activation row stride in a stage (bank-conflict free B fragments)
constexpr int kSS = 80;                    // activation-scale row stride (16-B aligned, conflict free)
constexpr int kWB = 16 * kKC * 256;        // weight bytes a stage (16 tiles)
constexpr int kWSB = 16 * kKC * 16;        // weight-scale bytes a stage
constexpr int kXB = kNP * kXS;             // activation bytes a stage
constexpr int kStageB = kWB + kWSB + kXB;
constexpr int kSmem = kST * kStageB + kNP * kSS;

__device__ __forceinline__ void cp_async4(void* smem, const void* gmem, bool ok) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(s), "l"(gmem), "r"(ok ? 4 : 0));
}

// The (expert, first pair) tiles of the call, by expert: one block, thread e = expert e.
__global__ void __launch_bounds__(kExperts) k_tiles37(const int* __restrict__ off, int2* tiles, int* ntiles) {
  __shared__ int ws[kExperts / 32];
  const int e = threadIdx.x, lane = e & 31, w = e >> 5;
  const int n = (off[e + 1] - off[e] + kNP - 1) / kNP;
  int incl = n;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int v = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += v;
  }
  if (lane == 31) ws[w] = incl;
  __syncthreads();
  int base = 0;
  for (int i = 0; i < w; i++) base += ws[i];
  base += incl - n;
  for (int j = 0; j < n; j++) tiles[base + j] = make_int2(e, j * kNP);
  if (e == kExperts - 1) *ntiles = base + n;
}

// grid (8, tiles bound): CTA = row block blockIdx.x × tile blockIdx.y (one expert, ≤ 64 of its pairs) — an expert's
// tiles are neighbours in the launch order (resident together), so they share the weight stream through L2.
// UP: rows [256·by, +256) of the expert's 2048 gate/up rows (= its blocks 4by … 4by+3), K 2048.
// DOWN: latent tiles 16·by … +15, K 1024.
template <bool UP>
__global__ void __launch_bounds__(256, 2) k_mm37(const uint2* __restrict__ W, const uint8_t* __restrict__ WS,
                                                 const uint8_t* __restrict__ xq, const uint8_t* __restrict__ xs,
                                                 const int* __restrict__ off, const int2* __restrict__ tiles,
                                                 const int* __restrict__ ntiles, const int* __restrict__ prow,
                                                 uint8_t* hq, uint8_t* hs, float* y) {
  if (blockIdx.y >= *ntiles) return;
  const int2 tile = tiles[blockIdx.y];
  const int by = blockIdx.x;
  constexpr int KB = UP ? kUpKB : kDnKB;            // k32 steps
  constexpr int XROW = UP ? kLatent : kInterR;      // activation bytes a row
  constexpr int SROW = XROW / 32;                   // activation scales a row
  constexpr int NS = KB / kKC;                      // stages
  const int e = tile.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int p0 = off[e], np = off[e + 1] - p0;
  const int rq = warp >> 1, ph = warp & 1;  // warp: tiles 4rq … 4rq+3 of the CTA's 16, pairs 32ph … 32ph+31
  extern __shared__ __align__(16) uint8_t smem[];
  uint8_t* sxs = smem + kST * kStageB;  // [kNP][kSS] activation scales of the tile
  // the CTA's 16 tiles: global tile index (UP: (c·4 + w) with c = 4bx + tile/4; DOWN: 16bx + tile)
  const uint2* wbase = W + (size_t)e * (UP ? kUpWords : kDnWords) + (size_t)(16 * by) * KB * 32;
  const uint8_t* wsbase = WS + (size_t)e * (UP ? kUpScales : kDnScales) + (size_t)(16 * by) * KB * 16;

  {
    const int q0 = tile.y;
    const int nq = min(kNP, np - q0);
    // this thread's activation source rows (one pair, one 16-B piece of a stage: 64 pairs × 4 pieces = 256)
    const int my_q = tid >> 2, my_c = tid & 3;
    const bool my_ok = my_q < nq;
    const int my_row = my_ok ? (UP ? prow[p0 + q0 + my_q] : p0 + q0 + my_q) : 0;
    const uint8_t* my_x = xq + (size_t)my_row * XROW;
    auto load = [&](int st, int buf) {
      uint8_t* sw = smem + buf * kStageB;
      const int kb0 = st * kKC;
#pragma unroll
      for (int i = tid; i < kWB / 16; i += 256) {  // tile i / (KC·16): KC·256 contiguous bytes
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
    // activation scales of the tile (with stage 0)
    for (int i = tid; i < kNP * (SROW / 4); i += 256) {
      const int q = i / (SROW / 4), c4 = i % (SROW / 4);
      const bool ok = q < nq;
      const int row = ok ? (UP ? prow[p0 + q0 + q] : p0 + q0 + q) : 0;
      cp_async4(sxs + q * kSS + c4 * 4, xs + (size_t)row * SROW + c4 * 4, ok);
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
    const int ntv = min(4, max(0, (nq - 32 * ph + 7) / 8));  // valid n8 tiles of this warp (warp-uniform)
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
            const uint8_t* p = sx + q * kXS + kbl * 32 + 4 * t;
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
    __syncthreads();  // all reads of the stages done before the next tile's prologue
    if constexpr (UP) {
      // block c = 4bx + rq: tiles 0, 1 gate dims g, g+8 / 16+g, 24+g; tiles 2, 3 the up rows of the same dims
      const int c = 4 * by + rq;
#pragma unroll
      for (int nt = 0; nt < 4; nt++) {
        if (nt < ntv) {
#pragma unroll
          for (int i = 0; i < 2; i++) {
            float h[4];
            h[0] = situ(acc[0][nt][i], acc[2][nt][i]);          // dim g
            h[1] = situ(acc[0][nt][2 + i], acc[2][nt][2 + i]);  // dim g + 8
            h[2] = situ(acc[1][nt][i], acc[3][nt][i]);          // dim 16 + g
            h[3] = situ(acc[1][nt][2 + i], acc[3][nt][2 + i]);  // dim 24 + g
            float mx = fmaxf(fmaxf(fabsf(h[0]), fabsf(h[1])), fmaxf(fabsf(h[2]), fabsf(h[3])));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 8));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 16));
            const uint32_t ex = act_scale_e8(mx, 1e-10f);
            const int q = 32 * ph + 8 * nt + 2 * t + i;
            if (q < nq) {
              const size_t p = (size_t)(p0 + q0 + q);
              uint8_t* d = hq + p * kInterR + 32 * c;
              d[g] = uint8_t(quant_e4m3(h[0], ex)), d[g + 8] = uint8_t(quant_e4m3(h[1], ex));
              d[16 + g] = uint8_t(quant_e4m3(h[2], ex)), d[24 + g] = uint8_t(quant_e4m3(h[3], ex));
              if (g == 0) hs[p * (kInterR / 32) + c] = uint8_t(ex);
            }
          }
        }
      }
    } else {
#pragma unroll
      for (int mt = 0; mt < 4; mt++) {
        const int r = 16 * (16 * by + 4 * rq + mt) + g;
#pragma unroll
        for (int nt = 0; nt < 4; nt++) {
          if (nt < ntv) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
              const int q = 32 * ph + 8 * nt + 2 * t + (i & 1);
              if (q < nq) y[(size_t)(p0 + q0 + q) * kLatent + r + (i < 2 ? 0 : 8)] = acc[mt][nt][i];
            }
          }
        }
      }
    }
  }
}

void run_mm(const MoeExperts* E, MoeScratch* S, const uint8_t* xq, const uint8_t* xs, int P, cudaStream_t s) {
  static bool attr = [] {
    CUDA_CHECK(cudaFuncSetAttribute(k_mm37<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
    CUDA_CHECK(cudaFuncSetAttribute(k_mm37<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
    return true;
  }();
  (void)attr;
  const int bound = cdiv(P, kNP) + std::min(P, kExperts);  // Σ_e ⌈n_e / 64⌉ ≤ P / 64 + experts used
  k_tiles37<<<1, kExperts, 0, s>>>(S->off, S->tiles, S->ntiles);
  k_mm37<true><<<dim3(8, bound), 256, kSmem, s>>>(E->up, E->ups, xq, xs, S->off, S->tiles, S->ntiles, S->prow, S->hq,
                                                   S->hs, nullptr);
  CUDA_CHECK(cudaGetLastError());
  k_mm37<false><<<dim3(8, bound), 256, kSmem, s>>>(E->dn, E->dns, S->hq, S->hs, S->off, S->tiles, S->ntiles, nullptr,
                                                    nullptr, nullptr, S->y);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace p037
