// moe.cu — ERNIE latent MoE, native W4A8 (contract: moe.h).
//
// Weights per expert on this rank, retiled at upload for swap-AB block-scaled MMA (weights = the M side, 16-row tiles;
// a lane reads one 8-byte word of packed e2m1 per k32 step and expands it to the byte-per-value operand):
//   up    [c 32][w 4][kb 64][lane] uint2 — CTA c owns inter dims [32c, 32c+32): warps 0, 1 the gate rows (w1) of
//         its two 16-row halves, warps 2, 3 the up rows (w3) of the same dims, so the epilogue has g and u together
//   down  [tile 128][kb 32][lane] uint2 — 2048 latent rows of this rank's w2 columns; CTA d owns tiles 4d … 4d+3
//   scales UE8M0 per (tile, kb): 16 bytes, row i of the tile at byte i (lane takes row g + 8·(lane & 1))
// A lane's word for rows g, g+8 of a tile at k32 step kb: {A[g][4t…4t+3], A[g+8][4t…], A[g][16+4t…], A[g+8][16+4t…]},
// 16 bits each (4 codes, lower k in the lower nibble — the checkpoint's packing).
#include "kernels/moe/moe.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <thread>
#include <vector>

#include "core/log.h"
#include "kernels/gemm/mma.cuh"
#include "model/patches.h"

namespace e55 {

namespace {
constexpr int kInterR = kExpInter / 2;          // 1024 inter dims per rank
constexpr int kUpKB = kLatent / 32;             // 64
constexpr int kDnKB = kInterR / 32;             // 32
constexpr int kUpWords = 32 * 4 * kUpKB * 32;   // uint2 per expert
constexpr int kDnTiles = kLatent / 16;          // 128
constexpr int kDnWords = kDnTiles * kDnKB * 32;
constexpr int kUpScales = 32 * 4 * kUpKB * 16;  // bytes per expert
constexpr int kDnScales = kDnTiles * kDnKB * 16;
constexpr int kStage = 512;                     // activation bytes per k stage (16 k32 steps)
constexpr int kPad = 16;
}  // namespace

struct MoeExperts {
  uint2* up = nullptr;     // [512][kUpWords]
  uint2* dn = nullptr;     // [512][kDnWords]
  uint8_t* ups = nullptr;  // [512][kUpScales]
  uint8_t* dns = nullptr;  // [512][kDnScales]
};

struct MoeScratch {
  int max_rows = 0;
  int *cnt = nullptr, *off = nullptr, *act = nullptr, *nact = nullptr;  // [512], [513], [512], [1]
  int *prow = nullptr, *pidx = nullptr;                                // [pairs] row of pair, [rows·10] pair of slot
  uint8_t *hq = nullptr, *hs = nullptr;                                 // [pairs][1024] e4m3, [pairs][32] UE8M0
  float* y = nullptr;                                                   // [pairs][2048]
  int2* tiles = nullptr;                                                // P037: (expert, first pair) tiles
  int* ntiles = nullptr;
  int* pfl = nullptr;                                                   // P065: [pairs] row·10 + slot of a pair
  float* run = nullptr;                                                 // P065: [rows][2048] running combine sums
  uint32_t* cst = nullptr;                                              // P065: [rows][8][4] chain states (zero)
  int* dcnt = nullptr;                                                  // P066 / P190 / P194: ticket / done counters (zero, 8 KB)
  uint8_t* dlat = nullptr;                                              // P066 / P190 / P194: the latent as MXFP8 (≤ 128 rows)
  int* phist = nullptr;                                                 // P065: [64 chunks][512] plan counts
};

// ------------------------------------------------------------------------------------------------------------ upload
namespace {
// packed nibbles of row r, k values [k0, k0+4) of a [rows, K/2] e2m1 matrix → 16 bits
inline uint16_t nib4(const uint8_t* w, int K, int r, int k0) {
  const uint8_t* p = w + (size_t)r * (K / 2) + k0 / 2;
  return uint16_t(p[0] | (p[1] << 8));
}
void put_tile(uint2* dst, uint8_t* sdst, const uint8_t* w, const uint8_t* s, int K, int row0, int kb) {
  for (int lane = 0; lane < 32; lane++) {
    const int g = lane >> 2, t = lane & 3, k = kb * 32;
    dst[lane].x = uint32_t(nib4(w, K, row0 + g, k + 4 * t)) | uint32_t(nib4(w, K, row0 + g + 8, k + 4 * t)) << 16;
    dst[lane].y = uint32_t(nib4(w, K, row0 + g, k + 16 + 4 * t)) | uint32_t(nib4(w, K, row0 + g + 8, k + 16 + 4 * t)) << 16;
  }
  for (int i = 0; i < 16; i++) sdst[i] = s[(size_t)(row0 + i) * (K / 32) + kb];
}
}  // namespace

MoeExperts* moe_upload(const Weights& W, int layer) {
  return moe_upload(W, "model.layers." + std::to_string(layer) + ".block_sparse_moe.experts.");
}

MoeExperts* moe_upload(const Weights& W, const std::string& prefix) {
  auto* E = new MoeExperts;
  CUDA_CHECK(cudaMalloc(&E->up, size_t(kExperts) * kUpWords * 8));
  CUDA_CHECK(cudaMalloc(&E->dn, size_t(kExperts) * kDnWords * 8));
  CUDA_CHECK(cudaMalloc(&E->ups, size_t(kExperts) * kUpScales));
  CUDA_CHECK(cudaMalloc(&E->dns, size_t(kExperts) * kDnScales));
  const std::string& pre = prefix;
  std::atomic<int> next{0};
  std::vector<std::thread> th;
  const int nt = std::max(1, std::min(load_threads(), 8));
  for (int i = 0; i < nt; i++)
    th.emplace_back([&] {
      std::vector<uint2> up(kUpWords), dn(kDnWords);
      std::vector<uint8_t> ups(kUpScales), dns(kDnScales);
      for (int e; (e = next.fetch_add(1)) < kExperts;) {
        const std::string p = pre + std::to_string(e) + ".";
        std::vector<int64_t> sh;
        auto w1 = W.slice(p + "w1.weight", &sh);
        if (sh[0] != kInterR || sh[1] != kLatent / 2) qlog::fail("moe", "%sw1 shape", p.c_str());
        auto w3 = W.slice(p + "w3.weight");
        auto w2 = W.slice(p + "w2.weight", &sh);
        if (sh[0] != kLatent || sh[1] != kInterR / 2) qlog::fail("moe", "%sw2 shape", p.c_str());
        auto s1 = W.slice(p + "w1.weight_scale_inv"), s3 = W.slice(p + "w3.weight_scale_inv");
        auto s2 = W.slice(p + "w2.weight_scale_inv");
        for (int c = 0; c < 32; c++)
          for (int w = 0; w < 4; w++)
            for (int kb = 0; kb < kUpKB; kb++) {
              const size_t i = ((size_t)(c * 4 + w) * kUpKB + kb);
              const uint8_t* src = w < 2 ? w1.data() : w3.data();
              const uint8_t* ss = w < 2 ? s1.data() : s3.data();
              put_tile(up.data() + i * 32, ups.data() + i * 16, src, ss, kLatent, 32 * c + 16 * (w & 1), kb);
            }
        for (int T = 0; T < kDnTiles; T++)
          for (int kb = 0; kb < kDnKB; kb++) {
            const size_t i = (size_t)T * kDnKB + kb;
            put_tile(dn.data() + i * 32, dns.data() + i * 16, w2.data(), s2.data(), kInterR, 16 * T, kb);
          }
        CUDA_CHECK(cudaMemcpy(E->up + size_t(e) * kUpWords, up.data(), size_t(kUpWords) * 8, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(E->dn + size_t(e) * kDnWords, dn.data(), size_t(kDnWords) * 8, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(E->ups + size_t(e) * kUpScales, ups.data(), kUpScales, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(E->dns + size_t(e) * kDnScales, dns.data(), kDnScales, cudaMemcpyHostToDevice));
      }
    });
  for (auto& x : th) x.join();
  return E;
}

void moe_free(MoeExperts* e) {
  if (!e) return;
  cudaFree(e->up), cudaFree(e->dn), cudaFree(e->ups), cudaFree(e->dns);
  delete e;
}

MoeScratch* moe_scratch(int max_rows) {
  auto* s = new MoeScratch;
  s->max_rows = max_rows;
  const size_t P = size_t(max_rows) * kTopK;
  CUDA_CHECK(cudaMalloc(&s->cnt, kExperts * 4));
  CUDA_CHECK(cudaMalloc(&s->off, (kExperts + 1) * 4));
  CUDA_CHECK(cudaMalloc(&s->act, kExperts * 4));
  CUDA_CHECK(cudaMalloc(&s->nact, 4));
  CUDA_CHECK(cudaMalloc(&s->prow, P * 4));
  CUDA_CHECK(cudaMalloc(&s->pidx, P * 4));
  CUDA_CHECK(cudaMalloc(&s->hq, P * kInterR));
  CUDA_CHECK(cudaMalloc(&s->hs, P * (kInterR / 32)));
  CUDA_CHECK(cudaMalloc(&s->y, P * kLatent * 4));
  CUDA_CHECK(cudaMalloc(&s->tiles, (P / 64 + 1 + kExperts) * sizeof(int2)));
  CUDA_CHECK(cudaMalloc(&s->ntiles, 4));
  CUDA_CHECK(cudaMalloc(&s->pfl, P * 4));
  CUDA_CHECK(cudaMalloc(&s->run, size_t(max_rows) * kLatent * 4));
  CUDA_CHECK(cudaMalloc(&s->cst, size_t(max_rows) * 32 * 4));
  CUDA_CHECK(cudaMemset(s->cst, 0, size_t(max_rows) * 32 * 4));
  CUDA_CHECK(cudaMalloc(&s->dcnt, 8192));  // P194's 1024 sub-expert counters (P066 / P190 use the first 4 KB)
  CUDA_CHECK(cudaMemset(s->dcnt, 0, 8192));
  CUDA_CHECK(cudaMalloc(&s->dlat, 128 * (kLatent + kLatent / 32) + 64));  // P066 ≤ 32 rows, P190 ≤ 64, P194 ≤ 128
  CUDA_CHECK(cudaMalloc(&s->phist, 64 * kExperts * 4));
  return s;
}

void moe_scratch_free(MoeScratch* s) {
  if (!s) return;
  cudaFree(s->cnt), cudaFree(s->off), cudaFree(s->act), cudaFree(s->nact), cudaFree(s->prow), cudaFree(s->pidx);
  cudaFree(s->hq), cudaFree(s->hs), cudaFree(s->y), cudaFree(s->tiles), cudaFree(s->ntiles);
  cudaFree(s->pfl), cudaFree(s->run), cudaFree(s->cst), cudaFree(s->dcnt), cudaFree(s->dlat), cudaFree(s->phist);
  delete s;
}

// ------------------------------------------------------------------------------------------------------------ routing
namespace {
__device__ __forceinline__ float sigmoid_ieee(float x) { return __fdiv_rn(1.0f, 1.0f + expf(-x)); }

// one warp per row: lane holds experts lane + 32j (j < 16). vLLM-ERNIE's routing (ernie5_5/nvidia/moe.py,
// _ernie5_5_routing_kernel): top 10 of s + bias by repeated argmax (ties → lower id); the slots then in
// torch.topk(sorted=False) order — experts whose s + bias is above the 10th value by id, then those equal to it by id;
// the renormalizing sum is tl.sum's tree over the 16 zero-padded slots ((a_j + a_j+8), then +4, +2, +1 — the same
// bits as torch's sum over 10); w = s / max(sum, 1e-20) (IEEE) · routed_scaling_factor_param.
__global__ void k_route(const float* __restrict__ lg, const float* __restrict__ bias, const float* __restrict__ scale,
                        int M, int* ids, float* w) {
  const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (row >= M) return;
  float s[16], sel[16];
#pragma unroll
  for (int j = 0; j < 16; j++) {
    const int e = lane + 32 * j;
    s[j] = sigmoid_ieee(lg[(size_t)row * 1024 + e]) + sigmoid_ieee(lg[(size_t)row * 1024 + 512 + e]);
    sel[j] = s[j] + bias[e];
  }
  int my_id[kTopK];
  float my_s[kTopK], my_v[kTopK];
  for (int k = 0; k < kTopK; k++) {
    float best = -INFINITY;
    int be = 1 << 30;
#pragma unroll
    for (int j = 0; j < 16; j++) {
      const int e = lane + 32 * j;
      if (sel[j] > best || (sel[j] == best && e < be)) best = sel[j], be = e;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ob = __shfl_xor_sync(0xffffffffu, best, o);
      const int oe = __shfl_xor_sync(0xffffffffu, be, o);
      if (ob > best || (ob == best && oe < be)) best = ob, be = oe;
    }
    float sv = 0.f;
#pragma unroll
    for (int j = 0; j < 16; j++)
      if (lane + 32 * j == be) sv = s[j], sel[j] = -INFINITY;
    sv = __shfl_sync(0xffffffffu, sv, be & 31);
    my_id[k] = be, my_s[k] = sv, my_v[k] = best;
  }
  if (lane < kTopK) {  // lane k places selection k at its topk(sorted=False) slot
    int key[kTopK];
#pragma unroll
    for (int k = 0; k < kTopK; k++) key[k] = my_v[k] > my_v[kTopK - 1] ? my_id[k] : my_id[k] + kExperts;
    int mine = 0, slot = 0;
    float ms = 0.f;
#pragma unroll
    for (int k = 0; k < kTopK; k++)
      if (k == lane) mine = key[k], ms = my_s[k];
#pragma unroll
    for (int k = 0; k < kTopK; k++) slot += key[k] < mine;
    float a[16];
#pragma unroll
    for (int i = 0; i < 16; i++) a[i] = 0.f;
#pragma unroll
    for (int k = 0; k < kTopK; k++) {  // every lane forms the slot array (ranks are a permutation)
      int r = 0;
#pragma unroll
      for (int j = 0; j < kTopK; j++) r += key[j] < key[k];
#pragma unroll
      for (int i = 0; i < kTopK; i++)
        if (i == r) a[i] = my_s[k];
    }
#pragma unroll
    for (int i = 0; i < 8; i++) a[i] = a[i] + a[i + 8];
#pragma unroll
    for (int i = 0; i < 4; i++) a[i] = a[i] + a[i + 4];
#pragma unroll
    for (int i = 0; i < 2; i++) a[i] = a[i] + a[i + 2];
    const float tot = fmaxf(a[0] + a[1], 1e-20f);
    const int id = mine % kExperts;
    ids[(size_t)row * kTopK + slot] = id;
    w[(size_t)row * kTopK + slot] = __fdiv_rn(ms, tot) * scale[id];
  }
}

// block e: the pairs (row, slot) routed to expert e, in (row, slot) order — pass 0 counts, pass 1 writes
__global__ void k_plan(const int* __restrict__ ids, int P, int pass, int* cnt, const int* off, int* prow, int* pidx) {
  const int e = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  __shared__ int wbase[33];
  int base = pass ? off[e] : 0;
  for (int c0 = 0; c0 < P; c0 += blockDim.x) {
    const int i = c0 + threadIdx.x;
    const bool hit = i < P && ids[i] == e;
    const unsigned b = __ballot_sync(0xffffffffu, hit);
    if (lane == 0) wbase[warp] = __popc(b);
    __syncthreads();
    if (threadIdx.x == 0) {
      int a = 0;
      for (int k = 0; k < nw; k++) {
        const int v = wbase[k];
        wbase[k] = a, a += v;
      }
      wbase[32] = a;
    }
    __syncthreads();
    if (pass && hit) {
      const int p = base + wbase[warp] + __popc(b & ((1u << lane) - 1));
      prow[p] = i / kTopK;
      pidx[i] = p;
    }
    base += wbase[32];
    __syncthreads();
  }
  if (!pass && threadIdx.x == 0) cnt[e] = base;
}

__global__ void k_scan(const int* cnt, int* off, int* act, int* nact) {
  if (threadIdx.x) return;
  int a = 0, n = 0;
  for (int e = 0; e < kExperts; e++) {
    off[e] = a, a += cnt[e];
    if (cnt[e]) act[n++] = e;
  }
  off[kExperts] = a;
  *nact = n;
}

// ------------------------------------------------------------------------------------------------------------ experts
__device__ __forceinline__ float situ(float g, float u) {
  return 4.0f * tanhf(g * 0.25f) * __fdiv_rn(1.0f, 1.0f + expf(-g)) * (25.0f * tanhf(__fdiv_rn(u, 25.0f)));
}

// up: grid (32 inter blocks, expert slots); MT pairs per tile
template <int MT>
__global__ void __launch_bounds__(128) k_up(const uint2* __restrict__ W, const uint8_t* __restrict__ WS,
                                            const uint8_t* __restrict__ xq, const uint8_t* __restrict__ xs,
                                            const int* __restrict__ off, const int* __restrict__ act,
                                            const int* __restrict__ nact, const int* __restrict__ prow, uint8_t* hq,
                                            uint8_t* hs) {
  if (blockIdx.y >= *nact) return;
  const int e = act[blockIdx.y], c = blockIdx.x;
  const int p0 = off[e], np = off[e + 1] - p0;
  extern __shared__ __align__(16) uint8_t smem[];
  constexpr int RB = kStage + kPad;
  uint8_t* sx = smem;                       // [2][MT][RB]
  uint8_t* ssc = smem + 2 * MT * RB;        // [2][MT][16]
  float* gu = reinterpret_cast<float*>(smem + 2 * MT * RB + 2 * MT * 16);  // [2][32][MT]
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const uint2* wt = W + (size_t)e * kUpWords + ((size_t)(c * 4 + warp) * kUpKB) * 32 + lane;
  const uint8_t* wst = WS + (size_t)e * kUpScales + ((size_t)(c * 4 + warp) * kUpKB) * 16 + g + 8 * (lane & 1);
  constexpr int nst = kLatent / kStage;  // 4
  for (int q0 = 0; q0 < np; q0 += MT) {
    auto load = [&](int st, int buf) {
      uint8_t* d = sx + buf * MT * RB;
      for (int i = threadIdx.x; i < MT * (kStage / 16); i += 128) {
        const int r = i / (kStage / 16), ch = i % (kStage / 16);
        const bool ok = q0 + r < np;
        const int row = ok ? prow[p0 + q0 + r] : 0;
        cp_async16(d + r * RB + ch * 16, xq + (size_t)row * kLatent + st * kStage + ch * 16, ok);
      }
      uint8_t* sd = ssc + buf * MT * 16;
      for (int i = threadIdx.x; i < MT; i += 128) {
        const bool ok = q0 + i < np;
        const int row = ok ? prow[p0 + q0 + i] : 0;
        cp_async16(sd + i * 16, xs + (size_t)row * (kLatent / 32) + st * 16, ok);
      }
      cp_async_commit();
    };
    float acc[MT / 8][4];
#pragma unroll
    for (int i = 0; i < MT / 8; i++) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
    load(0, 0);
    for (int st = 0; st < nst; st++) {
      const int buf = st & 1;
      uint2 wf[kStage / 32];
      uint32_t sf[kStage / 32];
#pragma unroll
      for (int kb = 0; kb < kStage / 32; kb++) {
        wf[kb] = __ldg(wt + (size_t)(st * (kStage / 32) + kb) * 32);
        sf[kb] = __ldg(wst + (size_t)(st * (kStage / 32) + kb) * 16);
      }
      if (st + 1 < nst) {
        load(st + 1, buf ^ 1);
        cp_async_wait<1>();
      } else {
        cp_async_wait<0>();
      }
      __syncthreads();
      const uint8_t* bx = sx + buf * MT * RB;
      const uint8_t* bs = ssc + buf * MT * 16;
#pragma unroll
      for (int kb = 0; kb < kStage / 32; kb++) {
        const uint32_t a0 = e2m1x4_expand(wf[kb].x & 0xFFFF), a1 = e2m1x4_expand(wf[kb].x >> 16);
        const uint32_t a2 = e2m1x4_expand(wf[kb].y & 0xFFFF), a3 = e2m1x4_expand(wf[kb].y >> 16);
#pragma unroll
        for (int nt = 0; nt < MT / 8; nt++) {
          const uint8_t* p = bx + (nt * 8 + g) * RB + kb * 32 + 4 * t;
          mma_e2m1_e4m3(acc[nt], a0, a1, a2, a3, *reinterpret_cast<const uint32_t*>(p),
                        *reinterpret_cast<const uint32_t*>(p + 16), sf[kb], bs[(nt * 8 + g) * 16 + kb]);
        }
      }
      __syncthreads();
    }
    // gate (warps 0, 1) / up (2, 3) values of inter dims 16·(warp & 1) + g / + 8 → shared [gate|up][32][MT]
    float* dst = gu + (warp >> 1) * 32 * MT;
    const int ra = 16 * (warp & 1) + g;
#pragma unroll
    for (int nt = 0; nt < MT / 8; nt++) {
      const int tk = nt * 8 + 2 * t;
      dst[ra * MT + tk] = acc[nt][0], dst[ra * MT + tk + 1] = acc[nt][1];
      dst[(ra + 8) * MT + tk] = acc[nt][2], dst[(ra + 8) * MT + tk + 1] = acc[nt][3];
    }
    __syncthreads();
    // h = SiTU(g, u), MXFP8 per pair over these 32 inter dims: one warp per pair, lane = dim
    for (int r = warp; r < MT && q0 + r < np; r += 4) {
      const float h = situ(gu[lane * MT + r], gu[32 * MT + lane * MT + r]);
      const uint32_t ex = act_scale_e8(warp_max(fabsf(h)), 1e-10f);
      const size_t p = (size_t)(p0 + q0 + r);
      hq[p * kInterR + 32 * c + lane] = uint8_t(quant_e4m3(h, ex));
      if (lane == 0) hs[p * (kInterR / 32) + c] = uint8_t(ex);
    }
    __syncthreads();
  }
}

// down: grid (32 latent blocks of 64 rows, expert slots) → y[pair][2048] fp32
template <int MT>
__global__ void __launch_bounds__(128) k_down(const uint2* __restrict__ W, const uint8_t* __restrict__ WS,
                                              const uint8_t* __restrict__ hq, const uint8_t* __restrict__ hs,
                                              const int* __restrict__ off, const int* __restrict__ act,
                                              const int* __restrict__ nact, float* y) {
  if (blockIdx.y >= *nact) return;
  const int e = act[blockIdx.y];
  const int p0 = off[e], np = off[e + 1] - p0;
  extern __shared__ __align__(16) uint8_t smem[];
  constexpr int RB = kStage + kPad;
  uint8_t* sx = smem;
  uint8_t* ssc = smem + 2 * MT * RB;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int T = blockIdx.x * 4 + warp;
  const uint2* wt = W + (size_t)e * kDnWords + ((size_t)T * kDnKB) * 32 + lane;
  const uint8_t* wst = WS + (size_t)e * kDnScales + ((size_t)T * kDnKB) * 16 + g + 8 * (lane & 1);
  constexpr int nst = kInterR / kStage;  // 2
  for (int q0 = 0; q0 < np; q0 += MT) {
    auto load = [&](int st, int buf) {
      uint8_t* d = sx + buf * MT * RB;
      for (int i = threadIdx.x; i < MT * (kStage / 16); i += 128) {
        const int r = i / (kStage / 16), ch = i % (kStage / 16);
        const bool ok = q0 + r < np;
        const size_t p = ok ? (size_t)(p0 + q0 + r) : 0;
        cp_async16(d + r * RB + ch * 16, hq + p * kInterR + st * kStage + ch * 16, ok);
      }
      uint8_t* sd = ssc + buf * MT * 16;
      for (int i = threadIdx.x; i < MT; i += 128) {
        const bool ok = q0 + i < np;
        const size_t p = ok ? (size_t)(p0 + q0 + i) : 0;
        cp_async16(sd + i * 16, hs + p * (kInterR / 32) + st * 16, ok);
      }
      cp_async_commit();
    };
    float acc[MT / 8][4];
#pragma unroll
    for (int i = 0; i < MT / 8; i++) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
    load(0, 0);
    for (int st = 0; st < nst; st++) {
      const int buf = st & 1;
      uint2 wf[kStage / 32];
      uint32_t sf[kStage / 32];
#pragma unroll
      for (int kb = 0; kb < kStage / 32; kb++) {
        wf[kb] = __ldg(wt + (size_t)(st * (kStage / 32) + kb) * 32);
        sf[kb] = __ldg(wst + (size_t)(st * (kStage / 32) + kb) * 16);
      }
      if (st + 1 < nst) {
        load(st + 1, buf ^ 1);
        cp_async_wait<1>();
      } else {
        cp_async_wait<0>();
      }
      __syncthreads();
      const uint8_t* bx = sx + buf * MT * RB;
      const uint8_t* bs = ssc + buf * MT * 16;
#pragma unroll
      for (int kb = 0; kb < kStage / 32; kb++) {
        const uint32_t a0 = e2m1x4_expand(wf[kb].x & 0xFFFF), a1 = e2m1x4_expand(wf[kb].x >> 16);
        const uint32_t a2 = e2m1x4_expand(wf[kb].y & 0xFFFF), a3 = e2m1x4_expand(wf[kb].y >> 16);
#pragma unroll
        for (int nt = 0; nt < MT / 8; nt++) {
          const uint8_t* p = bx + (nt * 8 + g) * RB + kb * 32 + 4 * t;
          mma_e2m1_e4m3(acc[nt], a0, a1, a2, a3, *reinterpret_cast<const uint32_t*>(p),
                        *reinterpret_cast<const uint32_t*>(p + 16), sf[kb], bs[(nt * 8 + g) * 16 + kb]);
        }
      }
      __syncthreads();
    }
    const int ra = 16 * T + g;
#pragma unroll
    for (int nt = 0; nt < MT / 8; nt++) {
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int q = q0 + nt * 8 + 2 * t + (i & 1), r = i < 2 ? ra : ra + 8;
        if (q < np) y[(size_t)(p0 + q) * kLatent + r] = acc[nt][i];
      }
    }
  }
}

// part[row][c] = Σ_slot w[row][slot] · y[pair(row, slot)][c], slot order (fp32; the bf16 form rounds that sum once)
__device__ __forceinline__ void put_out(float* p, float v) { *p = v; }
__device__ __forceinline__ void put_out(bf16* p, float v) { *p = __float2bfloat16_rn(v); }
template <class OutT>
__global__ void k_combine(const float* __restrict__ y, const int* __restrict__ pidx, const float* __restrict__ w,
                          OutT* part, int ld) {
  const int row = blockIdx.x;
  int p[kTopK];
  float wv[kTopK];
#pragma unroll
  for (int k = 0; k < kTopK; k++) p[k] = pidx[row * kTopK + k], wv[k] = w[row * kTopK + k];
  for (int c = threadIdx.x; c < kLatent; c += blockDim.x) {
    float a = 0.f;
#pragma unroll
    for (int k = 0; k < kTopK; k++) a += wv[k] * y[(size_t)p[k] * kLatent + c];
    put_out(part + (size_t)row * ld + c, a);
  }
}

template <int MT>
void run_experts(const MoeExperts* E, MoeScratch* S, const uint8_t* xq, const uint8_t* xs, int slots, cudaStream_t s) {
  constexpr int RB = kStage + kPad;
  const size_t sm_up = 2 * MT * RB + 2 * MT * 16 + 2 * 32 * MT * 4, sm_dn = 2 * MT * RB + 2 * MT * 16;
  static bool attr = false;
  if (!attr) {
    CUDA_CHECK(cudaFuncSetAttribute(k_up<MT>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(sm_up)));
    CUDA_CHECK(cudaFuncSetAttribute(k_down<MT>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(sm_dn)));
    attr = true;
  }
  k_up<MT><<<dim3(32, slots), 128, sm_up, s>>>(E->up, E->ups, xq, xs, S->off, S->act, S->nact, S->prow, S->hq, S->hs);
  CUDA_CHECK(cudaGetLastError());
  k_down<MT><<<dim3(kDnTiles / 4, slots), 128, sm_dn, s>>>(E->dn, E->dns, S->hq, S->hs, S->off, S->act, S->nact, S->y);
  CUDA_CHECK(cudaGetLastError());
}
}  // namespace

#include "kernels/moe/p035_moe_decode.cuh"  // P035 moe_decode: decode experts in two launches (same bits)
#include "kernels/moe/p036_route_fast.cuh"  // P036 route_fast: routing with a short critical path (same bits)
#include "kernels/moe/p037_moe_prefill.cuh"  // P037 moe_prefill: pipelined 256-row expert tiles (same bits)
#include "kernels/moe/p065_moe_combine.cuh"  // P065 moe_combine: prefill combine in the down epilogue (same bits)
#include "kernels/moe/p066_moe_fused.cuh"    // P066 moe_fused: decode route + latent quant + experts, one launch
#include "kernels/moe/p190_moe_fused_mid.cuh"  // P190 moe_fused64: P066's launch for 33 … 64 rows (same bits)
#include "kernels/moe/p194_moe_fused128.cuh"  // P194 moe_fused128: P190's items for 65 … 128 rows, two row halves (same bits)
#include "kernels/moe/p086_moe_dn.cuh"       // P086 moe_pf_dn: prefill down + combine, coalesced chains

void moe_route(const float* logits, const float* bias, const float* scale, int M, int* ids, float* w, cudaStream_t s) {
  if (M <= 0) return;
  if (patch::on(patch::route_fast)) {
    p036::k_route36<<<cdiv(M, 4), 128, 0, s>>>(logits, bias, scale, M, ids, w);
    CUDA_CHECK(cudaGetLastError());
    return;
  }
  k_route<<<cdiv(M, 4), 128, 0, s>>>(logits, bias, scale, M, ids, w);
  CUDA_CHECK(cudaGetLastError());
}

namespace {
template <class OutT>
void experts(const MoeExperts* E, MoeScratch* S, const uint8_t* lat_q, const uint8_t* lat_s, const int* ids,
             const float* w, int M, OutT* part, int ld, cudaStream_t s) {
  if (M <= 0) return;
  if (M > S->max_rows) qlog::fail("moe", "M %d > scratch %d", M, S->max_rows);
  if (M <= p035::kMaxM && patch::on(patch::moe_decode)) return p035::run(E, S, lat_q, lat_s, ids, w, M, part, ld, s);
  if (M > 64 && patch::on(patch::moe_pf_dn)) return p086::run(E, S, lat_q, lat_s, ids, w, M, part, ld, s);
  if (M > 64 && patch::on(patch::moe_combine)) return p065::run(E, S, lat_q, lat_s, ids, w, M, part, ld, s);
  const int P = M * kTopK;
  k_plan<<<kExperts, 256, 0, s>>>(ids, P, 0, S->cnt, S->off, S->prow, S->pidx);
  k_scan<<<1, 32, 0, s>>>(S->cnt, S->off, S->act, S->nact);
  k_plan<<<kExperts, 256, 0, s>>>(ids, P, 1, S->cnt, S->off, S->prow, S->pidx);
  CUDA_CHECK(cudaGetLastError());
  const int slots = std::min(kExperts, P);
  if (M <= 4) run_experts<8>(E, S, lat_q, lat_s, slots, s);
  else if (M <= 64) run_experts<16>(E, S, lat_q, lat_s, slots, s);
  else if (patch::on(patch::moe_prefill)) p037::run_mm(E, S, lat_q, lat_s, P, s);
  else run_experts<64>(E, S, lat_q, lat_s, slots, s);
  k_combine<<<M, 256, 0, s>>>(S->y, S->pidx, w, part, ld);
  CUDA_CHECK(cudaGetLastError());
}
}  // namespace

void moe_experts(const MoeExperts* E, MoeScratch* S, const uint8_t* lat_q, const uint8_t* lat_s, const int* ids,
                 const float* w, int M, float* part, int ld, cudaStream_t s) {
  experts(E, S, lat_q, lat_s, ids, w, M, part, ld, s);
}
// test only (tests/bind/moe.cpp): the routing plan alone — P065's two launches, or the base's four (k_plan ×2, k_scan,
// P037's k_tiles37)
void moe_test_plan(MoeScratch* S, const int* ids, int M, bool fast, cudaStream_t s) {
  const int P = M * kTopK;
  if (fast) {
    p065::k_plan65a<<<p065::kPC, 256, 0, s>>>(ids, P, S->phist);
    p065::k_plan65b<<<p065::kPC, kExperts, 0, s>>>(ids, P, S->phist, S->cnt, S->off, S->act, S->nact, S->prow, S->pidx,
                                                   S->pfl, S->tiles, S->ntiles);
  } else {
    k_plan<<<kExperts, 256, 0, s>>>(ids, P, 0, S->cnt, S->off, S->prow, S->pidx);
    k_scan<<<1, 32, 0, s>>>(S->cnt, S->off, S->act, S->nact);
    k_plan<<<kExperts, 256, 0, s>>>(ids, P, 1, S->cnt, S->off, S->prow, S->pidx);
    p037::k_tiles37<<<1, kExperts, 0, s>>>(S->off, S->tiles, S->ntiles);
  }
  CUDA_CHECK(cudaGetLastError());
}

// P066 up to 32 rows; P190 moe_fused64 (with P066) 33 … 64 rows (DSpark verify / draft steps of 5–8 sequences);
// P194 moe_fused128 (with P066) 65 … 128 rows (9–16 sequences)
bool moe_fused_ok(int M) {
  if (M < 1 || !patch::on(patch::moe_fused)) return false;
  if (M <= p066::kMaxM) return true;
  if (M <= p190::kMaxM) return patch::on(patch::moe_fused64);
  return M <= p194::kMaxM && patch::on(patch::moe_fused128);
}
namespace {
template <class OutT>
void fused(const MoeExperts* E, MoeScratch* S, const float* logits, const float* bias, const float* scale,
           const bf16* lat, int ldl, const uint8_t* lat_q, const uint8_t* lat_s, int M, int* ids, float* w, OutT* part,
           int ld, cudaStream_t s) {
  if (!moe_fused_ok(M) || !lat == !lat_q) qlog::fail("moe", "moe_fused: M %d", M);
  if (M <= p066::kMaxM) return p066::run(E, S, logits, bias, scale, lat, ldl, lat_q, lat_s, M, ids, w, part, ld, s);
  if (M <= p190::kMaxM) return p190::run(E, S, logits, bias, scale, lat, ldl, lat_q, lat_s, M, ids, w, part, ld, s);
  p194::run(E, S, logits, bias, scale, lat, ldl, lat_q, lat_s, M, ids, w, part, ld, s);  // P194
}
}  // namespace
void moe_fused(const MoeExperts* E, MoeScratch* S, const float* logits, const float* bias, const float* scale,
               const bf16* lat, int ldl, const uint8_t* lat_q, const uint8_t* lat_s, int M, int* ids, float* w,
               bf16* part, int ld, cudaStream_t s) {
  fused(E, S, logits, bias, scale, lat, ldl, lat_q, lat_s, M, ids, w, part, ld, s);
}
void moe_fused_f32(const MoeExperts* E, MoeScratch* S, const float* logits, const float* bias, const float* scale,
                   const bf16* lat, int ldl, const uint8_t* lat_q, const uint8_t* lat_s, int M, int* ids, float* w,
                   float* part, int ld, cudaStream_t s) {
  fused(E, S, logits, bias, scale, lat, ldl, lat_q, lat_s, M, ids, w, part, ld, s);
}
void moe_experts_bf16(const MoeExperts* E, MoeScratch* S, const uint8_t* lat_q, const uint8_t* lat_s, const int* ids,
                      const float* w, int M, bf16* part, int ld, cudaStream_t s) {
  experts(E, S, lat_q, lat_s, ids, w, M, part, ld, s);
}

}  // namespace e55
