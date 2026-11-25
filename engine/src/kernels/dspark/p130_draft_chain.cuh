// p130_draft_chain.cuh — P130 draft_chain (patches.def; included by dspark.cu inside namespace e55): the greedy Markov
// chain of the DSpark draft as one launch and one all-reduce a step, instead of k_markov + k_argmax + all-reduce + k_pick.
//   - the vocabulary sweep is k_markov<true>'s (P082 markov8: w2 as e4m3 + UE8M0), element for element: the same
//     products, the same fma chain over a lane's 32 elements, the same xor-shuffle sum, multimax and first-maximum rule
//   - the previous step's pick (k_pick's rule on its all-reduced candidates) is taken in the sweep's prologue — every
//     CTA for its own items, CTA 0 also writes the drafts of that step — so no launch sits between the all-reduce and
//     the sweep; the last step's pick stays dsp_pick
//   - the last CTA to finish (a counter per step, reset by that CTA) reduces the CTAs' (value, id) partials of each item
//     and writes this rank's candidate into the step's exchange rows as k_argmax does. better() is a total order on
//     (value, id), so the maximum is the same in any reduction order
//   - odd steps sweep the vocabulary blocks in reverse CTA order: a step starts on the w2 rows the previous step read
//     last, part of which L2 still holds (26 MB a sweep, 24 MB of L2; one GPU, 7 steps: 85 → 60 µs a step)
//   - launched as a programmatic dependent: each thread requests its first pass's w2 bytes and scale (written once at
//     load) before griddepcontrol.wait, so the sweep's first wave streams while the previous step's all-reduce (P077:
//     it lets its dependent launch right away) waits on the network; the chain's inputs (exchange rows, base logits)
//     after the wait; each next pass's bytes are requested while the current pass is computed
// Same drafts and confidences as the base chain.
#pragma once

namespace p130 {

constexpr int kRows = kMkRows;  // 128 vocabulary rows per CTA, as k_markov
constexpr int kNT = 256;

__device__ __forceinline__ int pick(const float* ex, int K, int k) {  // k_pick's rule
  const float v0 = ex[2 * k], v1 = ex[(K + k) * 2];
  const int i0 = int(ex[2 * k + 1]), i1 = int(ex[(K + k) * 2 + 1]);
  return i0 < 0 ? i1 : i1 < 0 ? i0 : better(v1, i1, v0, i0) ? i1 : i0;
}

struct Args {
  const float* base;  // [*, Vh] this rank's base logits
  const int *brow, *seq;  // this step's items: base row, sequence (batch index)
  int K;
  const int* pseq;    // the previous step's items' sequences (step > 0)
  int pK;
  const float* pex;   // the previous step's all-reduced exchange rows [2][pK][2] (step > 0), else null
  const int* last;    // [B] the anchors' tokens (step 0's previous token)
  int* drafts;        // [B][kDspMaxN]
  int step;
  const bf16* w1t;
  const uint8_t *w2q, *w2s;
  int Vh, off;
  const float* mm;
  float2* part;       // [K][gridDim.x]
  unsigned* cnt;      // this step's counter (0 between launches)
  int rank;
  float* ex;          // this step's exchange rows [2][K][2]
};

__global__ void __launch_bounds__(kNT, 3) k_chain(Args a) {
  __shared__ float e[kDspMaxN][8 * 33];
  __shared__ float2 red[8][kDspMaxN];
  __shared__ float rt[8];
  __shared__ int tok[kDspMaxN];
  __shared__ float sv[kNT / 32];
  __shared__ int si[kNT / 32];
  __shared__ int s_last;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gi = lane >> 3, li = lane & 7;
  // odd steps sweep the vocabulary blocks backwards: they start on the w2 rows the previous step read last
  const int nb = gridDim.x, vb = (a.step & 1) ? nb - 1 - int(blockIdx.x) : int(blockIdx.x), v0 = vb * kRows;
  // this thread's w2 bytes of a pass (independent of the chain): pass 0's requested before the wait, each next pass's
  // while the current one is computed
  uint4 wq[2][2];
  uint32_t scb[2];
  auto load = [&](int pass, int buf) {
    const int v = v0 + pass * 32 + warp * 4 + gi;
    if (v < a.Vh) {
      const uint4* src = reinterpret_cast<const uint4*>(a.w2q + (size_t)v * kDspRank + 32 * li);
      wq[buf][0] = src[0], wq[buf][1] = src[1];
      scb[buf] = a.w2s[(size_t)v * (kDspRank / 32) + li];
    } else {
      wq[buf][0] = wq[buf][1] = make_uint4(0, 0, 0, 0), scb[buf] = 0;
    }
  };
  load(0, 0);
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // the next (the all-reduce) waits for our end
  if (tid < a.K) {  // the previous token of each item: the anchor (step 0) or the previous step's pick
    const int b = a.seq[tid];
    int t;
    if (!a.pex) {
      t = a.last[b];
    } else {
      int kp = 0;
      while (kp < a.pK - 1 && a.pseq[kp] != b) kp++;
      t = pick(a.pex, a.pK, kp);
    }
    tok[tid] = t;
  }
  if (a.pex && blockIdx.x == 0 && tid < a.pK) a.drafts[a.pseq[tid] * kDspMaxN + a.step - 1] = pick(a.pex, a.pK, tid);
  if (tid < 8) rt[tid] = a.mm[tid];
  __syncthreads();
  for (int i = tid; i < a.K * kDspRank; i += kNT) {
    const int k = i / kDspRank, c = i % kDspRank;
    e[k][(c >> 5) * 33 + (c & 31)] = __bfloat162float(a.w1t[(size_t)tok[k] * kDspRank + c]);
  }
  __syncthreads();
  float bv[2] = {-INFINITY, -INFINITY};
  int bi[2] = {INT_MAX, INT_MAX};
#pragma unroll
  for (int pass = 0; pass < kRows / 32; pass++) {
    const int v = v0 + pass * 32 + warp * 4 + gi, cb = pass & 1;
    if (pass + 1 < kRows / 32) load(pass + 1, cb ^ 1);
    float w[32];
    if (v < a.Vh) {
      const float sc = ue8m0_f32(scb[cb]);
#pragma unroll
      for (int q2 = 0; q2 < 2; q2++) {
        const uint4 u = wq[cb][q2];
        const uint32_t x[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int j = 0; j < 16; j++) {
          __nv_fp8_e4m3 f;
          f.__x = (x[j >> 2] >> (8 * (j & 3))) & 0xFF;
          w[16 * q2 + j] = float(f) * sc;
        }
      }
    } else {
#pragma unroll
      for (int j = 0; j < 32; j++) w[j] = 0.f;
    }
#pragma unroll
    for (int k = 0; k < kDspMaxN; k++) {
      if (k >= a.K) break;
      float acc = 0.f;
#pragma unroll
      for (int j = 0; j < 32; j++) acc = fmaf(w[j], e[k][33 * li + j], acc);
      acc += __shfl_xor_sync(0xffffffffu, acc, 1);
      acc += __shfl_xor_sync(0xffffffffu, acc, 2);
      acc += __shfl_xor_sync(0xffffffffu, acc, 4);
      if ((k & 7) == li && v < a.Vh) {
        const float y = mmax(__fadd_rn(a.base[(size_t)a.brow[k] * a.Vh + v], acc), rt);
        if (better(y, a.off + v, bv[k >> 3], bi[k >> 3])) bv[k >> 3] = y, bi[k >> 3] = a.off + v;
      }
    }
  }
#pragma unroll
  for (int h = 0; h < 2; h++)
#pragma unroll
    for (int o = 8; o < 32; o <<= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv[h], o);
      const int oi = __shfl_xor_sync(0xffffffffu, bi[h], o);
      if (better(ov, oi, bv[h], bi[h])) bv[h] = ov, bi[h] = oi;
    }
  if (gi == 0)
#pragma unroll
    for (int h = 0; h < 2; h++)
      if (8 * h + li < a.K) red[warp][8 * h + li] = make_float2(bv[h], __int_as_float(bi[h]));
  __syncthreads();
  if (tid < a.K) {
    float v = red[0][tid].x;
    int i = __float_as_int(red[0][tid].y);
    for (int w = 1; w < 8; w++) {
      const float ov = red[w][tid].x;
      const int oi = __float_as_int(red[w][tid].y);
      if (better(ov, oi, v, i)) v = ov, i = oi;
    }
    a.part[(size_t)tid * nb + vb] = make_float2(v, __int_as_float(i));
    __threadfence();
  }
  __syncthreads();
  if (tid == 0) s_last = atomicAdd(a.cnt, 1u) == unsigned(nb - 1);
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  for (int k = 0; k < a.K; k++) {  // k_argmax over the CTAs' partials
    float v = -INFINITY;
    int i = INT_MAX;
    for (int b = tid; b < nb; b += kNT) {
      const float2 p = __ldcg(a.part + (size_t)k * nb + b);
      if (better(p.x, __float_as_int(p.y), v, i)) v = p.x, i = __float_as_int(p.y);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, v, o);
      const int oi = __shfl_xor_sync(0xffffffffu, i, o);
      if (better(ov, oi, v, i)) v = ov, i = oi;
    }
    if (lane == 0) sv[warp] = v, si[warp] = i;
    __syncthreads();
    if (tid == 0) {
      for (int w = 1; w < kNT / 32; w++)
        if (better(sv[w], si[w], v, i)) v = sv[w], i = si[w];
      float* me = a.ex + ((size_t)a.rank * a.K + k) * 2;
      float* other = a.ex + ((size_t)(1 - a.rank) * a.K + k) * 2;
      me[0] = v, me[1] = float(i == INT_MAX ? -1 : i);
      other[0] = 0.f, other[1] = 0.f;
    }
    __syncthreads();
  }
  if (tid == 0) *a.cnt = 0u;
}

inline void launch(const Args& a, int nb, cudaStream_t s) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(nb);
  cfg.blockDim = dim3(kNT);
  cfg.stream = s;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, k_chain, a));
}

}  // namespace p130
