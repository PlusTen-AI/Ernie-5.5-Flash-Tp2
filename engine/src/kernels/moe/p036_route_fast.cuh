// p036_route_fast.cuh — P036 route_fast: moe_route with a short critical path, included by moe.cu (bitwise k_route).
// One warp per row as k_route, but: lane L holds experts 16L … 16L+15 (vector loads); each of the 10 argmax rounds is
// two warp reductions (redux.sync max of the order-preserving key of s + bias, then min of the ids holding it) and
// only the winning lane rescans its 16 values; the topk(sorted=False) slots and the 16-slot tree sum are a shared-
// memory scatter and the xor 8, 4, 2, 1 butterfly — the tree tl.sum uses (k_route's order, so the same bits).
// Same sigmoid / sum / division / scale expressions as k_route; ties → lower id as k_route (−0 never occurs: s > 0).
namespace p036 {

__device__ __forceinline__ uint32_t okey(float x) {  // order-preserving, > 0 for every non-NaN float
  const uint32_t b = __float_as_uint(x);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__global__ void __launch_bounds__(128) k_route36(const float* __restrict__ lg, const float* __restrict__ bias,
                                                 const float* __restrict__ scale, int M, int* ids, float* w) {
  __shared__ float sa[4][16];
  const int wi = threadIdx.x >> 5, row = blockIdx.x * 4 + wi, lane = threadIdx.x & 31;
  if (row >= M) return;
  const float4* l0 = reinterpret_cast<const float4*>(lg + (size_t)row * 1024 + 16 * lane);
  const float4* l1 = reinterpret_cast<const float4*>(lg + (size_t)row * 1024 + 512 + 16 * lane);
  const float4* bb = reinterpret_cast<const float4*>(bias + 16 * lane);
  float s[16];
  uint32_t key[16];
#pragma unroll
  for (int q = 0; q < 4; q++) {
    const float4 a = l0[q], b = l1[q], c = bb[q];
    const float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w}, cv[4] = {c.x, c.y, c.z, c.w};
#pragma unroll
    for (int i = 0; i < 4; i++) {
      s[4 * q + i] = sigmoid_ieee(av[i]) + sigmoid_ieee(bv[i]);
      key[4 * q + i] = okey(s[4 * q + i] + cv[i]);
    }
  }
  auto local_best = [&](uint32_t& bk, int& bj) {
    bk = 0, bj = 0;
#pragma unroll
    for (int j = 0; j < 16; j++)
      if (key[j] > bk) bk = key[j], bj = j;  // strict: the lower id of a tie
  };
  uint32_t bk;
  int bj;
  local_best(bk, bj);
  int my_id = 0;  // lane k < 10 keeps selection k
  uint32_t my_v = 0;
  float my_s = 0.f;
#pragma unroll 1
  for (int k = 0; k < kTopK; k++) {
    const uint32_t vmax = __reduce_max_sync(0xffffffffu, bk);
    const int id = int(__reduce_min_sync(0xffffffffu, bk == vmax ? uint32_t(16 * lane + bj) : 0xffffu));
    float park = 0.f;
    if (lane == (id >> 4)) {  // the owner takes the winner out and rescans
#pragma unroll
      for (int j = 0; j < 16; j++)
        if (j == bj) park = s[j], key[j] = 0;
      local_best(bk, bj);
    }
    const float sv = __shfl_sync(0xffffffffu, park, id >> 4);
    if (lane == k) my_id = id, my_v = vmax, my_s = sv;
  }
  // topk(sorted=False) slot of selection `lane`: above the 10th value by id, then equal to it by id
  const uint32_t v10 = __shfl_sync(0xffffffffu, my_v, kTopK - 1);
  const int kk = my_v > v10 ? my_id : my_id + kExperts;
  int slot = 0;
#pragma unroll
  for (int j = 0; j < kTopK; j++) {
    const int kj = __shfl_sync(0xffffffffu, kk, j);
    slot += kj < kk;
  }
  if (lane < 16) sa[wi][lane] = 0.f;
  __syncwarp();
  if (lane < kTopK) sa[wi][slot] = my_s;
  __syncwarp();
  const int sid = __shfl_sync(0xffffffffu, my_id, [&] {  // lane L ← id of the selection in slot L
    int src = 0;
#pragma unroll
    for (int j = 0; j < kTopK; j++)
      if (__shfl_sync(0xffffffffu, slot, j) == lane) src = j;
    return src;
  }());
  const float a = sa[wi][lane & 15];
  float t = a + __shfl_xor_sync(0xffffffffu, a, 8);
  t = t + __shfl_xor_sync(0xffffffffu, t, 4);
  t = t + __shfl_xor_sync(0xffffffffu, t, 2);
  t = t + __shfl_xor_sync(0xffffffffu, t, 1);
  const float tot = fmaxf(t, 1e-20f);
  if (lane < kTopK) {
    ids[(size_t)row * kTopK + lane] = sid;
    w[(size_t)row * kTopK + lane] = __fdiv_rn(a, tot) * scale[sid];
  }
}

}  // namespace p036
