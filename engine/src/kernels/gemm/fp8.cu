// fp8.cu — ERNIE FP8 block linears on block-scaled FP8 tensor cores (contract: fp8.h).
#include "kernels/gemm/fp8.h"

#include <cooperative_groups.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cstring>
#include <type_traits>

#include "core/log.h"
#include "kernels/gemm/mma.cuh"
#include "model/patches.h"

namespace e55 {
namespace {

constexpr int kCtaRows = 64;  // 4 warps × one 16-row weight tile
constexpr int kStage = 256;   // k bytes per shared-memory stage (8 k32 steps, 2 scale groups)
constexpr int kPad = 16;      // row padding of the staged activations (conflict-free fragment reads)

struct Args {
  const uint4* w;
  const uint8_t* ws;
  int n, k, split, rows_per_group;
  const uint8_t* xq;
  const uint8_t* xs;
  int ld;  // activation row stride (bytes); scales per row = ld / 128
  long gx, gs;
  int M;
  void* y;
  int ldy;
  float* part;  // split > 1: [split][M][n]
};

template <int MT, bool F32>
__global__ void __launch_bounds__(128) k_f8(Args a) {
  extern __shared__ __align__(16) uint8_t smem[];
  constexpr int RB = kStage + kPad;            // staged row bytes
  uint8_t* sx = smem;                          // [2][MT][RB]
  uint8_t* ssc = smem + 2 * MT * RB;           // [2][MT][2] scales of the stage's two groups
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int T = blockIdx.x * 4 + warp;         // this warp's 16-row weight tile
  const int r0 = blockIdx.x * kCtaRows;
  const int grp = r0 / a.rows_per_group;
  const uint8_t* xq = a.xq + grp * a.gx;
  const uint8_t* xs = a.xs + grp * a.gs;
  const int t0 = blockIdx.y * MT;
  const int KB = a.k >> 5, KG = a.k >> 7;
  const int kz = blockIdx.z, klen = a.k / a.split, kbeg = kz * klen;
  const int nst = klen / kStage;
  const int sld = a.ld >> 7;

  auto load_stage = [&](int st, int buf) {
    const int kk = kbeg + st * kStage;
    uint8_t* dst = sx + buf * MT * RB;
    for (int c = threadIdx.x; c < MT * (kStage / 16); c += 128) {
      const int row = c / (kStage / 16), ch = c % (kStage / 16);
      const int tok = t0 + row;
      const bool ok = tok < a.M;
      const uint8_t* src = xq + (size_t)(ok ? tok : 0) * a.ld + kk + ch * 16;
      cp_async16(dst + row * RB + ch * 16, src, ok);
    }
    uint8_t* sd = ssc + buf * MT * 2;
    for (int c = threadIdx.x; c < MT * 2; c += 128) {
      const int row = c >> 1, j = c & 1, tok = t0 + row;
      sd[c] = tok < a.M ? xs[(size_t)tok * sld + (kk >> 7) + j] : 0;
    }
    cp_async_commit();
  };

  float acc[MT / 8][4];
#pragma unroll
  for (int i = 0; i < MT / 8; i++) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;

  const uint4* wt = a.w + ((size_t)T * KB + (kbeg >> 5)) * 32 + lane;
  const uint8_t* wst = a.ws + (size_t)T * KG;
  uint4 wn = __ldg(wt);
  load_stage(0, 0);
  for (int st = 0; st < nst; st++) {
    const int buf = st & 1;
    if (st + 1 < nst) {
      load_stage(st + 1, buf ^ 1);
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();
    const uint8_t* bx = sx + buf * MT * RB;
    const uint8_t* bs = ssc + buf * MT * 2;
#pragma unroll
    for (int kb = 0; kb < kStage / 32; kb++) {
      const uint4 wa = wn;
      const int kbg = (kbeg >> 5) + st * (kStage / 32) + kb;  // global k32 index
      if (kbg + 1 < (kbeg >> 5) + (klen >> 5)) wn = __ldg(wt + (size_t)(st * (kStage / 32) + kb + 1) * 32);
      const uint32_t sfa = wst[kbg >> 2];
#pragma unroll
      for (int nt = 0; nt < MT / 8; nt++) {
        const uint8_t* p = bx + (nt * 8 + g) * RB + kb * 32 + 4 * t;
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(p);
        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(p + 16);
        const uint32_t sfb = bs[(nt * 8 + g) * 2 + (kb >> 2)];
        mma_e4m3_e4m3(acc[nt], wa.x, wa.y, wa.z, wa.w, b0, b1, sfa, sfb);
      }
    }
    __syncthreads();
  }
  // D[row g / g+8][token 2t / 2t+1] → y[token][row]
  const int ra = 16 * T + g, rb = ra + 8;
#pragma unroll
  for (int nt = 0; nt < MT / 8; nt++) {
    const int tk = t0 + nt * 8 + 2 * t;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int r = i < 2 ? ra : rb, m = tk + (i & 1);
      if (r >= a.n || m >= a.M) continue;
      if (a.split > 1) {
        a.part[((size_t)kz * a.M + m) * a.n + r] = acc[nt][i];
      } else if (F32) {
        reinterpret_cast<float*>(a.y)[(size_t)m * a.ldy + r] = acc[nt][i];
      } else {
        reinterpret_cast<bf16*>(a.y)[(size_t)m * a.ldy + r] = __float2bfloat16_rn(acc[nt][i]);
      }
    }
  }
}

template <bool F32>
__global__ void k_reduce(const float* part, int split, int M, int n, void* y, int ldy) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)M * n) return;
  const int m = int(i / n), r = int(i % n);
  float v = part[i];
  for (int z = 1; z < split; z++) v += part[(size_t)z * M * n + i];
  if (F32) reinterpret_cast<float*>(y)[(size_t)m * ldy + r] = v;
  else reinterpret_cast<bf16*>(y)[(size_t)m * ldy + r] = __float2bfloat16_rn(v);
}

// one thread per 4 consecutive elements; a group of G/4 threads (a whole warp for G = 128) shares the scale
__global__ void k_act_quant(const bf16* x, int ldx, int M, int k, int G, uint8_t* q, uint8_t* sc, int ld) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int q4 = k >> 2;
  const bool ok = i < (size_t)M * q4;
  const int row = ok ? int(i / q4) : 0, c = ok ? int(i % q4) * 4 : 0;
  float v[4] = {0.f, 0.f, 0.f, 0.f};
  if (ok) {
    const uint2 u = *reinterpret_cast<const uint2*>(x + (size_t)row * ldx + c);
    v[0] = bf16_lo(u.x), v[1] = bf16_hi(u.x), v[2] = bf16_lo(u.y), v[3] = bf16_hi(u.y);
  }
  float m = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
  for (int o = (G >> 3); o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  const uint32_t e = act_scale_e8(m, 1e-10f);
  if (!ok) return;
  uint32_t out = 0;
#pragma unroll
  for (int j = 0; j < 4; j++) out |= quant_e4m3(v[j], e) << (8 * j);
  *reinterpret_cast<uint32_t*>(q + (size_t)row * ld + c) = out;
  if ((c % G) == 0) sc[(size_t)row * (ld / G) + c / G] = uint8_t(e);
}

int pick_split(int npad, int k) {
  const int stages = k / kStage, ctas = npad / kCtaRows;
  for (int d = 1; d <= std::min(16, stages); d++)
    if (stages % d == 0 && ctas * d >= 96) return d;
  int best = 1;
  for (int d = 1; d <= std::min(16, stages); d++)
    if (stages % d == 0) best = d;
  return best;
}

template <int MT, bool F32>
void launch(const F8Lin& L, const Args& a, cudaStream_t s) {
  const size_t sm = 2 * MT * (kStage + kPad) + 2 * MT * 2;
  static bool attr = false;
  if (!attr) {
    CUDA_CHECK(cudaFuncSetAttribute(k_f8<MT, F32>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(sm)));
    attr = true;
  }
  dim3 grid(L.npad / kCtaRows, cdiv(a.M, MT), L.split);
  k_f8<MT, F32><<<grid, 128, sm, s>>>(a);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace
void act_quant(const bf16* x, int ldx, int M, int k, int group, F8Act a, cudaStream_t s);
namespace {
#include "kernels/gemm/p060_fuse_ops.cuh"  // P060 f8_fuse_ops: device helpers, the M > 8 prep / epilogue kernels
#include "kernels/gemm/p030_f8_decode.cuh"  // P030 f8_decode (M ≤ 32) + P032 f8_fuse_act + P060 / P064 in its kernel
#include "kernels/gemm/p031_f8_prefill.cuh"  // P031 f8_prefill (M > 32)
#include "kernels/gemm/p103_f8_tile.cuh"  // P103 f8_tile (M ≥ 256) + P104 f8_tile_epi in its store
#include "kernels/gemm/p133_f8_seq.cuh"   // P133 f8_tile_seq (M ≥ 256, K split 2 / 4: chains in one CTA)

// P133: does P133 take weight L at M rows with activation x (patch on, the shape runs and its measured choice)?
bool seq_takes(const F8Lin& L, int M, const F8Act& x, long gx, long gs) {
  return patch::on(patch::f8_tile_seq) && p133::ok(L, M, x, gx, gs) && p133::cfg_of(L, M) >= 0;
}

template <bool F32>
void forward(const F8Lin& L, const F8Act& x0, int M, void* y, int ldy, float* ws, cudaStream_t s, long gx, long gs) {
  if (M <= 0) return;
  F8Act x = x0;
  const bool dec = patch::on(patch::f8_decode);
  if (x.x) {  // P032: unquantized activation — fused into the decode kernel, else act_quant into x.q / x.s first
    if (L.groups > 1) qlog::fail("gemm", "a grouped GEMM takes a quantized activation");
    if (dec && p030::fuse_ok(L, M)) return p030::forward<F32>(L, x, M, y, ldy, s, gx, gs);
    if (x.pro) qlog::fail("gemm", "P060 prologue op %d without the decode kernel (M %d, %d×%d)", x.pro, M, L.n, L.k);
    act_quant(x.x, x.ldx, M, L.k, 128, x, s);
    x.x = nullptr;
  }
  if (dec && p030::ok(L, M)) return p030::forward<F32>(L, x, M, y, ldy, s, gx, gs);
  if (seq_takes(L, M, x, gx, gs)) return p133::forward<F32>(L, x, M, y, ldy, s, gx, gs);
  if (patch::on(patch::f8_tile) && p103::ok(L, M, x, gx, gs, ws) && p103::prefer(L, M))
    return p103::forward<F32>(L, x, M, y, ldy, ws, s, gx, gs);
  if (patch::on(patch::f8_prefill) && p031::ok(L, M)) return p031::forward<F32>(L, x, M, y, ldy, s, gx, gs);
  Args a{reinterpret_cast<const uint4*>(L.w), L.s, L.n, L.k, L.split, L.n / L.groups, x.q, x.s, x.ld, gx, gs, M,
         y, ldy, ws};
  if (L.split > 1 && !ws) qlog::fail("gemm", "split %d needs a workspace", L.split);
  if (M <= 8) launch<8, F32>(L, a, s);
  else if (M <= 16) launch<16, F32>(L, a, s);
  else if (M <= 32) launch<32, F32>(L, a, s);
  else if (M <= 64) launch<64, F32>(L, a, s);
  else launch<128, F32>(L, a, s);
  if (L.split > 1) {
    const size_t tot = (size_t)M * L.n;
    k_reduce<F32><<<unsigned((tot + 255) / 256), 256, 0, s>>>(ws, L.split, M, L.n, y, ldy);
    CUDA_CHECK(cudaGetLastError());
  }
}

// P104: may epilogue o.epi of weight L run in P103's store — or P133's (P104 on, P133 takes the shape) — (patches on,
// the kernel takes the shape; x: the activation, or nullptr for the shape alone)?
bool tile_epi(const F8Lin& L, int M, const F8Out& o, const F8Act* x, long gx, long gs, const float* ws) {
  if (!patch::on(patch::f8_tile_epi)) return false;
  F8Act d{};
  d.q = reinterpret_cast<uint8_t*>(uintptr_t(256)), d.s = d.q, d.ld = L.k;  // aligned stand-in
  if (x && (x->x || x->pro)) return false;
  if (p133::epi_ok(L, o.epi) && seq_takes(L, M, x ? *x : d, gx, gs)) return true;
  if (o.epi == kEpiSituQ128) return false;  // (P133 only)
  if (!patch::on(patch::f8_tile) || !p103::epi_ok(L, o.epi, o.e, o.lde)) return false;
  if (o.epi == kEpiMulQ128 && ((o.e == nullptr && x) || M < 2048)) return false;  // (see p103::epi_ok)
  return p103::ok(L, M, x ? *x : d, gx, gs, ws);
}
// a GEMM on the tiled kernels with its epilogue o (if any) in the store (tile_epi said yes) or a P105 chain member:
// P133 where it takes the shape (and the epilogue), else P103
void tiled(const F8Lin& L, const F8Act& x, int M, bf16* y, int ldy, float* ws, cudaStream_t s, long gx, long gs,
           const F8Out* o, bool late) {
  if (seq_takes(L, M, x, gx, gs) && (!o || p133::epi_ok(L, o->epi)))
    return p133::forward<false>(L, x, M, y, ldy, s, gx, gs, o, late);
  p103::forward<false>(L, x, M, y, ldy, ws, s, gx, gs, o, late);
}

}  // namespace

F8Lin f8_make(int n, int k, const uint8_t* w, const std::vector<const float*>& s, const std::vector<int>& parts,
              int groups, int split) {
  if (n % 16 || k % kStage) qlog::fail("gemm", "FP8 shape %d×%d not supported (n %% 16, k %% %d)", n, k, kStage);
  if (groups > 1 && (n % groups || (n / groups) % kCtaRows)) qlog::fail("gemm", "groups %d of %d rows", groups, n);
  F8Lin L;
  L.n = n, L.k = k, L.groups = groups, L.npad = cdiv(n, kCtaRows) * kCtaRows;
  L.split = split > 0 ? split : pick_split(L.npad, k);
  const int KB = k / 32, KG = k / 128, tiles = L.npad / 16;
  std::vector<uint8_t> pw((size_t)tiles * KB * 512, 0), ps((size_t)tiles * KG, 0);
  int p = 0, pstart = 0;
  for (int T = 0; T < tiles; T++) {
    const int r0 = 16 * T;
    if (r0 >= n) break;
    while (r0 >= pstart + parts[p]) pstart += parts[p++];
    if (r0 + 16 > pstart + parts[p]) qlog::fail("gemm", "part boundary inside a 16-row tile");
    for (int kg = 0; kg < KG; kg++) {
      const float f = s[p][((r0 - pstart) / 128) * KG + kg];
      uint32_t b;
      std::memcpy(&b, &f, 4);
      if ((b & 0x807FFFFFu) || !(b >> 23)) qlog::fail("gemm", "block scale %g is not a positive power of two", f);
      ps[(size_t)T * KG + kg] = uint8_t(b >> 23);
    }
    for (int kb = 0; kb < KB; kb++) {
      uint8_t* blk = pw.data() + ((size_t)T * KB + kb) * 512;
      for (int lane = 0; lane < 32; lane++) {
        const int g = lane >> 2, t = lane & 3;
        const uint8_t* ra = w + (size_t)(r0 + g) * k + kb * 32;
        const uint8_t* rb = w + (size_t)(r0 + g + 8) * k + kb * 32;
        uint8_t* d = blk + lane * 16;
        std::memcpy(d, ra + 4 * t, 4);
        std::memcpy(d + 4, rb + 4 * t, 4);
        std::memcpy(d + 8, ra + 16 + 4 * t, 4);
        std::memcpy(d + 12, rb + 16 + 4 * t, 4);
      }
    }
  }
  CUDA_CHECK(cudaMalloc(&L.w, pw.size()));
  CUDA_CHECK(cudaMemcpy(L.w, pw.data(), pw.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&L.s, ps.size()));
  CUDA_CHECK(cudaMemcpy(L.s, ps.data(), ps.size(), cudaMemcpyHostToDevice));
  return L;
}

F8Lin f8_upload(const Weights& W, const std::vector<std::string>& names, int groups) {
  std::vector<uint8_t> wb;
  std::vector<std::vector<uint8_t>> sb;
  std::vector<int> parts;
  int k = -1, n = 0;
  for (const auto& name : names) {
    const bool bare = name.size() >= 19 && name.compare(name.size() - 19, 19, "linear_o_group_proj") == 0;
    const std::string wn = bare ? name : name + ".weight", sn = bare ? name + "_scale_inv" : name + ".weight_scale_inv";
    const auto& ti = W.info(wn);
    if (ti.dtype != "F8_E4M3") qlog::fail("gemm", "%s: %s is not FP8", wn.c_str(), ti.dtype.c_str());
    std::vector<int64_t> sh, ss;
    auto b = W.slice(wn, &sh);
    auto sc = W.slice(sn, &ss);
    if (sh.size() != 2 || (k >= 0 && sh[1] != k)) qlog::fail("gemm", "%s: shape mismatch", wn.c_str());
    if (ss.size() != 2 || ss[0] != (sh[0] + 127) / 128 || ss[1] != sh[1] / 128)
      qlog::fail("gemm", "%s: scale shape mismatch", sn.c_str());
    k = int(sh[1]), n += int(sh[0]);
    parts.push_back(int(sh[0]));
    wb.insert(wb.end(), b.begin(), b.end());
    sb.push_back(std::move(sc));
  }
  std::vector<const float*> sp;
  for (auto& v : sb) sp.push_back(reinterpret_cast<const float*>(v.data()));
  return f8_make(n, k, wb.data(), sp, parts, groups);
}

void f8_free(F8Lin& L) {
  cudaFree(L.w), cudaFree(L.s);
  L = F8Lin{};
}

void act_quant(const bf16* x, int ldx, int M, int k, int group, F8Act a, cudaStream_t s) {
  if (M <= 0) return;
  if ((group != 32 && group != 128) || k % 128) qlog::fail("gemm", "act_quant group %d k %d", group, k);
  const size_t thr = (size_t)M * (k / 4);
  k_act_quant<<<unsigned((thr + 255) / 256), 256, 0, s>>>(x, ldx, M, k, group, a.q, a.s, a.ld);
  CUDA_CHECK(cudaGetLastError());
}

void f8_forward(const F8Lin& L, const F8Act& a, int M, bf16* y, int ldy, float* ws, cudaStream_t s, long gx, long gs) {
  forward<false>(L, a, M, y, ldy, ws, s, gx, gs);
}
void f8_forward_f32(const F8Lin& L, const F8Act& a, int M, float* y, int ldy, float* ws, cudaStream_t s, long gx,
                    long gs) {
  forward<true>(L, a, M, y, ldy, ws, s, gx, gs);
}
size_t f8_ws(const F8Lin& L, int M) { return L.split > 1 ? (size_t)L.split * M * L.n * 4 : 0; }
// P060: one or more weights on one activation, with the activation op (F8Act::pro), the side RMSNorm (aux) and each
// weight's epilogue (F8Out::epi) — at M ≤ 8 all inside the decode launch; else the op + quantization (and aux) in one
// launch first (x.q / x.s / x.ld must be the caller's buffers), the epilogues in p030's store (M ≤ 32) or one launch
// each after the GEMM. Without any of them: P034's one launch, else one f8_forward per weight (as before).
void f8_forward_multi(std::initializer_list<F8Out> outs, const F8Act& a, int M, float* ws, cudaStream_t s,
                      const F8Aux* aux) {
  const int n = int(outs.size());
  if (M <= 0 || n < 1) return;
  if (n > p030::kParts) qlog::fail("gemm", "f8_forward_multi: %d weights", n);
  const F8Out* ob = outs.begin();
  const F8Lin* Ls[p030::kParts];
  void* ys[p030::kParts];
  int ldys[p030::kParts];
  bool epi = false;
  for (int i = 0; i < n; i++) {
    Ls[i] = ob[i].L, ys[i] = ob[i].y, ldys[i] = ob[i].ldy, epi |= ob[i].epi != kEpiNone;
    if (ob[i].epi == kEpiQ32 && (ob[i].L->n % 32 || ob[i].ldy % 32))
      qlog::fail("gemm", "P060 MXFP8 epilogue: n %d, ldy %d (multiples of 32)", ob[i].L->n, ob[i].ldy);
  }
  if (aux && aux->n != 512 && aux->n != 1024 && aux->n != 2048) qlog::fail("gemm", "P060 side RMSNorm n %d", aux->n);
  if (a.pro && (!a.x || (a.pro == kProRms && Ls[0]->k != 512 && Ls[0]->k != 1024 && Ls[0]->k != 2048)))
    qlog::fail("gemm", "P060 activation op %d over k %d", a.pro, Ls[0]->k);
  const bool dec = patch::on(patch::f8_decode), mul = patch::on(patch::f8_multi);
  auto p030_ok = [&](const F8Act& x) {  // one p030 launch for all weights with activation x?
    if (!dec) return false;
    if (n == 1) return Ls[0]->groups == 1 && (x.x ? p030::fuse_ok(*Ls[0], M) : p030::ok(*Ls[0], M));
    return mul && p030::multi_ok(Ls, n, x, M);
  };
  F8Act x = a;
  if ((x.pro || aux) && !(x.x && p030_ok(x))) {  // M > 8 (or no fused decode launch): op + quantization first
    if (!x.q || !x.s || x.ld < Ls[0]->k) qlog::fail("gemm", "P060 activation op at M %d needs F8Act::q / s / ld", M);
    p060::prep(x, M, Ls[0]->k, aux, s);
    x.x = nullptr, x.pro = 0, aux = nullptr;
  }
  bool in_store = p030_ok(x);  // epilogues in p030's store: kEpiQ32 needs the cluster path (a K split, 64-row blocks)
  for (int i = 0; i < n; i++)
    if ((ob[i].epi == kEpiQ32 && (Ls[i]->split < 2 || Ls[i]->n % 64)) || ob[i].epi == kEpiMulQ128) in_store = false;
  if (in_store && (n >= 2 || epi || x.pro || aux))
    return p030::forward_parts<false>(Ls, ys, ldys, n, x, M, s, 0, 0, epi ? ob : nullptr, aux);
  if (x.pro || aux) qlog::fail("gemm", "P060: no decode launch for the fused activation (M %d)", M);  // not reached
  // P105 f8_tile_chain: every weight on P103 / P133 with its epilogue in the store, programmatic launches, at most one
  // on P103 with a K split (the workspace; P133 needs none) — the 2nd / 3rd GEMM waits for its predecessor only after
  // its main loop
  bool chain = n > 1 && patch::on(patch::f8_tile_chain) && patch::on(patch::f8_pdl);
  int nws = 0;
  for (int i = 0; chain && i < n; i++) {
    const bool sq = seq_takes(*Ls[i], M, x, 0, 0) && p133::epi_ok(*Ls[i], ob[i].epi) &&
                    (ob[i].epi == kEpiNone || tile_epi(*Ls[i], M, ob[i], &x, 0, 0, ws));
    chain = sq || (patch::on(patch::f8_tile) && p103::ok(*Ls[i], M, x, 0, 0, ws) &&
                   (ob[i].epi == kEpiNone || tile_epi(*Ls[i], M, ob[i], &x, 0, 0, ws)));
    nws += !sq && Ls[i]->split > 1;
  }
  chain = chain && nws <= 1;
  for (int i = 0; i < n; i++) {
    const bool fe = ob[i].epi != kEpiNone && tile_epi(*Ls[i], M, ob[i], &x, 0, 0, ws);  // P104: in the tile's store
    if (fe || (chain && i > 0)) {
      tiled(*Ls[i], x, M, ob[i].y, ob[i].ldy, ws, s, 0, 0, fe ? &ob[i] : nullptr, chain && i > 0);
      continue;
    }
    if (ob[i].epi == kEpiMulQ128 || ob[i].epi == kEpiSituQ128)
      qlog::fail("gemm", "epilogue %d without its tile kernel (M %d)", ob[i].epi, M);
    f8_forward(*Ls[i], x, M, ob[i].y, ob[i].ldy, ws, s);
    if (ob[i].epi != kEpiNone) p060::epi(ob[i], M, s);
  }
}
void f8_forward_out(const F8Out& o, const F8Act& a, int M, float* ws, cudaStream_t s, long gx, long gs) {
  if (!gx && !gs) return f8_forward_multi({o}, a, M, ws, s);
  // grouped weight (wo_a): a quantized activation, no prologue
  if (a.x || a.pro) qlog::fail("gemm", "a grouped GEMM takes a quantized activation");
  if (o.epi != kEpiNone && tile_epi(*o.L, M, o, &a, gx, gs, ws)) return tiled(*o.L, a, M, o.y, o.ldy, ws, s, gx, gs, &o, false);
  if (o.epi == kEpiMulQ128 || o.epi == kEpiSituQ128) qlog::fail("gemm", "epilogue %d without its tile kernel (M %d)", o.epi, M);
  f8_forward(*o.L, a, M, o.y, o.ldy, ws, s, gx, gs);
  if (o.epi != kEpiNone) p060::epi(o, M, s);
}
bool f8_epi_fused(const F8Lin& L, int M, int epi, const float* ws) {
  F8Out o{&L, nullptr, 0, epi};
  return tile_epi(L, M, o, nullptr, 0, 0, ws);
}
const char* f8_test_path(const F8Lin& L, int M, int epi) {  // tests: the kernel a call takes (aligned activation)
  F8Act d{};
  d.q = reinterpret_cast<uint8_t*>(uintptr_t(256)), d.s = d.q, d.ld = L.k;
  float* ws = reinterpret_cast<float*>(uintptr_t(256));
  if (patch::on(patch::f8_decode) && p030::ok(L, M)) return "P030";
  if (epi != kEpiNone) {
    const F8Out o{&L, nullptr, 0, epi};
    if (tile_epi(L, M, o, &d, 0, 0, ws)) return seq_takes(L, M, d, 0, 0) && p133::epi_ok(L, epi) ? "P133+epi" : "P103+epi";
  }
  if (seq_takes(L, M, d, 0, 0)) return "P133";
  if (patch::on(patch::f8_tile) && p103::ok(L, M, d, 0, 0, ws) && p103::prefer(L, M)) return "P103";
  if (patch::on(patch::f8_prefill) && p031::ok(L, M)) return "P031";
  return "base";
}
int f8_test_seq(int c) {  // tests: P133's config for every shape (−1: the measured choice); returns the previous one
  const int o = p133::g_cfg;
  p133::g_cfg = c;
  return o;
}
int f8_test_pfn(int n) {  // tests: P064's prefetch depth (k32 blocks); returns the previous one
  const int o = p030::g_pfn;
  p030::g_pfn = n;
  return o;
}
bool f8_defer_act(int M) { return M <= 8 && patch::on(patch::f8_fuse_act) && patch::on(patch::f8_decode); }
bool f8_fuse_ok(int M) { return M >= 1 && patch::on(patch::f8_fuse_ops); }

}  // namespace e55
