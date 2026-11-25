// rdma_ar.cu — P001 rdma_ar (contract in rdma_ar.h): GPU kernels, the CPU proxy thread and the ibverbs setup.
#include "kernels/comm/rdma_ar.h"

#include <arpa/inet.h>
#include <cuda.h>  // P131: stream memory operation types (the entry points are taken at run time: no -lcuda)
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <pthread.h>
#include <sched.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include "core/log.h"
#include "model/patches.h"

namespace e55 {
namespace {

constexpr int kMaxRails = 4;
constexpr int kMaxChunks = 64;
constexpr size_t kLLMaxCap = size_t(512) << 10;  // largest LL call (data bytes); LL lines take twice that
constexpr size_t kLLRegion = 2 * kLLMaxCap;     // per parity, send and recv
constexpr size_t kFusedMax = size_t(1) << 20;   // data bytes ≤: one fused kernel (copy · signal · spin · add)
constexpr size_t kAgLL8 = size_t(16) << 10;     // 8-byte aligned all-gathers: LL up to this (Comm::ag_flat's limit)
constexpr size_t kAuto = ~size_t(0);           // a small-call plan setting left to P120
constexpr size_t kBigChunk = size_t(1) << 20;
constexpr size_t kHdrBytes = size_t(64) << 10;  // flags + doorbells
constexpr size_t kBulk = kHdrBytes + 4 * kLLRegion;  // bulk staging send[2], recv[2] from here (page aligned)
constexpr int kThreads = 512;
constexpr int kSqDepth = 512;
constexpr uint32_t kMagic = 0x51385242;          // "Q8RB"
constexpr int kMaxZc = 4;                        // P176: zc_alloc regions
constexpr unsigned long long kDbZc = 2;          // P176 doorbell mode: a put's chunk (region in bits 8– / 16–)
constexpr int kDbZcSrc = 8, kDbZcDst = 16;
constexpr int kZcStaged = 255;                   // P176 source "region": the staging (send[p] offsets)

struct alignas(64) Flag {  // written by the peer's NIC (inline RDMA WRITE after the chunk's data, same QP)
  unsigned long long seq;
  unsigned long long pad[7];
};
struct alignas(64) Db {    // written by our GPU (seq last, release), read by our proxy
  unsigned long long seq;
  unsigned long long src, dst, bytes;  // byte offsets in the (same-layout) regions of both ranks
  long long flag;                      // peer flag slot to write after the data, -1 = none (LL)
  unsigned long long last;
  unsigned long long mode;             // P118 ar_bf_post: data and flag as two single-WQE posts (BlueFlame each);
                                       // P176: kDbZc, src / dst as offsets in zc regions (kDbZcSrc / kDbZcDst)
  unsigned long long pad[1];
};
struct Hdr {
  Flag flag[2][kMaxChunks];
  Db db[2][kMaxChunks];
  Flag zc_ready;  // P176: the last put the peer reached (written by the peer's proxy)
  Flag zc_sent;   // P176: the last put whose WRITEs completed here (written by this rank's proxy; the stream waits)
};
static_assert(sizeof(Hdr) <= kHdrBytes, "header");

__host__ __device__ constexpr size_t ll_send_off(int p) { return kHdrBytes + size_t(p) * kLLRegion; }
__host__ __device__ constexpr size_t ll_recv_off(int p) { return kHdrBytes + size_t(2 + p) * kLLRegion; }

struct Ctl {                      // device memory, zeroed once
  unsigned long long seq;         // last call issued on the GPU (1, 2, …); parity = seq & 1
  unsigned int cnt[kMaxChunks];   // blocks that finished copying their part of a chunk
  unsigned int done;              // blocks that finished the kernel
  unsigned long long ll_swept[2]; // last LL call per parity that cleared stale lines
};

struct Args {
  bf16* buf;         // the add's destination (and own operand)
  const bf16* src;   // what is sent: buf (all-reduce), or the peer's half (P094 reduce-scatter)
  size_t n;          // elements of this call
  size_t chunk;      // elements per chunk (bulk) or LL lines per chunk
  int C;             // chunks
  char* base;        // host-pinned region, GPU view
  size_t stride;     // bulk bytes per parity buffer
  Ctl* ctl;
  long long timeout_ns;
  int skip;          // tests (RdmaAR::tune xfer_only): large calls move the bytes but skip the copy / add
  unsigned long long* stamp;  // tests (RdmaAR::probe): block 0's phase times, else null
  int mode;          // Db::mode of this call's doorbells
  int bpc;           // fused path: blocks per chunk (the chunk's last block to finish its copy rings)
  unsigned long long seq;  // P119 ar_host_seq: the call's sequence number from the host (0: read from ctl)
  int sweep;         // P119, LL: clear the stale lines this call (the host's count of calls per parity)
  __device__ size_t send_off(int p) const { return kBulk + size_t(p) * stride; }
  __device__ size_t recv_off(int p) const { return kBulk + size_t(2 + p) * stride; }
  __device__ Hdr* hdr() const { return reinterpret_cast<Hdr*>(base); }
};

// ---- device ------------------------------------------------------------------------------------------------------
__device__ __forceinline__ unsigned long long ld_relaxed_sys(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_sys(unsigned long long* p, unsigned long long v) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// tests (RdmaAR::probe): block 0's thread 0 sums its phase times (globaltimer ns): [0] start (after the PDL wait) →
// doorbell rung, [1] rung → the peer's data in (LL: + the add), [2] → the end; [3] calls; [4] gaps end → next start
// ([5] the last end). One call in flight at a time, so plain adds.
__device__ __forceinline__ void stamp(const Args& a, int i, unsigned long long& t) {
  if (!a.stamp || blockIdx.x || threadIdx.x) return;
  const unsigned long long n = gtimer();
  if (i == 0) {
    if (a.stamp[5]) a.stamp[4] += n - a.stamp[5];
    a.stamp[3]++;
  } else {
    a.stamp[i - 1] += n - t;
  }
  if (i == 3) a.stamp[5] = n;
  t = n;
}

__device__ void wait_flag(const Args& a, int p, int k, unsigned long long seq, bool nap) {
  const unsigned long long* f = &a.hdr()->flag[p][k].seq;
  if ((long long)(ld_relaxed_sys(f) - seq) < 0) {
    const unsigned long long t0 = gtimer();
    unsigned it = 0;
    while ((long long)(ld_relaxed_sys(f) - seq) < 0) {
      if (nap) __nanosleep(200);
      if (a.timeout_ns && (++it & 1023) == 0 && (long long)(gtimer() - t0) > a.timeout_ns) {
        printf("[rdma_ar] GPU wait timeout · call %llu chunk %d · flag %llu\n", seq, k, ld_relaxed_sys(f));
        __trap();
      }
    }
  }
  __threadfence_system();  // acquire: the data written before the flag is read after this
}

// rows of chunk k handled by this block: [e0, e1), multiples of 64 elements except at n
__device__ __forceinline__ void block_part(const Args& a, size_t c0, size_t c1, size_t& e0, size_t& e1) {
  const size_t per = ((c1 - c0 + gridDim.x - 1) / gridDim.x + 63) & ~size_t(63);
  e0 = min(c1, c0 + size_t(blockIdx.x) * per);
  e1 = min(c1, e0 + per);
}

__device__ __forceinline__ void copy_part(char* dst, const bf16* __restrict__ src, size_t e0, size_t e1, bool vec) {
  if (vec && e1 > e0) {
    const uint4* __restrict__ s = reinterpret_cast<const uint4*>(src + e0);
    uint4* d = reinterpret_cast<uint4*>(dst + 2 * e0);
    const size_t nv = (e1 - e0) >> 3;
    size_t i = threadIdx.x;
    for (; i + 3 * kThreads < nv; i += 4 * kThreads) {  // streaming (evict-first): keep compute's L2 working set
      const uint4 v0 = __ldcs(s + i), v1 = __ldcs(s + i + kThreads), v2 = __ldcs(s + i + 2 * kThreads),
                  v3 = __ldcs(s + i + 3 * kThreads);
      __stcs(d + i, v0), __stcs(d + i + kThreads, v1), __stcs(d + i + 2 * kThreads, v2), __stcs(d + i + 3 * kThreads, v3);
    }
    for (; i < nv; i += kThreads) __stcs(d + i, __ldcs(s + i));
    e0 += nv << 3;
  }
  bf16* d = reinterpret_cast<bf16*>(dst);
  for (size_t e = e0 + threadIdx.x; e < e1; e += kThreads) d[e] = src[e];
}

// One 32-bit word of each side. bf16 (F32 false): two elements, bf16(float(a) + float(b)) — one rounding of the exact
// fp32 sum = correctly rounded bf16 add (= NCCL's sum). fp32 (F32 true): one element, the fp32 add (= NCCL's 2-rank
// fp32 sum; commutative, so both ranks get the same bits). An fp32 call moves its bytes as twice as many bf16 slots:
// every chunk / block / line boundary is a multiple of 64 slots (or n), so no fp32 is ever split.
template <bool F32>
__device__ __forceinline__ uint32_t add2(uint32_t a, uint32_t b) {
  if constexpr (F32) return __float_as_uint(__fadd_rn(__uint_as_float(a), __uint_as_float(b)));
  __nv_bfloat162 x, y;
  memcpy(&x, &a, 4), memcpy(&y, &b, 4);
  const float2 fx = __bfloat1622float2(x), fy = __bfloat1622float2(y);
  const __nv_bfloat162 r = __floats2bfloat162_rn(__fadd_rn(fx.x, fy.x), __fadd_rn(fx.y, fy.y));
  uint32_t u;
  memcpy(&u, &r, 4);
  return u;
}
template <bool F32>
__device__ __forceinline__ uint4 add8(uint4 a, uint4 b) {
  return make_uint4(add2<F32>(a.x, b.x), add2<F32>(a.y, b.y), add2<F32>(a.z, b.z), add2<F32>(a.w, b.w));
}

// buf[e] += recv[e] for e in [e0, e1); recv is written by the NIC — plain loads (no read-only/nc path)
template <bool F32>
__device__ __forceinline__ void add_part(bf16* buf, const char* rcv, size_t e0, size_t e1, bool vec) {
  if (vec && e1 > e0) {
    uint4* b = reinterpret_cast<uint4*>(buf + e0);
    const uint4* r = reinterpret_cast<const uint4*>(rcv + 2 * e0);
    const size_t nv = (e1 - e0) >> 3;
    size_t i = threadIdx.x;
    for (; i + 3 * kThreads < nv; i += 4 * kThreads) {  // streaming (evict-first) like the copy
      const uint4 a0 = __ldcs(b + i), a1 = __ldcs(b + i + kThreads), a2 = __ldcs(b + i + 2 * kThreads),
                  a3 = __ldcs(b + i + 3 * kThreads);
      const uint4 r0 = __ldcs(r + i), r1 = __ldcs(r + i + kThreads), r2 = __ldcs(r + i + 2 * kThreads),
                  r3 = __ldcs(r + i + 3 * kThreads);
      __stcs(b + i, add8<F32>(a0, r0)), __stcs(b + i + kThreads, add8<F32>(a1, r1)),
          __stcs(b + i + 2 * kThreads, add8<F32>(a2, r2)), __stcs(b + i + 3 * kThreads, add8<F32>(a3, r3));
    }
    for (; i < nv; i += kThreads) __stcs(b + i, add8<F32>(__ldcs(b + i), __ldcs(r + i)));
    e0 += nv << 3;
  }
  if constexpr (F32) {  // whole fp32 words (e0, e1 even)
    uint32_t* bw = reinterpret_cast<uint32_t*>(buf);
    const uint32_t* rw = reinterpret_cast<const uint32_t*>(rcv);
    for (size_t w = e0 / 2 + threadIdx.x; w < e1 / 2; w += kThreads) bw[w] = add2<true>(bw[w], rw[w]);
    return;
  }
  const bf16* r = reinterpret_cast<const bf16*>(rcv);
  for (size_t e = e0 + threadIdx.x; e < e1; e += kThreads)
    buf[e] = __float2bfloat16_rn(__fadd_rn(__bfloat162float(buf[e]), __bfloat162float(r[e])));
}

__device__ __forceinline__ void ring(const Args& a, int p, int k, size_t src, size_t dst, size_t bytes, long long flag,
                                     unsigned long long seq) {
  Db* d = &a.hdr()->db[p][k];
  d->src = src, d->dst = dst, d->bytes = bytes, d->flag = flag, d->last = (k == a.C - 1), d->mode = a.mode;
  st_release_sys(&d->seq, seq);  // release (system scope) covers the staging writes this thread has observed
}

// thread 0, after __syncthreads(): the last block to finish its part of chunk k rings the doorbell
__device__ void publish(const Args& a, int p, int k, unsigned long long seq, unsigned parts) {
  if (parts > 1) {
    __threadfence();
    if (atomicAdd(&a.ctl->cnt[k], 1u) != parts - 1) return;
    a.ctl->cnt[k] = 0;
    __threadfence();
  }
  const size_t c0 = size_t(k) * a.chunk, b = 2 * (min(a.n, c0 + a.chunk) - c0);
  ring(a, p, k, a.send_off(p) + 2 * c0, a.recv_off(p) + 2 * c0, b, p * kMaxChunks + k, seq);
}

// the last block of the kernel records the call as issued (every block read ctl->seq before counting itself done)
__device__ __forceinline__ void finish(const Args& a, unsigned long long seq) {
  if (threadIdx.x == 0) {
    if (gridDim.x == 1) {
      a.ctl->seq = seq;
      return;
    }
    __threadfence();
    if (atomicAdd(&a.ctl->done, 1u) == gridDim.x - 1) a.ctl->done = 0, a.ctl->seq = seq;
  }
}

// ---- LL (small calls): 16-byte lines {d0, flag, d1, flag} carry 8 data bytes; a line is valid when both flags match
// (the NIC writes 8-byte words atomically — NCCL's LL protocol relies on the same). No flag WRITE; the receiver polls
// the data itself (no flag → data round trip).
__device__ __forceinline__ uint4 ld_volatile4(const uint4* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint2 ld_line_data(const bf16* buf, size_t i, size_t n, bool vec) {  // elements 4i..4i+3
  if (vec && 4 * i + 4 <= n) return *reinterpret_cast<const uint2*>(buf + 4 * i);
  uint16_t h[4] = {0, 0, 0, 0};
  for (int j = 0; j < 4; j++)
    if (4 * i + j < n) h[j] = __bfloat16_as_ushort(buf[4 * i + j]);
  return make_uint2(h[0] | uint32_t(h[1]) << 16, h[2] | uint32_t(h[3]) << 16);
}
// Block b owns lines [b·per, …) and rings its own doorbell(s) (chunk b, or 2 chunks — one per rail — for a single
// block): no cross-block sync before sending; each block then polls exactly the lines it sent.
// P077: launched as a programmatic dependent, the grid waits for the preceding kernel before anything (counters, buf);
// launched normally the wait returns at once.
__device__ __forceinline__ void pdl_enter() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  // let a programmatic dependent launch now: it waits for this grid's completion (griddepcontrol.wait) before reading
  // the result, but can stream its own weights while this one waits on the network (no effect otherwise)
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

// the call's sequence number: P119 ar_host_seq — from the host (block 0 stores it for the large path's kernels, which
// read ctl->seq; nothing else reads it, so no grid count at the end); else ctl->seq + 1, advanced by finish()
__device__ __forceinline__ unsigned long long begin_seq(const Args& a) {
  if (a.seq) {
    if (blockIdx.x == 0 && threadIdx.x == 0) a.ctl->seq = a.seq;
    return a.seq;
  }
  __shared__ unsigned long long s_seq;
  if (threadIdx.x == 0) s_seq = a.ctl->seq + 1;
  __syncthreads();
  return s_seq;
}

template <bool F32>
__global__ void __launch_bounds__(kThreads) k_ll(Args a) {
  pdl_enter();
  const unsigned long long seq = begin_seq(a);
  unsigned long long ts = 0;
  stamp(a, 0, ts);
  const int p = int(seq & 1);
  const uint32_t f = uint32_t(seq % 0xffffffffull) + 1;  // never 0
  const size_t lines = (a.n + 3) / 4;
  const int cpb = a.C / int(gridDim.x);                 // chunks per block (1 or 2)
  const size_t l0 = min(lines, size_t(blockIdx.x) * cpb * a.chunk), l1 = min(lines, l0 + cpb * a.chunk);
  const bool vec = ((reinterpret_cast<uintptr_t>(a.buf) | reinterpret_cast<uintptr_t>(a.src)) & 7) == 0;
  uint4* snd = reinterpret_cast<uint4*>(a.base + ll_send_off(p));
  for (size_t i = l0 + threadIdx.x; i < l1; i += kThreads) {
    const uint2 d = ld_line_data(a.src, i, a.n, vec);
    snd[i] = make_uint4(d.x, f, d.y, f);
  }
  __syncthreads();
  if (threadIdx.x == 0)
    for (int c = 0; c < cpb; c++) {
      const int k = int(blockIdx.x) * cpb + c;
      const size_t c0 = min(lines, size_t(k) * a.chunk), c1 = min(lines, c0 + a.chunk);
      ring(a, p, k, ll_send_off(p) + 16 * c0, ll_recv_off(p) + 16 * c0, 16 * (c1 - c0), -1, seq);
    }
  stamp(a, 1, ts);
  uint4* rcv = reinterpret_cast<uint4*>(a.base + ll_recv_off(p));
  for (size_t i = l0 + threadIdx.x; i < l1; i += kThreads) {
    const uint2 mine = ld_line_data(a.buf, i, a.n, vec);  // before polling: off the critical path
    uint4 v = ld_volatile4(rcv + i);
    if (v.y != f || v.w != f) {
      const unsigned long long t0 = gtimer();
      for (unsigned it = 0; v.y != f || v.w != f; it++) {
        v = ld_volatile4(rcv + i);
        if (a.timeout_ns && (it & 1023) == 1023 && (long long)(gtimer() - t0) > a.timeout_ns) {
          printf("[rdma_ar] GPU wait timeout · LL call %llu line %llu\n", seq, (unsigned long long)i);
          __trap();
        }
      }
    }
    const uint2 r = make_uint2(add2<F32>(mine.x, v.x), add2<F32>(mine.y, v.z));
    if (vec && 4 * i + 4 <= a.n) {
      *reinterpret_cast<uint2*>(a.buf + 4 * i) = r;
    } else {
      const uint16_t h[4] = {uint16_t(r.x), uint16_t(r.x >> 16), uint16_t(r.y), uint16_t(r.y >> 16)};
      for (int j = 0; j < 4; j++)
        if (4 * i + j < a.n) a.buf[4 * i + j] = __ushort_as_bfloat16(h[j]);
    }
  }
  // Stale lines keep old flags; a false match needs a line untouched for 2^32-1 calls. Every 2^20 calls (per parity)
  // an LL call clears the lines beyond its own, so no stale flag gets that old. (The peer writes recv[p] again only
  // after we finished the next call — no race.)
  if (blockIdx.x == 0 && (a.seq ? a.sweep != 0 : seq - a.ctl->ll_swept[p] >= (1u << 20))) {
    for (size_t i = lines + threadIdx.x; i < kLLRegion / 16; i += kThreads) rcv[i] = make_uint4(0, 0, 0, 0);
    __syncthreads();
    if (threadIdx.x == 0) a.ctl->ll_swept[p] = seq;
  }
  if (a.seq && !a.stamp) return;  // P119: nothing to count
  __syncthreads();
  stamp(a, 2, ts);
  if (!a.seq) finish(a, seq);
  stamp(a, 3, ts);
}

// mid-size calls: one kernel; block k owns chunk k — its own WRITE, so the NIC starts on a slice as soon as its writes
// have drained (no cross-block sync); chunks alternate rails; a block adds exactly what it copied (in place is safe)
template <bool F32>
__global__ void __launch_bounds__(kThreads) k_fused(Args a) {
  pdl_enter();
  const unsigned long long seq = begin_seq(a);
  unsigned long long ts = 0;
  stamp(a, 0, ts);
  const int p = int(seq & 1), k = int(blockIdx.x) / a.bpc;
  const bool vec = ((reinterpret_cast<uintptr_t>(a.buf) | reinterpret_cast<uintptr_t>(a.src)) & 15) == 0;
  const size_t pb = a.chunk / a.bpc, e0 = min(a.n, size_t(blockIdx.x) * pb), e1 = min(a.n, e0 + pb);
  copy_part(a.base + a.send_off(p), a.src, e0, e1, vec);
  __syncthreads();
  if (threadIdx.x == 0) {
    publish(a, p, k, seq, min(unsigned(a.bpc), gridDim.x - unsigned(k * a.bpc)));
    stamp(a, 1, ts);
    wait_flag(a, p, k, seq, false);
    stamp(a, 2, ts);
  }
  __syncthreads();
  add_part<F32>(a.buf, a.base + a.recv_off(p), e0, e1, vec);
  if (a.seq && !a.stamp) return;  // P119: nothing to count
  __syncthreads();
  if (!a.seq) finish(a, seq);
  stamp(a, 3, ts);
}

// large calls: copy one group of chunks (one per rail) + doorbells; the first group's kernel issues the call …
__global__ void __launch_bounds__(kThreads) k_copy(Args a, int k0, int k1, int first) {
  __shared__ unsigned long long s_seq;
  if (threadIdx.x == 0) s_seq = a.ctl->seq + (first ? 1 : 0);
  __syncthreads();
  const unsigned long long seq = s_seq;
  const int p = int(seq & 1);
  const bool vec = ((reinterpret_cast<uintptr_t>(a.buf) | reinterpret_cast<uintptr_t>(a.src)) & 15) == 0;
  char* snd = a.base + a.send_off(p);
  for (int k = k0; k < k1; k++) {
    size_t e0, e1;
    block_part(a, size_t(k) * a.chunk, min(a.n, size_t(k + 1) * a.chunk), e0, e1);
    if (!a.skip) copy_part(snd, a.src, e0, e1, vec);
    __syncthreads();
    if (threadIdx.x == 0) publish(a, p, k, seq, gridDim.x);
  }
  if (first) finish(a, seq);
}
// … then per group of chunks (one per rail): one sleeping warp waits for the flags …
__global__ void k_wait(Args a, int k0, int k1) {
  if (threadIdx.x) return;
  const unsigned long long seq = a.ctl->seq;
  for (int k = k0; k < k1; k++) wait_flag(a, int(seq & 1), k, seq, true);
}
// … and the add runs only once the data is there
template <bool F32>
__global__ void __launch_bounds__(kThreads) k_add(Args a, int k0, int k1) {
  if (a.skip) return;
  const unsigned long long seq = a.ctl->seq;
  const int p = int(seq & 1);
  size_t e0, e1;
  block_part(a, size_t(k0) * a.chunk, min(a.n, size_t(k1) * a.chunk), e0, e1);
  add_part<F32>(a.buf, a.base + a.recv_off(p), e0, e1, (reinterpret_cast<uintptr_t>(a.buf) & 15) == 0);
}

// ---- P076 all-gather: the pieces of a row as one byte stream (rows in order, pieces in order within a row) ----------
struct AgDesc {
  const char* src[kAgMaxPieces];
  char* own[kAgMaxPieces];   // dst + off + rank·stride
  char* peer[kAgMaxPieces];  // dst + off + (1 − rank)·stride
  long long sld[kAgMaxPieces], dld[kAgMaxPieces];
  int pre[kAgMaxPieces];     // stream offset of piece j in a row
  int W, np;                 // row bytes (Σ widths), pieces
  int fast;                  // P120 ar_plan: the row / offset of a stream byte by a 32-bit division
};

// stream byte e (a multiple of 8; widths are multiples of 16 — of 8 on the LL path —, so a unit never straddles a
// piece) → its source and its two destinations
__device__ __forceinline__ void ag_map(const AgDesc& g, size_t e, const char*& s, char*& own, char*& peer) {
  size_t m;
  int r;
  if (g.fast) {  // P120: a call's stream is < 4 GiB (≤ the staging), so 32 bits do (a 64-bit division costs ~4×)
    const uint32_t eu = uint32_t(e), mu = eu / uint32_t(g.W);
    m = mu, r = int(eu - mu * uint32_t(g.W));
  } else {
    m = e / size_t(g.W), r = int(e - m * size_t(g.W));
  }
  const char* sp = g.src[0];
  char *op = g.own[0], *pp = g.peer[0];
  long long sl = g.sld[0], dl = g.dld[0];
  int b = r;
#pragma unroll
  for (int j = 1; j < kAgMaxPieces; j++)
    if (j < g.np && r >= g.pre[j]) sp = g.src[j], op = g.own[j], pp = g.peer[j], sl = g.sld[j], dl = g.dld[j], b = r - g.pre[j];
  s = sp + size_t(m) * sl + b;
  own = op + size_t(m) * dl + b;
  peer = pp + size_t(m) * dl + b;
}

// stream bytes [b0, b1) (multiples of 16): to the send staging and to this rank's own slots
__device__ __forceinline__ void ag_pack(char* stage, const AgDesc& g, size_t b0, size_t b1) {
  for (size_t b = b0 + 16 * size_t(threadIdx.x); b < b1; b += 16 * size_t(kThreads)) {
    const char* s;
    char *o, *p;
    ag_map(g, b, s, o, p);
    const uint4 v = *reinterpret_cast<const uint4*>(s);
    __stcs(reinterpret_cast<uint4*>(stage + b), v);
    if (o != s) *reinterpret_cast<uint4*>(o) = v;  // (P094 gathers in place: the own part is already there)
  }
}
// received stream bytes [b0, b1) → the peer's slots (rcv written by the NIC: plain loads)
__device__ __forceinline__ void ag_place(const char* rcv, const AgDesc& g, size_t b0, size_t b1) {
  for (size_t b = b0 + 16 * size_t(threadIdx.x); b < b1; b += 16 * size_t(kThreads)) {
    const char* s;
    char *o, *p;
    ag_map(g, b, s, o, p);
    *reinterpret_cast<uint4*>(p) = __ldcs(reinterpret_cast<const uint4*>(rcv + b));
  }
}

// LL all-gather: as k_ll, 8 stream bytes per line (a.n = stream bytes / 2)
__global__ void __launch_bounds__(kThreads) k_ll_ag(Args a, AgDesc g) {
  pdl_enter();
  const unsigned long long seq = begin_seq(a);
  unsigned long long ts = 0;
  stamp(a, 0, ts);
  const int p = int(seq & 1);
  const uint32_t f = uint32_t(seq % 0xffffffffull) + 1;  // never 0
  const size_t lines = a.n / 4;
  const int cpb = a.C / int(gridDim.x);
  const size_t l0 = min(lines, size_t(blockIdx.x) * cpb * a.chunk), l1 = min(lines, l0 + cpb * a.chunk);
  uint4* snd = reinterpret_cast<uint4*>(a.base + ll_send_off(p));
  for (size_t i = l0 + threadIdx.x; i < l1; i += kThreads) {
    const char* s;
    char *o, *pp;
    ag_map(g, 8 * i, s, o, pp);
    const uint2 d = *reinterpret_cast<const uint2*>(s);
    snd[i] = make_uint4(d.x, f, d.y, f);
    if (o != s) *reinterpret_cast<uint2*>(o) = d;
  }
  __syncthreads();
  if (threadIdx.x == 0)
    for (int c = 0; c < cpb; c++) {
      const int k = int(blockIdx.x) * cpb + c;
      const size_t c0 = min(lines, size_t(k) * a.chunk), c1 = min(lines, c0 + a.chunk);
      ring(a, p, k, ll_send_off(p) + 16 * c0, ll_recv_off(p) + 16 * c0, 16 * (c1 - c0), -1, seq);
    }
  stamp(a, 1, ts);
  uint4* rcv = reinterpret_cast<uint4*>(a.base + ll_recv_off(p));
  for (size_t i = l0 + threadIdx.x; i < l1; i += kThreads) {
    const char* s;
    char *o, *pp;
    ag_map(g, 8 * i, s, o, pp);
    uint4 v = ld_volatile4(rcv + i);
    if (v.y != f || v.w != f) {
      const unsigned long long t0 = gtimer();
      for (unsigned it = 0; v.y != f || v.w != f; it++) {
        v = ld_volatile4(rcv + i);
        if (a.timeout_ns && (it & 1023) == 1023 && (long long)(gtimer() - t0) > a.timeout_ns) {
          printf("[rdma_ar] GPU wait timeout · LL gather %llu line %llu\n", seq, (unsigned long long)i);
          __trap();
        }
      }
    }
    *reinterpret_cast<uint2*>(pp) = make_uint2(v.x, v.z);
  }
  if (blockIdx.x == 0 && (a.seq ? a.sweep != 0 : seq - a.ctl->ll_swept[p] >= (1u << 20))) {  // as k_ll
    for (size_t i = lines + threadIdx.x; i < kLLRegion / 16; i += kThreads) rcv[i] = make_uint4(0, 0, 0, 0);
    __syncthreads();
    if (threadIdx.x == 0) a.ctl->ll_swept[p] = seq;
  }
  if (a.seq && !a.stamp) return;  // P119
  __syncthreads();
  stamp(a, 2, ts);
  if (!a.seq) finish(a, seq);
  stamp(a, 3, ts);
}

// mid-size all-gather: as k_fused (block k owns chunk k of the stream; a.chunk slots = 2 bytes each)
__global__ void __launch_bounds__(kThreads) k_fused_ag(Args a, AgDesc g) {
  pdl_enter();
  const unsigned long long seq = begin_seq(a);
  unsigned long long ts = 0;
  stamp(a, 0, ts);
  const int p = int(seq & 1), k = int(blockIdx.x);
  const size_t e0 = size_t(k) * a.chunk, e1 = min(a.n, e0 + a.chunk);
  ag_pack(a.base + a.send_off(p), g, 2 * e0, 2 * e1);
  __syncthreads();
  if (threadIdx.x == 0) {
    publish(a, p, k, seq, 1);
    stamp(a, 1, ts);
    wait_flag(a, p, k, seq, false);
    stamp(a, 2, ts);
  }
  __syncthreads();
  ag_place(a.base + a.recv_off(p), g, 2 * e0, 2 * e1);
  if (a.seq && !a.stamp) return;  // P119
  __syncthreads();
  if (!a.seq) finish(a, seq);
  stamp(a, 3, ts);
}

// large all-gathers: as k_copy / k_add (k_wait shared)
__global__ void __launch_bounds__(kThreads) k_pack_ag(Args a, AgDesc g, int k0, int k1, int first) {
  __shared__ unsigned long long s_seq;
  if (threadIdx.x == 0) s_seq = a.ctl->seq + (first ? 1 : 0);
  __syncthreads();
  const unsigned long long seq = s_seq;
  const int p = int(seq & 1);
  char* snd = a.base + a.send_off(p);
  for (int k = k0; k < k1; k++) {
    size_t e0, e1;
    block_part(a, size_t(k) * a.chunk, min(a.n, size_t(k + 1) * a.chunk), e0, e1);
    ag_pack(snd, g, 2 * e0, 2 * e1);
    __syncthreads();
    if (threadIdx.x == 0) publish(a, p, k, seq, gridDim.x);
  }
  if (first) finish(a, seq);
}
__global__ void __launch_bounds__(kThreads) k_place_ag(Args a, AgDesc g, int k0, int k1) {
  const unsigned long long seq = a.ctl->seq;
  const int p = int(seq & 1);
  size_t e0, e1;
  block_part(a, size_t(k0) * a.chunk, min(a.n, size_t(k1) * a.chunk), e0, e1);
  ag_place(a.base + a.recv_off(p), g, 2 * e0, 2 * e1);
}

// ---- host helpers ------------------------------------------------------------------------------------------------
void send_all(int fd, const void* p, size_t n) {
  const char* c = static_cast<const char*>(p);
  while (n) {
    ssize_t k = ::send(fd, c, n, 0);
    if (k <= 0) qlog::fail("rdma", "bootstrap send failed");
    c += k, n -= k;
  }
}
void recv_all(int fd, void* p, size_t n) {
  char* c = static_cast<char*>(p);
  while (n) {
    ssize_t k = ::recv(fd, c, n, 0);
    if (k <= 0) qlog::fail("rdma", "bootstrap receive failed");
    c += k, n -= k;
  }
}
int tcp_pair(int rank, const std::string& ip, int port) {
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(port);
  if (inet_pton(AF_INET, ip.c_str(), &a.sin_addr) != 1) qlog::fail("rdma", "bad address: %s", ip.c_str());
  if (rank == 0) {
    int ls = socket(AF_INET, SOCK_STREAM, 0), one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    if (bind(ls, (sockaddr*)&a, sizeof a) || listen(ls, 1)) qlog::fail("rdma", "cannot listen on %s:%d", ip.c_str(), port);
    int fd = accept(ls, nullptr, nullptr);
    close(ls);
    return fd;
  }
  for (int i = 0; i < 3000; i++) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (connect(fd, (sockaddr*)&a, sizeof a) == 0) return fd;
    close(fd);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  qlog::fail("rdma", "cannot reach the head %s:%d (300 s)", ip.c_str(), port);
}

std::vector<std::string> hca_names() {
  const char* e = std::getenv("E55_RDMA_HCA");
  if (!e || !*e) e = std::getenv("NCCL_IB_HCA");
  if (!e || !*e) qlog::fail("rdma", "no HCA list (E55_RDMA_HCA or NCCL_IB_HCA)");
  std::string s = e;
  if (s[0] == '=') s = s.substr(1);
  if (s[0] == '^') qlog::fail("rdma", "HCA exclude list (^) not supported: %s", e);
  std::vector<std::string> v;
  for (size_t i = 0; i <= s.size();) {
    size_t j = s.find(',', i);
    if (j == std::string::npos) j = s.size();
    std::string h = s.substr(i, j - i);
    if (h.find(':') != std::string::npos) h = h.substr(0, h.find(':'));
    if (!h.empty()) v.push_back(h);
    i = j + 1;
  }
  if (v.empty() || v.size() > size_t(kMaxRails)) qlog::fail("rdma", "HCA list has %zu entries (1–%d): %s", v.size(), kMaxRails, e);
  return v;
}

int env_int(const char* k, int def) {
  const char* e = std::getenv(k);
  return (e && *e) ? std::atoi(e) : def;
}

// proxy CPU candidates: E55_RDMA_CPU if set, else every CPU by capacity (big cores first; GB10: X925 > A725), ties by
// higher index (CPU 0 takes most interrupts)
std::vector<int> cpu_candidates() {
  const int c = env_int("E55_RDMA_CPU", -2);
  if (c != -2) return c >= 0 ? std::vector<int>{c} : std::vector<int>{};
  std::vector<std::pair<int, int>> v;
  const long n = sysconf(_SC_NPROCESSORS_CONF);
  for (int i = 0; i < n; i++) {
    int cap = 0;
    char path[96];
    snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpu_capacity", i);
    if (FILE* f = std::fopen(path, "r")) {
      if (std::fscanf(f, "%d", &cap) != 1) cap = 0;
      std::fclose(f);
    }
    v.push_back({-cap, -i});
  }
  std::sort(v.begin(), v.end());
  std::vector<int> r;
  for (auto& x : v) r.push_back(-x.second);
  return r;
}

struct Wire {  // bootstrap exchange (both ranks)
  uint32_t magic, rails;
  uint64_t stride, base;  // staging bytes per parity · virtual address of the registered region
  struct { uint32_t qpn, psn, rkey, mtu; uint8_t gid[16]; } r[kMaxRails];
};

struct Rail {
  std::string name;
  ibv_context* ctx = nullptr;
  ibv_pd* pd = nullptr;
  ibv_cq* cq = nullptr;
  ibv_qp* qp = nullptr;
  ibv_mr* mr = nullptr;
  int gidx = -1;
  ibv_gid gid{};
  ibv_mtu mtu = IBV_MTU_1024;
  uint32_t psn = 0, peer_rkey = 0;
  int outstanding = 0;
};

}  // namespace

// ---- RdmaAR ------------------------------------------------------------------------------------------------------
struct RdmaAR::Impl {
  int rank = 0, nr = 0, cpu = -1, copy_blocks = 2, add_blocks = 4, ahead = 1, xfer_only = 0;
  size_t max_elems = 0, stride = 0, bytes = 0, ll_max = 16 << 10, ag_ll_max = 16 << 10, chunk_bytes = kBigChunk;
  size_t ag_fused_max = size_t(16) << 20;
  int fused_chunks = 0;  // all-reduce fused path: at most this many chunks (0: one per block)
  size_t ll_split = 4 << 10;  // LL with one block: two chunks (one per rail) above this many data bytes
  // the small-call plan (P120 ar_plan picks the defaults; kAuto: by the patch, else set by env / tests): all-gathers
  // LL up to ag_ll_max bytes, LL data bytes per block, the fused all-reduce's data bytes per block
  size_t ll_block = kAuto, fused_block = kAuto;
  // P119 ar_host_seq: calls issued so far (every path counts; the GPU runs them in this order), the last LL call per
  // parity that swept the stale lines, and whether a stream was ever captured (then the device count only: a graph
  // replay advances it without the host)
  unsigned long long hseq = 0, hswept[2] = {0, 0};
  bool captured = false;
  long long timeout_ns = 0;
  // P131 rdma_ce: large calls by the copy engines and stream memory operations (a call on this path knows its seq at
  // issue: hseq above — the one count of P119 and P131 — is ctl->seq on the GPU after it, the proxy's count)
  bool ce = false;
  CUresult (*wr64)(CUstream, CUdeviceptr, cuuint64_t, unsigned) = nullptr;
  CUresult (*wt64)(CUstream, CUdeviceptr, cuuint64_t, unsigned) = nullptr;
  size_t ce_chunk = size_t(4) << 20;  // P131 chunk bytes (E55_RDMA_CE_KB), ≥ 4 chunks a call; groups ahead; add blocks
  int ce_ahead = 2, ce_add = 8;
  // P132 rdma_numa: the region from cuMemCreate on the host NUMA node (numa_size > 0) instead of cudaHostAlloc
  CUmemGenericAllocationHandle numa_h = 0;
  size_t numa_size = 0;
  // P176 zc_a2a: NIC-registered activation regions (zc_alloc) and the peer's matching ones; the bootstrap connection
  // stays open for their exchange
  struct Zc {
    char* base = nullptr;
    size_t size = 0;
    CUmemGenericAllocationHandle h = 0;
    ibv_mr* mr[kMaxRails] = {};
    uint64_t peer_base = 0;
    uint32_t peer_rkey[kMaxRails] = {};
  };
  Zc zc[kMaxZc];
  int nzc = 0, fd = -1;
  Rail rail[kMaxRails];
  char* host = nullptr;    // region: Hdr | LL send[2] | LL recv[2] | send[2] | recv[2]
  char* dev = nullptr;     // GPU view of the same memory
  Ctl* ctl = nullptr;
  uint64_t peer_base = 0;
  std::thread th;
  std::atomic<bool> stop{false};
  // tests (RdmaAR::probe): the GPU phase stamps (device, 8 words) and the proxy's post cost
  unsigned long long* stamp = nullptr;
  bool stamp_on = false;
  std::atomic<bool> probe_on{false};
  std::atomic<unsigned long long> post_ns{0}, posts{0};

  Hdr* hdr() const { return reinterpret_cast<Hdr*>(host); }

  void reap(int r) {
    ibv_wc wc[16];
    int n = ibv_poll_cq(rail[r].cq, 16, wc);
    if (n < 0) qlog::fail("rdma", "%s CQ error", rail[r].name.c_str());
    for (int i = 0; i < n; i++) {
      if (wc[i].status != IBV_WC_SUCCESS)
        qlog::fail("rdma", "%s RDMA WRITE failed: %s", rail[r].name.c_str(), ibv_wc_status_str(wc[i].status));
      rail[r].outstanding -= int(wc[i].wr_id);  // WQEs this completion covers
    }
  }

  // data WRITE (src → peer dst); then, if flag ≥ 0, an inline 8-byte WRITE of seq to the peer's flag slot (same QP).
  // mode 1 (P118 ar_bf_post): the two WQEs as two posts — libmlx5 sends a lone WQE by BlueFlame (the WQE itself in
  // the MMIO write), a chain of two by a doorbell the NIC answers by fetching the WQEs from host memory first
  void post(int r, uint64_t src, uint64_t dst, uint64_t len, long long flag, unsigned long long seq, uint64_t mode) {
    Rail& R = rail[r];
    uint64_t la = reinterpret_cast<uint64_t>(host) + src, ra = peer_base + dst;
    uint32_t lk = R.mr->lkey, rk = R.peer_rkey;
    if (mode & kDbZc) {  // P176: a put's chunk — from this rank's zc region or staging straight into the peer's region
      const int is = int((mode >> kDbZcSrc) & 255);
      const Zc& zd = zc[(mode >> kDbZcDst) & 255];
      if (is != kZcStaged) la = reinterpret_cast<uint64_t>(zc[is].base) + src, lk = zc[is].mr[r]->lkey;  // (else staging)
      ra = zd.peer_base + dst, rk = zd.peer_rkey[r];
    }
    while (R.outstanding + 2 > kSqDepth) reap(r);
    ibv_sge sd{la, uint32_t(len), lk};
    unsigned long long v = seq;
    ibv_sge sf{reinterpret_cast<uint64_t>(&v), 8, R.mr->lkey};
    ibv_send_wr w[2]{}, *bad = nullptr;
    w[0].sg_list = &sd, w[0].num_sge = 1, w[0].opcode = IBV_WR_RDMA_WRITE;
    w[0].wr.rdma.remote_addr = ra, w[0].wr.rdma.rkey = rk;
    int nw = 1;
    if (flag >= 0) {
      w[0].next = &w[1];
      w[1].sg_list = &sf, w[1].num_sge = 1, w[1].opcode = IBV_WR_RDMA_WRITE;
      w[1].send_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED, w[1].wr_id = 2;
      w[1].wr.rdma.remote_addr = peer_base + offsetof(Hdr, flag) + size_t(flag) * sizeof(Flag);
      w[1].wr.rdma.rkey = R.peer_rkey;
      nw = 2;
    } else {
      w[0].send_flags = IBV_SEND_SIGNALED, w[0].wr_id = 1;  // a lone WQE goes out by BlueFlame
    }
    if (nw == 2 && (mode & 1)) {
      w[0].next = nullptr;
      if (int e = ibv_post_send(R.qp, &w[0], &bad)) qlog::fail("rdma", "%s post_send failed: %s", R.name.c_str(), strerror(e));
      if (int e = ibv_post_send(R.qp, &w[1], &bad)) qlog::fail("rdma", "%s post_send failed: %s", R.name.c_str(), strerror(e));
    } else if (int e = ibv_post_send(R.qp, w, &bad)) {
      qlog::fail("rdma", "%s post_send failed: %s", R.name.c_str(), strerror(e));
    }
    R.outstanding += nw;
  }

  // P176: a put's rendezvous — an inline WRITE of seq to the peer's zc_ready (the doorbell means this rank's stream
  // reached the put: its destination is free), then wait for the peer's. false: stopped while waiting
  bool zc_ready(unsigned long long seq) {
    Rail& R = rail[seq % nr];
    while (R.outstanding + 1 > kSqDepth) reap(int(seq % nr));
    unsigned long long v = seq;
    ibv_sge sf{reinterpret_cast<uint64_t>(&v), 8, R.mr->lkey};
    ibv_send_wr w{}, *bad = nullptr;
    w.sg_list = &sf, w.num_sge = 1, w.opcode = IBV_WR_RDMA_WRITE, w.wr_id = 1;
    w.send_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;
    w.wr.rdma.remote_addr = peer_base + offsetof(Hdr, zc_ready), w.wr.rdma.rkey = R.peer_rkey;
    if (int e = ibv_post_send(R.qp, &w, &bad)) qlog::fail("rdma", "%s post_send failed: %s", R.name.c_str(), strerror(e));
    R.outstanding += 1;
    for (unsigned spin = 0; (long long)(__atomic_load_n(&hdr()->zc_ready.seq, __ATOMIC_ACQUIRE) - seq) < 0; spin++) {
      if ((spin & 255) == 255) {
        if (stop.load(std::memory_order_relaxed)) return false;
        for (int r = 0; r < nr; r++)
          if (rail[r].outstanding) reap(r);
      }
#if defined(__aarch64__)
      asm volatile("yield" ::: "memory");
#endif
    }
    return true;
  }
  // P176: after a put's last chunk — every WRITE completed (RC: in order per QP; the peer holds the data, this rank's
  // sources are read), then `sent` = seq for the stream's wait. false: stopped while waiting
  bool zc_done(unsigned long long seq) {
    for (unsigned spin = 0;; spin++) {
      bool any = false;
      for (int r = 0; r < nr; r++)
        if (rail[r].outstanding) reap(r), any = any || rail[r].outstanding;
      if (!any) break;
      if ((spin & 255) == 255 && stop.load(std::memory_order_relaxed)) return false;
#if defined(__aarch64__)
      asm volatile("yield" ::: "memory");
#endif
    }
    __atomic_store_n(&hdr()->zc_sent.seq, seq, __ATOMIC_RELEASE);
    return true;
  }

  // Follows the GPU's calls in order: per call, chunks 0.. until `last`; chunk k of call seq goes on rail (seq+k)%nr.
  void proxy() {
    using clk = std::chrono::steady_clock;
    auto active = clk::now();
    unsigned long long seq = 0;
    for (;;) {
      ++seq;
      const int p = int(seq & 1);
      for (int k = 0;; k++) {
        Db* d = &hdr()->db[p][k];
        for (unsigned spin = 0; __atomic_load_n(&d->seq, __ATOMIC_ACQUIRE) != seq; spin++) {
          if ((spin & 255) == 255) {
            if (stop.load(std::memory_order_relaxed)) return;
            for (int r = 0; r < nr; r++)
              if (rail[r].outstanding) reap(r);
            // idle: 200 ms → 20 µs naps; 1 s → 500 µs naps (an idle server near 0% CPU; the first all-reduce after a
            // long pause waits ≤ 0.5 ms more; while steps run, calls come within µs and the loop keeps spinning)
            const auto idle = clk::now() - active;
            if (idle > std::chrono::seconds(1)) std::this_thread::sleep_for(std::chrono::microseconds(500));
            else if (idle > std::chrono::milliseconds(200)) std::this_thread::sleep_for(std::chrono::microseconds(20));
          }
#if defined(__aarch64__)
          asm volatile("yield" ::: "memory");
#endif
        }
        const uint64_t src = d->src, dst = d->dst, len = d->bytes, last = d->last, mode = d->mode;
        const long long flag = d->flag;
        if ((mode & kDbZc) && k == 0 && !zc_ready(seq)) return;  // P176: both streams at the put (stop: false)
        if (probe_on.load(std::memory_order_relaxed)) {
          const auto t = clk::now();
          post(int((seq + k) % nr), src, dst, len, flag, seq, mode);
          post_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - t).count(),
                            std::memory_order_relaxed);
          posts.fetch_add(1, std::memory_order_relaxed);
        } else {
          post(int((seq + k) % nr), src, dst, len, flag, seq, mode);
        }
        if (last) {
          if ((mode & kDbZc) && !zc_done(seq)) return;  // P176: this rank's sources read → `sent` (stop: false)
          break;
        }
      }
      active = clk::now();
    }
  }
};

RdmaAR::RdmaAR(int rank, const std::string& head_ip, int port, size_t max_elems) : d_(std::make_unique<Impl>()) {
  Impl& I = *d_;
  auto t0 = std::chrono::steady_clock::now();
  I.rank = rank;
  I.max_elems = std::max<size_t>(64, (max_elems + 63) & ~size_t(63));
  I.stride = (I.max_elems * 2 + (size_t(64) << 10) - 1) & ~((size_t(64) << 10) - 1);
  I.bytes = kBulk + 4 * I.stride;
  I.timeout_ns = (long long)(std::atof(std::getenv("E55_RDMA_TIMEOUT_S") ? std::getenv("E55_RDMA_TIMEOUT_S") : "0") * 1e9);
  I.copy_blocks = std::max(1, env_int("E55_RDMA_COPY_BLOCKS", 2));  // few blocks: the copy only has to stay ahead of
  I.add_blocks = std::max(1, env_int("E55_RDMA_ADD_BLOCKS", 4));    // the NIC; more starve its DMA (DRAM-bound)
  I.chunk_bytes = size_t(std::max(64, env_int("E55_RDMA_CHUNK_KB", int(kBigChunk >> 10)))) << 10;
  I.ahead = std::max(1, env_int("E55_RDMA_AHEAD", 1));
  I.ll_max = env_int("E55_RDMA_LL_MAX", -1) < 0 ? kAuto : std::min<size_t>(kLLMaxCap, size_t(env_int("E55_RDMA_LL_MAX", 0)));
  I.ag_ll_max = env_int("E55_RDMA_AG_LL_MAX", -1) < 0 ? kAuto : std::min<size_t>(kLLMaxCap, size_t(env_int("E55_RDMA_AG_LL_MAX", 0)));
  I.ag_fused_max = size_t(std::max(0, env_int("E55_RDMA_AG_FUSED_KB", 16384))) << 10;
  I.fused_chunks = std::max(0, env_int("E55_RDMA_FUSED_CHUNKS", 0));
  I.ce_chunk = size_t(std::max(64, env_int("E55_RDMA_CE_KB", 4096))) << 10;  // P131 (measured: 4 MiB, 2 ahead, 8 adds)
  I.ce_ahead = std::max(1, env_int("E55_RDMA_CE_AHEAD", 2));
  I.ce_add = std::max(1, env_int("E55_RDMA_CE_ADD_BLOCKS", 8));

  if (patch::on(patch::rdma_numa)) {
    // P132: after Pico (DeepSeekV4.1-Flash-Pico, MIT, Copyright (c) 2026 Andy Shu; engine/comm/ring.cpp alloc_region):
    // pinned host memory from cuMemCreate on the host NUMA node, mapped for the GPU and the CPU at one address — the
    // GPU reads it at DRAM speed (Pico measured cudaHostAlloc memory 178 vs 236 GB/s), the NIC registers it
    auto sym = [](const char* n) {
      void* f = nullptr;
      cudaDriverEntryPointQueryResult q;
      if (cudaGetDriverEntryPointByVersion(n, &f, 12000, cudaEnableDefault, &q) != cudaSuccess ||
          q != cudaDriverEntryPointSuccess)
        qlog::fail("rdma", "P132: driver entry point %s", n);
      return f;
    };
    auto gran_f = reinterpret_cast<CUresult (*)(size_t*, const CUmemAllocationProp*, CUmemAllocationGranularity_flags)>(
        sym("cuMemGetAllocationGranularity"));
    auto create = reinterpret_cast<CUresult (*)(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*,
                                                unsigned long long)>(sym("cuMemCreate"));
    auto reserve = reinterpret_cast<CUresult (*)(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long)>(
        sym("cuMemAddressReserve"));
    auto map = reinterpret_cast<CUresult (*)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle,
                                             unsigned long long)>(sym("cuMemMap"));
    auto access = reinterpret_cast<CUresult (*)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t)>(
        sym("cuMemSetAccess"));
    auto ck = [](CUresult r, const char* what) {
      if (r != CUDA_SUCCESS) qlog::fail("rdma", "P132 %s failed (%d)", what, int(r));
    };
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
    prop.location.id = 0;
    size_t gran = 0;
    ck(gran_f(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM), "granularity");
    I.numa_size = (I.bytes + gran - 1) / gran * gran;
    ck(create(&I.numa_h, I.numa_size, &prop, 0), "cuMemCreate");
    CUdeviceptr va = 0;
    ck(reserve(&va, I.numa_size, gran, 0, 0), "cuMemAddressReserve");
    ck(map(va, I.numa_size, 0, I.numa_h, 0), "cuMemMap");
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUmemAccessDesc acc[2]{};
    acc[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE, acc[0].location.id = dev;
    acc[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    acc[1].location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA, acc[1].location.id = 0;
    acc[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    ck(access(va, I.numa_size, acc, 2), "cuMemSetAccess");
    I.host = reinterpret_cast<char*>(va);
  } else {
    CUDA_CHECK(cudaHostAlloc(&I.host, I.bytes, cudaHostAllocMapped | cudaHostAllocPortable));
  }
  std::memset(I.host, 0, I.bytes);
  if (I.numa_size) {
    I.dev = I.host;  // (P132: one address for both)
  } else {
    void* dp = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&dp, I.host, 0));
    I.dev = static_cast<char*>(dp);
  }
  CUDA_CHECK(cudaMalloc(&I.ctl, sizeof(Ctl)));
  CUDA_CHECK(cudaMemset(I.ctl, 0, sizeof(Ctl)));

  // rails: device, RoCE v2 IPv4 GID, PD, MR over the whole region, CQ, RC QP → INIT
  auto names = hca_names();
  I.nr = int(names.size());
  int ndev = 0;
  ibv_device** devs = ibv_get_device_list(&ndev);
  if (!devs) qlog::fail("rdma", "ibv_get_device_list failed (/dev/infiniband?)");
  std::mt19937 rng(std::random_device{}());
  const int gid_env = env_int("E55_RDMA_GID_INDEX", -1);
  for (int r = 0; r < I.nr; r++) {
    Rail& R = I.rail[r];
    R.name = names[r];
    for (int i = 0; i < ndev && !R.ctx; i++)
      if (R.name == ibv_get_device_name(devs[i])) R.ctx = ibv_open_device(devs[i]);
    if (!R.ctx) qlog::fail("rdma", "HCA %s not found", R.name.c_str());
    ibv_port_attr pa{};
    if (ibv_query_port(R.ctx, 1, &pa) || pa.state != IBV_PORT_ACTIVE) qlog::fail("rdma", "%s port 1 not active", R.name.c_str());
    R.mtu = pa.active_mtu;
    if (gid_env >= 0) {
      R.gidx = gid_env;
    } else {
      ibv_gid_entry ge[64];
      ssize_t ng = ibv_query_gid_table(R.ctx, ge, 64, 0);
      for (ssize_t i = 0; i < ng && R.gidx < 0; i++) {
        const uint8_t* g = ge[i].gid.raw;
        bool v4 = g[10] == 0xff && g[11] == 0xff;
        for (int b = 0; b < 10; b++) v4 = v4 && g[b] == 0;
        if (ge[i].port_num == 1 && ge[i].gid_type == IBV_GID_TYPE_ROCE_V2 && v4) R.gidx = int(ge[i].gid_index);
      }
    }
    if (R.gidx < 0 || ibv_query_gid(R.ctx, 1, R.gidx, &R.gid)) qlog::fail("rdma", "%s has no RoCE v2 IPv4 GID", R.name.c_str());
    if (!(R.pd = ibv_alloc_pd(R.ctx))) qlog::fail("rdma", "%s PD failed", R.name.c_str());
    R.mr = ibv_reg_mr(R.pd, I.host, I.bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!R.mr) qlog::fail("rdma", "%s MR registration failed (%.0f MiB): %s", R.name.c_str(), I.bytes / 1048576.0, strerror(errno));
    if (!(R.cq = ibv_create_cq(R.ctx, kSqDepth, nullptr, nullptr, 0))) qlog::fail("rdma", "%s CQ failed", R.name.c_str());
    ibv_qp_init_attr qa{};
    qa.send_cq = qa.recv_cq = R.cq;
    qa.qp_type = IBV_QPT_RC;
    qa.cap.max_send_wr = kSqDepth, qa.cap.max_recv_wr = 1, qa.cap.max_send_sge = 1, qa.cap.max_recv_sge = 1;
    qa.cap.max_inline_data = 64;
    if (!(R.qp = ibv_create_qp(R.pd, &qa))) qlog::fail("rdma", "%s QP failed: %s", R.name.c_str(), strerror(errno));
    ibv_qp_attr at{};
    at.qp_state = IBV_QPS_INIT, at.pkey_index = 0, at.port_num = 1;
    at.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
    if (ibv_modify_qp(R.qp, &at, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS))
      qlog::fail("rdma", "%s QP INIT failed", R.name.c_str());
    R.psn = rng() & 0xffffff;
  }
  ibv_free_device_list(devs);

  // exchange QPs and the region address with the peer
  Wire me{}, pe{};
  me.magic = kMagic, me.rails = uint32_t(I.nr), me.stride = I.stride, me.base = reinterpret_cast<uint64_t>(I.host);
  for (int r = 0; r < I.nr; r++) {
    me.r[r].qpn = I.rail[r].qp->qp_num, me.r[r].psn = I.rail[r].psn, me.r[r].rkey = I.rail[r].mr->rkey;
    me.r[r].mtu = uint32_t(I.rail[r].mtu);
    std::memcpy(me.r[r].gid, I.rail[r].gid.raw, 16);
  }
  const int fd = tcp_pair(rank, head_ip, port);
  send_all(fd, &me, sizeof me);
  recv_all(fd, &pe, sizeof pe);
  if (pe.magic != kMagic || pe.rails != me.rails || pe.stride != me.stride)
    qlog::fail("rdma", "settings differ from the peer · rails %u/%u · buffer %llu/%llu B", me.rails, pe.rails, (unsigned long long)me.stride,
               (unsigned long long)pe.stride);
  I.peer_base = pe.base;
  for (int r = 0; r < I.nr; r++) {
    Rail& R = I.rail[r];
    if (std::memcmp(me.r[r].gid + 12, pe.r[r].gid + 12, 3))  // same /24 per rail (direct cables)
      qlog::fail("rdma", "rail %d subnets differ (%u.%u.%u.x ↔ %u.%u.%u.x) — check the HCA order", r, me.r[r].gid[12],
                 me.r[r].gid[13], me.r[r].gid[14], pe.r[r].gid[12], pe.r[r].gid[13], pe.r[r].gid[14]);
    R.peer_rkey = pe.r[r].rkey;
    ibv_qp_attr at{};
    at.qp_state = IBV_QPS_RTR;
    at.path_mtu = ibv_mtu(std::min<uint32_t>(uint32_t(R.mtu), pe.r[r].mtu));
    at.dest_qp_num = pe.r[r].qpn, at.rq_psn = pe.r[r].psn, at.max_dest_rd_atomic = 1, at.min_rnr_timer = 12;
    at.ah_attr.is_global = 1, at.ah_attr.port_num = 1;
    std::memcpy(at.ah_attr.grh.dgid.raw, pe.r[r].gid, 16);
    at.ah_attr.grh.sgid_index = uint8_t(R.gidx), at.ah_attr.grh.hop_limit = 64;
    if (int e = ibv_modify_qp(R.qp, &at, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                             IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER))
      qlog::fail("rdma", "%s QP RTR failed: %s", R.name.c_str(), strerror(e));
    at = ibv_qp_attr{};
    at.qp_state = IBV_QPS_RTS, at.timeout = 14, at.retry_cnt = 7, at.rnr_retry = 7, at.sq_psn = R.psn;
    at.max_rd_atomic = 1;
    if (int e = ibv_modify_qp(R.qp, &at, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                                             IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC))
      qlog::fail("rdma", "%s QP RTS failed: %s", R.name.c_str(), strerror(e));
  }
  char b = 1;  // both sides are RTS before anyone writes
  send_all(fd, &b, 1);
  recv_all(fd, &b, 1);
  I.fd = fd;  // (P176: zc_alloc exchanges its regions over it; closed by the destructor)

  {  // P131: the driver's stream memory operations (64-bit write / wait), if the device has them
    void *w = nullptr, *t = nullptr, *ga = nullptr;
    cudaDriverEntryPointQueryResult q1, q2, q3;
    if (cudaGetDriverEntryPointByVersion("cuStreamWriteValue64", &w, 12000, cudaEnableDefault, &q1) == cudaSuccess &&
        cudaGetDriverEntryPointByVersion("cuStreamWaitValue64", &t, 12000, cudaEnableDefault, &q2) == cudaSuccess &&
        cudaGetDriverEntryPointByVersion("cuDeviceGetAttribute", &ga, 12000, cudaEnableDefault, &q3) == cudaSuccess &&
        q1 == cudaDriverEntryPointSuccess && q2 == cudaDriverEntryPointSuccess && q3 == cudaDriverEntryPointSuccess) {
      int dev = 0, ok = 0;
      CUDA_CHECK(cudaGetDevice(&dev));
      auto get = reinterpret_cast<CUresult (*)(int*, CUdevice_attribute, CUdevice)>(ga);
      if (get(&ok, CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS, dev) == CUDA_SUCCESS && ok) {
        I.wr64 = reinterpret_cast<decltype(I.wr64)>(w), I.wt64 = reinterpret_cast<decltype(I.wt64)>(t);
        I.ce = true;
      }
    }
    if (!I.ce) qlog::warn("rdma", "rank%d · no 64-bit stream memory operations: P131 rdma_ce unavailable", rank);
  }
  I.th = std::thread([&I] { I.proxy(); });
  for (int c : cpu_candidates()) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(c, &set);
    if (pthread_setaffinity_np(I.th.native_handle(), sizeof set, &set) == 0) {
      I.cpu = c;
      break;
    }
  }
  std::string rl;
  for (int r = 0; r < I.nr; r++)
    rl += (r ? "," : "") + I.rail[r].name + "/" + std::to_string(I.rail[r].gidx);
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  qlog::ok("rdma", "rank%d · RDMA AR · %d rails %s · staging %.0f MiB · proxy CPU %d · LL ≤ %zu KiB · %.0f ms", rank, I.nr,
           rl.c_str(), I.bytes / 1048576.0, I.cpu,
           (I.ll_max != kAuto ? I.ll_max : size_t(patch::on(patch::ar_plan) ? 8 << 10 : 16 << 10)) >> 10, ms);
}

// P176: driver entry points (no -lcuda), and a zc region's unmap / release
static void* drv_sym(const char* n) {
  void* f = nullptr;
  cudaDriverEntryPointQueryResult q;
  if (cudaGetDriverEntryPointByVersion(n, &f, 12000, cudaEnableDefault, &q) != cudaSuccess ||
      q != cudaDriverEntryPointSuccess)
    return nullptr;
  return f;
}
static void zc_unmap(char* base, size_t size, CUmemGenericAllocationHandle h) {
  auto unmap = reinterpret_cast<CUresult (*)(CUdeviceptr, size_t)>(drv_sym("cuMemUnmap"));
  auto afree = reinterpret_cast<CUresult (*)(CUdeviceptr, size_t)>(drv_sym("cuMemAddressFree"));
  auto release = reinterpret_cast<CUresult (*)(CUmemGenericAllocationHandle)>(drv_sym("cuMemRelease"));
  const CUdeviceptr va = CUdeviceptr(reinterpret_cast<uintptr_t>(base));
  if (base && size && unmap) unmap(va, size);
  if (base && size && afree) afree(va, size);
  if (h && release) release(h);
}

RdmaAR::~RdmaAR() {
  if (!d_) return;
  Impl& I = *d_;
  I.stop = true;
  if (I.th.joinable()) I.th.join();
  for (int r = 0; r < I.nr; r++) {
    Rail& R = I.rail[r];
    if (R.qp) ibv_destroy_qp(R.qp);
    if (R.cq) ibv_destroy_cq(R.cq);
    if (R.mr) ibv_dereg_mr(R.mr);
    for (int z = 0; z < I.nzc; z++)  // P176
      if (I.zc[z].mr[r]) ibv_dereg_mr(I.zc[z].mr[r]), I.zc[z].mr[r] = nullptr;
    if (R.pd) ibv_dealloc_pd(R.pd);
    if (R.ctx) ibv_close_device(R.ctx);
  }
  for (int z = 0; z < I.nzc; z++) zc_unmap(I.zc[z].base, I.zc[z].size, I.zc[z].h);  // P176
  if (I.fd >= 0) close(I.fd);
  if (I.numa_size) {  // P132
    auto sym = [](const char* n) {
      void* f = nullptr;
      cudaDriverEntryPointQueryResult q;
      cudaGetDriverEntryPointByVersion(n, &f, 12000, cudaEnableDefault, &q);
      return f;
    };
    auto unmap = reinterpret_cast<CUresult (*)(CUdeviceptr, size_t)>(sym("cuMemUnmap"));
    auto afree = reinterpret_cast<CUresult (*)(CUdeviceptr, size_t)>(sym("cuMemAddressFree"));
    auto release = reinterpret_cast<CUresult (*)(CUmemGenericAllocationHandle)>(sym("cuMemRelease"));
    const CUdeviceptr va = CUdeviceptr(reinterpret_cast<uintptr_t>(I.host));
    if (unmap) unmap(va, I.numa_size);
    if (afree) afree(va, I.numa_size);
    if (release) release(I.numa_h);
  } else if (I.host) {
    cudaFreeHost(I.host);
  }
  if (I.ctl) cudaFree(I.ctl);
  if (I.stamp) cudaFree(I.stamp);
}

int RdmaAR::rails() const { return d_->nr; }
void RdmaAR::tune_small(int ll_max, int ag_ll_max, int fused_chunks, int ll_split, int ll_block, int fused_block) {
  Impl& I = *d_;
  if (fused_block >= (8 << 10) || fused_block == -2) I.fused_block = fused_block == -2 ? kAuto : size_t(fused_block);
  if (ll_block > 0 || ll_block == -2) I.ll_block = ll_block == -2 ? kAuto : size_t(ll_block);
  if (ag_ll_max == -2) I.ag_ll_max = kAuto;
  if (ll_max == -2) I.ll_max = kAuto;
  if (ll_split >= 0) I.ll_split = size_t(ll_split);
  if (fused_chunks >= 0) I.fused_chunks = fused_chunks;
  if (ll_max >= 0) I.ll_max = std::min<size_t>(kLLMaxCap, size_t(ll_max));
  if (ag_ll_max >= 0) I.ag_ll_max = std::min<size_t>(kLLMaxCap, size_t(ag_ll_max));
}
void RdmaAR::tune(int chunk_kb, int copy_blocks, int add_blocks, int ahead, int xfer_only) {
  Impl& I = *d_;
  if (chunk_kb > 0) I.chunk_bytes = size_t(std::max(64, chunk_kb)) << 10;
  if (copy_blocks > 0) I.copy_blocks = copy_blocks;
  if (add_blocks > 0) I.add_blocks = add_blocks;
  if (ahead > 0) I.ahead = ahead;
  if (xfer_only >= 0) I.xfer_only = xfer_only;
}
size_t RdmaAR::max_elems() const { return d_->max_elems; }
int RdmaAR::proxy_cpu() const { return d_->cpu; }
void RdmaAR::probe(bool on) {
  Impl& I = *d_;
  CUDA_CHECK(cudaDeviceSynchronize());
  if (!I.stamp) CUDA_CHECK(cudaMalloc(&I.stamp, 8 * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(I.stamp, 0, 8 * sizeof(unsigned long long)));
  I.post_ns = 0, I.posts = 0;
  I.stamp_on = on;
  I.probe_on = on;
}
std::vector<double> RdmaAR::probe_read() {
  Impl& I = *d_;
  CUDA_CHECK(cudaDeviceSynchronize());
  unsigned long long h[8] = {};
  if (I.stamp) CUDA_CHECK(cudaMemcpy(h, I.stamp, sizeof h, cudaMemcpyDeviceToHost));
  const double c = double(std::max<unsigned long long>(1, h[3])), p = double(std::max<unsigned long long>(1, I.posts.load()));
  return {double(h[3]), h[0] / c, h[1] / c, h[2] / c, h[3] > 1 ? h[4] / (c - 1) : 0.0, double(I.posts.load()),
          double(I.post_ns.load()) / p};
}

// the first kernel of a small call: a programmatic dependent of the preceding kernel with P077 ar_pdl (its kernels
// run griddepcontrol.wait first)
template <typename... P, typename... A>
static void launch_first(void (*k)(P...), int grid, cudaStream_t s, A... args) {
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t cfg{dim3(grid), dim3(kThreads), 0, s, at, patch::on(patch::ar_pdl) ? 1u : 0u};
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, k, args...));
}

// LL layout of a small call (lines of 8 data bytes): G blocks, a.C chunks of a.chunk lines (~8 KiB of data per block,
// ≤ 16 blocks, one chunk per block; a single block sends 2 chunks — one per rail — above 4 KiB)
static int ll_plan(const RdmaAR::Impl& I, Args& a, size_t bytes, size_t ll_block) {
  const size_t lines = (a.n + 3) / 4;
  int G = int(std::min<size_t>(16, std::max<size_t>(1, bytes / ll_block)));
  if (G == 1) {
    a.C = (I.nr > 1 && bytes > I.ll_split && lines >= 2) ? 2 : 1;  // (no empty chunk: mlx5 reads length 0 as 2 GiB)
    a.chunk = (lines + a.C - 1) / a.C;
  } else {  // no empty chunks
    a.chunk = (lines + G - 1) / G;
    a.C = G = int((lines + a.chunk - 1) / a.chunk);
  }
  return G;
}

// ---- P131 rdma_ce: the large path without SM copy / wait kernels ------------------------------------------------
// The same protocol, buffers and bytes as k_copy / k_pack_ag → proxy → k_wait → k_add / k_place_ag: the staging copies
// are copy-engine memcpys (2-D per piece for the all-gather's row pieces), each chunk's doorbell is written by stream
// memory operations (its descriptor, then seq; each write fenced), the wait for the peer's flag is a stream wait
// (GEQ, the wraparound compare of wait_flag). An all-gather then needs no SM at all; an all-reduce / reduce-scatter
// keeps its add kernels (k_add, unchanged: the same sums). Under compute load the kernels' short launches waited for SM
// slots (the copies and the 1-warp wait cannot share an SM with register-heavy GEMM / attention CTAs).
bool ce_ok(const RdmaAR::Impl& I) { return I.ce && !I.captured && patch::on(patch::rdma_ce); }  // (not once captured)
CUdeviceptr dptr(const void* p) { return CUdeviceptr(reinterpret_cast<uintptr_t>(p)); }
void ce_check(CUresult r) {
  if (r != CUDA_SUCCESS) qlog::fail("rdma", "P131 stream memory operation failed (%d)", int(r));
}
// chunk k's doorbell of call seq (parity p): descriptor fields (P118's post mode too), then seq (the proxy reads seq with acquire)
void ce_ring(RdmaAR::Impl& I, int p, int k, int C, size_t src, size_t dst, size_t bytes, unsigned long long seq,
             cudaStream_t s, unsigned long long zc_mode = 0) {
  const CUstream cs = reinterpret_cast<CUstream>(s);
  const char* d = I.dev + offsetof(Hdr, db) + (size_t(p) * kMaxChunks + k) * sizeof(Db);
  ce_check(I.wr64(cs, dptr(d + offsetof(Db, src)), src, 0));
  ce_check(I.wr64(cs, dptr(d + offsetof(Db, dst)), dst, 0));
  ce_check(I.wr64(cs, dptr(d + offsetof(Db, bytes)), bytes, 0));
  ce_check(I.wr64(cs, dptr(d + offsetof(Db, flag)), cuuint64_t(p * kMaxChunks + k), 0));
  ce_check(I.wr64(cs, dptr(d + offsetof(Db, last)), k == C - 1, 0));
  ce_check(I.wr64(cs, dptr(d + offsetof(Db, mode)), (patch::on(patch::ar_bf_post) ? 1 : 0) | zc_mode, 0));  // (P118, as ring(); P176)
  ce_check(I.wr64(cs, dptr(d + offsetof(Db, seq)), seq, 0));
}
void ce_wait(RdmaAR::Impl& I, int p, int k, unsigned long long seq, cudaStream_t s) {
  const char* f = I.dev + offsetof(Hdr, flag) + (size_t(p) * kMaxChunks + k) * sizeof(Flag);
  ce_check(I.wt64(reinterpret_cast<CUstream>(s), dptr(f), seq, CU_STREAM_WAIT_VALUE_GEQ));
}
// the call's seq on the GPU (the kernels that follow — this call's adds, later calls — read ctl->seq)
void ce_seq(RdmaAR::Impl& I, unsigned long long seq, cudaStream_t s) {
  ce_check(I.wr64(reinterpret_cast<CUstream>(s), dptr(&I.ctl->seq), seq, 0));
}

// all-gather call of mc rows (AgDesc g, row bytes W): chunks of whole rows (~chunk_bytes), groups of one chunk per
// rail, packs one group ahead of the waits / places (as k_pack_ag / k_place_ag)
void ag_ce(RdmaAR::Impl& I, const AgDesc& g, int mc, unsigned long long seq, cudaStream_t s) {
  const int p = int(seq & 1);
  const size_t W = size_t(g.W);
  // whole rows of ≤ ce_chunk bytes, at least 2 groups (pipelining) where the rows allow, ≤ kMaxChunks chunks
  int rpc = int(std::max<size_t>(1, std::min(I.ce_chunk, size_t(mc) * W / (2 * I.nr)) / W));
  if ((mc + rpc - 1) / rpc > kMaxChunks) rpc = (mc + kMaxChunks - 1) / kMaxChunks;
  const int C = (mc + rpc - 1) / rpc, P = (C + I.nr - 1) / I.nr;
  const size_t so = kBulk + size_t(p) * I.stride, ro = kBulk + size_t(2 + p) * I.stride;
  ce_seq(I, seq, s);
  auto width = [&](int j) { return size_t((j + 1 < g.np ? g.pre[j + 1] : g.W) - g.pre[j]); };
  auto pack = [&](int k) {
    const int r0 = k * rpc, n = std::min(mc, r0 + rpc) - r0;
    for (int j = 0; j < g.np; j++) {
      const char* sj = g.src[j] + size_t(r0) * g.sld[j];
      CUDA_CHECK(cudaMemcpy2DAsync(I.dev + so + size_t(r0) * W + g.pre[j], W, sj, g.sld[j], width(j), n,
                                   cudaMemcpyDefault, s));
      if (g.own[j] != g.src[j] || g.sld[j] != g.dld[j])  // this rank's own part (in place: already there)
        CUDA_CHECK(cudaMemcpy2DAsync(g.own[j] + size_t(r0) * g.dld[j], g.dld[j], sj, g.sld[j], width(j), n,
                                     cudaMemcpyDefault, s));
    }
    ce_ring(I, p, k, C, so + size_t(r0) * W, ro + size_t(r0) * W, size_t(n) * W, seq, s);
  };
  auto place = [&](int k) {
    const int r0 = k * rpc, n = std::min(mc, r0 + rpc) - r0;
    ce_wait(I, p, k, seq, s);
    for (int j = 0; j < g.np; j++)
      CUDA_CHECK(cudaMemcpy2DAsync(g.peer[j] + size_t(r0) * g.dld[j], g.dld[j], I.dev + ro + size_t(r0) * W + g.pre[j],
                                   W, width(j), n, cudaMemcpyDefault, s));
  };
  const int A = std::min(I.ce_ahead, P);  // packs A groups ahead of the waits / places
  for (int k = 0; k < std::min(C, A * I.nr); k++) pack(k);
  for (int j = 0; j < P; j++) {
    for (int k = (j + A) * I.nr; k < std::min(C, (j + A + 1) * I.nr); k++) pack(k);
    for (int k = j * I.nr; k < std::min(C, (j + 1) * I.nr); k++) place(k);
  }
}

// P120 ar_plan — the small-call plan, measured on the decode / verify collectives (test_comm.py dec, P118 + P119 on):
// LL only up to 8 KiB, all-reduces and all-gathers (the fused path above: 12 / 16 KiB all-reduces −0.5 / −0.6 µs,
// 10–16 KiB gathers −1.2…2.8 µs), LL calls with 4 KiB of data per block (8 / 12 KiB on 2 / 3 blocks: the sysmem
// writes and doorbell fences in parallel, −0.6 µs), the fused all-reduce with 8 KiB per block up to 64 KiB (−0.3…0.5
// µs; above it 16 KiB as before — 8 KiB measured slower at 128 / 256 KiB), a gather's stream byte mapped by a 32-bit
// division. Off: LL up to 16 KiB, 8 KiB per LL block, 16 KiB per fused block, 64-bit.
struct Plan {
  size_t ll, ag_ll, ll_block, fused_small, fused_block;  // fused_small: per block up to 64 KiB, fused_block above
  int fast;
};
static Plan small_plan(const RdmaAR::Impl& I) {
  const bool p = patch::on(patch::ar_plan);
  const size_t fb = I.fused_block;
  return {I.ll_max != kAuto ? I.ll_max : size_t(p ? 8 << 10 : 16 << 10),
          I.ag_ll_max != kAuto ? I.ag_ll_max : size_t(p ? 8 << 10 : 16 << 10),
          I.ll_block != kAuto ? I.ll_block : size_t(p ? 4 << 10 : 8 << 10),
          fb != kAuto ? fb : size_t(p ? 8 << 10 : 16 << 10), fb != kAuto ? fb : size_t(16 << 10), p ? 1 : 0};
}

// P119 / P131: the next call's sequence number — ONE host count, advanced exactly once per call on every path (the GPU
// runs the calls in issue order, so after call n ctl->seq = n on either rank); a.seq for P119's small calls (0: the
// kernels count on the device) and the LL sweep. Once a stream was captured (a graph replay advances ctl->seq without
// the host) neither host-seq path is taken again (P119: the device count; P131: the kernel path).
static unsigned long long call_seq(RdmaAR::Impl& I, Args& a, cudaStream_t s, bool ll) {
  const unsigned long long seq = ++I.hseq;
  a.seq = 0, a.sweep = 0;
  const bool hs = patch::on(patch::ar_host_seq);
  if (I.captured || !(hs || ce_ok(I))) return seq;
  cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
  CUDA_CHECK(cudaStreamIsCapturing(s, &st));
  if (st != cudaStreamCaptureStatusNone) return I.captured = true, seq;
  if (!hs) return seq;
  a.seq = seq;
  if (ll && seq - I.hswept[seq & 1] >= (1u << 20)) a.sweep = 1, I.hswept[seq & 1] = seq;
  return seq;
}

// n bf16 slots at buf (an fp32 call: 2 slots per element, F32 adds); src: what is sent (buf for an all-reduce)
template <bool F32>
void ar_run(RdmaAR::Impl& I, const bf16* src, bf16* buf, size_t n, cudaStream_t s) {
  for (size_t o = 0; o < n; o += I.max_elems) {
    Args a{};
    a.buf = buf + o, a.src = src + o, a.n = std::min(I.max_elems, n - o);
    a.base = I.dev, a.stride = I.stride, a.ctl = I.ctl, a.timeout_ns = I.timeout_ns;
    a.stamp = I.stamp_on ? I.stamp : nullptr;
    a.mode = patch::on(patch::ar_bf_post) ? 1 : 0;
    const size_t bytes = a.n * 2;
    const Plan pl = small_plan(I);
    const unsigned long long seq = call_seq(I, a, s, bytes <= pl.ll);  // (P119 / P131: the one count)
    if (bytes > kFusedMax) a.seq = 0;  // (the large path counts on the device)
    if (bytes <= pl.ll) {  // LL: ll_block of data per block (≤ 16 blocks), one chunk per block; one block → 2 rails
      const int G = ll_plan(I, a, bytes, pl.ll_block);
      launch_first(k_ll<F32>, G, s, a);
    } else if (bytes <= kFusedMax) {
      // fused: 8 or 16 KiB per block (the plan; 2–32 blocks), each block's slice is its own chunk (no cross-block sync;
      // the NIC starts on a slice as soon as its writes have drained), chunks alternate rails
      // (fused_chunks > 0: at most that many chunks — a chunk's blocks counted, its last one rings)
      const size_t fb = bytes <= (64 << 10) ? pl.fused_small : pl.fused_block;
      const int G0 = int(std::min<size_t>((256 << 10) / fb, std::max<size_t>(I.nr, bytes / fb)));
      const size_t pb = ((a.n + G0 - 1) / G0 + 63) & ~size_t(63);
      const int G = int((a.n + pb - 1) / pb);
      a.bpc = (I.fused_chunks > 0 && G > I.fused_chunks) ? (G + I.fused_chunks - 1) / I.fused_chunks : 1;
      a.chunk = pb * a.bpc;
      a.C = (G + a.bpc - 1) / a.bpc;
      launch_first(k_fused<F32>, G, s, a);
    } else {
      // large: groups of one chunk per rail. Copies run one group ahead of the waits/adds in short bursts — one big
      // copy up front saturates DRAM and starves the NIC's DMA (measured: almost no transfer until it ended).
      int C = std::min<int>(kMaxChunks, 2 * int((bytes + 2 * I.chunk_bytes - 1) / (2 * I.chunk_bytes)));
      a.chunk = ((a.n + C - 1) / C + 63) & ~size_t(63);
      a.C = int((a.n + a.chunk - 1) / a.chunk);
      const int P = (a.C + I.nr - 1) / I.nr, A = std::min(I.ahead, P);  // copies run A groups ahead
      a.skip = I.xfer_only;
      auto group = [&](int j, int& k0, int& k1) { k0 = j * I.nr, k1 = std::min(a.C, k0 + I.nr); };
      int k0, k1;
      if (ce_ok(I) && !a.skip) {  // P131: copy-engine packs, stream-memop doorbells and waits, the same adds
        // chunks of ≤ ce_chunk bytes, at least 2 groups (4 chunks on 2 rails)
        const size_t cb = std::min(I.ce_chunk, std::max<size_t>(64 << 10, bytes / (2 * I.nr)));
        C = std::min<int>(kMaxChunks, std::max(2 * I.nr, I.nr * int((bytes + I.nr * cb - 1) / (I.nr * cb))));
        a.chunk = ((a.n + C - 1) / C + 63) & ~size_t(63);
        a.C = int((a.n + a.chunk - 1) / a.chunk);
        const int P = (a.C + I.nr - 1) / I.nr, A = std::min(I.ce_ahead, P);
        const int p = int(seq & 1);
        const size_t so = kBulk + size_t(p) * I.stride, ro = kBulk + size_t(2 + p) * I.stride;
        ce_seq(I, seq, s);
        auto pack = [&](int j) {
          group(j, k0, k1);
          for (int k = k0; k < k1; k++) {
            const size_t c0 = size_t(k) * a.chunk, c1 = std::min(a.n, c0 + a.chunk);
            CUDA_CHECK(cudaMemcpyAsync(I.dev + so + 2 * c0, a.src + c0, 2 * (c1 - c0), cudaMemcpyDefault, s));
            ce_ring(I, p, k, a.C, so + 2 * c0, ro + 2 * c0, 2 * (c1 - c0), seq, s);
          }
        };
        for (int j = 0; j < A; j++) pack(j);
        for (int j = 0; j < P; j++) {
          if (j + A < P) pack(j + A);
          group(j, k0, k1);
          for (int k = k0; k < k1; k++) ce_wait(I, p, k, seq, s);
          k_add<F32><<<I.ce_add, kThreads, 0, s>>>(a, k0, k1);
        }
        CUDA_CHECK(cudaGetLastError());
        continue;
      }
      for (int j = 0; j < A; j++) {
        group(j, k0, k1);
        k_copy<<<I.copy_blocks, kThreads, 0, s>>>(a, k0, k1, j == 0);
      }
      for (int j = 0; j < P; j++) {
        if (j + A < P) {
          group(j + A, k0, k1);
          k_copy<<<I.copy_blocks, kThreads, 0, s>>>(a, k0, k1, 0);
        }
        group(j, k0, k1);
        k_wait<<<1, 32, 0, s>>>(a, k0, k1);
        k_add<F32><<<I.add_blocks, kThreads, 0, s>>>(a, k0, k1);
      }
    }
    CUDA_CHECK(cudaGetLastError());
  }
}

// P076: rows in calls of ≤ max_elems·2 stream bytes; each call takes the AR's path for its size (LL lines up to
// ag_ll_max, then fused, then large)
static void ag_run(RdmaAR::Impl& I, const AgPiece* pc, int np, int M, cudaStream_t s, bool bg) {
  const size_t fused_max = bg ? std::min(I.ag_fused_max, kFusedMax) : I.ag_fused_max;
  if (np < 1 || np > kAgMaxPieces || M < 0) qlog::fail("rdma", "all_gather: %d pieces, %d rows", np, M);
  if (M == 0) return;
  int W = 0;
  uint64_t bits = 0;  // alignment: 16 bytes everywhere, or 8 when every call takes the LL path (8-byte lines)
  for (int j = 0; j < np; j++) {
    const AgPiece& q = pc[j];
    bits |= reinterpret_cast<uintptr_t>(q.src) | reinterpret_cast<uintptr_t>(q.dst) | uint64_t(q.src_ld) |
            uint64_t(q.dst_ld) | uint64_t(q.off) | uint64_t(q.stride) | uint64_t(q.width);
    if (q.width <= 0) qlog::fail("rdma", "all_gather piece %d: width %d", j, q.width);
    W += q.width;
  }
  // LL up to the plan's ag_ll bytes a call; 8-byte aligned gathers (P063's 520-byte rows) always LL, up to kAgLL8
  const Plan pl = small_plan(I);
  const size_t ll_lim = (bits & 15) ? std::max(pl.ag_ll, kAgLL8) : pl.ag_ll;
  if ((bits & 7) || ((bits & 15) && size_t(M) * W > ll_lim))
    qlog::fail("rdma", "all_gather: addresses / strides / widths must be 16-byte aligned (8 up to %zu bytes a call)", ll_lim);
  const size_t cap = I.max_elems * 2;
  if (size_t(W) > cap) qlog::fail("rdma", "all_gather row of %d bytes > staging %zu", W, cap);
  const int rows = int(std::min<size_t>(size_t(M), cap / size_t(W)));
  for (int m0 = 0; m0 < M; m0 += rows) {
    const int mc = std::min(rows, M - m0);
    AgDesc g{};
    g.W = W, g.np = np, g.fast = pl.fast;
    for (int j = 0, pre = 0; j < np; j++) {
      const AgPiece& q = pc[j];
      char* d = static_cast<char*>(q.dst) + size_t(m0) * q.dst_ld + q.off;
      g.src[j] = static_cast<const char*>(q.src) + size_t(m0) * q.src_ld;
      g.own[j] = d + I.rank * q.stride, g.peer[j] = d + (1 - I.rank) * q.stride;
      g.sld[j] = q.src_ld, g.dld[j] = q.dst_ld, g.pre[j] = pre;
      pre += q.width;
    }
    Args a{};
    const size_t bytes = size_t(mc) * W;
    a.n = bytes / 2;
    a.base = I.dev, a.stride = I.stride, a.ctl = I.ctl, a.timeout_ns = I.timeout_ns;
    a.stamp = I.stamp_on ? I.stamp : nullptr;
    a.mode = patch::on(patch::ar_bf_post) ? 1 : 0;
    const unsigned long long seq = call_seq(I, a, s, bytes <= ll_lim);  // (P119 / P131: the one count)
    if (bytes > fused_max) a.seq = 0;  // (the large path counts on the device)
    if (bytes <= ll_lim) {
      const int G = ll_plan(I, a, bytes, pl.ll_block);
      launch_first(k_ll_ag, G, s, a, g);
    } else if (bytes <= fused_max) {
      // fused up to 16 MiB (logits rows): ≥ 32 KiB per block, ≤ kMaxChunks blocks — the NIC starts on the first chunks
      // while the others are copied (the large path's two copy blocks and one group in flight: R = 16 logits rows
      // 463 µs, slower than NCCL's 403)
      const int G = int(std::min<size_t>(kMaxChunks, std::max<size_t>(I.nr, bytes >> 15)));
      a.chunk = ((a.n + G - 1) / G + 63) & ~size_t(63);
      a.C = int((a.n + a.chunk - 1) / a.chunk);
      launch_first(k_fused_ag, a.C, s, a, g);
    } else if (ce_ok(I)) {  // P131
      ag_ce(I, g, mc, seq, s);
    } else {
      int C = std::min<int>(kMaxChunks, 2 * int((bytes + 2 * I.chunk_bytes - 1) / (2 * I.chunk_bytes)));
      a.chunk = ((a.n + C - 1) / C + 63) & ~size_t(63);
      a.C = int((a.n + a.chunk - 1) / a.chunk);
      const int P = (a.C + I.nr - 1) / I.nr;
      auto group = [&](int j, int& k0, int& k1) { k0 = j * I.nr, k1 = std::min(a.C, k0 + I.nr); };
      int k0, k1;
      group(0, k0, k1);
      k_pack_ag<<<I.copy_blocks, kThreads, 0, s>>>(a, g, k0, k1, 1);
      for (int j = 0; j < P; j++) {
        if (j + 1 < P) {
          group(j + 1, k0, k1);
          k_pack_ag<<<I.copy_blocks, kThreads, 0, s>>>(a, g, k0, k1, 0);
        }
        group(j, k0, k1);
        k_wait<<<1, 32, 0, s>>>(a, k0, k1);
        k_place_ag<<<I.add_blocks, kThreads, 0, s>>>(a, g, k0, k1);
      }
    }
    CUDA_CHECK(cudaGetLastError());
  }
}

void RdmaAR::all_reduce_bf16(bf16* buf, size_t n, cudaStream_t s) { ar_run<false>(*d_, buf, buf, n, s); }
void RdmaAR::reduce_scatter_bf16(bf16* buf, size_t n, cudaStream_t s) {  // P094
  const int r = d_->rank;
  ar_run<false>(*d_, buf + size_t(1 - r) * n, buf + size_t(r) * n, n, s);
}
void RdmaAR::all_gather(const AgPiece* p, int np, int M, cudaStream_t s, bool bg) { ag_run(*d_, p, np, M, s, bg); }
void RdmaAR::all_reduce_f32(float* buf, size_t n, cudaStream_t s) {
  ar_run<true>(*d_, reinterpret_cast<bf16*>(buf), reinterpret_cast<bf16*>(buf), 2 * n, s);
}

// ---- P176 zc_a2a (rdma_ar.h) --------------------------------------------------------------------------------------
void* RdmaAR::zc_alloc(size_t bytes) {
  Impl& I = *d_;
  struct Wz {
    uint32_t magic, ok;
    uint64_t size, base;
    uint32_t rkey[kMaxRails];
  } me{}, pe{};
  Impl::Zc z;
  bool ok = I.nzc < kMaxZc && I.fd >= 0 && bytes > 0;
  auto gran_f = reinterpret_cast<CUresult (*)(size_t*, const CUmemAllocationProp*, CUmemAllocationGranularity_flags)>(
      drv_sym("cuMemGetAllocationGranularity"));
  auto create = reinterpret_cast<CUresult (*)(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*,
                                              unsigned long long)>(drv_sym("cuMemCreate"));
  auto reserve = reinterpret_cast<CUresult (*)(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long)>(
      drv_sym("cuMemAddressReserve"));
  auto map = reinterpret_cast<CUresult (*)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long)>(
      drv_sym("cuMemMap"));
  auto access = reinterpret_cast<CUresult (*)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t)>(drv_sym("cuMemSetAccess"));
  ok = ok && gran_f && create && reserve && map && access;
  CUmemAllocationProp prop{};  // pinned on the host NUMA node, mapped for this GPU and the CPU at one address (P132's)
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  prop.location.id = 0;
  size_t gran = 0;
  if (ok) ok = gran_f(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) == CUDA_SUCCESS && gran;
  if (ok) z.size = (bytes + gran - 1) / gran * gran, ok = create(&z.h, z.size, &prop, 0) == CUDA_SUCCESS;
  CUdeviceptr va = 0;
  if (ok) ok = reserve(&va, z.size, gran, 0, 0) == CUDA_SUCCESS;
  if (ok) {
    z.base = reinterpret_cast<char*>(va);
    ok = map(va, z.size, 0, z.h, 0) == CUDA_SUCCESS;
    if (!ok) {  // (reserved, not mapped)
      if (auto afree = reinterpret_cast<CUresult (*)(CUdeviceptr, size_t)>(drv_sym("cuMemAddressFree"))) afree(va, z.size);
      z.base = nullptr;
    }
  }
  if (ok) {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUmemAccessDesc acc[2]{};
    acc[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE, acc[0].location.id = dev;
    acc[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    acc[1].location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA, acc[1].location.id = 0;
    acc[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    ok = access(va, z.size, acc, 2) == CUDA_SUCCESS;
  }
  if (ok) {
    CUDA_CHECK(cudaMemset(z.base, 0, z.size));
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int r = 0; r < I.nr && ok; r++) {
      z.mr[r] = ibv_reg_mr(I.rail[r].pd, z.base, z.size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
      ok = z.mr[r] != nullptr;
      if (!ok) qlog::warn("rdma", "rank%d · P176: %s MR registration failed (%.0f MiB): %s", I.rank, I.rail[r].name.c_str(),
                          z.size / 1048576.0, strerror(errno));
    }
  }
  me.magic = kMagic, me.ok = ok, me.size = z.size, me.base = reinterpret_cast<uint64_t>(z.base);
  for (int r = 0; r < I.nr; r++) me.rkey[r] = z.mr[r] ? z.mr[r]->rkey : 0;
  if (I.fd >= 0) {
    send_all(I.fd, &me, sizeof me);
    recv_all(I.fd, &pe, sizeof pe);
  }
  if (!ok || pe.magic != kMagic || !pe.ok || pe.size != me.size) {  // either side failed: neither uses it
    for (int r = 0; r < I.nr; r++)
      if (z.mr[r]) ibv_dereg_mr(z.mr[r]);
    zc_unmap(z.base, z.base ? z.size : 0, z.h);
    qlog::warn("rdma", "rank%d · P176: no zero-copy region of %.0f MiB (this rank %s, peer %s)", I.rank, bytes / 1048576.0,
               ok ? "ok" : "failed", pe.ok ? "ok" : "failed");
    return nullptr;
  }
  z.peer_base = pe.base;
  for (int r = 0; r < I.nr; r++) z.peer_rkey[r] = pe.rkey[r];
  I.zc[I.nzc] = z;
  std::atomic_thread_fence(std::memory_order_release);  // (the proxy reads it once a put's doorbell arrives)
  I.nzc++;
  return z.base;
}

// region of [p, p + n) (−1: not inside one) and its offset
static int zc_find(const RdmaAR::Impl& I, const void* p, size_t n, size_t& off) {
  const char* c = static_cast<const char*>(p);
  for (int z = 0; z < I.nzc; z++)
    if (c >= I.zc[z].base && n <= I.zc[z].size && size_t(c - I.zc[z].base) <= I.zc[z].size - n)
      return off = size_t(c - I.zc[z].base), z;
  return -1;
}

bool RdmaAR::put_ready() const { return d_->nzc && ce_ok(*d_); }

// dst inside one region; src inside one (then no copy) or anywhere the GPU reads (then staged through send[p], ≤ the
// staging per call: put() splits a longer range in several calls)
bool RdmaAR::put_ok(const ZcPut* p, int np) const {
  const Impl& I = *d_;
  if (!put_ready() || np < 1) return false;
  for (int j = 0; j < np; j++) {
    size_t o;
    if (!p[j].bytes || zc_find(I, p[j].dst, p[j].bytes, o) < 0) return false;
  }
  return true;
}

// Calls of ≤ kMaxChunks chunks: a source in a region goes in one chunk per rail; a staged one (copy engine → send[p],
// then its doorbell: the NIC starts on a chunk while the next is copied) in chunks of ≤ ce_chunk, ≤ the staging a call.
// Then the waits for the peer's chunks (its pieces have the same sizes: the same chunks) and, if a source is in a
// region, for `sent` (this rank's WRITEs read it).
void RdmaAR::put(const ZcPut* pc, int np, cudaStream_t s) {
  Impl& I = *d_;
  if (!put_ok(pc, np)) qlog::fail("rdma", "put: destination outside the zero-copy regions or the stream-memop path off");
  struct Ch {
    const char* src;  // staged: the GPU address copied; else null
    size_t so, dst, bytes;
    unsigned long long mode;
  };
  std::vector<std::vector<Ch>> calls(1);
  size_t staged = 0;
  for (int j = 0; j < np; j++) {
    size_t so = 0, dof = 0;
    const int rs = zc_find(I, pc[j].src, pc[j].bytes, so), rd = zc_find(I, pc[j].dst, pc[j].bytes, dof);
    const size_t B = pc[j].bytes;
    const unsigned long long m = kDbZc | (unsigned long long)(rs < 0 ? kZcStaged : rs) << kDbZcSrc |
                                 (unsigned long long)rd << kDbZcDst;
    const size_t per = rs >= 0 ? ((B + I.nr - 1) / I.nr + 63) & ~size_t(63)
                               : std::min(I.ce_chunk, std::max<size_t>(64 << 10, ((B + 2 * I.nr - 1) / (2 * I.nr) + 63) & ~size_t(63)));
    for (size_t o = 0; o < B; o += per) {
      const size_t n = std::min(per, B - o);
      if (calls.back().size() == size_t(kMaxChunks) || (rs < 0 && staged + n > I.stride)) calls.emplace_back(), staged = 0;
      calls.back().push_back({rs < 0 ? static_cast<const char*>(pc[j].src) + o : nullptr, so + o, dof + o, n, m});
      if (rs < 0) staged += n;
    }
  }
  for (auto& ch : calls) {
    Args a{};
    const unsigned long long seq = call_seq(I, a, s, false);  // (P119 / P131: the one count)
    if (I.captured) qlog::fail("rdma", "put inside a stream capture");
    const int p = int(seq & 1), C = int(ch.size());
    const size_t sb = kBulk + size_t(p) * I.stride;  // send[p]
    ce_seq(I, seq, s);
    bool any_zc = false;
    size_t off = 0;
    const int A = I.ce_ahead * I.nr;  // staged copies run ≤ A chunks ahead of the peer's arrivals (P131's pacing: a
    int w = 0;                        // burst of copies starves the NIC's DMA)
    for (int k = 0; k < C; k++) {
      size_t src = ch[k].so;
      if (ch[k].src) {  // staged: copy engine → send[p] at the call's running offset
        if (k >= A) ce_wait(I, p, w, seq, s), w++;
        CUDA_CHECK(cudaMemcpyAsync(I.dev + sb + off, ch[k].src, ch[k].bytes, cudaMemcpyDefault, s));
        src = sb + off, off += ch[k].bytes;
      } else {
        any_zc = true;
      }
      ce_ring(I, p, k, C, src, ch[k].dst, ch[k].bytes, seq, s, ch[k].mode);
    }
    for (; w < C; w++) ce_wait(I, p, w, seq, s);
    if (any_zc)
      ce_check(I.wt64(reinterpret_cast<CUstream>(s), dptr(I.dev + offsetof(Hdr, zc_sent)), seq, CU_STREAM_WAIT_VALUE_GEQ));
  }
}

}  // namespace e55
