// rdma_ar.h — P001 rdma_ar: two-rank in-place bf16 / fp32 all-reduce (sum) over the CX7 RoCE rails with RDMA WRITE.
//
// Data path (no host callbacks, CUDA-graph capturable; both ranks must issue the same calls in the same order, and the
// calls of one RdmaAR must be ordered on the GPU — one in flight at a time, like the engine's s_/cs_ event chains):
//   1. GPU: copy buf → pinned host staging (parity p = call & 1), then ring a doorbell per chunk {seq, src, dst, len}
//      in host memory (release, system scope)
//   2. CPU proxy thread (busy-polls the doorbells): per chunk one RDMA WRITE to the peer's staging (chunks alternate
//      the rails), plus — except for LL — an inline 8-byte WRITE of seq to the peer's flag slot on the same QP
//   3. GPU: wait until the peer's data is there, buf = bf16(float(buf) + float(peer)) — the same bits as NCCL's 2-rank
//      bf16 sum (commutative; one rounding of the exact fp32 sum = the correctly rounded bf16 add)
// Three paths by size (data bytes): ≤ 16 KiB LL — 16-byte lines {d0, flag, d1, flag} (NCCL's LL format; relies on
// the NIC writing 8-byte words atomically), one kernel, no flag WRITE, the GPU polls the lines; ≤ 1 MiB fused — one
// kernel, one chunk per block (~16 KiB), flag after each chunk; larger — 1 MiB chunks, a 2-block copy kernel per pair
// of chunks running one pair ahead, a 1-warp wait kernel (nanosleep) and a 4-block add kernel per pair: while the NIC
// moves data no SMs are held, and the short copy/add bursts leave DRAM bandwidth to the NIC's DMA.
// Parity double buffering makes reuse safe without credits: a rank can only start call i+2 after the peer finished
// call i+1, which it started after finishing call i (so peer recv[p] is consumed and our send[p] has been read).
// The NIC only touches the pinned host region (cudaHostAlloc, GPU-mapped): no GPUDirect / dma-buf needed
// (GB10: cudaMalloc memory cannot be exported as dma-buf).
//
// Rails: env E55_RDMA_HCA, else NCCL_IB_HCA (comma list, same rail order on both ranks, e.g. C "roceP2p1s0f0,
// rocep1s0f0" ↔ D "roceP2p1s0f1,rocep1s0f1"); port 1; RoCE v2 IPv4 GID found per HCA (E55_RDMA_GID_INDEX overrides).
// Proxy CPU: the highest-capacity CPU allowed (E55_RDMA_CPU=n to choose, -1 = unpinned). E55_RDMA_TIMEOUT_S=s makes a
// GPU wait longer than s seconds trap instead of hanging (default 0 = wait forever, like NCCL). Tuning (defaults are
// the measured best): E55_RDMA_LL_MAX (bytes, 16384), E55_RDMA_CHUNK_KB (1024), E55_RDMA_COPY_BLOCKS (2),
// E55_RDMA_ADD_BLOCKS (4), E55_RDMA_AHEAD (copy groups ahead of the adds, 1), E55_RDMA_AG_LL_MAX (bytes, all-gathers,
// P120's plan: 8192, off 16384), E55_RDMA_AG_FUSED_KB (all-gathers up to this size in one kernel, 16384).
// Memory: pinned host 4 × max_elems × 2 B + 4 MiB (LL lines, flags, doorbells), registered on every rail; calls with
// n > max_elems are split into several calls of ≤ max_elems. Device: 1 KiB of counters.
//
// P076 rdma_gather — all-gather of row pieces over the same transport (same doorbells, proxy, parity and LL / fused /
// large paths; the add becomes a placement): per row m and piece j, this rank's `width` bytes at src + m·src_ld land at
// dst + m·dst_ld + off + rank·stride on both ranks (the own copy written by the sending kernel, the peer's from the
// received bytes). Pieces of a row travel together (one call for several outputs, e.g. different buffers / dtypes).
// Bytes move unchanged — no arithmetic.
// P077 ar_pdl — the first kernel of a small call (LL / fused, the decode sizes) is launched as a programmatic
// dependent of the preceding kernel: it is resident when that one ends and runs griddepcontrol.wait before touching
// anything (counters included), so it only hides the launch gap; results unchanged.
// P094 sp_resid — reduce-scatter of two equal halves: the all-reduce's path with a separate source — each rank sends
// the peer's half and adds the received half into its own (the same bf16(float(a) + float(b)) per element).
// Small calls (decode / verify; per-call phases measured with probe(), test_comm.py dec — ~2 µs GPU start → doorbell,
// 4–6 µs doorbell → the peer's data on the GPU, ~1 µs to the next kernel):
// P118 ar_bf_post — a chunk's data WRITE and flag WRITE (fused / large paths) as two single-WQE posts: libmlx5 sends a lone WQE
// by BlueFlame (inside the MMIO write), a chain of two by a doorbell that the NIC answers by fetching the WQEs first.
// P119 ar_host_seq — the call's sequence number passed from the host (calls run on the GPU in issue order): no device
// counter read behind a block barrier at the start, no grid count (done atomics) or sweep-counter read at the end. The
// device counter is still stored (block 0) for the large path, and used alone once a stream was captured (a graph
// replay advances it without the host).
// P120 ar_plan — the small-call plan: all-gathers LL only up to 8 KiB (fused above), LL calls with 4 KiB of data per
// block, a gather's stream offsets by 32-bit division.
// All three move the same bytes into the same places: results unchanged.
// P176 zc_a2a — zero-copy puts: zc_alloc() gives NIC-registered memory (cuMemCreate on the host NUMA node, GPU + CPU
// mapped at one address — streams at cudaMalloc speed, but its lines do not stay in L2 across kernels (8 MiB written
// by one kernel, read by the next: 215 vs 835 GB/s), so only receive rows go there; registered on every rail, base +
// rkeys exchanged with the peer over the bootstrap connection). put(): each rank RDMA-WRITEs byte ranges into the
// peer's region with no copy on the receiving side (sources outside a region staged by the copy engine as P131): the
// call's doorbells (stream memory operations, after a system-scope fence: the producers' writes are visible to the
// NIC) → the proxy writes a ready seq to the peer and waits for the peer's (the peer's stream reached the call: its
// destination is no longer read by earlier work) → data WRITEs + flag WRITEs → the proxy waits for the WRITEs'
// completions and stores `sent`; the stream waits for the peer's flags (and for `sent` when a source is in a region:
// reusable then). The same bytes land where the receiver reads them.
#pragma once
#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <vector>

#include "core/common.h"

namespace e55 {

// one piece of a row-wise all-gather (bytes; width, strides, offsets and addresses multiples of 16)
struct AgPiece {
  const void* src;   // this rank's part: rows at src + m·src_ld
  void* dst;         // full rows at dst + m·dst_ld; rank r's part at + off + r·stride
  long long src_ld, dst_ld, off, stride;
  int width;
};
constexpr int kAgMaxPieces = 5;

// P176: one byte range of a put — src: GPU-readable bytes of this rank (in zc_alloc memory: sent from there; else staged
// by the copy engine); dst = where it lands on the peer, given as this rank's address of the same place (the same
// offset in the same-numbered region)
struct ZcPut {
  const void* src;
  const void* dst;
  size_t bytes;
};

class RdmaAR {
 public:
  // TCP bootstrap only for the QP exchange: rank 0 listens on head_ip:port, rank 1 connects (retries for 300 s).
  RdmaAR(int rank, const std::string& head_ip, int port, size_t max_elems);
  ~RdmaAR();  // call after the GPU work that uses it has finished
  RdmaAR(const RdmaAR&) = delete;
  RdmaAR& operator=(const RdmaAR&) = delete;

  void all_reduce_bf16(bf16* buf, size_t n, cudaStream_t s);  // in place, sum, stream-ordered
  void all_reduce_f32(float* buf, size_t n, cudaStream_t s);  // the same for fp32 (fp32 adds; max_elems counts bf16)
  // P094: buf [2·n] bf16; rank r's half [r·n, (r+1)·n) ← its own + the peer's (the peer's half is sent, not changed)
  void reduce_scatter_bf16(bf16* buf, size_t n, cudaStream_t s);
  // P076: rows [M] of np ≤ kAgMaxPieces pieces (rows split in several calls when a call exceeds the staging).
  // bg (P094: gathers under compute): the one-kernel fused path only up to 1 MiB, as the all-reduce — above it the
  // large path, so no CTAs sit on SMs (spinning, at the comm stream's priority) while the NIC moves the data
  void all_gather(const AgPiece* p, int np, int M, cudaStream_t s, bool bg = false);
  // P176 (above). zc_alloc: both ranks, the same sizes in the same order (nullptr on both if either fails); freed and
  // deregistered by the destructor. put_ok: every range inside one region (src and dst) and the stream-memop path
  // usable (P131's: no stream ever captured). put: both ranks, the same piece sizes, in call order like the others.
  void* zc_alloc(size_t bytes);
  bool put_ready() const;  // a region exists and the stream-memop path is usable (no stream ever captured)
  bool put_ok(const ZcPut* p, int np) const;
  void put(const ZcPut* p, int np, cudaStream_t s);
  int rails() const;
  size_t max_elems() const;
  // tests: large-call tuning in-process (> 0 / ≥ 0 to change): chunk KiB, copy / add blocks, copy groups ahead of the
  // adds, xfer_only (1: move the bytes without the copy / add — the transport's own rate; results are garbage)
  void tune(int chunk_kb, int copy_blocks, int add_blocks, int ahead, int xfer_only);
  // tests: per-call phase times of the small paths (LL / fused; block 0's globaltimer) and the proxy's post cost.
  // probe(true) zeroes and starts, probe(false) stops; probe_read() → {calls, start → doorbell ns, doorbell → peer data
  // in ns, data in → end ns, gap end → next start ns, posts, ns per post}
  void probe(bool on);
  // tests: LL limits (bytes), the fused all-reduce's chunk cap (0: one per block), the one-block LL call's rail split
  // (bytes), LL / fused all-reduce data bytes per block; ≥ 0 (blocks: > 0) to change, −2: P120's plan (gathers' LL
  // limit and the block sizes)
  void tune_small(int ll_max, int ag_ll_max, int fused_chunks, int ll_split, int ll_block, int fused_block);
  std::vector<double> probe_read();
  int proxy_cpu() const;

  struct Impl;

 private:
  std::unique_ptr<Impl> d_;
};

}  // namespace e55
