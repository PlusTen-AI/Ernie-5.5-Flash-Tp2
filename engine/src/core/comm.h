// comm.h — TP=2 collectives between C (rank 0) and D (rank 1) over the CX7 link.
// Bootstrap: rank 0 listens on <head_ip>:<port>, rank 1 connects and receives the NCCL unique id. NCCL then runs over
// RoCE (NCCL_IB_HCA / NCCL_IB_GID_INDEX / NCCL_SOCKET_IFNAME from the environment, as in tools/dump/run_dump.sh).
// All-reduces (bf16 and fp32) go over P001 rdma_ar (dual-rail RDMA WRITE, kernels/comm/rdma_ar.h; its own QPs,
// bootstrap on port + 1) when the patch is on, else NCCL; all-gathers too with P076 rdma_gather. Both are the same sums
// (2 ranks: one commutative add) and the same bytes, so the patches change no bits.
//
// P075 ar_lanes — asynchronous collectives: *_async issues the collective on the comm stream (highest priority, so its
// short kernels get SMs as soon as compute CTAs retire) after the work queued on s so far and returns a ticket;
// join(ticket, s) makes s wait for it (before anything reads the result or writes the buffer). While tickets are
// outstanding, a plain collective also runs on the comm stream (then s waits for it): collectives keep their issue
// order on the GPU — the transport takes one call at a time — and both ranks must issue the same calls in the same
// order. With the patch off, *_async runs the plain collective on s and returns −1 (join(−1) does nothing).
//
// P094 sp_resid — reduce_scatter_bf16: buf [2·n] bf16 (two partials of rows split in halves): rank r's half
// [r·n, (r+1)·n) ends as the all-reduce would leave it — the same bf16(float(a) + float(b)) of the same two values
// (RDMA: rdma_ar's add with the peer's half as the source; NCCL: an all-reduce of both halves). The peer's half is
// left as sent (this rank's partial).
//
// P176 zc_a2a — zero-copy puts (rdma_ar.h): zc_alloc gives NIC-registered activation memory (nullptr without the RDMA
// transport or if either rank cannot register it: then the caller keeps cudaMalloc memory); put / put_async move byte
// ranges of it straight into the peer's (ZcPut: dst as this rank's address of the same place) — the call's bytes land
// where the equivalent all-gather would place them, with no staging copies. zc_ok: the transport can take these
// ranges now (both ranks decide alike: same patches, same regions).
//
// E55_COMM_TRACE=<label> (measurement; read at every async issue, so a test switches it per window): a wall-time
// timeline of the asynchronous collectives from timing events, no synchronization added — per call when s reaches it,
// when the comm stream takes it and when it ends, per join the time s waits; mark() adds a point of s. Written at close
// to E55_COMM_TRACE_FILE.rank<r> (TSV; events pre-created at start when that is set).
#pragma once
#include <cuda_runtime.h>
#include <nccl.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "core/common.h"
#include "kernels/comm/rdma_ar.h"  // AgPiece, ZcPut

namespace e55 {

class Comm {
 public:
  Comm(int rank, const std::string& head_ip, int port);
  ~Comm();
  int rank() const { return rank_; }
  void all_reduce_bf16(bf16* buf, size_t n, cudaStream_t s);                     // in place, sum
  void reduce_scatter_bf16(bf16* buf, size_t n, cudaStream_t s);                 // P094 (above): n per half
  void all_gather_bf16(const bf16* send, bf16* recv, size_t n_per_rank, cudaStream_t s);
  void all_reduce_f32(float* buf, size_t n, cudaStream_t s);
  void all_gather_f32(const float* send, float* recv, size_t n_per_rank, cudaStream_t s);  // recv = [rank0 | rank1]
  // rows [M] of np ≤ kAgMaxPieces pieces (AgPiece, rdma_ar.h: rank r's part of row m lands at dst + m·dst_ld + off +
  // r·stride on both ranks; 16-byte aligned). RDMA with P076, else NCCL (packed through a scratch buffer).
  void all_gather_rows(const AgPiece* p, int np, int M, cudaStream_t s);
  void barrier(cudaStream_t s);

  // P075 (above)
  int all_reduce_bf16_async(bf16* buf, size_t n, cudaStream_t s);
  int all_reduce_f32_async(float* buf, size_t n, cudaStream_t s);
  int reduce_scatter_bf16_async(bf16* buf, size_t n, cudaStream_t s);
  int all_gather_rows_async(const AgPiece* p, int np, int M, cudaStream_t s);  // (RDMA: rdma_ar's bg path)
  // P176 (above)
  void* zc_alloc(size_t bytes);
  bool zc_ok(const ZcPut* p, int np) const;
  bool zc_ready() const;  // puts possible now (RDMA, a region, the stream-memop path)
  void put(const ZcPut* p, int np, cudaStream_t s);
  int put_async(const ZcPut* p, int np, cudaStream_t s);
  void join(int ticket, cudaStream_t s);
  void mark(const char* what, int i, cudaStream_t s);  // E55_COMM_TRACE (above); `what` a string literal
  bool rdma() const { return bool(rd_); }
  RdmaAR* rdma_ar() const { return rd_.get(); }  // tests (tuning)

 private:
  static constexpr int kTickets = 64;
  void issue(cudaStream_t s, const std::function<void(cudaStream_t)>& f, char kind = '?', size_t bytes = 0);  // plain: on s, or via the comm stream
  int issue_async(cudaStream_t s, const std::function<void(cudaStream_t)>& f, char kind = '?', size_t bytes = 0);
  void ar_bf16(bf16* buf, size_t n, cudaStream_t s);
  void ar_f32(float* buf, size_t n, cudaStream_t s);
  void rs_bf16(bf16* buf, size_t n, cudaStream_t s);
  void ag_rows(const AgPiece* p, int np, int M, cudaStream_t s, bool bg = false);
  void ag_flat(const void* send, void* recv, size_t bytes, ncclDataType_t t, size_t n, cudaStream_t s);
  bool use_rdma() const;
  int rank_;
  ncclComm_t nc_ = nullptr;
  std::unique_ptr<RdmaAR> rd_;
  cudaStream_t cs_ = nullptr;
  cudaEvent_t ready_ = nullptr, done_[kTickets] = {};
  bool open_[kTickets] = {};
  int next_ = 0, n_open_ = 0;
  void* scratch_ = nullptr;  // NCCL all_gather_rows: [2][M·W] packed rows
  size_t scratch_bytes_ = 0;
  // E55_COMM_TRACE (above): records {kind (A all-reduce, F fp32 all-reduce, R reduce-scatter, G gather, Z put; J join;
  // M mark),
  // label, bytes / ticket / mark number, events}
  struct Trace {
    char kind;
    int label;         // tr_labels_ index
    const char* what;  // marks
    long long v;
    int e[3];          // tr_pool_ indices (−1: none)
  };
  const char* trace_on();
  int tr_ev(cudaStream_t s);
  void trace_write();
  std::vector<cudaEvent_t> tr_pool_;
  std::vector<Trace> tr_;
  std::vector<std::string> tr_labels_;
  int tr_next_ = 0, tr_rec_[kTickets] = {};
};

}  // namespace e55
