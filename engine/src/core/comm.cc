#include "core/comm.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>

#include "core/log.h"
#include "kernels/comm/rdma_ar.h"
#include "model/patches.h"

#define NCCL_CHECK(x)                                                                       \
  do {                                                                                      \
    ncclResult_t r_ = (x);                                                                  \
    if (r_ != ncclSuccess) qlog::fail("nccl", "%s · %s:%d", ncclGetErrorString(r_), __FILE__, __LINE__); \
  } while (0)

namespace e55 {

static size_t ag_bytes(const AgPiece* p, int np, int M) {  // (E55_COMM_TRACE: what a gather sends)
  size_t w = 0;
  for (int j = 0; j < np; j++) w += size_t(p[j].width);
  return w * size_t(std::max(M, 0));
}
static void send_all(int fd, const void* p, size_t n) {
  const char* c = static_cast<const char*>(p);
  while (n) {
    ssize_t k = ::send(fd, c, n, 0);
    if (k <= 0) qlog::fail("comm", "bootstrap send failed");
    c += k, n -= k;
  }
}
static void recv_all(int fd, void* p, size_t n) {
  char* c = static_cast<char*>(p);
  while (n) {
    ssize_t k = ::recv(fd, c, n, 0);
    if (k <= 0) qlog::fail("comm", "bootstrap receive failed");
    c += k, n -= k;
  }
}

Comm::Comm(int rank, const std::string& head_ip, int port) : rank_(rank) {
  if (head_ip.empty()) qlog::fail("comm", "no head address (EngineCfg.head_ip; tests: E55_HEAD_IP)");
  ncclUniqueId id;
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(port);
  if (inet_pton(AF_INET, head_ip.c_str(), &a.sin_addr) != 1) qlog::fail("comm", "bad address: %s", head_ip.c_str());
  auto t0 = std::chrono::steady_clock::now();
  if (rank == 0) {
    NCCL_CHECK(ncclGetUniqueId(&id));
    int ls = socket(AF_INET, SOCK_STREAM, 0), one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    if (bind(ls, (sockaddr*)&a, sizeof a) || listen(ls, 1)) qlog::fail("comm", "cannot listen on %s:%d", head_ip.c_str(), port);
    int fd = accept(ls, nullptr, nullptr);
    send_all(fd, &id, sizeof id);
    close(fd);
    close(ls);
  } else {
    int fd = -1;
    for (int i = 0; i < 3000; i++) {  // the head may start later (e.g. still compiling): retry for 300 s
      fd = socket(AF_INET, SOCK_STREAM, 0);
      if (connect(fd, (sockaddr*)&a, sizeof a) == 0) break;
      close(fd), fd = -1;
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (fd < 0) qlog::fail("comm", "cannot reach the head %s:%d (300 s)", head_ip.c_str(), port);
    recv_all(fd, &id, sizeof id);
    close(fd);
  }
  NCCL_CHECK(ncclCommInitRank(&nc_, 2, id, rank));
  double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  qlog::ok("comm", "rank%d · NCCL TP2 · %s:%d · %.0f ms", rank, head_ip.c_str(), port, ms);
  {  // P075's comm stream at the highest priority, events for its tickets
    int lo = 0, hi = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
    CUDA_CHECK(cudaStreamCreateWithPriority(&cs_, cudaStreamNonBlocking, hi));
    CUDA_CHECK(cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming));
    for (auto& e : done_) CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
  }
  std::fill(tr_rec_, tr_rec_ + kTickets, -1);
  if (std::getenv("E55_COMM_TRACE_FILE")) {  // E55_COMM_TRACE: the timing events up front (none created while traced)
    const char* n = std::getenv("E55_COMM_TRACE_EVENTS");
    tr_pool_.resize(size_t(std::max(1024, n ? std::atoi(n) : 32768)));
    for (auto& e : tr_pool_) CUDA_CHECK(cudaEventCreate(&e));
  }
  if (patch::on(patch::rdma_ar) || std::getenv("E55_RDMA_INIT")) {  // E55_RDMA_INIT=1: tests switch P001 per call
    // staging per call (pinned, 4 buffers; larger calls are split in calls of this size, the pipeline draining at each
    // boundary): 8 MiB, or with P079 ar_staging 32 MiB — a 4096-row chunk's lane all-reduces (VHA z 32 MiB, the shared
    // expert's 32 MiB) in one call each; E55_RDMA_STAGING_MB overrides
    const char* st = std::getenv("E55_RDMA_STAGING_MB");
    const size_t mb = st ? size_t(std::max(1, std::atoi(st))) : patch::on(patch::ar_staging) ? 32 : 8;
    t0 = std::chrono::steady_clock::now();
    rd_ = std::make_unique<RdmaAR>(rank, head_ip, port + 1, mb << 19);
    ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    qlog::ok("rdma", "rank%d · all-reduce over %d rail%s · %.0f ms", rank, rd_->rails(), rd_->rails() > 1 ? "s" : "", ms);
  }
}

Comm::~Comm() {
  if (cs_) cudaStreamSynchronize(cs_);
  trace_write();
  for (auto& e : tr_pool_) cudaEventDestroy(e);
  rd_.reset();
  if (nc_) ncclCommDestroy(nc_);
  for (auto& e : done_)
    if (e) cudaEventDestroy(e);
  if (ready_) cudaEventDestroy(ready_);
  if (cs_) cudaStreamDestroy(cs_);
  cudaFree(scratch_);
}

// the transport of every collective, chosen per call (tests switch P001 in-process; the RDMA queue pairs exist only
// when the patch was on at start)
bool Comm::use_rdma() const { return rd_ && patch::on(patch::rdma_ar); }

void Comm::ar_bf16(bf16* buf, size_t n, cudaStream_t s) {
  if (use_rdma()) return rd_->all_reduce_bf16(buf, n, s);
  NCCL_CHECK(ncclAllReduce(buf, buf, n, ncclBfloat16, ncclSum, nc_, s));
}
void Comm::ar_f32(float* buf, size_t n, cudaStream_t s) {
  if (use_rdma()) return rd_->all_reduce_f32(buf, n, s);
  NCCL_CHECK(ncclAllReduce(buf, buf, n, ncclFloat32, ncclSum, nc_, s));
}
void Comm::rs_bf16(bf16* buf, size_t n, cudaStream_t s) {  // P094: NCCL — the all-reduce itself (same bits)
  if (use_rdma()) return rd_->reduce_scatter_bf16(buf, n, s);
  NCCL_CHECK(ncclAllReduce(buf, buf, 2 * n, ncclBfloat16, ncclSum, nc_, s));
}
void Comm::ag_rows(const AgPiece* p, int np, int M, cudaStream_t s, bool bg) {
  if (M <= 0) return;
  if (use_rdma() && patch::on(patch::rdma_gather)) return rd_->all_gather(p, np, M, s, bg);
  // NCCL: this rank's pieces packed as rows of W bytes into scratch[rank], all-gather, then both parts placed
  if (np < 1 || np > kAgMaxPieces) qlog::fail("comm", "all_gather_rows: %d pieces", np);
  size_t W = 0;
  for (int j = 0; j < np; j++) W += size_t(p[j].width);
  const size_t need = 2 * W * size_t(M);
  if (need > scratch_bytes_) {
    CUDA_CHECK(cudaStreamSynchronize(s));
    cudaFree(scratch_);
    CUDA_CHECK(cudaMalloc(&scratch_, need));
    scratch_bytes_ = need;
  }
  char* sc = static_cast<char*>(scratch_);
  const size_t part = W * size_t(M);
  for (int j = 0, pre = 0; j < np; j++) {
    const AgPiece& q = p[j];
    CUDA_CHECK(cudaMemcpy2DAsync(sc + rank_ * part + pre, W, q.src, q.src_ld, q.width, M, cudaMemcpyDeviceToDevice, s));
    pre += q.width;
  }
  NCCL_CHECK(ncclAllGather(sc + rank_ * part, sc, part, ncclUint8, nc_, s));
  for (int j = 0, pre = 0; j < np; j++) {
    const AgPiece& q = p[j];
    for (int r = 0; r < 2; r++)
      CUDA_CHECK(cudaMemcpy2DAsync(static_cast<char*>(q.dst) + q.off + r * q.stride, q.dst_ld, sc + r * part + pre, W,
                                   q.width, M, cudaMemcpyDeviceToDevice, s));
    pre += q.width;
  }
}
// recv = [rank0 | rank1] of `bytes` each: over RDMA as rows (≤ 1 MiB each when it divides evenly; 16-byte aligned, or
// 8 for ≤ 16 KiB), else NCCL
void Comm::ag_flat(const void* send, void* recv, size_t bytes, ncclDataType_t t, size_t n, cudaStream_t s) {
  const uint64_t al = reinterpret_cast<uintptr_t>(send) | reinterpret_cast<uintptr_t>(recv) | bytes;
  if (use_rdma() && patch::on(patch::rdma_gather) && (!(al & 15) || (!(al & 7) && bytes <= (16 << 10)))) {
    size_t w = bytes;
    int M = 1;
    while (w > (size_t(1) << 20) && w % 32 == 0) w /= 2, M *= 2;
    if (w <= rd_->max_elems() * 2 && w < (size_t(1) << 31)) {
      AgPiece q{send, recv, (long long)w, (long long)w, 0, (long long)bytes, int(w)};
      return rd_->all_gather(&q, 1, M, s);
    }
  }
  NCCL_CHECK(ncclAllGather(send, recv, n, t, nc_, s));
}

// P075: a plain collective runs on s unless asynchronous ones are outstanding (then through the comm stream, s waits)
void Comm::issue(cudaStream_t s, const std::function<void(cudaStream_t)>& f, char kind, size_t bytes) {
  if (!n_open_) return f(s);
  join(issue_async(s, f, kind, bytes), s);
}
int Comm::issue_async(cudaStream_t s, const std::function<void(cudaStream_t)>& f, char kind, size_t bytes) {
  if (!n_open_ && !patch::on(patch::ar_lanes)) {  // patch off: the plain collective on s
    f(s);
    return -1;
  }
  if (!tr_pool_.empty() && std::getenv("E55_COMM_SKIP"))  // tests (with E55_COMM_TRACE_FILE): timing without the
    return -1;                                             // asynchronous transfers — results garbage
  const int t = next_;
  if (open_[t]) qlog::fail("comm", "%d collectives outstanding (join them)", kTickets);
  next_ = (next_ + 1) % kTickets;
  tr_rec_[t] = -1;
  const char* tl = trace_on();  // E55_COMM_TRACE: s reaches the call · the comm stream takes it · it ends
  Trace r{kind, 0, nullptr, (long long)bytes, {-1, -1, -1}};
  if (tl && tr_next_ + 3 <= int(tr_pool_.size())) {
    if (tr_labels_.empty() || tr_labels_.back() != tl) tr_labels_.push_back(tl);
    r.label = int(tr_labels_.size()) - 1, r.e[0] = tr_ev(s);
  }
  CUDA_CHECK(cudaEventRecord(ready_, s));
  CUDA_CHECK(cudaStreamWaitEvent(cs_, ready_, 0));
  if (r.e[0] >= 0) r.e[1] = tr_ev(cs_);
  f(cs_);
  CUDA_CHECK(cudaEventRecord(done_[t], cs_));
  if (r.e[0] >= 0) r.e[2] = tr_ev(cs_), tr_rec_[t] = int(tr_.size()), tr_.push_back(r);
  open_[t] = true, n_open_++;
  return t;
}
void Comm::join(int t, cudaStream_t s) {
  if (t < 0) return;
  if (t >= kTickets || !open_[t]) qlog::fail("comm", "join: ticket %d is not outstanding", t);
  const int rec = tr_rec_[t];
  Trace j{'J', rec >= 0 ? tr_[rec].label : 0, nullptr, rec, {-1, -1, -1}};
  if (rec >= 0 && tr_next_ + 2 <= int(tr_pool_.size())) j.e[0] = tr_ev(s);  // E55_COMM_TRACE: s's wait
  CUDA_CHECK(cudaStreamWaitEvent(s, done_[t], 0));
  if (j.e[0] >= 0) j.e[1] = tr_ev(s), tr_.push_back(j);
  tr_rec_[t] = -1;
  open_[t] = false, n_open_--;
}

// ---- E55_COMM_TRACE (comm.h) --------------------------------------------------------------------------------------
const char* Comm::trace_on() {
  if (tr_pool_.empty()) return nullptr;
  const char* e = std::getenv("E55_COMM_TRACE");
  return e && *e ? e : nullptr;
}
int Comm::tr_ev(cudaStream_t s) {
  if (tr_next_ >= int(tr_pool_.size())) return -1;
  CUDA_CHECK(cudaEventRecord(tr_pool_[tr_next_], s));
  return tr_next_++;
}
void Comm::mark(const char* what, int i, cudaStream_t s) {
  const char* tl = trace_on();
  if (!tl || tr_next_ + 1 > int(tr_pool_.size())) return;
  if (tr_labels_.empty() || tr_labels_.back() != tl) tr_labels_.push_back(tl);
  tr_.push_back(Trace{'M', int(tr_labels_.size()) - 1, what, i, {tr_ev(s), -1, -1}});
}
// kind · label · what · v (bytes; a join: its call's record; a mark: its number) · ms of the events from the first
void Comm::trace_write() {
  const char* f = std::getenv("E55_COMM_TRACE_FILE");
  if (!f || tr_.empty()) return;
  cudaDeviceSynchronize();
  const std::string path = std::string(f) + ".rank" + std::to_string(rank_);
  FILE* o = std::fopen(path.c_str(), "w");
  if (!o) return qlog::warn("comm", "trace: cannot write %s", path.c_str());
  auto ms = [&](int e) {
    float x = -1;
    if (e >= 0 && cudaEventElapsedTime(&x, tr_pool_[0], tr_pool_[e]) != cudaSuccess) x = -1;
    return x;
  };
  for (const Trace& r : tr_)
    std::fprintf(o, "%c\t%s\t%s\t%lld\t%.4f\t%.4f\t%.4f\n", r.kind, tr_labels_.at(r.label).c_str(),
                 r.what ? r.what : "-", r.v, ms(r.e[0]), ms(r.e[1]), ms(r.e[2]));
  std::fclose(o);
  qlog::ok("comm", "trace · %zu records → %s", tr_.size(), path.c_str());
}

void Comm::all_reduce_bf16(bf16* buf, size_t n, cudaStream_t s) {
  issue(s, [=](cudaStream_t x) { ar_bf16(buf, n, x); }, 'A', 2 * n);
}
void Comm::all_reduce_f32(float* buf, size_t n, cudaStream_t s) {
  issue(s, [=](cudaStream_t x) { ar_f32(buf, n, x); }, 'F', 4 * n);
}
void Comm::reduce_scatter_bf16(bf16* buf, size_t n, cudaStream_t s) {
  issue(s, [=](cudaStream_t x) { rs_bf16(buf, n, x); }, 'R', 2 * n);
}
void Comm::all_gather_bf16(const bf16* send, bf16* recv, size_t n, cudaStream_t s) {
  issue(s, [=](cudaStream_t x) { ag_flat(send, recv, n * 2, ncclBfloat16, n, x); });
}
void Comm::all_gather_f32(const float* send, float* recv, size_t n, cudaStream_t s) {
  issue(s, [=](cudaStream_t x) { ag_flat(send, recv, n * 4, ncclFloat32, n, x); });
}
void Comm::all_gather_rows(const AgPiece* p, int np, int M, cudaStream_t s) {
  std::vector<AgPiece> v(p, p + np);
  issue(s, [=](cudaStream_t x) { ag_rows(v.data(), np, M, x); }, 'G', ag_bytes(p, np, M));
}
int Comm::all_reduce_bf16_async(bf16* buf, size_t n, cudaStream_t s) {
  return issue_async(s, [=](cudaStream_t x) { ar_bf16(buf, n, x); }, 'A', 2 * n);
}
int Comm::all_reduce_f32_async(float* buf, size_t n, cudaStream_t s) {
  return issue_async(s, [=](cudaStream_t x) { ar_f32(buf, n, x); }, 'F', 4 * n);
}
int Comm::reduce_scatter_bf16_async(bf16* buf, size_t n, cudaStream_t s) {
  return issue_async(s, [=](cudaStream_t x) { rs_bf16(buf, n, x); }, 'R', 2 * n);
}
int Comm::all_gather_rows_async(const AgPiece* p, int np, int M, cudaStream_t s) {
  std::vector<AgPiece> v(p, p + np);
  return issue_async(s, [=](cudaStream_t x) { ag_rows(v.data(), np, M, x, true); }, 'G', ag_bytes(p, np, M));
}
// P176 (comm.h)
void* Comm::zc_alloc(size_t bytes) { return rd_ ? rd_->zc_alloc(bytes) : nullptr; }
bool Comm::zc_ok(const ZcPut* p, int np) const { return use_rdma() && rd_->put_ok(p, np); }
bool Comm::zc_ready() const { return use_rdma() && rd_->put_ready(); }
static size_t put_bytes(const ZcPut* p, int np) {
  size_t b = 0;
  for (int j = 0; j < np; j++) b += p[j].bytes;
  return b;
}
void Comm::put(const ZcPut* p, int np, cudaStream_t s) {
  std::vector<ZcPut> v(p, p + np);
  issue(s, [=](cudaStream_t x) { rd_->put(v.data(), np, x); }, 'Z', put_bytes(p, np));
}
int Comm::put_async(const ZcPut* p, int np, cudaStream_t s) {
  std::vector<ZcPut> v(p, p + np);
  return issue_async(s, [=](cudaStream_t x) { rd_->put(v.data(), np, x); }, 'Z', put_bytes(p, np));
}

void Comm::barrier(cudaStream_t s) {
  static float* d = nullptr;
  if (!d) CUDA_CHECK(cudaMalloc(&d, sizeof(float)));
  all_reduce_f32(d, 1, s);
  CUDA_CHECK(cudaStreamSynchronize(s));
}

}  // namespace e55
