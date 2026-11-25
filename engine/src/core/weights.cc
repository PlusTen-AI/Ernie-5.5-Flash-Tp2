#include "core/weights.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <fstream>
#include <sstream>
#include <thread>

#include "core/log.h"

namespace e55 {

static std::vector<std::string> split(const std::string& s, char d) {
  std::vector<std::string> r;
  std::string x;
  std::istringstream is(s);
  while (std::getline(is, x, d)) r.push_back(x);
  if (!s.empty() && s.back() == d) r.emplace_back();
  return r;
}

static size_t itemsize(const std::string& dt) {
  if (dt == "BF16" || dt == "F16" || dt == "I16") return 2;
  if (dt == "F32" || dt == "I32" || dt == "U32") return 4;
  if (dt == "F64" || dt == "I64" || dt == "U64") return 8;
  return 1;  // U8, I8, F8_E4M3, BOOL
}

Weights::Weights(const std::string& model_dir, const std::string& table_tsv, int rank) : dir_(model_dir), rank_(rank) {
  std::ifstream f(table_tsv);
  if (!f) qlog::fail("weights", "no tensor table: %s", table_tsv.c_str());
  std::string line;
  std::getline(f, line);
  auto hdr = split(line, '\t');
  auto col = [&](const char* n) {
    for (size_t i = 0; i < hdr.size(); i++) if (hdr[i] == n) return int(i);
    qlog::fail("weights", "tensor table has no %s column", n);
  };
  int c_name = col("name"), c_file = col("file"), c_off = col("offset"), c_nb = col("nbytes"), c_dt = col("dtype"),
      c_sh = col("shape"), c_tp = col("tp"), c_bl = col("block"), c_pa = col("part");
  while (std::getline(f, line)) {
    auto v = split(line, '\t');
    if (v[c_tp] == "skip") continue;
    TensorInfo ti{v[c_file], v[c_dt], v[c_tp], v[c_bl], v[c_pa], std::stoull(v[c_off]), std::stoull(v[c_nb]), {}};
    if (v[c_sh] != "-") for (auto& d : split(v[c_sh], 'x')) ti.shape.push_back(std::stoll(d));
    if (ti.tp == "-") qlog::fail("weights", "tp column is empty — rebuild the table with tools/tp_plan.py");
    t_.emplace(v[c_name], std::move(ti));
  }
  for (auto& [n, ti] : t_) {
    if (fds_.count(ti.file)) continue;
    int fd = open((dir_ + "/" + ti.file).c_str(), O_RDONLY);
    if (fd < 0) qlog::fail("weights", "cannot open: %s", ti.file.c_str());
    fds_[ti.file] = fd;
  }
  drop_cache();  // start from an empty page cache for the shard files (left by earlier readers)
}

int load_threads() { return std::max(1, std::min<int>(16, std::thread::hardware_concurrency())); }

void Weights::drop_cache() const {
  std::unordered_map<std::string, int> seen;
  for (auto& [n, ti] : t_) seen.emplace(ti.file, 0);
  for (auto& [f, _] : seen) {
    int fd = open((dir_ + "/" + f).c_str(), O_RDONLY);
    if (fd >= 0) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED), close(fd);
  }
}

Weights::~Weights() {
  for (auto& [k, fd] : fds_) close(fd);
}

const TensorInfo& Weights::info(const std::string& name) const {
  auto it = t_.find(name);
  if (it == t_.end()) qlog::fail("weights", "no such tensor: %s", name.c_str());
  return it->second;
}

void Weights::pread_all(const TensorInfo& ti, uint64_t off, uint64_t len, uint8_t* dst) const {
  const int fd = fds_.at(ti.file);
  const uint64_t base = ti.offset + off;
  auto part = [&](uint64_t a, uint64_t b) {  // bytes [a, b) of the range
    for (uint64_t done = a; done < b;) {
      ssize_t r = pread(fd, dst + done, b - done, base + done);
      if (r <= 0) qlog::fail("weights", "read failed: %s @%llu", ti.file.c_str(), (unsigned long long)(base + done));
      done += r;
    }
    // drop what we just read from the page cache (no root needed): on GB10 the GPU driver needs free pages, and ~50 GB
    // of cached shard pages can stall cudaMalloc (a note in the reference deployment's recipe)
    posix_fadvise(fd, base + a, b - a, POSIX_FADV_DONTNEED);
  };
  constexpr uint64_t kPiece = uint64_t(8) << 20;
  const int nt = int(std::min<uint64_t>(load_threads(), len / kPiece));
  if (nt < 2) return part(0, len);
  std::atomic<uint64_t> next{0};
  std::vector<std::thread> th;
  for (int i = 0; i < nt; i++)
    th.emplace_back([&] {
      for (uint64_t a; (a = next.fetch_add(kPiece)) < len;) part(a, std::min(len, a + kPiece));
    });
  for (auto& x : th) x.join();
}

std::vector<uint8_t> Weights::raw(const std::string& name) const {
  const auto& ti = info(name);
  std::vector<uint8_t> b(ti.nbytes);
  pread_all(ti, 0, ti.nbytes, b.data());
  return b;
}

// dim0 rows [r0, r1) of a row-major tensor with `rb` bytes per row
static void take_rows(const std::vector<uint8_t>& src, uint64_t rb, int64_t r0, int64_t r1, std::vector<uint8_t>& dst) {
  dst.insert(dst.end(), src.begin() + r0 * rb, src.begin() + r1 * rb);
}

std::vector<uint8_t> Weights::slice(const std::string& name, std::vector<int64_t>* shape_out) const {
  const auto& ti = info(name);
  std::vector<int64_t> shape = ti.shape;
  const std::string& tp = ti.tp;
  std::vector<uint8_t> out;
  const int r = rank_;

  if (tp == "rep") {
    out = raw(name);
  } else if (tp.rfind("own:", 0) == 0) {
    if (std::stoi(tp.substr(4)) == r) out = raw(name);
    else shape.assign(1, 0);
  } else if (tp == "col") {
    uint64_t h = ti.nbytes / 2;
    out.resize(h);
    pread_all(ti, r * h, h, out.data());
    shape[0] /= 2;
  } else if (tp == "row") {
    auto all = raw(name);
    uint64_t rb = ti.nbytes / shape[0], hb = rb / 2;
    out.resize(shape[0] * hb);
    for (int64_t i = 0; i < shape[0]; i++) std::memcpy(out.data() + i * hb, all.data() + i * rb + r * hb, hb);
    shape[1] /= 2;
  } else if (tp.rfind("seg:", 0) == 0) {
    auto all = raw(name);
    uint64_t rb = ti.nbytes / shape[0];
    int64_t base = 0;
    for (auto& s : split(tp.substr(4), ',')) {
      int64_t n = std::stoll(s);
      take_rows(all, rb, base + r * n / 2, base + (r + 1) * n / 2, out);
      base += n;
    }
    if (base != shape[0]) qlog::fail("weights", "%s: seg sum %lld ≠ dim0 %lld", name.c_str(), (long long)base, (long long)shape[0]);
    shape[0] /= 2;
  } else {
    qlog::fail("weights", "%s: unknown tp rule %s", name.c_str(), tp.c_str());
  }
  if (shape_out) *shape_out = shape;
  return out;
}

}  // namespace e55
