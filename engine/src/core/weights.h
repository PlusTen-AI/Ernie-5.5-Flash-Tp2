// weights.h — tensor table + this rank's slice of each tensor (host bytes), same rules as tools/tp_plan.py:
//   rep | col (dim0 halves) | row (dim1 halves) | seg:a,b,c (dim0 segments, each halved) | own:r | skip
#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace e55 {

struct TensorInfo {
  std::string file, dtype, tp, block, part;
  uint64_t offset = 0, nbytes = 0;
  std::vector<int64_t> shape;
};

class Weights {
 public:
  Weights(const std::string& model_dir, const std::string& table_tsv, int rank);
  ~Weights();
  bool has(const std::string& name) const { return t_.count(name) != 0; }
  const TensorInfo& info(const std::string& name) const;
  // This rank's bytes of `name` (row-major). `shape` gets the local shape.
  std::vector<uint8_t> slice(const std::string& name, std::vector<int64_t>* shape = nullptr) const;
  // Whole tensor as stored (no split, no decode).
  std::vector<uint8_t> raw(const std::string& name) const;
  int rank() const { return rank_; }
  void drop_cache() const;  // posix_fadvise(DONTNEED) on every shard file (no root): GB10's driver wants free pages
  size_t count() const { return t_.size(); }

 private:
  // Safe from several threads (the shard files are opened once, in the constructor). A long range is read in 8 MiB
  // pieces on load_threads() threads: one stream reads NVMe at ~1.5 GB/s, 16 at ~5 GB/s.
  void pread_all(const TensorInfo& ti, uint64_t off, uint64_t len, uint8_t* dst) const;
  std::string dir_;
  int rank_;
  std::unordered_map<std::string, TensorInfo> t_;
  std::unordered_map<std::string, int> fds_;
};

// Threads for loading (reads, host-side layout): the CPU's, at most 16.
int load_threads();


}  // namespace e55
