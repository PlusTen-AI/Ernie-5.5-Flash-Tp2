// lindec_spin.cu — test-only (test_lindec.py): a stand-in for the kernel before a decode linear in the engine — one
// warp that lets its programmatic dependents start at once (griddepcontrol.launch_dependents), then spins `ns`
// nanoseconds (globaltimer) without touching memory: the next kernel's prologue runs under it, its griddepcontrol.wait
// returns when this one ends.
#include <cuda_runtime.h>

#include <cstdint>

namespace {
__global__ void k_spin(long ns) {
  asm volatile("griddepcontrol.launch_dependents;\n" :::);
  uint64_t t0, t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  do {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  } while (t - t0 < uint64_t(ns));
}
}  // namespace

void lindec_spin(long ns, cudaStream_t s) { k_spin<<<1, 32, 0, s>>>(ns); }
