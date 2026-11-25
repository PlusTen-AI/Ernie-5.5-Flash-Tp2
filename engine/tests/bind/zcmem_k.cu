// zcmem_k.cu — kernels and allocation for tests/bind/zcmem.cpp (P176 feasibility: GPU access to cuMemCreate
// HOST_NUMA memory — the NIC-registrable kind — against cudaMalloc memory: streaming read / write / copy, L2 reuse).
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>

namespace {
__global__ void k_read(const uint4* __restrict__ p, size_t n, int reps, unsigned* sink) {
  unsigned acc = 0;
  for (int r = 0; r < reps; r++)
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
      uint4 v;  // (asm volatile: not hoisted out of the reps loop)
      asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p + i));
      acc += v.x ^ v.y ^ v.z ^ v.w;
    }
  if (acc == 0x12345678u) sink[0] = acc;
}
__global__ void k_write(uint4* p, size_t n, unsigned v) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    p[i] = make_uint4(v, v + 1, v + 2, unsigned(i));
}
__global__ void k_copy(const uint4* __restrict__ s, uint4* d, size_t n) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) d[i] = s[i];
}
}  // namespace

// kind 0: cudaMalloc · 1: cuMemCreate pinned on the host NUMA node, mapped for this GPU and the CPU (P132's region)
void* zc_alloc(int kind, size_t bytes, unsigned long long* handle, size_t* mapped) {
  if (kind == 0) {
    void* p = nullptr;
    cudaMalloc(&p, bytes);
    *mapped = 0;
    return p;
  }
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  prop.location.id = 0;
  size_t gran = 0;
  if (cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) != CUDA_SUCCESS) return nullptr;
  const size_t sz = (bytes + gran - 1) / gran * gran;
  CUmemGenericAllocationHandle h = 0;
  if (cuMemCreate(&h, sz, &prop, 0) != CUDA_SUCCESS) return nullptr;
  CUdeviceptr va = 0;
  cuMemAddressReserve(&va, sz, gran, 0, 0);
  cuMemMap(va, sz, 0, h, 0);
  int dev = 0;
  cudaGetDevice(&dev);
  CUmemAccessDesc acc[2]{};
  acc[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE, acc[0].location.id = dev;
  acc[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  acc[1].location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA, acc[1].location.id = 0;
  acc[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  cuMemSetAccess(va, sz, acc, 2);
  *handle = h, *mapped = sz;
  std::printf("[zcmem] HOST_NUMA granularity %zu KiB · %zu MiB\n", gran >> 10, sz >> 20);
  return reinterpret_cast<void*>(va);
}
void zc_free(void* p, unsigned long long h, size_t mapped) {
  if (!mapped) return void(cudaFree(p));
  cuMemUnmap(CUdeviceptr(p), mapped);
  cuMemAddressFree(CUdeviceptr(p), mapped);
  cuMemRelease(h);
}

// op 0 read · 1 write · 2 copy (first half → second half) · 3 L2 reuse (read the first `bytes` reps times)
void zc_run(int op, void* p, size_t bytes, int reps, unsigned* sink, cudaStream_t s) {
  const size_t n = bytes / 16;
  const int G = 48 * 8, T = 512;
  if (op == 0) k_read<<<G, T, 0, s>>>(static_cast<const uint4*>(p), n, 1, sink);
  if (op == 1) k_write<<<G, T, 0, s>>>(static_cast<uint4*>(p), n, 7u);
  if (op == 2) k_copy<<<G, T, 0, s>>>(static_cast<const uint4*>(p), static_cast<uint4*>(p) + n / 2, n / 2);
  if (op == 3) k_read<<<G, T, 0, s>>>(static_cast<const uint4*>(p), n, reps, sink);
}
