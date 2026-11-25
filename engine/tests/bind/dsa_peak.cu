// dsa_peak.cu — test-only (test_dsa.py --long): the FP4 MMA's measured peak — a pure loop of the indexer's k64
// instruction (mma.sync m16n8k64 kind::mxf4nvf4, e2m1 × e2m1, UE8M0 scale_vec::2X, fp32 accumulators), register
// operands, `chains` independent accumulators per warp, no memory traffic inside the loop. The floor of dsa_select's
// logits work is this rate.
#include <cuda_runtime.h>

#include <cstdint>

namespace {
template <int CH>
__global__ void k_mma_peak(int iters, float* out, long long* clk) {
  const uint32_t lane = threadIdx.x & 31;
  uint32_t a[4], b[2];
#pragma unroll
  for (int i = 0; i < 4; i++) a[i] = 0x13572468u ^ (lane * 0x01010101u) ^ (0x11111111u * i);
  b[0] = 0x24681357u ^ (lane * 0x00010001u), b[1] = 0x31415926u ^ lane;
  const uint32_t sa = 0x7F7F7F7Fu, sb = 0x7F7F7F7Fu;  // all scales 1
  float acc[CH][4];
#pragma unroll
  for (int c = 0; c < CH; c++) acc[c][0] = acc[c][1] = acc[c][2] = acc[c][3] = 0.f;
  const long long t0 = clock64();
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int c = 0; c < CH; c++)
      asm volatile(
          "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};\n"
          : "+f"(acc[c][0]), "+f"(acc[c][1]), "+f"(acc[c][2]), "+f"(acc[c][3])
          : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb));
  }
  const long long t1 = clock64();
  float s = 0.f;
#pragma unroll
  for (int c = 0; c < CH; c++) s += acc[c][0] + acc[c][1] + acc[c][2] + acc[c][3];
  if (s == 1234.5f) out[blockIdx.x * blockDim.x + threadIdx.x] = s;  // keeps the chains
  if (blockIdx.x == 0 && threadIdx.x == 0) *clk = t1 - t0;
}
}  // namespace

// ms of the loop, SM cycles of CTA 0, MMAs issued: grid ctas × threads, `chains` (2, 4, 8) accumulators a warp
void dsa_mma_peak(int ctas, int threads, int chains, int iters, float* ms, long long* cycles, double* mmas) {
  float* out = nullptr;
  long long* clk = nullptr;
  cudaMalloc(&out, sizeof(float) * ctas * threads);
  cudaMalloc(&clk, sizeof(long long));
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0), cudaEventCreate(&e1);
  auto launch = [&]() {
    if (chains == 2) k_mma_peak<2><<<ctas, threads>>>(iters, out, clk);
    else if (chains == 4) k_mma_peak<4><<<ctas, threads>>>(iters, out, clk);
    else k_mma_peak<8><<<ctas, threads>>>(iters, out, clk);
  };
  launch();  // warm-up (clocks up)
  cudaEventRecord(e0);
  launch();
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  cudaEventElapsedTime(ms, e0, e1);
  cudaMemcpy(cycles, clk, sizeof(long long), cudaMemcpyDeviceToHost);
  *mmas = double(ctas) * (threads / 32) * chains * double(iters);
  cudaEventDestroy(e0), cudaEventDestroy(e1);
  cudaFree(out), cudaFree(clk);
}
