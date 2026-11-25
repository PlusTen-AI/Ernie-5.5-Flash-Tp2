// Test binding for the mHC family (engine/src/kernels/mhc/mhc.cu): mixers from the checkpoint, pre / post + pre /
// layer-0 pre / post on copies of the inputs, patch switches, CUDA-event timing over a rotation of mixers (cold weights).
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <memory>

#include "core/log.h"
#include "kernels/mhc/mhc.h"
#include "model/patches.h"

void lindec_spin(long ns, cudaStream_t s);  // tests/bind/lindec_spin.cu

namespace {
std::string g_dir, g_table;
std::unique_ptr<e55::Weights> g_w;
std::vector<e55::MhcW> g_m;

void open(const std::string& model_dir, const std::string& table) { g_dir = model_dir, g_table = table; }

// kind 0: self_attention_hyper_connection + input_layernorm, 1: mlp_hyper_connection + post_attention_layernorm
int upload(int layer, int kind) {
  if (!g_w) g_w = std::make_unique<e55::Weights>(g_dir, g_table, 0);
  const std::string p = "model.layers." + std::to_string(layer) + ".";
  g_m.push_back(kind == 0 ? e55::mhc_upload(*g_w, p + "self_attention_hyper_connection.", p + "input_layernorm.weight")
                          : e55::mhc_upload(*g_w, p + "mlp_hyper_connection.", p + "post_attention_layernorm.weight"));
  return int(g_m.size()) - 1;
}

void set_patch(const std::string& name, bool on) { e55::patch::set(name, on); }
void p164_rows(int r) { e55::mhc_p164_rows(r); }

cudaStream_t cur() { return c10::cuda::getCurrentCUDAStream().stream(); }
e55::bf16* bp(torch::Tensor& t) { return reinterpret_cast<e55::bf16*>(t.data_ptr()); }

// the device fn_b [4096][24] of a mixer (layer 0 attention)
torch::Tensor fnb(int h, torch::Tensor like) {
  auto t = torch::empty({4096, 24}, like.options().dtype(torch::kFloat32));
  cudaMemcpy(t.data_ptr<float>(), g_m.at(h).fnb, 4096 * 24 * 4, cudaMemcpyDeviceToDevice);
  return t;
}

// what: 0 pre (R), 1 post + pre, 2 layer-0 pre, 3 post alone
void call(int what, int h, torch::Tensor& out, torch::Tensor& R, torch::Tensor& post, torch::Tensor& comb,
          torch::Tensor& mix, torch::Tensor& inp, cudaStream_t st) {
  const int M = int(R.size(0));
  float *mx = mix.data_ptr<float>(), *po = post.data_ptr<float>(), *co = comb.data_ptr<float>();
  switch (what) {
    case 0: e55::mhc_pre(g_m.at(h), bp(R), M, mx, po, co, bp(inp), st); break;
    case 1: e55::mhc_post_pre(g_m.at(h), bp(out), bp(R), M, mx, po, co, bp(inp), st); break;
    case 2: e55::mhc_pre_embed(g_m.at(h), bp(R), M, mx, po, co, bp(inp), st); break;
    default: e55::mhc_post(bp(out), bp(R), po, co, M, bp(R), st);
  }
}

// inputs are copied; returns R', post, comb, inp
std::vector<torch::Tensor> run(int what, int h, torch::Tensor out, torch::Tensor R, torch::Tensor post,
                               torch::Tensor comb) {
  const int M = int(R.size(0));
  auto Ro = R.clone(), po = post.clone(), co = comb.clone();
  auto mix = torch::zeros({M, 72}, post.options());
  auto inp = torch::zeros({M, 4096}, R.options());
  call(what, h, out, Ro, po, co, mix, inp, cur());
  return {Ro, po, co, inp};
}

// ms per call of each rep (a rep = one call per handle, in order; inputs restored before each rep, outside the timing)
std::vector<double> bench(int what, const std::vector<int>& hs, torch::Tensor out, torch::Tensor R, torch::Tensor post,
                          torch::Tensor comb, int reps) {
  const int M = int(R.size(0));
  auto st = cur();
  auto mix = torch::zeros({M, 72}, post.options());
  auto inp = torch::empty({M, 4096}, R.options());
  auto Rs = R.clone(), ps = post.clone(), cs = comb.clone();
  for (int h : hs) call(what, h, out, Rs, ps, cs, mix, inp, st);  // warm-up
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0), cudaEventCreate(&e1);
  std::vector<double> r;
  for (int i = 0; i < reps; i++) {
    Rs.copy_(R), ps.copy_(post), cs.copy_(comb);
    cudaEventRecord(e0, st);
    for (int h : hs) call(what, h, out, Rs, ps, cs, mix, inp, st);
    cudaEventRecord(e1, st);
    cudaEventSynchronize(e1);
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    r.push_back(ms / hs.size());
  }
  cudaEventDestroy(e0), cudaEventDestroy(e1);
  return r;
}

// µs per call of each round: an event, then for each handle [lindec_spin(spin_ns)] + the call, an event; minus the
// spins, divided by the handles (one interval over all of them: no per-call event granularity)
std::vector<double> timed_batch(int what, const std::vector<int>& hs, torch::Tensor out, torch::Tensor R,
                                torch::Tensor post, torch::Tensor comb, int rounds, long spin_ns) {
  const int M = int(R.size(0)), n = int(hs.size());
  auto st = cur();
  auto mix = torch::zeros({M, 72}, post.options());
  auto inp = torch::empty({M, 4096}, R.options());
  auto Rs = R.clone(), ps = post.clone(), cs = comb.clone();
  for (int h : hs) call(what, h, out, Rs, ps, cs, mix, inp, st);  // warm-up
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0), cudaEventCreate(&e1);
  std::vector<double> us;
  for (int r = 0; r < rounds; r++) {
    Rs.copy_(R), ps.copy_(post), cs.copy_(comb);
    cudaEventRecord(e0, st);
    for (int h : hs) {
      if (spin_ns > 0) lindec_spin(spin_ns, st);
      call(what, h, out, Rs, ps, cs, mix, inp, st);
    }
    cudaEventRecord(e1, st);
    cudaEventSynchronize(e1);
    float f;
    cudaEventElapsedTime(&f, e0, e1);
    us.push_back((f * 1000.0 - n * spin_ns / 1000.0) / n);
  }
  cudaEventDestroy(e0), cudaEventDestroy(e1);
  return us;
}

std::vector<int64_t> device() {
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  int clk = 0;
  cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
  return {p.multiProcessorCount, p.l2CacheSize, (int64_t)p.sharedMemPerMultiprocessor, (int64_t)p.sharedMemPerBlockOptin,
          clk};
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("open", &open);
  m.def("upload", &upload);
  m.def("set_patch", &set_patch);
  m.def("p164_rows", &p164_rows);
  m.def("fnb", &fnb);
  m.def("run", &run);
  m.def("bench", &bench);
  m.def("timed_batch", &timed_batch);
  m.def("device", &device);
}
