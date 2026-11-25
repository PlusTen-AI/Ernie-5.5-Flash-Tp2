// layer.h — one decoder layer's weights on this rank (model.cc loads the target's, dspark.cc the draft stages').
#pragma once
#include "model/model.h"

namespace e55 {

struct Engine::HcaW {
  F8Lin qkv, qup, gate, woa, oproj;
  Lin comp;
  F8Lin qv, kvh;  // P078 tp_split views of qkv: q_down (every row) and this rank's half of the kv rows
  Lin cvh, cgh;   // P078: this rank's halves of the compressor's wkv / wgate rows
  bf16 *qn = nullptr, *kvn = nullptr, *cnw = nullptr;
  float *ape = nullptr, *sink = nullptr, *U = nullptr, *V = nullptr;
};
struct Engine::DsaW {
  F8Lin qkv, qb, gate, oproj, iq;
  Lin ikw;
  bf16 *qan = nullptr, *kvan = nullptr, *knw = nullptr, *knb = nullptr, *wuk = nullptr, *wuv = nullptr;
  uint8_t *wuk8 = nullptr, *wuv8 = nullptr;  // P073 dsa_bmm_f8: kv_b_proj's e4m3 bytes (W_UK transposed, W_UV as stored)
  float* kvs = nullptr;                      // P073: its block scales, fp32 [32·448 / 128][4]
  float *sink = nullptr, *U = nullptr, *V = nullptr;
};
struct Engine::FfnW {
  bool dense = false;
  F8Lin gu, down, fc1, fc2;
  Lin router;
  Lin r0h, r1h;      // P078 tp_split: this rank's halves of the router's gate.weight / gate.weight_1 rows
  F8Lin fc1h, fc2h;  // P078: this rank's halves of fc1 / fc2 (output rows)
  F8Lin gup, downp;  // P136 sp_shared: the peer's shared-expert halves (gate/up rows, down K half) as the peer uploads them
  float *bias = nullptr, *rscale = nullptr;
  bf16* lnorm = nullptr;
  MoeExperts* ex = nullptr;
};
// P078 tp_split views (model.cc): rows [r0, r0 + n) of an uploaded weight, the same tiles and K split
F8Lin f8_rows(const F8Lin& L, int r0, int n);
Lin lin_rows(const Lin& L, int r0, int n);

struct Engine::Layer {
  MhcW ha, hf;
  bool dsa = false;
  HcaW h;
  DsaW d;
  FfnW f;
};

}  // namespace e55
