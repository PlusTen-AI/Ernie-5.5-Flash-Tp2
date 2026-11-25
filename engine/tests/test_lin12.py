#!/usr/bin/env python3
"""test_lin12.py — P098 lin12_prefill / P097 lm12_only (kernels/linear p098_lin12_prefill.cuh) on the checkpoint's
router (gate.weight ‖ gate.weight_1) and HCA compressor (linear_wkv ‖ linear_wgate) of layers 1–4 (rank 0), fp32 out:
1. bitwise: the 12-bit-only Lin through P098's prefill kernel (both tiles) == the bf16 Lin's current prefill GEMM (P098
   off) at M = 1024 and 4096; lm_head (no K split) at M = 96
2. ms per call (CUDA events; the 4 layers back to back, median of 20 rounds): current → P098 at M = 1024 and 4096
  tools/agent.sh mem --gpu python3 engine/tests/test_lin12.py --model /models/ERNIE-5.5-Flash-Preview --table /work/cache/table.tsv"""
import argparse
import os
import statistics
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ext  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--table", required=True)
a = ap.parse_args()
m = ext.load("lin12", ["src/kernels/linear/linear.cu", "src/core/weights.cc", "src/model/patches.cc"],
             log="/tmp/build-lin12.log")
m.open(a.model, a.table, 0)
LAY = [1, 2, 3, 4]
NAMES = {
    "router": lambda i: [f"model.layers.{i}.block_sparse_moe.gate.weight", f"model.layers.{i}.block_sparse_moe.gate.weight_1"],
    "compressor": lambda i: [f"model.layers.{i}.self_attn.core_attention.compressor.linear_wkv.weight",
                             f"model.layers.{i}.self_attn.core_attention.compressor.linear_wgate.weight"],
}


def bits(t):
    return t.view(torch.int32)


def tflops(n, k, M, ms):
    return 2.0 * n * k * M / ms / 1e9


ok = True
torch.manual_seed(0)
print("[lin12] router / compressor: current (bf16 copy, P098 off) vs 12-bit only (P098, tile 128×128 · 128×64)")
for tag, names in NAMES.items():
    hb = [m.lin(names(i), 1) for i in LAY]  # as today: bf16 + P062's 12-bit copy
    hp = [m.lin(names(i), 2) for i in LAY]  # P098: the 12-bit copy only
    n, k, sp, hw, hk, nx = m.lin_info(hp[0])
    assert hw == 0 and hk == 1, "12-bit-only Lin still has a bf16 copy"
    print(f"  {tag}: {n}×{k}, {sp} K splits, layer 1 {nx} exceptions")
    for M in (1024, 4096):
        x = torch.randn(M, k, device="cuda").bfloat16()
        same = True
        for hb_, hp_ in zip(hb, hp):
            m.set_patch("lin12_prefill", False)
            ref = bits(m.fwd(hb_, x))
            m.set_patch("lin12_prefill", True)
            for t in (0, 1):
                m.tile(t)
                same &= torch.equal(bits(m.fwd(hp_, x)), ref)
        m.tile(-1)
        m.set_patch("lin12_prefill", False)
        t0 = statistics.median(m.bench(hb, x, 20))
        m.set_patch("lin12_prefill", True)
        tt = []
        for t in (0, 1):
            m.tile(t)
            tt.append(statistics.median(m.bench(hp, x, 20)))
        m.tile(-1)
        ta = statistics.median(m.bench(hp, x, 20))
        ok &= same
        print(f"    M={M}: bitwise {'identical' if same else 'DIFFERENT'} · current {t0:.3f} ms "
              f"({tflops(n, k, M, t0):.0f} TF) → P098 {ta:.3f} ms ({tflops(n, k, M, ta):.0f} TF) "
              f"[128×128 {tt[0]:.3f} · 128×64 {tt[1]:.3f}]")
    m.free_all()

# lm_head (no K split, P097): R > 64 rows go through P098's kernel
hb, hp = m.lin(["lm_head.weight"], 1), m.lin(["lm_head.weight"], 2)
x = torch.randn(96, 4096, device="cuda").bfloat16()
m.set_patch("lin12_prefill", False)
ref = bits(m.fwd(hb, x))
m.set_patch("lin12_prefill", True)
same = True
for t in (0, 1):
    m.tile(t)
    same &= torch.equal(bits(m.fwd(hp, x)), ref)
m.tile(-1)
ok &= same
print(f"  lm_head half M=96: bitwise {'identical' if same else 'DIFFERENT'}")
m.free_all()
print("[lin12] " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
