#!/usr/bin/env python3
"""test_markov_back.py — P161 markov_back / P163 markov_pipe on one GPU (node B): the sampled Markov sweep (k_markov,
P082 e4m3 w2) run backwards and in CTA parts gives every q value and argmax partial bit for bit as the one forward
launch; and the µs of a 7-step chain of sweeps (one item / four items) forward every step vs alternating (L2 reuse),
one launch vs 2 / 4 parts (CUDA events, warm). Synthetic weights at the draft's shapes (Vh = 100608, rank 256).
  tools/agent.sh dec --gpu python3 engine/tests/test_markov_back.py"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ext  # noqa: E402

E = ext.load("dspark", ["src/kernels/dspark/dspark.cu", "src/kernels/attn/hca.cu", "src/model/patches.cc"],
             log="/work/dev/dec/build_markov.log")
dev = "cuda"
torch.manual_seed(0)
Vh, V, R = 100608, 201216, 256
NB = E.markov_blocks(Vh)
w1t = (torch.randn(V, R, device=dev) * 0.05).to(torch.bfloat16)
w2q = torch.randint(0, 120, (Vh, R), dtype=torch.uint8, device=dev)  # e4m3 bytes (finite, positive range)
w2q[:, ::3] |= 0x80  # some negatives
w2s = torch.randint(118, 128, (Vh, R // 32), dtype=torch.uint8, device=dev)
mm = torch.tensor([1.0, 0.5, -0.25, 2.0, 0.0, 0.0, 0.0, 0.0], device=dev)
ok = True


def run(K, g_parts, back, out, part):
    base = run.base[:K]
    for g0, ng in g_parts:
        E.markov(base, run.brow, run.seq, run.prev, w1t, w2q, w2s, mm, K, out, run.orow, part, g0, ng, back)


run.base = torch.randn(16, Vh, device=dev) * 4
run.brow = torch.arange(16, dtype=torch.int32, device=dev)
run.seq = torch.arange(16, dtype=torch.int32, device=dev)
run.orow = torch.arange(16, dtype=torch.int32, device=dev)
run.prev = torch.randint(0, V, (16,), dtype=torch.int32, device=dev)


def parts(n):
    return [(NB * c // n, NB * (c + 1) // n - NB * c // n) for c in range(n)]


for K in (1, 4):
    ref_o = torch.empty(K, Vh, device=dev)
    ref_p = torch.empty(K * NB * 2, device=dev)
    run(K, [(0, NB)], False, ref_o, ref_p)
    for back in (False, True):
        for n in (1, 2, 4):
            o = torch.full((K, Vh), float("nan"), device=dev)
            p = torch.full((K * NB * 2,), float("nan"), device=dev)
            run(K, parts(n), back, o, p)
            same = torch.equal(o.view(torch.int32), ref_o.view(torch.int32)) and \
                torch.equal(p.view(torch.int32), ref_p.view(torch.int32))
            ok &= same
            print(f"[markov] K {K} · {'backwards' if back else 'forwards'} · {n} part(s): q rows and partials "
                  f"{'bitwise equal' if same else 'DIFFER'} to one forward launch", flush=True)
torch.cuda.synchronize()

# timing: 7 steps a chain, the L2 cold before each chain (a 64 MB sweep of other memory)
flush = torch.empty(64 << 20, dtype=torch.uint8, device=dev)


def chain_ms(K, alt, n, reps=20):
    o = torch.empty(K, Vh, device=dev)
    p = torch.empty(K * NB * 2, device=dev)
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    tot = 0.0
    for r in range(reps + 3):
        flush.add_(1)
        ev[0].record()
        for i in range(7):
            run(K, parts(n), alt and (i & 1) == 1, o, p)
        ev[1].record()
        torch.cuda.synchronize()
        if r >= 3:
            tot += ev[0].elapsed_time(ev[1])
    return tot / reps / 7 * 1e3


for K in (1, 4):
    res = {(alt, n): chain_ms(K, alt, n) for alt in (False, True) for n in (1, 2, 4)}
    print(f"[markov] K {K} · µs a step (7-step chain, warm launches, cold L2 at the chain's start): " +
          " · ".join(f"{'alternating' if alt else 'forwards'} {n} part(s) {v:.1f}" for (alt, n), v in res.items()),
          flush=True)
print(f"[markov] all: {'PASS' if ok else 'FAIL'}", flush=True)
