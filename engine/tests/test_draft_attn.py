#!/usr/bin/env python3
"""test_draft_attn.py — P129 draft_attn on one GPU (node B): the fused draft-stage attention launch vs the base chain
(hca_q_prep + hca_kv_rows + dsp_attn + vha_partial) on synthetic rows — o and the VHA partial z bit for bit, and µs per
call (CUDA events, warm, back to back).
  tools/agent.sh draft --gpu python3 engine/tests/test_draft_attn.py"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ext  # noqa: E402

E = ext.load("dspark", ["src/kernels/dspark/dspark.cu", "src/kernels/attn/hca.cu", "src/model/patches.cc"],
             log="/work/dev/draft/build_attn.log")
print(f"[draft_attn] device: {E.props()}", flush=True)
torch.manual_seed(0)
dev = "cuda"
SLOTS, RING, RB = 4, 256, 584
inv = torch.tensor([1.0 / (8e6 ** (2 * i / 64)) for i in range(32)], dtype=torch.float32, device=dev)
ring = torch.zeros(3 * SLOTS * RING * RB, dtype=torch.uint8, device=dev)
stage_b = SLOTS * RING * RB


def bf(*s):
    return torch.randn(*s, device=dev).to(torch.bfloat16)


def case(M, P, slot=1, stage=1, reps=200):
    K = min(P, 128)
    pos = torch.arange(P - K, P, dtype=torch.int32, device=dev)
    E.ctx_rows(bf(K, 1536), pos, torch.full((K,), slot, dtype=torch.int32, device=dev), inv, ring, SLOTS)
    rg = ring.narrow(0, stage * stage_b, stage_b)
    meta = torch.tensor([[P + i, slot, 0, P, P + M - 1] for i in range(M)], dtype=torch.int32, device=dev)
    q0 = bf(M, 32, 512) * 3
    kv = bf(M, 512)
    sink = torch.randn(32, device=dev)
    U = torch.randn(32 * 16, device=dev) * 0.2
    q = q0.clone()
    rows = torch.zeros(M * RB, dtype=torch.uint8, device=dev)
    o, zp = torch.empty(M, 32, 512, dtype=torch.bfloat16, device=dev), torch.empty(M, 16, 512, device=dev)
    o2, zp2 = torch.empty_like(o), torch.empty_like(zp)
    E.base(q, kv, meta, rg, sink, inv, U, rows, o, zp)
    E.fused(q0, kv, meta, rg, sink, inv, U, o2, zp2)
    torch.cuda.synchronize()
    same = torch.equal(o.view(torch.int16), o2.view(torch.int16)) and torch.equal(zp.view(torch.int32),
                                                                                 zp2.view(torch.int32))

    def t(fn):
        for _ in range(10):
            fn()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            fn()
        b.record()
        torch.cuda.synchronize()
        return a.elapsed_time(b) / reps * 1e3

    tb = t(lambda: E.base(q, kv, meta, rg, sink, inv, U, rows, o, zp))
    parts = [t(lambda w=w: E.base_one(w, q, kv, meta, rg, sink, inv, U, rows, o, zp)) for w in range(4)]
    tf = t(lambda: E.fused(q0, kv, meta, rg, sink, inv, U, o2, zp2))
    print(f"[draft_attn] M={M} P={P}: bitwise {'YES' if same else 'NO'} · base {tb:.1f} µs (q_prep {parts[0]:.1f} · "
          f"kv_rows {parts[1]:.1f} · dsp_attn {parts[2]:.1f} · vha_partial {parts[3]:.1f}) · fused {tf:.1f} µs",
          flush=True)
    return same


ok = all([case(7, 600), case(4, 600), case(1, 600), case(7, 20, slot=3, stage=2), case(15, 300)])
print(f"[draft_attn] all bitwise: {'YES' if ok else 'NO'}")

# P130: the greedy Markov chain of one sequence (no exchange on one GPU), base launches vs dsp_chain
Vh, V = 100608, 201216
base = torch.randn(7, Vh, device=dev) * 4
w1t = (torch.randn(V, 256, device=dev) * 0.5).to(torch.bfloat16)
w2q = torch.randint(0, 0x7E, (Vh, 256), dtype=torch.uint8, device=dev)
w2q = torch.where(torch.rand(Vh, 256, device=dev) < 0.5, w2q, w2q | 0x80)
w2s = torch.randint(118, 124, (Vh, 8), dtype=torch.uint8, device=dev)
mm = torch.tensor([-8., 8., -12., 12., 0.1, -0.1, 0.01, -0.01], device=dev)
ws = torch.zeros(8192, dtype=torch.int32, device=dev)
part = torch.zeros(16 * 2 * 1024, device=dev)
for N in (4, 7):
    d0, d1 = torch.zeros(256, dtype=torch.int32, device=dev), torch.zeros(256, dtype=torch.int32, device=dev)
    E.chain(False, base, w1t, w2q, w2s, mm, 1234, N, ws, part, d0)
    E.chain(True, base, w1t, w2q, w2s, mm, 1234, N, ws, part, d1)
    torch.cuda.synchronize()
    same = torch.equal(d0[:N], d1[:N])
    tt = []
    for fused in (False, True):
        for _ in range(5):
            E.chain(fused, base, w1t, w2q, w2s, mm, 1234, N, ws, part, d1)
        a_, b_ = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a_.record()
        for _ in range(50):
            E.chain(fused, base, w1t, w2q, w2s, mm, 1234, N, ws, part, d1)
        b_.record()
        torch.cuda.synchronize()
        tt.append(a_.elapsed_time(b_) / 50 * 1e3)
    print(f"[draft_chain] N={N}: drafts {d0[:N].tolist()} · bitwise {'YES' if same else 'NO'} · base {tt[0]:.1f} µs · "
          f"chain {tt[1]:.1f} µs ({tt[0] / N:.1f} → {tt[1] / N:.1f} µs a step)", flush=True)
