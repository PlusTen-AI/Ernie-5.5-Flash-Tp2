#!/usr/bin/env python3
"""test_bigxfer.py — the RDMA transport's LARGE calls under load (TP=2; through tools/tp2.sh, which appends --rank and
--head-ip): GB/s of 32 MiB all-reduces (bf16) and row all-gathers issued async on the comm stream (as the engine's
prefill lanes do), alone and while a side stream runs a load: GEMM (bf16 4096³ matmuls, compute-bound) or MEM (1 GiB
copies, DRAM-bound). --variants: RdmaAR::tune tuples chunkKB/copy/add/ahead/xfer (P001's large-call knobs).
--bits: the transfers' results checked against the expected values (all-reduce of known inputs, gathered rows).
Rank 0 prints one line per measurement."""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ext  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, required=True)
ap.add_argument("--head-ip", required=True)
ap.add_argument("--port", type=int, default=29671)
ap.add_argument("--mib", type=int, default=32)
ap.add_argument("--calls", type=int, default=8)
ap.add_argument("--variants", default="1024/2/4/1/0")
ap.add_argument("--patches", default="", help="A/B sets 'a;b' of +name,-name switched in-process (each measured)")
ap.add_argument("--bits", action="store_true")
a = ap.parse_args()
os.environ.setdefault("E55_RDMA_TIMEOUT_S", "30")
R0 = a.rank == 0


def say(*x):
    if R0:
        print(*x, flush=True)


m = ext.load("commtest", ["src/core/comm.cc", "src/kernels/comm/rdma_ar.cu", "src/model/patches.cc", "tests/comm_k.cu"],
             log=f"/tmp/bigxfer_build{a.rank}.log", ldflags=["-lnccl", "-libverbs", "-lcuda"])
m.init(a.rank, a.head_ip, a.port)
for p in ("rdma_ar", "rdma_gather", "ar_pdl", "ar_lanes"):
    m.set_patch(p, True)
dev = "cuda"
B = a.mib << 20
W = 32768                       # gather row bytes (the P100 exchange's HCA row)
M = B // W
z = torch.zeros(B // 2, device=dev, dtype=torch.bfloat16)
src = torch.zeros(M, W, device=dev, dtype=torch.uint8)
dst = torch.zeros(M, 2 * W, device=dev, dtype=torch.uint8)
A_ = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
B_ = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
C_ = torch.empty(4096, 4096, device=dev, dtype=torch.bfloat16)
X_ = torch.empty(1 << 29, device=dev, dtype=torch.int16)
Y_ = torch.empty_like(X_)
side = torch.cuda.Stream()


def ev():
    return torch.cuda.Event(enable_timing=True)


def load(kind, n):
    with torch.cuda.stream(side):
        for _ in range(n):
            if kind == "gemm":
                torch.matmul(A_, B_, out=C_)
            else:
                Y_.copy_(X_)


def calls(op):
    for _ in range(a.calls):
        tk = m.ar_async(z) if op == "ar" else m.ag([src], [dst], [0], [W], M, True)
        m.join(tk)


def timed(op, kind):
    """ms for a.calls transfers on the comm stream (each joined), with the side load running"""
    m.barrier()
    torch.cuda.synchronize()
    if kind:
        load(kind, 400 if kind == "gemm" else 60)
    e0, e1 = ev(), ev()
    e0.record()
    calls(op)
    e1.record()
    e1.synchronize()
    ms = e0.elapsed_time(e1)
    torch.cuda.synchronize()
    return ms


SETS = [""] + [x for x in a.patches.split(";") if x]


def apply(pv):
    for p in filter(None, pv.split(",")):
        m.set_patch(p[1:], p[0] == "+")


if a.bits:  # the large paths' results: all-reduce (bf16 sums of mixed values; rank 1 adds rank 0's in the other
    # order, so both must equal the same bits), gathered rows = each rank's bytes, a 2-piece gather (strided rows)
    for pv in SETS:
        apply(pv)
        fails = 0
        g = torch.Generator(device=dev).manual_seed(99)
        x0 = torch.randn(B // 2, generator=g, device=dev).to(torch.bfloat16)
        x1 = torch.randn(B // 2, generator=g, device=dev).to(torch.bfloat16)
        exp = (x0.float() + x1.float()).to(torch.bfloat16)
        zz = (x0 if a.rank == 0 else x1).clone()
        m.join(m.ar_async(zz))
        fails += int(not torch.equal(zz.view(torch.int16), exp.view(torch.int16)))
        s2 = (torch.arange(M * W, device=dev, dtype=torch.int64) * 7 + a.rank * 13).to(torch.uint8).view(M, W)
        d2 = torch.zeros(M, 2 * W, device=dev, dtype=torch.uint8)
        m.join(m.ag([s2], [d2], [0], [W], M, True))
        for r in (0, 1):
            e = (torch.arange(M * W, device=dev, dtype=torch.int64) * 7 + r * 13).to(torch.uint8).view(M, W)
            fails += int(not torch.equal(d2[:, r * W:(r + 1) * W], e))
        # two pieces of 3072 / 4096 B from wider rows into rows of [p0 r0 | p0 r1] / [p1 r0 | p1 r1] (attn_front-like)
        Mp = 2048
        a0 = (torch.arange(Mp * 4096, device=dev, dtype=torch.int64) * 3 + a.rank).to(torch.uint8).view(Mp, 4096)
        a1 = (torch.arange(Mp * 8192, device=dev, dtype=torch.int64) * 5 + a.rank).to(torch.uint8).view(Mp, 8192)
        o0 = torch.zeros(Mp, 2 * 3072, device=dev, dtype=torch.uint8)
        o1 = torch.zeros(Mp, 8192 + 4096, device=dev, dtype=torch.uint8)
        m.join(m.ag([a0[:, :3072], a1[:, :4096]], [o0, o1], [0, 0], [3072, 4096], Mp, True))
        for r in (0, 1):
            e0 = (torch.arange(Mp * 4096, device=dev, dtype=torch.int64) * 3 + r).to(torch.uint8).view(Mp, 4096)
            e1 = (torch.arange(Mp * 8192, device=dev, dtype=torch.int64) * 5 + r).to(torch.uint8).view(Mp, 8192)
            fails += int(not torch.equal(o0[:, r * 3072:(r + 1) * 3072], e0[:, :3072]))
            fails += int(not torch.equal(o1[:, r * 4096:(r + 1) * 4096], e1[:, :4096]))
        torch.cuda.synchronize()
        say(f"[bits] large all-reduce / all-gathers ({pv or 'defaults'}): {'OK' if not fails else f'{fails} FAIL'}")

for kind in ("gemm", "mem"):  # the loads alone (ms for their length), so a timed window stays inside one
    torch.cuda.synchronize()
    e0, e1 = ev(), ev()
    e0.record(side)
    load(kind, 400 if kind == "gemm" else 60)
    e1.record(side)
    e1.synchronize()
    say(f"[load] {kind}: {e0.elapsed_time(e1):.0f} ms")

for v in a.variants.split(","):
    t = tuple(int(x) for x in v.split("/"))
    m.tune(*t)
    for pv in SETS:
        apply(pv)
        out = []
        for op in ("ar", "ag"):
            for _ in range(2):
                timed(op, None)  # warm-up
            for kind in (None, "gemm", "mem"):
                ms = timed(op, kind)
                out.append(f"{op}{'+' + kind if kind else ''} {a.calls * B / ms / 1e6:.1f}")
        say(f"[xfer] {a.mib} MiB · tune {v}{' · ' + pv if pv else ''} · GB/s per rank: " + " · ".join(out))
m.tune(1024, 2, 4, 1, 0)
m.barrier()
m.close()
