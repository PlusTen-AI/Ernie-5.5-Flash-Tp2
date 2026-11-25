#!/usr/bin/env python3
"""zcmem.py — P176 feasibility (one GPU): GPU kernels on cuMemCreate HOST_NUMA memory (NIC-registrable on GB10) vs
cudaMalloc memory — streaming read / write / copy of a large buffer and L2 reuse (a small set read 32×; a set written
by one kernel and read by the next). Prints GB/s per op and kind (best of --iters, CUDA events)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ext  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--mib", type=int, default=512)
ap.add_argument("--l2-mib", type=int, default=8)
ap.add_argument("--iters", type=int, default=5)
ap.add_argument("--log", default="/tmp/zcmem.log")
a = ap.parse_args()
M = ext.load("zcmem", ["tests/bind/zcmem_k.cu"], log=a.log, ldflags=["-lcuda"])
GB = 1e9
names = ["read", "write", "copy (r+w)", f"set {a.l2_mib} MiB read ×32 in a kernel", f"set {a.l2_mib} MiB after a read",
         f"set {a.l2_mib} MiB after a write"]
by = [a.mib << 20, a.mib << 20, a.mib << 20, 32 * (a.l2_mib << 20), a.l2_mib << 20, a.l2_mib << 20]
res = {k: M.bench(k, a.mib, a.l2_mib, a.iters) for k in (0, 1)}
print(f"[zcmem] {'op':28s} {'cudaMalloc':>12s} {'HOST_NUMA':>12s}  (GB/s; ms)")
for i, n in enumerate(names):
    r0, r1 = res[0][i], res[1][i]
    print(f"[zcmem] {n:28s} {by[i] / r0 / 1e6:8.1f} {r0:6.3f} {by[i] / r1 / 1e6:8.1f} {r1:6.3f}")
