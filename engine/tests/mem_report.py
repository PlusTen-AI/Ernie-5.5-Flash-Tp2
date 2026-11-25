#!/usr/bin/env python3
"""mem_report.py — where a rank's memory goes, at the server's shapes (TP=2, through tools/tp2.sh, which appends
--rank and --head-ip):  tools/tp2.sh mem python3 /work/src/engine/tests/mem_report.py [--slots 4 --ctx 1048576 --chunk 8192]
Per rank, at each point: /proc/meminfo MemAvailable / MemFree / Cached and the CUDA free memory (GB10: one pool with the
host), this process's host RSS and the node's AnonPages (other processes on the node move MemFree too), as GiB and as
the change since the previous point: before init · after init (target weights, fixed per-slot state, workspaces, the
P125 pool's warm-up room) · after dspark_init (the draft) · after the pool (P125 kv_pool_init as the server calls it: a
warm-up of the serving paths, then the pool takes min MemFree of both nodes − --reserve) · after one prefill chunk and
one verify-shaped step, and (ctx > 78K) a chunk at position 70,000 (nothing should be allocated any more) · after close (the engine destroyed, torch's cached
blocks released: should be back at "before init"). With P125 off (E55_PATCHES=-kv_pool): every slot's ctx positions at
init, no pool point."""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ext  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--slots", type=int, default=4)
ap.add_argument("--ctx", type=int, default=65536, help="tokens per request at most (P125: address ranges only)")
ap.add_argument("--chunk", type=int, default=4096)
ap.add_argument("--block", type=int, default=7, help="DSpark block: logit rows = slots × (block + 1)")
ap.add_argument("--model", default="/models/ERNIE-5.5-Flash-Preview")
ap.add_argument("--table", default="/work/cache/table.tsv")
ap.add_argument("--prompts", default="/work/results/ref/v2/prompts.json")
ap.add_argument("--reserve", type=float, default=15.0, help="GiB kept free (server.memory_free_gib)")
ap.add_argument("--cap", type=float, default=0.0, help="pool GiB at most (server.kv_pool_gib; 0 = no cap)")
ap.add_argument("--log", default="", help="the engine build log (default /tmp/build<rank>.log)")
ap.add_argument("--rank", type=int, required=True)
ap.add_argument("--head-ip", required=True)
ap.add_argument("--port", type=int, default=29611)
a = ap.parse_args()
G = 1024 ** 3
prev = {}


def point(tag):
    import torch
    m = {}
    for ln in open("/proc/meminfo"):
        k, v = ln.split(":")
        m[k] = int(v.split()[0]) * 1024
    torch.cuda.synchronize()
    free, tot = torch.cuda.mem_get_info()
    rss = next(int(ln.split()[1]) * 1024 for ln in open("/proc/self/status") if ln.startswith("VmRSS:"))
    cur = {"avail": m["MemAvailable"], "free": m["MemFree"], "cached": m["Cached"], "cuda free": free, "rss": rss,
           "anon": m["AnonPages"]}
    d = " · ".join(f"{k} {v / G:.2f}" + (f" ({(v - prev[k]) / G:+.2f})" if k in prev else "") for k, v in cur.items())
    print(f"[mem] rank{a.rank} {tag:16s} {d}", flush=True)
    prev.update(cur)


import torch  # noqa: E402

torch.cuda.init()
point("before init")
E = ext.load("engine", ext.engine_sources(), log=(a.log or "/tmp/build{r}.log").format(r=a.rank),
             ldflags=ext.ENGINE_LDFLAGS)
lr = a.slots * (a.block + 1)
pool = E.patch_on("kv_pool")
E.init(a.model, a.table, a.rank, a.head_ip, a.slots, a.ctx, max(a.chunk, lr), lr, a.port, pool)
point("after init")
E.dspark_init()
point("after dspark")
if pool:
    n = E.kv_pool_init(a.reserve, a.cap, True)
    point("after pool")
    lo, hi = 0, a.ctx  # the longest single request the pool holds
    while lo < hi:
        m = (lo + hi + 1) // 2
        lo, hi = (m, hi) if E.kv_need(m) <= n else (lo, m - 1)
    per = 6 * (656 + 68) + 37 * 584 / 128
    print(f"[mem] rank{a.rank} pool {n:,} chunks · {n * 2 / 1024:.2f} GiB · ≈ {n * 2**21 / per / 1e6:.2f}M tokens "
          f"({per:.0f} B a token) · one request ≤ {lo:,} tokens · free {E.kv_free():,} · MemFree "
          f"{prev['free'] / G:.2f} GiB vs reserve {a.reserve:g}", flush=True)
base = json.load(open(a.prompts))["doc_long"]
ids = (base * (a.chunk // len(base) + 2))[:a.chunk + 8]
E.reset(0)
E.fill(ids[:a.chunk], list(range(a.chunk)), [0], [0, a.chunk])
E.step(ids[a.chunk:a.chunk + 8], list(range(a.chunk, a.chunk + 8)), [0], [0, 8], True)
if a.ctx > 70000 + a.chunk:  # a chunk at position 70,000 too (P046's long-context select; slot 1, history set)
    E.reset(1)
    E.set_history(1, 70000)
    E.fill(ids[:a.chunk], list(range(70000, 70000 + a.chunk)), [1], [0, a.chunk])
point("after steps")
if pool:
    print(f"[mem] rank{a.rank} slot 0 holds {E.kv_held(0):,} chunks ({a.chunk + 8:,} tokens, need "
          f"{E.kv_need(a.chunk + 8):,}) · slot 1 {E.kv_held(1):,} ({70000 + a.chunk:,} positions) · free "
          f"{E.kv_free():,} of {E.kv_total():,}", flush=True)
E.close()
torch.cuda.empty_cache()  # the step's returned logits (torch's caching allocator), not the engine's
point("after close")
