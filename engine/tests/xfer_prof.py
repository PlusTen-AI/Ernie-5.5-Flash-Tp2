#!/usr/bin/env python3
"""xfer_prof.py — one prefill chunk's wall time under schedule / transport variants, A / B in one TP=2 process (through
tools/tp2.sh, which appends --rank and --head-ip):
  tools/tp2.sh x python3 engine/tests/xfer_prof.py --rows 8192 --at 0 --reps 2 \\
      --var 'base' --var 'late4:E55_SP_LATE=4' --var 'noce:-rdma_ce' [--staged base] [--trace base]
A variant is name[:item,…], an item ENV=value (set in this process: the engine reads these per step) or +patch / -patch
(switched in-process). Every rep runs every variant once (alternating), each the chunk at --at after a reset; rank 0
prints each wall time and per variant the min / mean. --staged NAME: that variant also once staged (sync mode 2, every
stage line). --trace NAME: that variant's reps with E55_COMM_TRACE (comm.h) set — the timeline file written at close
(--trace-file, per rank). --bits: per variant the chunk once more through step() with the logits of 14 rows around the
lane / part bounds, compared bitwise to the first variant's (a quick check; patch_check.py is the gate; --bits-vars:
only these variants). --at a,b,…: the whole set at each position (one process)."""
import argparse
import collections
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ext  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, required=True)
ap.add_argument("--head-ip", required=True)
ap.add_argument("--port", type=int, default=29631)
ap.add_argument("--model", default="/models/ERNIE-5.5-Flash-Preview")
ap.add_argument("--table", default="/work/cache/table.tsv")
ap.add_argument("--prompts", default="/work/results/ref/v3/prompts.json")
ap.add_argument("--rows", type=int, default=8192)
ap.add_argument("--at", default="0", help="positions, comma-separated")
ap.add_argument("--reps", type=int, default=2)
ap.add_argument("--var", action="append", default=[])
ap.add_argument("--staged", action="append", default=[])
ap.add_argument("--trace", action="append", default=[])
ap.add_argument("--trace-file", default="/work/dev/xfer/trace")
ap.add_argument("--bits", action="store_true", help="each variant's logits of rows around the lane bounds == the first's")
ap.add_argument("--bits-vars", default="", help="comma-separated variant names for --bits (default: all)")
ap.add_argument("--first-only", default="", help="comma-separated variants timed at the first --at only (and skip*)")
ap.add_argument("--deadline", type=float, default=0, help="seconds: dump the stacks and exit if not done by then "
                "(faulthandler's own thread: a hung GPU wait never holds the nodes)")
a = ap.parse_args()
if a.deadline > 0:
    import faulthandler
    faulthandler.dump_traceback_later(a.deadline, exit=True)
ATS = [int(x) for x in a.at.split(",") if x != ""]
R0 = a.rank == 0


def say(*x):
    if R0:
        print(*x, flush=True)


if a.trace:
    os.environ["E55_COMM_TRACE_FILE"] = a.trace_file
    os.environ.pop("E55_COMM_TRACE", None)
E = ext.load("engine", ext.engine_sources(), log=f"/tmp/build{a.rank}.log", ldflags=ext.ENGINE_LDFLAGS)
E.init(a.model, a.table, a.rank, a.head_ip, 1, max(ATS) + a.rows + 64, max(a.rows, 4096), 16, a.port)
base = json.load(open(a.prompts))["doc_long"]
ids = (base * ((max(ATS) + a.rows) // len(base) + 1))[:max(ATS) + a.rows]
import torch  # noqa: E402

VARS = []
for v in a.var or ["base"]:
    name, _, items = v.partition(":")
    VARS.append((name, [x for x in items.split(",") if x]))
DEFAULT_PATCHES = {}


def apply(items, on):
    """on: set this variant's items; off: restore what they changed"""
    for it in items:
        if it[0] in "+-":
            p = it[1:]
            if on:
                DEFAULT_PATCHES.setdefault(p, E.patch_on(p))
                E.patch(p, it[0] == "+")
            else:
                E.patch(p, DEFAULT_PATCHES[p])
        else:
            k, _, val = it.partition("=")
            if on:
                os.environ[k] = val
            else:
                os.environ.pop(k, None)


GROUPS = [("attention", ("hca_", "dsa_", "vha_", "mul_bf16", "sigmoid")), ("FP8 GEMM", ("f8_forward", "act")),
          ("bf16 GEMM", ("lin_forward",)), ("MoE", ("moe_",)), ("collectives", ("comm_",)),
          ("mHC", ("mhc_",)), ("norms / FFN act", ("rmsnorm", "situ", "add_bf16")),
          ("head / logits", ("hc_head", "lm_head", "logits"))]


def group(name):
    return next((g for g, pre in GROUPS if name.startswith(pre)), "other")


def chunk(at, staged=False, label=None):
    E.reset(0)
    for c0 in range(0, at, 4096):
        c1 = min(at, c0 + 4096)
        E.fill(ids[c0:c1], list(range(c0, c1)), [0], [0, c1 - c0])
    torch.cuda.synchronize()
    if label:
        os.environ["E55_COMM_TRACE"] = label
    if staged:
        E.sync_mode(2)
        E.prof()
    t = time.time()
    E.fill(ids[at:at + a.rows], list(range(at, at + a.rows)), [0], [0, a.rows])
    torch.cuda.synchronize()
    dt = 1e3 * (time.time() - t)
    os.environ.pop("E55_COMM_TRACE", None)
    if not staged:
        return dt, None
    rows = E.prof()
    E.sync_mode(0)
    return dt, rows


chunk(ATS[0])  # warm-up (first-call setup)
BV = [x for x in a.bits_vars.split(",") if x] or [n for n, _ in VARS]
for at in ATS:
    walls = collections.defaultdict(list)
    for r in range(a.reps):
        for name, items in VARS:
            if at != ATS[0] and (name.startswith("skip") or name in a.first_only.split(",")):  # (first position only)
                continue
            apply(items, True)
            dt, _ = chunk(at, label=f"{name}@{at}#{r}" if name in a.trace and at == ATS[0] else None)
            apply(items, False)
            walls[name].append(dt)
            say(f"[xfer] at {at:,} · rep {r} · {name}: {dt:.0f} ms")
    say(f"[xfer] prefill {a.rows} rows at {at:,}, wall ms (min / mean of {a.reps}):")
    for name, items in VARS:
        w = walls[name]
        if w:
            say(f"  {name:12s} {min(w):7.1f} / {sum(w) / len(w):7.1f}   {' '.join(items) or 'defaults'}")
    if a.bits:
        sel = [r for r in (0, 1, 1023, 1024, 2047, 2048, 3071, 3072, 4095, 4096, 6143, 6144, 8190, 8191) if r < a.rows]
        ref = None
        for name, items in VARS:
            if name not in BV:
                continue
            apply(items, True)
            E.reset(0)
            for c0 in range(0, at, 4096):
                c1 = min(at, c0 + 4096)
                E.fill(ids[c0:c1], list(range(c0, c1)), [0], [0, c1 - c0])
            lg = E.step(ids[at:at + a.rows], list(range(at, at + a.rows)), [0], [0, a.rows], False, sel).cpu()
            apply(items, False)
            if ref is None:
                ref, rn = lg, name
            same = lg.shape == ref.shape and torch.equal(lg.contiguous().view(torch.uint8), ref.contiguous().view(torch.uint8))
            say(f"[bits] at {at:,} · {name}: logits of {len(sel)} rows {'bitwise == ' + rn if same else 'DIFFER from ' + rn}")
for name, items in VARS:
    if name not in a.staged:
        continue
    apply(items, True)
    dt, rows = chunk(ATS[0], staged=True)
    apply(items, False)
    g = collections.Counter()
    for n, line, ms, c in rows:
        g[group(n)] += ms
    say(f"[prof] {name}: prefill {a.rows} rows at {ATS[0]:,} staged {dt:.0f} ms")
    say("  " + " · ".join(f"{k} {v:.0f} ms ({100 * v / dt:.0f}%)" for k, v in g.most_common()))
    top = sorted(rows, key=lambda x: -x[2])
    say("  top: " + " · ".join(f"{n}:{ln} {ms:.1f}×{c}" for n, ln, ms, c in top))
E.close()
