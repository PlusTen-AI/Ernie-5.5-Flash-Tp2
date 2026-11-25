#!/usr/bin/env python3
"""wide_check.py — a copy round's wide verify at long context (TP2; tools/tp2.sh appends --rank / --head-ip): two slots
prefilled with the same --ctx-token prompt in 8192-row chunks (indexer candidates past P142's 65,536 and P153's
131,072), then --rows positions teacher-forced — slot 0 as one --rows-row step (all logits), slot 1 as one-row steps —
every row's logits bitwise equal (the batch invariance copy drafts rely on, at the context where the select paths
switch)."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ext  # noqa: E402
import torch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, required=True)
ap.add_argument("--head-ip", required=True)
ap.add_argument("--port", type=int, default=29693)
ap.add_argument("--model", default="/models/ERNIE-5.5-Flash-Preview")
ap.add_argument("--table", default="/work/cache/table.tsv")
ap.add_argument("--prompts", default="/work/results/ref/v3/prompts.json")
ap.add_argument("--ctx", type=int, default=140000)
ap.add_argument("--rows", type=int, default=64)
a = ap.parse_args()
R0 = a.rank == 0
CH = 8192


def say(*x):
    if R0:
        print(*x, flush=True)


E = ext.load("engine", ext.engine_sources(), log=f"/tmp/build-wide{a.rank}.log", ldflags=ext.ENGINE_LDFLAGS)
E.init(a.model, a.table, a.rank, a.head_ip, 2, a.ctx + a.rows + 64, CH, a.rows, a.port)
d = json.load(open(a.prompts))["doc_long"]
T = (d * ((a.ctx + a.rows) // len(d) + 2))[:a.ctx + a.rows]
t0 = time.time()
for s in (0, 1):
    E.reset(s)
    for c0 in range(0, a.ctx, CH):
        c1 = min(c0 + CH, a.ctx)
        E.step(T[c0:c1], list(range(c0, c1)), [s], [0, c1 - c0], False)
say(f"[wide] two slots prefilled to {a.ctx:,} · {time.time() - t0:.0f} s")
P = a.ctx
wide = E.step(T[P:P + a.rows], list(range(P, P + a.rows)), [0], [0, a.rows], True).cpu()
bad = [i for i in range(a.rows)
       if not torch.equal(wide[i], E.step([T[P + i]], [P + i], [1], [0, 1], False).cpu()[0])]
say(f"[wide] {a.rows}-row step at {P:,} vs one-row steps: " + (f"bitwise equal {a.rows}/{a.rows}" if not bad else
    f"DIFFER at rows {bad[:12]} ({len(bad)}/{a.rows})"))
E.close()
