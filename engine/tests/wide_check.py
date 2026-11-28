#!/usr/bin/env python3
"""wide_check.py — a copy round's wide verify at long context (TP2; tools/tp2.sh appends --rank / --head-ip): two slots
prefilled with the same --ctx-token prompt in 8192-row chunks (indexer candidates past P142's 65,536 and P153's
131,072), then --rows positions teacher-forced — slot 0 as one --rows-row step (all logits), slot 1 as one-row steps —
every row's logits bitwise equal (the batch invariance copy drafts rely on, at the context where the select paths
switch). --rows 128: a 16-sequence verify's row count.
--seqs B (4 / 8 / 16): a B-sequence verify instead — 2B slots, slot b and slot B + b prefilled alike with prompt b
(--seq-base + b · --seq-step tokens: every sequence its own context, key-chunk and group counts), then one decode-kind
step of B × --seq-rows rows (8: block 7 + 1 → 32 / 64 / 128 rows, all logits) over slots 0 … B−1 and each sequence's
rows alone (one --seq-rows-row step on slot B + b): every row bitwise equal (a verify row never depends on the other
sequences in the step, at the M where the kernels switch: P066 / P190, P164 / P192, P078 / P191, P115 / P193, P030 /
P200 — run it with each patch set under test).
  tools/tp2.sh wide python3 engine/tests/wide_check.py --seqs 16"""
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
ap.add_argument("--seqs", type=int, default=0, help="B-sequence verify vs each sequence alone (0: the long-context check)")
ap.add_argument("--seq-rows", type=int, default=8, help="--seqs: rows a sequence (block 7 + 1)")
ap.add_argument("--seq-base", type=int, default=2048, help="--seqs: prompt 0's tokens")
ap.add_argument("--seq-step", type=int, default=777, help="--seqs: each next prompt's extra tokens")
a = ap.parse_args()
R0 = a.rank == 0
CH = 8192


def say(*x):
    if R0:
        print(*x, flush=True)


E = ext.load("engine", ext.engine_sources(), log=f"/tmp/build-wide{a.rank}.log", ldflags=ext.ENGINE_LDFLAGS)
if a.seqs:
    B, r = a.seqs, a.seq_rows
    n = [a.seq_base + b * a.seq_step for b in range(B)]
    E.init(a.model, a.table, a.rank, a.head_ip, 2 * B, max(n) + r + 64, CH, B * r, a.port)
    PJ = json.load(open(a.prompts))
    src = PJ["doc_long"] + PJ["en_tools"]
    src = src * ((max(n) + r + B * 997) // len(src) + 2)
    seqs = [src[b * 997:b * 997 + n[b] + r] for b in range(B)]  # prompt b: n[b] tokens, then its r verify-row tokens
    t0 = time.time()
    for b in range(B):
        for sl in (b, B + b):
            E.reset(sl)
            for c0 in range(0, n[b], CH):
                c1 = min(c0 + CH, n[b])
                E.fill(seqs[b][c0:c1], list(range(c0, c1)), [sl], [0, c1 - c0])
    say(f"[wide] {B} sequences × 2 slots prefilled ({n[0]:,} … {n[-1]:,} tokens) · {time.time() - t0:.0f} s")
    ids, pos, ss = [], [], [0]
    for b in range(B):
        ids += seqs[b][n[b]:n[b] + r]
        pos += list(range(n[b], n[b] + r))
        ss.append(ss[-1] + r)
    wide = E.step(ids, pos, list(range(B)), ss, True).cpu()  # (decode kind: a verify step's rows)
    bad = []
    for b in range(B):
        alone = E.step(seqs[b][n[b]:n[b] + r], list(range(n[b], n[b] + r)), [B + b], [0, r], True).cpu()
        bad += [b * r + i for i in range(r) if not torch.equal(wide[b * r + i], alone[i])]
    say(f"[wide] {B} × {r} = {B * r}-row step vs each sequence's {r} rows alone: " +
        (f"bitwise equal {B * r}/{B * r}" if not bad else f"DIFFER at rows {bad[:12]} ({len(bad)}/{B * r})"))
    E.close()
    sys.exit(0 if not bad else 1)
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
