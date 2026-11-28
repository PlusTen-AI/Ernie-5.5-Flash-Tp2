#!/usr/bin/env python3
"""patch_check.py — base vs patches, bitwise: one TP2 run of the engine (patch state from E55_PATCHES, read once per process); rank 0
saves the logits / greedy tokens of each stage to <out>/<label>.pt and, with --cmp, compares them bitwise to an earlier
run's file. Same calls in every run except the long prefill of slot 0 (--long one: one 16384-row step; chunks: four
4096-row steps). Stages:
  long   slot 0: 16384 positions (logits of 16 rows around the lane / part bounds)
  c1     slot 1: a 4096-row chunk (8 rows around the 1024-row lane bounds)
  c2     slot 1: a 3000-row chunk on top (8 rows around its lane bounds 768 / 1536 / 2240)
  dec    slot 1: 12 greedy decode steps through step() (logits), then 4 through run() (greedy; P063 when on) → tokens
  r5     slot 1: one 5-row step, all logits
  r16    slot 1 (10 rows) + slot 0 (6 rows) in one 16-row step, all logits
  w64    (--wide: 16 slots, 128 logit rows) slots 2 … 15 prefilled (600 + 211·s tokens each), then a verify-shaped step
         of 8 sequences × 8 rows (slots 0 … 7, all logits: B = 8 at block 7)
  w32    (--wide --w32) first the same over slots 0 … 3: 32 rows (B = 4; the later stages' positions move by 8)
  w128   the same over all 16 slots: 128 rows (B = 16) — the mid-M decode kernels (P190 – P193, P200) and their bases
  dsp    (--dspark) after dspark_init, slots 0 / 1 reset: one 2601-row step of both (1500 + 1101 rows: lanes, the last
         one odd; 10 logit rows around the lane / sequence bounds), then greedy drafts + confidences (depth 7) at both
         anchors — the draft's context rows of that step
Step kinds as the server's (core/step_kind.h): long, c1, c2, the --wide fills and dsp's step are prefill steps
(prefill = True); dec, r5, r16, w64, w128 and the drafts decode / verify-shaped decode steps.
Prefill-rule patches (pf_rule.py, rule of 2026-11-28: their own accumulation order, not the base's bits) cannot be
checked here. --pf off (default) | on forces all of them to that state in this run, after E55_PATCHES (the last token
wins), and saves the state with the stages: --cmp refuses a file whose prefill-rule state differs (every stage after a
prefill differs then). So a run compares the other patches bitwise with the prefill-rule kernels held off, or held on
(--pf on in both runs: also the prefill-rule kernels' own chunk-size invariance, --long one vs chunks); the
prefill-rule patches themselves pass engine/tests/gate.py --pf both (the reference at the noise floor) and
engine/tests/pf_check.py (determinism, chunk / checkpoint invariance, decode kind untouched) instead."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pf_rule  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, required=True)
ap.add_argument("--head-ip", required=True)
ap.add_argument("--port", type=int, default=29691)
ap.add_argument("--label", required=True)
ap.add_argument("--long", choices=["one", "chunks"], required=True)
ap.add_argument("--cmp", default="")
ap.add_argument("--out", default="/work/results/patch-check")
ap.add_argument("--model", default="/models/ERNIE-5.5-Flash-Preview")
ap.add_argument("--table", default="/work/cache/table.tsv")
ap.add_argument("--prompts", default="/work/results/ref/v3/prompts.json")
ap.add_argument("--dspark", action="store_true", help="also the dsp stage (after the others)")
ap.add_argument("--wide", action="store_true", help="16 slots / 128 logit rows and the w64 / w128 stages (both runs)")
ap.add_argument("--w32", action="store_true", help="with --wide: first a w32 stage (4 sequences × 8 rows: B = 4)")
ap.add_argument("--pf", choices=["off", "on"], default="off", help="the prefill-rule patches held at this state")
a = ap.parse_args()
R0 = a.rank == 0
PF = pf_rule.names()
os.environ["E55_PATCHES"] = pf_rule.env(a.pf == "on", PF)  # before the engine reads it (its first patch lookup)
import ext  # noqa: E402
import torch  # noqa: E402


def say(*x):
    if R0:
        print(*x, flush=True)


E = ext.load("engine", ext.engine_sources(), log=f"/tmp/build-run{a.rank}.log", ldflags=ext.ENGINE_LDFLAGS)
pf_state = [int(E.patch_on(n)) for n in PF]
if pf_state != [int(a.pf == "on")] * len(PF):
    raise SystemExit(f"[chk] prefill-rule patches {PF} are {pf_state}, not --pf {a.pf}")
say(f"[chk] prefill-rule patches {PF} held {a.pf} · E55_PATCHES={os.environ['E55_PATCHES']}")
LONG = 16384
E.init(a.model, a.table, a.rank, a.head_ip, 16 if a.wide else 2, LONG + 64, LONG, 128 if a.wide else 16, a.port)
P = json.load(open(a.prompts))
d, t = P["doc_long"], P["en_tools"]
T0 = (d * (LONG // len(d) + 2))[:LONG + 6]
T1 = (t * (8000 // len(t) + 2))[:8000]
res = {"pf_state": torch.tensor(pf_state, dtype=torch.int32)}

# long prefill, slot 0
sel = [0, 1, 2047, 2048, 4095, 4096, 6143, 6144, 8191, 8192, 10239, 10240, 12287, 12288, 16382, 16383]
E.reset(0)
if a.long == "one":
    res["long"] = E.step(T0[:LONG], list(range(LONG)), [0], [0, LONG], False, sel, prefill=True).cpu()
else:
    parts = []
    for c0 in range(0, LONG, 4096):
        c1 = c0 + 4096
        parts.append(E.step(T0[c0:c1], list(range(c0, c1)), [0], [0, 4096], False,
                            [r - c0 for r in sel if c0 <= r < c1], prefill=True).cpu())
    res["long"] = torch.cat(parts)
say(f"[chk] long done ({a.long})")
# slot 1: 4096 + 3000 rows
E.reset(1)
res["c1"] = E.step(T1[:4096], list(range(4096)), [1], [0, 4096], False,
                   [0, 1023, 1024, 2047, 2048, 3071, 3072, 4095], prefill=True).cpu()
res["c2"] = E.step(T1[4096:7096], list(range(4096, 7096)), [1], [0, 3000], False,
                   [0, 767, 768, 1535, 1536, 2239, 2240, 2999], prefill=True).cpu()
p = 7096
tok = int(res["c2"][-1].argmax())
toks, dl = [tok], []
for i in range(16):
    if i < 12:
        lg = E.step([tok], [p], [1], [0, 1], False).cpu()
        dl.append(lg)
        tok = int(lg[0].argmax())
    else:
        tok = E.run([tok], [p], [1], [0, 1], [0.0], [0], [1.0], [0.0], [1], [p + 1], [1.0], [0.0], [0.0])[0]
    toks.append(tok)
    p += 1
res["dec"] = torch.cat(dl)
res["dec_tokens"] = torch.tensor(toks)
say(f"[chk] greedy tokens {toks}")
res["r5"] = E.step(T1[p:p + 5], list(range(p, p + 5)), [1], [0, 5], True).cpu()
p += 5
res["r16"] = E.step(T1[p:p + 10] + T0[LONG:LONG + 6], list(range(p, p + 10)) + list(range(LONG, LONG + 6)), [1, 0],
                    [0, 10, 16], True).cpu()
if a.wide:  # verify-shaped steps of 8 and 16 sequences × 8 rows (decode kind), every sequence at its own position
    hist = {0: LONG + 6, 1: p + 10}
    for sl in range(2, 16):
        n = 600 + 211 * sl
        src = (T0 if sl % 2 else T1)[sl * 101:sl * 101 + n + 16]
        E.reset(sl)
        E.fill(src[:n], list(range(n)), [sl], [0, n])
        hist[sl] = n
    for B in ((4,) if a.w32 else ()) + (8, 16):
        ids, pos, ss = [], [], [0]
        for sl in range(B):
            h0 = hist[sl]
            ids += (T0 if sl % 2 else T1)[(h0 + 37 * sl) % 6000:(h0 + 37 * sl) % 6000 + 8]
            pos += list(range(h0, h0 + 8))
            ss.append(ss[-1] + 8)
            hist[sl] = h0 + 8
        res[f"w{8 * B}"] = E.step(ids, pos, list(range(B)), ss, True).cpu()
    say(f"[chk] wide done ({'w32, ' if a.w32 else ''}w64, w128)")
if a.dspark:
    E.dspark_init()
    E.reset(0)
    E.reset(1)
    n0, n1 = 1500, 1101
    res["dsp"] = E.step(T0[:n0] + T1[:n1], list(range(n0)) + list(range(n1)), [0, 1], [0, n0, n0 + n1], False,
                        [0, 639, 640, 1279, 1280, 1499, 1500, 1919, 1920, 2600], prefill=True).cpu()
    for sl, (seq, n) in enumerate([(T0, n0), (T1, n1)]):
        dr, cf = E.draft_at(sl, seq[n], n, 7)
        res[f"dsp_draft{sl}"] = torch.tensor(dr)
        res[f"dsp_conf{sl}"] = torch.tensor([cf], dtype=torch.float32)
    say(f"[chk] dsp drafts {res['dsp_draft0'].tolist()} · {res['dsp_draft1'].tolist()}")
E.close()

if R0:
    os.makedirs(a.out, exist_ok=True)
    torch.save(res, os.path.join(a.out, f"{a.label}.pt"))
    say(f"[chk] saved {a.out}/{a.label}.pt: " + " · ".join(f"{k} {tuple(v.shape)}" for k, v in res.items()))
    if a.cmp:
        ref = torch.load(a.cmp)
        theirs = ref.get("pf_state", torch.zeros(len(PF), dtype=torch.int32))  # (older files: all off then)
        if not torch.equal(theirs, res["pf_state"]):
            say(f"[chk] NOT COMPARED: prefill-rule patches {PF} were {theirs.tolist()} in {a.cmp}, {pf_state} here — "
                f"every stage after a prefill differs by the prefill rule, not bitwise; run both with the same --pf "
                f"(those patches pass gate.py --pf both and pf_check.py instead)")
            sys.exit(1)
        if "pf_state" not in ref:
            say(f"[chk] {a.cmp} has no pf_state: taken as all off")
        for k, v in res.items():
            if k not in ref:
                say(f"[chk] {k}: not in {a.cmp}")
                continue
            w = ref[k]
            if v.dtype == torch.float32:
                same = torch.equal(v.view(torch.int32), w.view(torch.int32))
                if same:
                    say(f"[chk] {k}: bitwise identical ({v.shape[0]} rows)")
                else:
                    diff = (v - w).abs()
                    rows = (v.view(torch.int32) != w.view(torch.int32)).any(1).nonzero().flatten().tolist()
                    say(f"[chk] {k}: DIFFERENT rows {rows} · max |Δ| {diff.max().item():.3g} · "
                        f"argmax equal {bool((v.argmax(1) == w.argmax(1)).all())}")
            else:
                say(f"[chk] {k}: {'identical' if torch.equal(v, w) else 'DIFFERENT'} {v.tolist()} vs {w.tolist()}")
