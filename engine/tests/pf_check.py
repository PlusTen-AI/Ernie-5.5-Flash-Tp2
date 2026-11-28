#!/usr/bin/env python3
"""pf_check.py — the prefill rule (2026-11-28) at the model level: with the prefill-rule patches (pf_rule.py) on, a
prompt row's logits must not depend on how the prompt is read, and decode-kind steps must never take those kernels.
TP2 (both ranks the same calls; rank 0 prints and saves):
  tools/tp2.sh pfchk python3 /work/src/engine/tests/pf_check.py [--prompts /work/results/ref/v3/prompts.json]
One long prompt on slot 0 (prompts.json's "long", tools/ref/prompts.py --long; else doc_long + en_tools repeated to
--len tokens), logits at probe rows — 0, 1, every 2048 bound ± 1, every 1024th row from 114,688 (HCA rows with ≥ 5 key
chunks: where the base takes P150 / P172), the last row — plus each step's last row. Per state of the prefill-rule
patches (--states on,off: "on" is the check, "off" the control that the base holds the same):
  c8192   chunks of 8192 (the server's); P146 checkpoints at --ck (a chunk end) and at the prompt's end
  rest    the --ck checkpoint restored, the rest read as 1 row (a 1-row prefill chunk: the prefill kernels), 2047,
          then 8192s
  dec     (state "on" only) from the end checkpoint: 4 single decode steps, one 8-row and one 128-row decode-kind step
          (all logits) with the patches on; restored; the same with them off — bitwise (decode / verify rows never
          take a prefill-rule kernel)
  c2048   chunks of 2048
  odd     pieces 8192, 1, 4095, 65, 3000, 1, 2047, 6000, 64, 7999, 333, 129 … (a checkpoint at the first piece end
          past --ck, off the 128-row compressor groups); then that checkpoint restored and the rest read in 8192s
  again   c8192 once more (determinism)
Every row seen in rest / c2048 / odd / its restore / again must equal c8192's, bitwise. Between the states only
reported (max |Δ|, top-1 agreement): those differ by the rule. Needs P125 + P146 (the KV pool, checkpoints) for
rest / dec / odd's restore; without them those parts are skipped. Summary → <out>/pf_check<tag>.txt."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ext  # noqa: E402
import pf_rule  # noqa: E402
import torch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, required=True)
ap.add_argument("--head-ip", required=True)
ap.add_argument("--port", type=int, default=29651)
ap.add_argument("--model", default="/models/ERNIE-5.5-Flash-Preview")
ap.add_argument("--table", default="/work/cache/table.tsv")
ap.add_argument("--prompts", default="/work/results/ref/v3/prompts.json")
ap.add_argument("--name", default="long")
ap.add_argument("--len", type=int, default=139264, help="the synthesized prompt's length (no --name in prompts.json)")
ap.add_argument("--states", default="on,off")
ap.add_argument("--ck", type=int, default=122880, help="the c8192 checkpoint (a multiple of 8192)")
ap.add_argument("--reserve", type=float, default=15.0, help="GiB the KV pool leaves free")
ap.add_argument("--out", default="/work/results/pf-check")
ap.add_argument("--tag", default="")
a = ap.parse_args()
R0 = a.rank == 0
PF = pf_rule.names()
CH = 8192
ODD = [8192, 1, 4095, 65, 3000, 1, 2047, 6000, 64, 7999, 333, 129]


def say(*x):
    if R0:
        print(*x, flush=True)


P = json.load(open(a.prompts))
if a.name in P:
    ids = P[a.name]
else:
    d, t = P["doc_long"], P["en_tools"]
    ids = ((d + t) * (a.len // len(d + t) + 1))[:a.len]
T = len(ids)
assert a.ck % CH == 0 and 0 < a.ck < T, "--ck: a chunk end inside the prompt"
probe = {0, 1, T - 1} | {x for k in range(2048, T, 2048) for x in (k - 1, k, k + 1) if x < T} | set(range(114688, T, 1024))
t0 = time.time()
E = ext.load("engine", ext.engine_sources(), log=f"/tmp/build-pfc{a.rank}.log", ldflags=ext.ENGINE_LDFLAGS)
E.init(a.model, a.table, a.rank, a.head_ip, 2, T + 256, CH, 128, a.port)
E.kv_pool_init(a.reserve, 0.0, False)
CK = E.ckpt_on()
if CK:
    E.ckpt_init(2)
start = {n: E.patch_on(n) for n in PF}
say(f"[pfc] {a.name if a.name in P else 'synthesized'}: {T} tokens, {len(probe)} probe rows · prefill-rule patches "
    f"{start} · checkpoints {'on' if CK else 'OFF (rest / dec / odd restore skipped)'} · {time.time() - t0:.0f} s")


def set_pf(on):
    for n in PF:
        E.patch(n, on)


def read(pieces, at=0, ck_after=None):
    """slot 0 from position at, prefill steps of the given sizes (the last size repeated, the last step cut to the
    prompt's end) → {pos: logits};
    ck_after: (position, entry) — a checkpoint after the first step ending at or past position"""
    got, p, i, saved = {}, at, 0, None
    while p < T:
        n = min(pieces[min(i, len(pieces) - 1)], T - p)
        rows = sorted({q - p for q in probe if p <= q < p + n} | {n - 1})
        lg = E.step(ids[p:p + n], list(range(p, p + n)), [0], [0, n], False, rows, prefill=True).cpu()
        for j, r in enumerate(rows):
            got[p + r] = lg[j]
        p += n
        i += 1
        if ck_after and saved is None and p >= ck_after[0] and CK:
            assert E.ckpt_save(0, ck_after[1]), "checkpoint save failed (KV pool full?)"
            saved = p
    return got, saved


def cmp(name, got, base):
    """every row of got that base has: bitwise"""
    keys = sorted(set(got) & set(base))
    bad = [k for k in keys if not torch.equal(got[k].view(torch.int32), base[k].view(torch.int32))]
    if bad:
        d = max((got[k] - base[k]).abs().max().item() for k in bad)
        line = f"{name}: DIFFERENT at {len(bad)} of {len(keys)} rows (first {bad[:6]}) · max |Δ| {d:.3g}"
    else:
        line = f"{name}: bitwise identical ({len(keys)} rows)"
    say(f"[pfc]   {line}")
    return not bad, line


def dec_stage():
    """from the end checkpoint: decode-kind steps with the patches on, then off — bitwise"""
    rng = torch.Generator().manual_seed(7)
    toks = torch.randint(1000, 100000, (4 + 8 + 64,), generator=rng).tolist()
    outs = {}
    for on in (True, False):
        set_pf(on)
        E.ckpt_restore(0, 1)
        p, lg = T, []
        for i in range(4):
            lg.append(E.step([toks[i]], [p], [0], [0, 1], False).cpu())
            p += 1
        for n, o in ((8, 4), (64, 12)):  # (64: a copy round's most rows for one sequence; the server sends no more)
            lg.append(E.step(toks[o:o + n], list(range(p, p + n)), [0], [0, n], True).cpu())
            p += n
        outs[on] = torch.cat(lg)
    set_pf(True)
    same = torch.equal(outs[True].view(torch.int32), outs[False].view(torch.int32))
    line = (f"dec (1 ×4, 8, 64 decode-kind rows from the prompt's end, patches on vs off): "
            f"{'bitwise identical' if same else 'DIFFERENT'} ({outs[True].shape[0]} rows)")
    say(f"[pfc]   {line}")
    return same, line


lines, ok, firsts = [], True, {}
for st in a.states.split(","):
    on = st == "on"
    set_pf(on)
    say(f"[pfc] state {st}")
    t = time.time()
    E.reset(0)
    base, _ = read([CH], 0, (a.ck, 0))
    if CK:
        assert E.ckpt_save(0, 1), "checkpoint save at the end failed"
    say(f"[pfc]   c8192: {len(base)} rows · {time.time() - t:.0f} s")
    firsts[st] = base
    res = []
    if CK:
        E.ckpt_restore(0, 0)
        res.append(cmp(f"{st} rest (restore at {a.ck}, then 1, 2047, 8192s)", read([1, 2047, CH], a.ck)[0], base))
        if on:
            res.append(dec_stage())
    E.reset(0)
    res.append(cmp(f"{st} c2048", read([2048])[0], base))
    E.reset(0)
    got, at = read(ODD * (T // sum(ODD) + 1), 0, (a.ck, 0))  # (the cycle repeated)
    res.append(cmp(f"{st} odd pieces", got, base))
    if CK and at is not None:
        E.ckpt_restore(0, 0)
        res.append(cmp(f"{st} odd restore (at {at}, then 8192s)", read([CH], at)[0], base))
    E.reset(0)
    res.append(cmp(f"{st} again (c8192)", read([CH])[0], base))
    for good, line in res:
        ok &= good
        lines.append(line)
    say(f"[pfc]   state {st} · {time.time() - t:.0f} s")
for n in PF:
    E.patch(n, start[n])
if "on" in firsts and "off" in firsts:
    keys = sorted(set(firsts["on"]) & set(firsts["off"]))
    a1 = torch.stack([firsts["on"][k] for k in keys])
    b1 = torch.stack([firsts["off"][k] for k in keys])
    far = [i for i, k in enumerate(keys) if k >= 114688]
    line = (f"on vs off (c8192, by the rule not bitwise): max |Δ| {(a1 - b1).abs().max().item():.3g} · top-1 agree "
            f"{(a1.argmax(-1) == b1.argmax(-1)).float().mean().item():.3f} ({len(keys)} rows)" +
            (f"; rows ≥ 114688 top-1 {(a1[far].argmax(-1) == b1[far].argmax(-1)).float().mean().item():.3f}"
             if far else ""))
    say(f"[pfc] {line}")
    lines.append(line)
E.close()
lines.append(f"pf_check {PF} at {T} tokens: {'PASS' if ok else 'FAIL'}")
say(f"[pfc] {lines[-1]} · {time.time() - t0:.0f} s")
if R0:
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, f"pf_check{a.tag}.txt"), "w").write("\n".join(lines) + "\n")
