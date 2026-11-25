#!/usr/bin/env python3
"""sched_ckpt.py — P146 prefix checkpoints through the server's scheduler (server/scheduler.py + kvguard.py), driven
through the engine binding like sched_dspark.py (both ranks run the same scheduler on the same requests; no HTTP). TP2:
  E55_GPU_LOCK=1 E55_TREE=ckpt tools/tp2.sh ckpt python3 engine/tests/sched_ckpt.py
(--dry: the prompts' token counts and common prefixes only, no engine — one GPU-less container.)
Serving shapes: 4 slots × 64K, chunk 8192, DSpark block 7, gate 0.3, the KV pool sized as the server sizes it (room
left − 15 GiB), checkpoints 8 a slot every 8192 tokens. A ≈ 32.9K-token prompt (thinking on) runs first (64 greedy
tokens); then, on the same scheduler:
  (a) the next turn as a client sends it without the last reply's reasoning (the template's empty think block): the
      prompt-end checkpoint restored — its time to first token through the scheduler is the measured number
  (b) the first prompt with one token changed at 20,000: the periodic checkpoint at 16,384 restored
  (c) the first prompt changed at 12,000 while slot 0 is busy: an empty slot copies slot 0's checkpoint at 8,192
Each is served again from 0 on a fresh scheduler (no checkpoints, the same slot): the last prompt row's logits, the 64
greedy DSpark tokens and every round's drafts, emitted tokens and verify logits (all rows) must be bitwise equal."""
import argparse
import hashlib
import os
import random
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, os.path.join(ROOT, "server"), os.path.join(ROOT, "tools")]
from chat import Chat, normalize_messages  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, default=0)
ap.add_argument("--head-ip", default="")
ap.add_argument("--port", type=int, default=29611)
ap.add_argument("--model", default="/models/ERNIE-5.5-Flash-Preview")
ap.add_argument("--table", default="/work/cache/table.tsv")
ap.add_argument("--slots", type=int, default=4)
ap.add_argument("--ctx", type=int, default=65536)
ap.add_argument("--chunk", type=int, default=8192)
ap.add_argument("--block", type=int, default=7)
ap.add_argument("--gate", type=float, default=0.3)
ap.add_argument("--reserve", type=float, default=15.0, help="server.memory_free_gib")
ap.add_argument("--tokens", type=int, default=32900, help="the first prompt's length (about)")
ap.add_argument("--gen", type=int, default=64)
ap.add_argument("--dry", action="store_true")
a = ap.parse_args()
R0 = a.rank == 0
SEED = 1234


def say(*x):
    if R0:
        print(*x, flush=True)


# ---- prompts -------------------------------------------------------------------------------------------------------
chat = Chat(a.model)
W = dict(place="Arden Brockley Calloway Dunmore Elstead Farrow Glenmoor Harwick Ivydale Jessop Kelmscott Larchfield".split(),
         thing="output throughput backlog latency yield turnover inventory demand headcount uptime".split(),
         adj="higher lower steadier noisier slower faster".split(),
         name="Mara Tobias Ingrid Rafael Selin Oduya Petra Hamish Ying Dario".split(),
         verb="drifted recovered stalled doubled halved flattened".split())


def report(n, rng):
    out = []
    for i in range(n):
        c = {k: rng.choice(v) for k, v in W.items()}
        out.append(f"Entry {i + 1}. On day {rng.randint(1, 365)} the team at {c['place']} measured {c['thing']} at "
                   f"{rng.randint(10, 9999)} units, {c['adj']} than the week before; {c['name']} noted that the "
                   f"{rng.choice(W['thing'])} had {c['verb']} and asked for a review of the {rng.choice(W['thing'])}.")
    return "\n".join(out)


def turn1(n):
    return [{"role": "user", "content": "Here is a field report.\n\n" + report(n, random.Random(7)) +
             "\n\nWhich three places appear most often in the report, and what changed there? Answer briefly."}]


def encode(msgs):
    return chat.encode_chat(chat.render(normalize_messages(msgs), None, thinking=True, level="high"))


n = max(1, a.tokens // 45)
for _ in range(4):  # entries for about --tokens tokens
    m1 = turn1(n)
    L = len(encode(m1))
    n = max(1, round(n * a.tokens / L))
m1 = turn1(n)
P1 = encode(m1)
ANSWER = ("The places named most often are Glenmoor, Harwick and Calloway. At Glenmoor the backlog drifted and was "
          "reviewed; at Harwick throughput doubled; at Calloway latency stalled before it recovered.")
m2 = m1 + [{"role": "assistant", "content": ANSWER},
           {"role": "user", "content": "Now list, for Harwick only, every entry where throughput was measured, with "
                                       "its day and value, as a table."}]
P2 = encode(m2)
EDIT = chat.encode(" banana")[0]


def edited(at):
    p = list(P1)
    p[at] = EDIT if p[at] != EDIT else EDIT + 1
    return p


P3, P4 = edited(20000), edited(12000)


def lcp(x, y):
    k = 0
    while k < min(len(x), len(y)) and x[k] == y[k]:
        k += 1
    return k


say(f"[prompts] P1 {len(P1):,} tokens ({n} entries) · P2 (next turn, reasoning dropped) {len(P2):,} · common prefix "
    f"{lcp(P1, P2):,} · P3 / P4 = P1 edited at 20,000 / 12,000")
if a.dry:
    sys.exit(0)

import ext  # noqa: E402
from kvguard import KvGuard  # noqa: E402
from scheduler import Req, Scheduler  # noqa: E402

E = ext.load("engine", ext.engine_sources(), log=f"/tmp/build{a.rank}.log", ldflags=ext.ENGINE_LDFLAGS)
LR = a.slots * (a.block + 1)
E.init(a.model, a.table, a.rank, a.head_ip, a.slots, a.ctx, max(a.chunk, LR), LR, a.port)
E.dspark_init()
E.kv_pool_init(a.reserve, 0.0, False)  # the pool to the room left (the server's sizing, without its warm-up)
say(f"[ckpt] engine + draft up · P146 prefix_ckpt {E.patch_on('prefix_ckpt')} · pool {E.kv_total():,} chunks")


def h(t):
    return hashlib.sha1(t.contiguous().cpu().numpy().tobytes()).hexdigest()


class Tap:
    """The engine as the scheduler sees it, recording the last prompt row's logits, every round's drafts / emitted
    tokens / verify logits, and the wall time of the checkpoint restore / copy calls."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.pf, self.rounds, self.ck_ms = None, [], 0.0

    def __getattr__(self, n):
        return getattr(E, n)

    def step(self, ids, pos, slot, ss, all_logits):
        out = E.step(ids, pos, slot, ss, all_logits)
        self.pf = out[0].clone()
        return out

    def spec_round(self, slots, draft_next, caps):
        d = [list(E.spec_drafts(s)) for s in slots]
        out = E.spec_round(slots, draft_next, caps)
        self.rounds.append((d, [list(t) for t in out], h(E.spec_logits())))
        return out

    def ckpt_restore(self, s, k):
        t = time.perf_counter()
        E.ckpt_restore(s, k)
        self.ck_ms = 1e3 * (time.perf_counter() - t)

    def ckpt_copy(self, d, s, k):
        t = time.perf_counter()
        ok = E.ckpt_copy(d, s, k)
        torch.cuda.synchronize()
        self.ck_ms = 1e3 * (time.perf_counter() - t)
        return ok


tap = Tap()


class Busy:  # a slot marked busy (the scheduler only checks slot.req)
    pass


def serve(s, prompt, busy=()):
    """one greedy request (thinking on) through scheduler s → (Req, tokens, the tap's records)"""
    for i in busy:
        s.slots[i].req = Busy()
    got = []
    r = Req(prompt, a.gen, 0.0, 0, 1.0, SEED, lambda t, f: got.extend(t), "ckpt")
    r.thinking = True
    tap.clear()
    s.submit(r)
    s.drain(time.time() + 1e9)
    s.run()
    for i in busy:
        s.slots[i].req = None
    return r, got, (tap.pf, list(tap.rounds), tap.ck_ms)


def sched(ck):
    kv = None
    if ck:  # the server's admission (no MemFree watch: its collective would not pair up across the two schedulers)
        kv = KvGuard(E, a.ctx, 32768, 4.0)
        kv.poll_s = float("inf")
    return Scheduler(tap, chat.eos, a.slots, a.ctx, a.chunk, think_end=chat.think_end, loop_guard=True, block=a.block,
                     gate=a.gate, kv=kv, checkpoints=8 if ck else 0, checkpoint_every=8192)


def cks(s, i):
    v = sorted(s.slots[i].ck.values())
    return " · ".join(f"{p:,} ({E.ckpt_bytes(p) / 2**20:.2f} MiB, {c} chunks)" for p, c in v), \
        sum(E.ckpt_bytes(p) for p, _ in v), sum(c for _, c in v)


S = sched(True)
t = time.time()
r1, out1, _ = serve(S, P1)
desc, by, ch = cks(S, 0)
say(f"[turn 1] P1 {len(P1):,} tokens from 0: first token {r1.t_first - r1.t0:.2f} s · {len(out1)} tokens · slot 0 "
    f"checkpoints: {desc} · per slot {by / 2**20:.2f} MiB in {ch} chunks ({ch * 2} MiB of the pool)")
CASES = [("a", "next turn, last reasoning dropped", P2, ()), ("b", "edit at 20,000", P3, ()),
         ("c", "edit at 12,000, slot 0 busy", P4, (0,))]
res = {}
for key, what, P, busy in CASES:
    r, out, rec = serve(S, P, busy)
    res[key] = (r, out, rec)
    say(f"[{key}] {what}: prompt {len(P):,} · slot {r.slot.i} · reused {r.cached:,} ({r.reuse}) · first token "
        f"{r.t_first - r.t0:.3f} s (restore / copy {rec[2]:.1f} ms, then {len(P) - r.cached:,} tokens read) · "
        f"{len(out)} tokens")
    if key == "a":
        d, by, ch = cks(S, r.slot.i)
        say(f"[a] slot {r.slot.i} checkpoints now: {d} · {by / 2**20:.2f} MiB in {ch} chunks")
allok = True
for key, what, P, busy in CASES:
    Sr = sched(False)
    r, out, rec = serve(Sr, P, busy)
    r0, out0, rec0 = res[key]
    pf = rec0[0] is not None and torch.equal(rec0[0], rec[0])
    same_r = rec0[1] == rec[1]
    ok = pf and out0 == out and same_r and r.slot.i == r0.slot.i and r.cached == 0 and r0.cached > 0
    allok &= ok
    nb = sum(x != y for x, y in zip(rec0[1], rec[1])) + abs(len(rec0[1]) - len(rec[1]))
    say(f"[{key}] vs the full prefill (slot {r.slot.i}, first token {r.t_first - r.t0:.2f} s): last prompt row logits "
        f"{'bitwise equal' if pf else 'DIFFER'} · {len(out)} tokens {'IDENTICAL' if out0 == out else 'DIFFER'} · "
        f"{len(rec[1])} rounds (drafts, tokens, verify logits) {'bitwise equal' if same_r else f'{nb} DIFFER'}")
    if not pf and rec0[0] is not None:
        say(f"  max |Δ| prompt row {float((rec0[0] - rec[0]).abs().max()):.3g}")
say(f"[ckpt] all: {'IDENTICAL' if allok else 'FAILED'} · {time.time() - t:.0f} s")
E.close()
