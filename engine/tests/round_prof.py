#!/usr/bin/env python3
"""round_prof.py — a DSpark serving round's ms and the decode-round patches P160–P163 (TP2, both ranks; tools/tp2.sh
appends --rank / --head-ip):
  E55_GPU_LOCK=1 E55_TREE=dec tools/tp2.sh dec-final python3 engine/tests/round_prof.py --do final
The server's scheduler (server/scheduler.py, decode.verify "cost", block 7) drives the engine binding on bench_lite's
decode case: one ≈ --ptok-token prompt (the checkpoint's README + bench_lite's question), thinking on (high), the
generation_config's sampling (T 1.0 / top_p 0.95), B = 1 (one stream) and B = 4 (four streams, seeds 1 … 4); only
rounds with all B streams count.
  final   greedy, one stream (≥ 64 rounds), all off vs all on: tokens identical, every kept verify row's logits
          bitwise, drafts + confidences identical; then per B the same requests (same seeds: the same rounds) with the
          patches switched on one by one in this process (all off, + P160 ctx_side, + P161 markov_back, + P162
          draft_qrows, all off again; + P163 markov_pipe last): per variant the round's ms (start to start, and the
          engine's events: verify target step, the draft's context rows, accept, commit, next drafts), and every
          variant's tokens and per-round drafts + confidences == all off's. The patches change nothing in the verify
          step, so each one's effect is read on the non-verify ms (start to start − verify); off vs off again and the
          verify's spread over the variants are the noise.
(The session-7 audit — the same rounds with the GPU held so the host had every launch queued, and the E55_DSP_PROF
stage / phase splits with expert counts — used hooks not kept in the tree; results/runs and the report have it.)"""
import argparse
import collections
import faulthandler
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, os.path.join(ROOT, "server"), os.path.join(ROOT, "tools")]
import torch  # noqa: E402

import ext  # noqa: E402
from chat import Chat, normalize_messages  # noqa: E402
from scheduler import Req, Scheduler  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, required=True)
ap.add_argument("--head-ip", required=True)
ap.add_argument("--port", type=int, default=29611)
ap.add_argument("--model", default="/models/ERNIE-5.5-Flash-Preview")
ap.add_argument("--table", default="/work/cache/table.tsv")
ap.add_argument("--slots", type=int, default=4)
ap.add_argument("--ctx", type=int, default=16384)
ap.add_argument("--chunk", type=int, default=8192)
ap.add_argument("--block", type=int, default=7)
ap.add_argument("--gate", type=float, default=0.3)
ap.add_argument("--ptok", type=int, default=8000, help="prompt tokens (≈)")
ap.add_argument("--mt1", type=int, default=384, help="tokens of the B = 1 stream")
ap.add_argument("--mt4", type=int, default=256, help="tokens a stream at B = 4")
ap.add_argument("--mtg", type=int, default=400, help="tokens of the greedy stream")
ap.add_argument("--do", default="final")
a = ap.parse_args()
R0 = a.rank == 0


def say(*x):
    if R0:
        print(*x, flush=True)


E = ext.load("engine", ext.engine_sources(), log=f"/tmp/build{a.rank}.log", ldflags=ext.ENGINE_LDFLAGS)
LR = a.slots * (a.block + 1)
E.init(a.model, a.table, a.rank, a.head_ip, a.slots, a.ctx, a.chunk, LR, a.port)
E.dspark_init()
chat = Chat(a.model)
D = chat.defaults
TEMP, TOPP, TOPK = float(D["temperature"]), float(D["top_p"]), (D["top_k"] if D["top_k"] > 0 else 0)
Q = "\n\nBased on the document above, explain in detail how its parts fit together and what a reader should check first."


def prompt(ntok):
    """bench_lite's decode prompt at ≈ ntok tokens: the README repeated, cut, + the question; thinking on (high)"""
    doc = open(os.path.join(a.model, "README.md"), encoding="utf-8").read()
    n = int(ntok * 4.4)
    for _ in range(6):
        body = (doc * (n // len(doc) + 1))[:n]
        ids = chat.encode_chat(chat.render(normalize_messages([{"role": "user", "content": body + Q}]), None,
                                           thinking=True, level="high"))
        if abs(len(ids) - ntok) <= 32:
            break
        n = int(n * ntok / len(ids))
    return ids


PROMPT = prompt(a.ptok)
say(f"[round] engine + draft up · prompt {len(PROMPT)} tokens (README + bench_lite's question, thinking on) · "
    f"T {TEMP} top_p {TOPP} top_k {TOPK} · block {a.block} · decode.verify cost · {a.slots} slots × {a.ctx}")


class RTap:
    """The engine as the scheduler sees it; per spec_round: B, rows, the engine's events (spec_stats), the start and
    the host wall of the call, the tokens, each slot's next drafts + confidences; keep(log entry, slots, out): the
    caller's look at the round's logits (greedy check)"""

    def __init__(self, keep=None):
        self.log, self.keep = [], keep

    def __getattr__(self, n):
        return getattr(E, n)

    def spec_round(self, slots, draft_next, caps):
        t0 = time.perf_counter()
        out = E.spec_round(slots, draft_next, caps)
        t1 = time.perf_counter()
        S = E.spec_stats(True)
        x = dict(B=len(slots), R=int(S["rows"]), t0=t0, wall=1e3 * (t1 - t0), S=S, toks=[list(y) for y in out],
                 dr=[(list(E.spec_drafts(s)), list(E.spec_conf(s))) for s in slots])
        if self.keep:
            self.keep(x, slots, out)
        self.log.append(x)
        return out


def drive(B, mt, temp=None, keep=None):
    """B streams (seeds 1 … B) through a fresh scheduler → (tap log, each stream's tokens)"""
    for s in range(a.slots):
        E.reset(s)
    tap = RTap(keep)
    sch = Scheduler(tap, chat.eos, a.slots, a.ctx, a.chunk, think_end=chat.think_end, loop_guard=True, block=a.block,
                    gate=a.gate, verify="cost")
    got = [[] for _ in range(B)]
    for i in range(B):
        def push(t, f, g=got[i]):
            g.extend(t)
        greedy = temp == 0.0
        r = Req(PROMPT, mt, 0.0 if greedy else TEMP, 0 if greedy else TOPK, 1.0 if greedy else TOPP, 1 + i, push, f"s{i}")
        r.thinking, r.budget = True, None
        sch.submit(r)
    sch.drain(time.time() + 1e9)
    sch.run()
    return tap.log, got


def full(log, B):
    """the rounds with all B streams (and a next one: start-to-start wall)"""
    return [(i, log[i]) for i in range(len(log) - 1) if log[i]["B"] == B]


def mean(x):
    return sum(x) / len(x) if x else float("nan")


PH = ("verify", "ctx", "accept", "commit", "draft")


def split(log, B):
    """per round: events and start-to-start; → dict of means"""
    rr = full(log, B)
    if not rr:
        return None
    o = collections.OrderedDict()
    o["rounds"] = len(rr)
    o["s2s"] = mean([1e3 * (log[i + 1]["t0"] - x["t0"]) for i, x in rr])
    o["wall"] = mean([x["wall"] for _, x in rr])
    for k in PH:
        o[k] = mean([x["S"]["ms_" + k] for _, x in rr])
    o["events"] = sum(o[k] for k in PH)
    o["gap"] = o["s2s"] - o["events"]
    o["rows"] = mean([x["R"] for _, x in rr])
    o["emitted"] = mean([x["S"]["emitted"] for _, x in rr])
    o["accepted"] = mean([x["S"]["accepted"] for _, x in rr])
    o["rest"] = o["s2s"] - o["verify"] - o["draft"]
    return o


def fmt(o, keys, f="{:.2f}"):
    return " · ".join(f"{k} " + f.format(o[k]) for k in keys if k in o)


# ---- final: greedy all off vs all on, then P160–P162 one by one, then P163 ------------------------------------------
MY = ("ctx_side", "markov_back", "draft_qrows", "markov_pipe")
LADDER = (("all off", 0), ("+ P160 ctx_side", 1), ("+ P161 markov_back", 2), ("+ P162 draft_qrows", 3),
          ("all off again", 0))


def setp(k):
    for j, n in enumerate(MY):
        E.patch(n, j < k)


def watched(*x, **kw):
    faulthandler.dump_traceback_later(300, exit=True)  # a hang: both ranks' stacks, then out
    r = drive(*x, **kw)
    faulthandler.cancel_dump_traceback_later()
    return r


def greedy_check():
    rows, bad = {}, [0, 0]

    def keep_off(x, slots, out):
        rows[len(rows)] = E.spec_logits()[:len(out[0])].cpu()

    def keep_on(x, slots, out):
        i = keep_on.i
        keep_on.i += 1
        L = E.spec_logits()[:len(out[0])].cpu()
        ref = rows.get(i)
        bad[1] += L.shape[0]
        bad[0] += ref is None or ref.shape != L.shape or not torch.equal(ref.view(torch.int32), L.view(torch.int32))
    keep_on.i = 0
    setp(0)
    logA, gotA = watched(1, a.mtg, 0.0, keep_off)
    setp(len(MY))
    logB, gotB = watched(1, a.mtg, 0.0, keep_on)
    dra, drb = [x["dr"] for x in logA], [x["dr"] for x in logB]
    ok = gotA == gotB and bad[0] == 0 and dra == drb and len(logA) >= 64
    say(f"[final] greedy, one stream, all off vs all on (P160–P163): {len(logA)} / {len(logB)} rounds · {len(gotA[0])} "
        f"tokens {'identical' if gotA == gotB else 'DIFFER'} · kept rows' logits bitwise equal {bad[1] - bad[0]}/{bad[1]} · "
        f"drafts + confidences {'identical' if dra == drb else 'DIFFER'} → {'PASS' if ok else 'FAIL'}")
    return ok


def variant(B, mt, name, k, ref):
    setp(k)
    log, got = watched(B, mt)
    o = split(log, B)
    rd = [(x["R"], x["toks"], x["dr"]) for x in log]
    same = ref is None or (got, rd) == ref
    o["nv"] = o["s2s"] - o["verify"]
    say(f"[final] B {B} · {name}: {o['rounds']} rounds of B {B} · rows {o['rows']:.2f} · emitted {o['emitted']:.2f} · "
        f"round {o['s2s']:.2f} ms start to start (events {o['events']:.2f}: " + fmt(o, PH) + f") · non-verify "
        f"{o['nv']:.2f} (draft {o['draft']:.2f} + rest {o['rest'] - 0:.2f}) · tokens, rows, drafts + confidences "
        + ("(the reference)" if ref is None else "identical to all off" if same else "DIFFER from all off"))
    return o, (got, rd), same


def final():
    dflt = {n: E.patch_on(n) for n in MY}
    allok = greedy_check()
    res, refs = {}, {}
    for B, mt in ((1, a.mt1), (4, a.mt4)):
        res[B] = []
        for name, k in LADDER:
            o, r, same = variant(B, mt, name, k, refs.get(B))
            refs.setdefault(B, r)
            allok &= same
            res[B].append((name, o))
    for B, mt in ((1, a.mt1), (4, a.mt4)):  # P163 last (its comm-stream gathers)
        o, r, same = variant(B, mt, "+ P163 markov_pipe (all on)", 4, refs[B])
        allok &= same
        res[B].insert(4, ("+ P163 markov_pipe", o))
    for B in (1, 4):
        R = res[B]
        off = {k: (R[0][1][k] + R[-1][1][k]) / 2 for k in ("s2s", "verify", "nv", "draft", "rest")}
        on = R[4][1]
        vs = [o["verify"] for _, o in R]
        say(f"[final] B {B} · all off (mean of the two) → all on: round {off['s2s']:.2f} → {on['s2s']:.2f} ms "
            f"({on['s2s'] - off['s2s']:+.2f}) · verify {off['verify']:.2f} → {on['verify']:.2f} · draft {off['draft']:.2f} → "
            f"{on['draft']:.2f} · rest {off['rest']:.2f} → {on['rest']:.2f} · non-verify {off['nv']:.2f} → {on['nv']:.2f} "
            f"({on['nv'] - off['nv']:+.2f}) · per patch (non-verify ms, step by step): " +
            " · ".join(f"{R[i][0].split(' ')[1]} {R[i][1]['nv'] - R[i - 1][1]['nv']:+.2f}" for i in range(1, 5)) +
            f" · noise: off vs off again non-verify {R[-1][1]['nv'] - R[0][1]['nv']:+.2f}, verify spread "
            f"{max(vs) - min(vs):.2f} over the variants")
    for n, v in dflt.items():
        E.patch(n, v)
    say(f"[final] all: {'PASS' if allok else 'FAIL'}")


for what in a.do.split(","):
    t = time.time()
    globals()[what]()
    say(f"[round] {what} · {time.time() - t:.0f} s")
E.close()
