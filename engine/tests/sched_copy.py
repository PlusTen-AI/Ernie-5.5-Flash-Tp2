#!/usr/bin/env python3
"""sched_copy.py — copy drafts (P168 copy_draft: engine spec_copy, server/copy_draft.py, the scheduler's copy rounds) on
TP2 (tools/tp2.sh appends --rank / --head-ip):
  E55_GPU_LOCK=1 E55_TREE=copy tools/tp2.sh copy python3 engine/tests/sched_copy.py
Both ranks run the same scheduler and matcher (no time in any decision), so their engine calls pair up as rank 1's
replay would. Serving shapes: 4 slots, chunk 8192, block 7, decode.verify "cost", logit rows max(4 × 8, 64). Each
segment prints as it goes; a failing segment does not stop the next ones.
  width   one sequence at ≈ 8K context (the README repeated): copy rounds of 16 / 32 / 64 rows (the text that follows as
          the copy; whatever is kept) — verify step ms (CUDA events: the target step with the draft's context rows) and
          the round's wall ms, --reps rounds each
  ref     the rewrite request (--file in the prompt, "repeat it with one change", thinking off, greedy) through the
          scheduler with copy off (today's DSpark rounds) → tokens, decode tok/s, tokens a round; plain greedy decoding
          by one-row step()s fed them → every position's logits fingerprint (Σ bits·w, int64) and argmax = next token
  forced  slot 1, teacher-forced copy rounds along those tokens: per width 16 / 32 / 64 rows, a copy wrong at a
          compressor group's closing position c ((c + 1) % 128 = 0) and one wrong right after it (c + 1), each with
          ≥ 129 positions of kept rows after the rollback before the next — every kept row's logits bitwise the one-row
          step's (by verify rows 2–8 / 9–16 / 17–32 / 33–64) and the emitted tokens the reference's; then the DSpark
          drafts + confidences at the end bitwise those after one-row steps' prefill to the same position (the draft's
          context ring). → the cap: the clean width with the most tokens per ms of a round accepted whole
  greedy  the request with copy on (that cap) → tokens identical, every kept row bitwise (by verify rows), decode tok/s
          (the checks' own time taken out), tokens a round, copy rounds and their rows, matcher µs a round without a copy
  sampled T / top_p from generation_config (1.0 / 0.95), the request with copy off vs on: decode tok/s, tokens a round,
          the text a sane rewrite (similarity to the file with the change)"""
import argparse
import difflib
import faulthandler
import os
import sys
import time
import traceback

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, os.path.join(ROOT, "server"), os.path.join(ROOT, "tools")]
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
ap.add_argument("--ctx", type=int, default=32768)
ap.add_argument("--chunk", type=int, default=8192)
ap.add_argument("--block", type=int, default=7)
ap.add_argument("--gate", type=float, default=0.3)
ap.add_argument("--rows", type=int, default=64, help="widest copy round (logit rows ≥ this)")
ap.add_argument("--match", type=int, default=8, help="decode.copy_match")
ap.add_argument("--file", default="server/link.py", help="the file the rewrite request repeats")
ap.add_argument("--mt", type=int, default=3000)
ap.add_argument("--wctx", type=int, default=8192)
ap.add_argument("--reps", type=int, default=3)
a = ap.parse_args()
R0 = a.rank == 0
SEED = 1234
faulthandler.enable()
T00 = time.time()


def say(*x):
    if R0:
        print(f"[{time.time() - T00:6.0f}s]", *x, flush=True)


E = ext.load("engine", ext.engine_sources(), log=f"/tmp/build{a.rank}.log", ldflags=ext.ENGINE_LDFLAGS)
LR = max(a.slots * (a.block + 1), a.rows)
E.init(a.model, a.table, a.rank, a.head_ip, a.slots, a.ctx, max(a.chunk, LR), LR, a.port)
E.dspark_init()
chat = Chat(a.model)
D = chat.defaults
say(f"[copy] engine + draft up · logit rows {LR} · P168 copy_draft {E.patch_on('copy_draft')}")
V = 201216
W = (torch.randint(-2**62, 2**62, (V,), generator=torch.Generator().manual_seed(7), dtype=torch.int64) | 1).cuda()


def fps(L):
    """fingerprints of logits rows [n, V] fp32: Σ_v bits_v · w_v (int64, wrapping) — equal rows, equal prints"""
    return ((L.contiguous().view(torch.int32).to(torch.int64) * W).sum(dim=1)).tolist()


def bucket(M):
    return "2-8" if M <= 8 else "9-16" if M <= 16 else "17-32" if M <= 32 else "33-64"


def prefill(slot, toks, last_logits=True):
    """toks from position 0 into slot (reset first); → the last row's logits (when last_logits)"""
    E.reset(slot)
    T = len(toks)
    for c0 in range(0, T, a.chunk):
        c1 = min(T, c0 + a.chunk)
        if c1 < T or not last_logits:
            E.fill(toks[c0:c1], list(range(c0, c1)), [slot], [0, c1 - c0])
        else:
            return E.step(toks[c0:c1], list(range(c0, c1)), [slot], [0, c1 - c0], False)
    return None


SRC = open(os.path.join(ROOT, a.file)).read()
WANT = SRC.replace("_alive(", "_keepalive(")
MSG = ("Here is a Python file:\n\n```python\n" + SRC + "```\n\nRewrite this file with one small change: rename the function "
       "`_alive` to `_keepalive` (its definition and every call). Output the complete updated file in one ```python code "
       "block and nothing else.")
PROMPT = chat.encode_chat(chat.render(normalize_messages([{"role": "user", "content": MSG}]), None, thinking=False,
                                      level="high"))


class Tap:
    """the engine as the scheduler sees it: every round's kept rows against the reference fingerprints (ref: position →
    print), by verify rows; per round (kind, rows, emitted, engine-call ms); the checks' own time"""

    def __init__(self, ref):
        self.ref, self.rows, self.bad, self.log, self.t_chk = ref, {}, {}, [], 0.0

    def __getattr__(self, n):
        return getattr(E, n)

    def _check(self, P, toks, M):
        t = time.perf_counter()
        if self.ref is not None:
            L = E.spec_logits()
            f = fps(L[:len(toks)])
            b = bucket(M)
            self.rows[b] = self.rows.get(b, 0) + len(toks)
            self.bad[b] = self.bad.get(b, 0) + sum(self.ref.get(P + j) != f[j] for j in range(len(toks)))
        self.t_chk += time.perf_counter() - t

    def spec_round(self, slots, draft_next, caps):
        P = E.spec_pos(slots[0])
        t = time.perf_counter()
        out = E.spec_round(slots, draft_next, caps)
        ms = 1e3 * (time.perf_counter() - t)
        M = len(out[0]) and (E.spec_verified()[0] + 1)
        self.log.append(("dsp", M, len(out[0]), ms))
        self._check(P, out[0], M)
        return out

    def spec_copy(self, slot, copy, cap, follow, draft_next):
        P = E.spec_pos(slot)
        t = time.perf_counter()
        out = E.spec_copy(slot, copy, cap, follow, draft_next)
        ms = 1e3 * (time.perf_counter() - t)
        self.log.append(("copy", len(copy) + 1, len(out), ms))
        self._check(P, out, len(copy) + 1)
        return out


def serve(eng, prompt, temp, copy, rows=64):
    """one request through a fresh scheduler (slots start over) → (Req, tokens, finish, decode tok/s, sched)"""
    s = Scheduler(eng, chat.eos, a.slots, a.ctx, a.chunk, think_end=chat.think_end, loop_guard=True, block=a.block,
                  gate=a.gate, verify="cost", copy=copy, copy_rows=rows, copy_match=a.match)
    got, fin, end = [], [], []

    def push(t, f):
        got.extend(t)
        if f:
            fin.append(f)
            end.append(time.time())
    top_k = D["top_k"] if D["top_k"] > 0 else 0
    r = Req(prompt, a.mt, temp, top_k if temp > 0 else 0, float(D["top_p"]) if temp > 0 else 1.0, SEED, push, "rewrite")
    s.submit(r)
    s.drain(time.time() + 1e9)
    s.run()
    dt = end[0] - r.t_first - (eng.t_chk if isinstance(eng, Tap) else 0.0)
    return r, got, (fin or ["?"])[0], (r.n_out - 1) / dt if dt > 0 else 0.0, s


def rounds_line(log):
    out = []
    for kind in ("copy", "dsp"):
        x = [e for e in log if e[0] == kind]
        if x:
            out.append(f"{kind} {len(x)} rounds · {sum(e[1] for e in x) / len(x):.1f} rows · "
                       f"{sum(e[2] for e in x) / len(x):.2f} tokens · {sum(e[3] for e in x) / len(x):.1f} ms a round")
    return " | ".join(out)


def segment(fn):
    try:
        return fn()
    except Exception:  # (both ranks see the same data: they fail alike)
        say(f"[{fn.__name__}] FAILED:\n{traceback.format_exc()}")
        return None


# ---- width: verify ms at 16 / 32 / 64 rows, one sequence at ≈ 8K ----------------------------------------------------
MS = {}


def width():
    text = open(os.path.join(a.model, "README.md"), encoding="utf-8").read()
    toks = chat.encode(text)
    while len(toks) < a.wctx + 4096:
        toks = toks + toks
    E.spec_stats(True)
    prefill(3, toks[:a.wctx])
    E.spec_begin(3, a.block, 0.0, SEED, 0, 0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, False)
    for rows in (16, 32, 64):
        if rows > a.rows:
            continue
        v, w, kept = [], [], 0
        for _ in range(a.reps + 1):  # (the first of each width: warm-up, not counted)
            p = E.spec_pos(3)
            cp = toks[p + 1:p + rows]
            E.spec_stats(True)
            t = time.perf_counter()
            out = E.spec_copy(3, cp, 0, -1, False)
            w.append(1e3 * (time.perf_counter() - t))
            S = E.spec_stats(True)
            v.append(S["ms_verify"] + S["ms_ctx"])
            kept += len(out)
        v, w = sorted(v[1:]), sorted(w[1:])
        MS[rows] = v[len(v) // 2]
        say(f"[width] {rows} rows at {E.spec_pos(3):,}: verify step {MS[rows]:.1f} ms (median of {a.reps}; "
            f"{', '.join(f'{x:.1f}' for x in v)}) · round wall {w[len(w) // 2]:.1f} ms · {MS[rows] / rows:.2f} ms a row")
    E.reset(3)


# ---- ref: copy off (today's rounds) + plain one-row decoding's logits ------------------------------------------------
G, REF, OFF = [], {}, {}


def ref():
    E.spec_stats(True)
    tap = Tap(None)
    r, got, fin, tps, _ = serve(tap, PROMPT, 0.0, False)
    S = E.spec_stats(True)
    G.extend(got)
    OFF.update(tps=tps, rounds=S["rounds"], emitted=S["emitted"])
    say(f"[ref] prompt {len(PROMPT):,} tokens ({len(chat.encode(SRC)):,} of them the file) · greedy, copy off: "
        f"{len(got):,} tokens ({fin}) · decode {tps:.1f} tok/s · {S['emitted'] / max(1, S['rounds']):.2f} tokens a "
        f"round ({S['rounds']:.0f} rounds) · {rounds_line(tap.log)}")
    T, seq = len(PROMPT), PROMPT + got
    REF[T - 1] = fps(prefill(0, PROMPT))[0]
    mism = 0
    for i in range(T, len(seq)):
        L = E.step([seq[i]], [i], [0], [0, 1], False)
        REF[i] = fps(L)[0]
        if i + 1 < len(seq):
            mism += int(L[0].argmax()) != seq[i + 1]
    E.reset(0)
    txt = chat.decode(got)
    say(f"[ref] plain one-row decoding over the {len(got):,} tokens: argmax = the next token at "
        f"{len(got) - 1 - mism}/{len(got) - 1} positions · the reply {'contains' if WANT.strip() in txt else 'differs from'}"
        f" the file with the change (similarity {difflib.SequenceMatcher(None, txt, WANT).ratio():.3f})")


# ---- forced: teacher-forced copy rounds, rollbacks at group ends, the draft's ring -----------------------------------
CLEAN = {}


def forced():
    T, seq = len(PROMPT), PROMPT + G
    end = len(seq) - 1  # positions with a reference row: T − 1 … end
    L = prefill(1, PROMPT)
    pf = fps(L)[0] == REF[T - 1]
    first = E.spec_begin(1, a.block, 0.0, SEED, 0, 0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, False)
    rows, bad, tok_bad, after, after_bad, cases = {}, {}, 0, 0, 0, []

    def rnd(k, wrong=None):
        nonlocal tok_bad, after, after_bad
        p = E.spec_pos(1)
        k = min(k, end - 1 - p)
        if k < 1 or (wrong is not None and wrong - p - 1 >= k):
            return False
        cp = list(seq[p + 1:p + 1 + k])
        if wrong is not None:
            cp[wrong - p - 1] = 1000 + (cp[wrong - p - 1] + 1) % 1000  # not the reference's token
        out = E.spec_copy(1, cp, 0, -1, False)
        want = k + 1 if wrong is None else wrong - p
        tok_bad += out != seq[p + 1:p + 1 + len(out)] or len(out) != want
        f = fps(E.spec_logits()[:len(out)])
        b = bucket(k + 1)
        rows[b] = rows.get(b, 0) + len(out)
        nb = sum(REF.get(p + j) != f[j] for j in range(len(out)))
        bad[b] = bad.get(b, 0) + nb
        if cases and p - cases[-1][2] < 129:  # rows after the last rollback
            after += sum(1 for j in range(len(out)) if p + j - cases[-1][2] < 129)
            after_bad += sum(REF.get(p + j) != f[j] for j in range(len(out)) if p + j - cases[-1][2] < 129)
        return True

    ok = first == G[0]
    for R in (64, 32, 16):  # (the widest first: the reference tokens may run out)
        if R > a.rows:
            continue
        k = R - 1
        for case in ("at c", "after c"):
            p = E.spec_pos(1)
            c = p + 2 + (129 if cases else 0) + k // 2  # the next closing position ≥ 129 past the last rollback, mid-copy
            c += (127 - c % 128) % 128
            while c - p > k // 2 + 1:  # all-right copies until c sits in the middle of the next one
                if not rnd(min(k, c - p - k // 2 - 1)):
                    break
                p = E.spec_pos(1)
            w = c if case == "at c" else c + 1
            if w > p + k or w < p + 1 or not rnd(k, w):
                say(f"[forced] {R} rows, wrong {case}: out of reference tokens at {p:,}")
                break
            cases.append((R, case, w, c))
    while E.spec_pos(1) - (cases[-1][2] if cases else 0) < 129 and rnd(63 if a.rows >= 64 else a.rows - 1):
        pass
    rb = ", ".join(f"{R} rows wrong {cs} = {w:,}" for R, cs, w, c in cases)
    say(f"[forced] slot 1 along the reference: prompt row {'bitwise' if pf else 'DIFFERS'} · first token "
        f"{'= ref' if ok else 'DIFFERS'} · rollbacks: {rb}"
        f" · tokens {'= ref' if not tok_bad else f'{tok_bad} rounds DIFFER'} · kept rows bitwise by verify rows: "
        + " · ".join(f"{b} {rows[b] - bad[b]}/{rows[b]}" for b in sorted(rows, key=lambda x: int(x.split('-')[0])))
        + f" · within 129 positions after a rollback {after - after_bad}/{after}")
    # the draft's context ring after copy rounds = after one-row steps' prefill to the same position
    p = E.spec_pos(1)
    d1, c1 = E.draft_at(1, seq[p], p, a.block)
    prefill(2, seq[:p], last_logits=False)
    d2, c2 = E.draft_at(2, seq[p], p, a.block)
    ring = d1 == d2 and list(c1) == list(c2)
    E.reset(1), E.reset(2)
    say(f"[forced] DSpark drafts at {p:,} after the copy rounds: {'bitwise' if ring else 'DIFFER'} (drafts "
        f"{'equal' if d1 == d2 else f'{d1} vs {d2}'}, confidences {'equal' if list(c1) == list(c2) else 'differ'})")
    lim = {"2-8": 8, "9-16": 16, "17-32": 32, "33-64": 64}
    for R in (16, 32, 64):
        CLEAN[R] = (ok and pf and not tok_bad and ring and not after_bad and
                    all(bad.get(b, 0) == 0 for b in lim if lim[b] <= R) and rows.get(bucket(R), 0) > 0)


# ---- greedy / sampled with copy on ------------------------------------------------------------------------------------
CAP = [0]


def pick():
    ok = [R for R in (16, 32, 64) if CLEAN.get(R) and R in MS]
    if not ok:
        say(f"[cap] no width is bitwise clean ({CLEAN}) — copy-on segments skipped")
        return
    CAP[0] = max(ok, key=lambda R: R / MS[R])
    say(f"[cap] bitwise clean widths {ok} · tokens a ms of a round accepted whole: "
        + " · ".join(f"{R} rows {R / MS[R]:.3f}" for R in ok) + f" → copy_rows {CAP[0]}")


def greedy():
    if not CAP[0]:
        return
    tap = Tap(REF)
    E.spec_stats(True)
    r, got, fin, tps, s = serve(tap, PROMPT, 0.0, True, CAP[0])
    S = E.spec_stats(True)
    cp = s.copy
    say(f"[greedy] copy on (≤ {CAP[0]} rows): {len(got):,} tokens ({fin}) · tokens "
        f"{'IDENTICAL' if got == G else 'DIFFER'} to copy off · decode {tps:.1f} tok/s (off {OFF['tps']:.1f}: "
        f"×{tps / max(OFF['tps'], 1e-9):.2f}) · {S['emitted'] / max(1, S['rounds']):.2f} tokens a round ({S['rounds']:.0f} "
        f"rounds; off {OFF['emitted'] / max(1, OFF['rounds']):.2f}) · {rounds_line(tap.log)}")
    if got != G:
        i = next((j for j in range(min(len(got), len(G))) if got[j] != G[j]), min(len(got), len(G)))
        say(f"  first difference at output token {i}: off {G[i:i + 4]} · on {got[i:i + 4]}")
    say(f"[greedy] kept rows' logits bitwise by verify rows: "
        + " · ".join(f"{b} {tap.rows[b] - tap.bad[b]}/{tap.rows[b]}" for b in sorted(tap.rows, key=lambda x: int(x.split('-')[0])))
        + f" · widest copy round {max((e[1] for e in tap.log if e[0] == 'copy'), default=0)} rows · matcher: "
        f"{cp.c['miss']} rounds without a copy {1e6 * cp.t['miss'] / max(1, cp.c['miss']):.1f} µs each, "
        f"{cp.c['hit']} copy rounds {1e6 * cp.t['hit'] / max(1, cp.c['hit']):.1f} µs, prefill chunk sync "
        f"{1e3 * cp.t['chunk'] / max(1, cp.c['chunk']):.2f} ms ({cp.c['chunk']}), {cp.c['whole']}/{cp.c['rounds']} copies "
        f"whole · index {cp.nbytes() / 2**20:.1f} MiB")


def sampled():
    if not CAP[0]:
        return
    T = float(D["temperature"])
    for on in (False, True):
        tap = Tap(None)
        E.spec_stats(True)
        r, got, fin, tps, s = serve(tap, PROMPT, T, on, CAP[0])
        S = E.spec_stats(True)
        txt = chat.decode(got)
        say(f"[sampled] T {T} top_p {D['top_p']} · copy {'on' if on else 'off'}: {len(got):,} tokens ({fin}) · decode "
            f"{tps:.1f} tok/s · {S['emitted'] / max(1, S['rounds']):.2f} tokens a round ({S['rounds']:.0f} rounds) · "
            f"{rounds_line(tap.log)} · similarity to the file with the change "
            f"{difflib.SequenceMatcher(None, txt, WANT).ratio():.3f} · `_keepalive(` ×{txt.count('_keepalive(')}"
            + (f" · matcher {s.copy.c['miss']} rounds without a copy {1e6 * s.copy.t['miss'] / max(1, s.copy.c['miss']):.1f}"
               f" µs each" if on else ""))


for fn in (width, ref, forced, pick, greedy, sampled):
    segment(fn)
say(f"[copy] done · logit rows {LR} (serving: max(slots × (block + 1) = {a.slots * (a.block + 1)}, copy_rows))")
E.close()
