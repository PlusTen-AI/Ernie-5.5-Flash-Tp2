#!/usr/bin/env python3
"""sched_dspark.py — the server's scheduler (server/scheduler.py) decoding with DSpark rounds, driven through the engine
binding: no HTTP, no command link — both ranks run the same scheduler on the same requests, so their engine calls pair
up as rank 1's replay of the link would make them. TP2 (tools/tp2.sh appends --rank / --head-ip):
  E55_GPU_LOCK=1 E55_TREE=dspserve tools/tp2.sh sched python3 engine/tests/sched_dspark.py
Serving shapes (4 slots × 64K, chunk 8192, block 7, gate 0.3); two prompts: thinking (en_prose, thinking on) and code
(py_code, thinking off).
  mem      main.py's memory check at these settings (MemFree before the engine) and the engine's MemFree drop
  exact    T = 0, per prompt: the scheduler's plain path (run()) → tokens; plain greedy decoding by step() fed them →
           every position's logits (argmax = the next token everywhere but the thinking budget's forced </think>); the
           scheduler's DSpark path (P121 / P122 on) → the same tokens, its prefill row and every kept verify row's
           logits bitwise those (thinking: budget --budget, then --answer tokens; code: --code tokens)
  measure  T 1.0 / top_p 0.95 (generation_config), single stream, --mlen tokens per prompt, P121 conf_gate off vs on:
           per round mean verify rows, accepted / emitted tokens, verify + draft ms (engine spec_stats)
  multi    the serving path with 4 slots at once (decode.verify --verify): five requests arriving by rounds (greedy
           thinking with a budget, greedy code, a greedy ~10K-token prompt (two chunks), greedy with repetition /
           frequency / presence penalties, sampled T 1.0 / top_p 0.95; the fifth waits for a slot), so rounds mix B = 1 … 4
           and prefill chunks cut in between rounds: every greedy request's tokens == its own plain decoding alone (the
           penalised one: with the same penalties), its prompt's last row and every kept verify row's logits bitwise
           plain greedy step()'s; the sampled one ends with sane text. --slots 8 / 16: greedy requests of their own
           (x0 …, distinct prompts) up to slots + 1, so rounds reach B = slots (verify 64 / 128 rows) — each one's
           tokens and kept rows as above; PASS needs a round with every slot
  cost     P143's table: sampled rounds over 4 sequences at context ≈ 9.4K, every draft verified up to a cap c (R = 4c,
           c = 1 2 3 4 6 8): verify step ms by total rows (+ the draft's context rows), and the rest of a round;
           --cost-b 4,8,16 (with --slots 16): the same at B = 8 / 16 (R ≤ logit rows: 64 / 128)
  rate     T 1.0 / top_p 0.95, thinking and code prompts, c = 1 (--mlen tokens) and c = 4 (--mlen4 a stream, seeds
           differ): decode.verify "gate" (P121, --gate) vs "cost" (P143, the table from `cost` + the gate runs) — rounds
           with B = c: emitted tokens per ms of rounds (derived tok/s; wall from a round's start to the next one's) and
           accepted / verified drafts per sequence and round
  kvwait   KvGuard + DSpark on a small pool (kv_pool_init --kv-gib: two of three short greedy requests fit): the third
           waits for room, then is placed and finishes with plain decoding's tokens; at the end the pool's chunks are
           all accounted for and all come back with the slots' resets (run it last: the pool stays small)"""
import argparse
import collections
import faulthandler
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, os.path.join(ROOT, "server"), os.path.join(ROOT, "tools")]
import ext  # noqa: E402
from chat import Chat, normalize_messages  # noqa: E402
import scheduler  # noqa: E402
from kvguard import KvGuard  # noqa: E402
from scheduler import Req, Scheduler  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, required=True)
ap.add_argument("--head-ip", required=True)
ap.add_argument("--port", type=int, default=29611)
ap.add_argument("--model", default="/models/ERNIE-5.5-Flash-Preview")
ap.add_argument("--table", default="/work/cache/table.tsv")
ap.add_argument("--slots", type=int, default=4)
ap.add_argument("--ctx", type=int, default=65536)
ap.add_argument("--chunk", type=int, default=8192)
ap.add_argument("--block", type=int, default=7)
ap.add_argument("--gate", type=float, default=0.3)
ap.add_argument("--reserve", type=float, default=15.0, help="server.memory_free_gib")
ap.add_argument("--budget", type=int, default=192)
ap.add_argument("--answer", type=int, default=256)
ap.add_argument("--code", type=int, default=320)
ap.add_argument("--mlen", type=int, default=512)
ap.add_argument("--mlen4", type=int, default=384, help="rate: tokens a stream at c = 4")
ap.add_argument("--verify", default="cost", help="multi / kvwait: decode.verify")
ap.add_argument("--kv-gib", type=float, default=0.2, help="kvwait: kv_pool_gib")
ap.add_argument("--do", default="exact,measure")
ap.add_argument("--cost-b", default="4", help="cost: sequences of the table's rounds (each ≤ --slots), e.g. 4,8,16")
ap.add_argument("--rate-c", default="1,4", help="rate: concurrent streams (each ≤ --slots), e.g. 1,4,8,16")
a = ap.parse_args()
R0 = a.rank == 0
SEED = 1234
PROMPTS = {  # test_dspark.py's en_prose / py_code
    "thinking": ([{"role": "user", "content": "Explain in detail how the printing press changed European society in the "
                                               "15th and 16th centuries."}], True),
    "code": ([{"role": "user", "content": "Write a Python module that implements a thread-safe LRU cache with a per-entry "
                                           "TTL, with type hints and docstrings, followed by pytest unit tests. Output "
                                           "only the code."}], False),
}
MORE = {  # multi's other requests
    "planets": ([{"role": "user", "content": "List the eight planets of the solar system, one line each with one fact "
                                              "about it."}], False),
    "story": ([{"role": "user", "content": "Write a short story about a lighthouse keeper who finds a message in a "
                                            "bottle."}], True),
}


def say(*x):
    if R0:
        print(*x, flush=True)


def memfree():
    with open("/proc/meminfo") as f:
        return next(int(x.split()[1]) for x in f if x.startswith("MemFree:")) / 1048576


# ---- main.py's memory check at these settings, before the engine -------------------------------------------------
f0 = memfree()
ENGINE_GIB = 87.2 + max(0, a.chunk - 4096) * 1.5 / 4096
need = a.slots * (a.ctx * (6 * 724 + 37 * 584 / 128) + 37 * (128 * 584 + 128 * 4096)) / 2**30
room = f0 - ENGINE_GIB - a.reserve
print(f"[mem] rank {a.rank}: MemFree {f0:.1f} GiB · main.py's check at {a.slots} × {a.ctx // 1024}K, chunk {a.chunk}: "
      f"caches {need:.2f} GiB ≤ room {room:.1f} (MemFree − {ENGINE_GIB:.1f} − {a.reserve:.0f}) → "
      f"{'passes' if need <= room else 'FAILS'}", flush=True)

E = ext.load("engine", ext.engine_sources(), log=f"/tmp/build{a.rank}.log", ldflags=ext.ENGINE_LDFLAGS)
LR = a.slots * (a.block + 1)
E.init(a.model, a.table, a.rank, a.head_ip, a.slots, a.ctx, max(a.chunk, LR), LR, a.port)
E.dspark_init()
chat = Chat(a.model)
D = chat.defaults
say(f"[sched] engine + draft up · logit rows {LR} · patches conf_gate {E.patch_on('conf_gate')} · cap_verify "
    f"{E.patch_on('cap_verify')}")


def encode(name):
    msgs, th = PROMPTS[name] if name in PROMPTS else MORE[name]
    return chat.encode_chat(chat.render(normalize_messages(msgs), None, thinking=th, level="high")), th


def encode_long():
    """a prompt of two chunks: the checkpoint's README (repeated) and a question, thinking off"""
    text = open(os.path.join(a.model, "README.md"), encoding="utf-8").read()
    while len(text) < 40000:
        text += "\n\n" + text
    msgs = [{"role": "user", "content": text[:40000] + "\n\nSummarize the text above in three sentences."}]
    return chat.encode_chat(chat.render(normalize_messages(msgs), None, thinking=False, level="high"))


def serve(eng, block, prompt, max_tokens, temp, top_k, top_p, thinking, budget):
    """One request through a fresh scheduler (its slots start over): → (Req, emitted tokens, finish)."""
    s = Scheduler(eng, chat.eos, a.slots, a.ctx, a.chunk, think_end=chat.think_end, loop_guard=True, block=block,
                  gate=a.gate)
    got, fin = [], []

    def push(t, f):
        got.extend(t)
        if f:
            fin.append(f)
    r = Req(prompt, max_tokens, temp, top_k, top_p, SEED, push, "test")
    r.thinking, r.budget = thinking, budget
    s.submit(r)
    s.drain(time.time() + 1e9)
    s.run()
    return r, got, (fin or ["?"])[0]


class Tap:
    """The engine as the DSpark scheduler sees it, checking (single stream) the prompt's last row and every kept verify
    row against plain decoding's logits by position."""

    def __init__(self, ref):
        self.ref, self.c = ref, dict(rows=0, bad=0, after=0, after_bad=0, pf=None, rounds=0, vrows=0)
        self.prev_rej = False

    def __getattr__(self, n):
        return getattr(E, n)

    def step(self, ids, pos, slot, ss, all_logits, rows=(), prefill=False):  # (prefill: the step's kind)
        out = E.step(ids, pos, slot, ss, all_logits, list(rows), prefill)
        self.c["pf"] = bool(torch.equal(out[0], self.ref[pos[-1]]))
        return out

    def spec_round(self, slots, draft_next, caps):
        P = E.spec_pos(slots[0])
        out = E.spec_round(slots, draft_next, caps)
        L, toks, c = E.spec_logits(), out[0], self.c
        for j in range(len(toks)):
            same = torch.equal(L[j], self.ref[P + j])
            c["rows"] += 1
            c["bad"] += not same
            if j == 0 and self.prev_rej:
                c["after"] += 1
                c["after_bad"] += not same
        self.prev_rej = len(toks) < L.shape[0]  # rows dropped: rejected, or past the cap
        c["rounds"] += 1
        c["vrows"] += L.shape[0]
        return out


def exact():
    allok = True
    for name in PROMPTS:
        prompt, th = encode(name)
        T = len(prompt)
        mt = a.budget + a.answer if th else a.code
        bud = a.budget if th else None
        r0, plain, f0_ = serve(E, 0, prompt, mt, 0.0, 0, 1.0, th, bud)
        # plain greedy decoding by step(), fed the plain path's tokens: logits at every position (row of position i
        # predicts token i + 1; the last emitted token runs too when the reply ended with an end token)
        E.reset(0)
        seq = prompt + plain
        ref = {T - 1: E.step(prompt, list(range(T)), [0], [0, T], False)[0]}
        for i in range(T, len(seq)):
            ref[i] = E.step([seq[i]], [i], [0], [0, 1], False)[0]
        mism = [i for i in range(T - 1, len(seq) - 1) if int(ref[i].argmax()) != seq[i + 1]]
        forced = [i for i in mism if seq[i + 1] == chat.think_end]
        tap = Tap(ref)
        E.spec_stats(True)
        r1, spec, f1_ = serve(tap, a.block, prompt, mt, 0.0, 0, 1.0, th, bud)
        S = E.spec_stats(True)
        c = tap.c
        ok = spec == plain and f1_ == f0_ and c["bad"] == 0 and c["pf"] and len(mism) == len(forced) <= 1
        allok &= ok
        say(f"[exact] {name}: prompt {T} · {len(plain)} tokens ({f0_}{', thinking closed (' + r0.forced + ')' if r0.forced else ''})"
            f" · plain step() argmax = the plain path's next token at {len(seq) - T - len(mism)}/{len(seq) - T} positions"
            f" (the rest: forced </think> {len(forced)}) · DSpark scheduler tokens "
            f"{'IDENTICAL' if spec == plain else 'DIFFER'} ({f1_}{', thinking closed (' + r1.forced + ')' if r1.forced else ''})"
            f" · prefill row {'bitwise equal' if c['pf'] else 'DIFFERS'} · kept rows' logits bitwise equal "
            f"{c['rows'] - c['bad']}/{c['rows']} · row 0 after a rejection {c['after'] - c['after_bad']}/{c['after']} · "
            f"{S['rounds']:.0f} rounds, {S['rows'] / max(1, S['rounds']):.2f} verify rows, "
            f"{S['emitted'] / max(1, S['rounds']):.2f} tokens a round")
        if spec != plain:
            i = next((j for j in range(min(len(spec), len(plain))) if spec[j] != plain[j]), min(len(spec), len(plain)))
            say(f"  first difference at output token {i}: plain {plain[i:i + 4]} · DSpark {spec[i:i + 4]}")
        del ref, tap
        torch.cuda.empty_cache()
        if name == "thinking":  # after prefill, plain steps, verify rounds and drafts: the engine's own footprint
            torch.cuda.synchronize()
            print(f"[mem] rank {a.rank}: MemFree drop since start {f0 - memfree():.1f} GiB (torch's cached blocks "
                  f"{torch.cuda.memory_reserved() / 2**30:.2f} of it) · main.py reckons {ENGINE_GIB:.1f} + caches "
                  f"{need:.2f}", flush=True)
    say(f"[exact] all: {'IDENTICAL tokens and bitwise logits' if allok else 'FAILED'}")


def measure():
    temp, top_p, top_k = float(D["temperature"]), float(D["top_p"]), (D["top_k"] if D["top_k"] > 0 else 0)
    for on in (False, True):
        E.patch("conf_gate", on)
        for name in PROMPTS:
            prompt, th = encode(name)
            E.spec_stats(True)
            r, got, fin = serve(E, a.block, prompt, a.mlen, temp, top_k, top_p, th, None)
            S = E.spec_stats(True)
            n = max(1, S["rounds"])
            say(f"[measure] T {temp} top_p {top_p} · P121 {'on (gate ' + str(a.gate) + ')' if on else 'off'} · {name}: "
                f"{len(got)} tokens ({fin}; thinking {r.n_think}) in {S['rounds']:.0f} rounds · per round: verify rows "
                f"{S['rows'] / n:.2f} · accepted {S['accepted'] / n:.2f} (of {S['verified'] / n:.2f} verified drafts) · "
                f"emitted {S['emitted'] / n:.2f} · verify {(S['ms_verify'] + S['ms_ctx']) / n:.1f} ms · draft "
                f"{S['ms_draft'] / n:.1f} ms · accept + commit {(S['ms_accept'] + S['ms_commit']) / n:.2f} ms")
    E.patch("conf_gate", True)


# ---- multi-slot serving (multi, kvwait), the cost table (cost), sampled rounds by verify mode (rate) ----------------
def mkreq(sp, push):
    r = Req(sp["prompt"], sp["mt"], sp.get("temp", 0.0), sp.get("top_k", 0), sp.get("top_p", 1.0), sp.get("seed", SEED),
            push, sp["name"])
    r.thinking, r.budget = sp["thinking"], sp.get("budget")
    r.pen = sp.get("pen", (1.0, 0.0, 0.0))
    return r


def drive(eng, specs, block, verify="gate", at=None, kv=None):
    """specs through one fresh scheduler (its slots start over); spec i is submitted at the start (at[i] = 0) or after
    round at[i] (eng: an MTap) → {name: (Req, emitted tokens, finish, round it finished)}"""
    s = Scheduler(eng, chat.eos, a.slots, a.ctx, a.chunk, think_end=chat.think_end, loop_guard=True, block=block,
                  gate=a.gate, kv=kv, verify=verify)
    res, later, ends = {}, [], {}
    f0 = s._finish

    def finish(r, why):  # the round a request ends in (its finish push is held until the next engine call)
        ends.setdefault(r.label, getattr(eng, "rounds", 0))
        f0(r, why)
    s._finish = finish
    for i, sp in enumerate(specs):
        got, fin = [], []

        def push(t, f, got=got, fin=fin):
            got.extend(t)
            if f:
                fin.append((f, getattr(eng, "rounds", 0)))
        r = mkreq(sp, push)
        res[sp["name"]] = (r, got, fin)
        w = (at or [0] * len(specs))[i]
        if w:
            later.append((w, r))
        else:
            s.submit(r)
    if isinstance(eng, MTap):
        eng.sched, eng.pending = s, sorted(later, key=lambda x: x[0])
    elif later:
        raise ValueError("arrivals by round need an MTap")
    s.drain(time.time() + 1e9)
    s.run()
    out = {n: (r, got, (fin or [("?", -1)])[0][0], ends.get(n, (fin or [("?", -1)])[0][1]))
           for n, (r, got, fin) in res.items()}
    if isinstance(eng, MTap) and eng.pending:
        say(f"  {len(eng.pending)} arrivals never submitted (every request ended first)")
    return out


def reference(sp, logits):
    """sp alone through the scheduler's plain path → (tokens, finish, Req); logits: plain greedy step() fed them →
    {position: logits row} (row of position i predicts token i + 1) and the positions where its argmax is not the
    next token (only the thinking budget's forced </think> may be)"""
    r, toks, fin, _ = drive(E, [sp], 0)[sp["name"]]
    if not logits:
        return toks, fin, r, None, []
    E.reset(0)
    P = sp["prompt"]
    T, seq, ref = len(P), sp["prompt"] + toks, {}
    for c0 in range(0, T, a.chunk):
        c1 = min(T, c0 + a.chunk)
        if c1 < T:
            E.fill(P[c0:c1], list(range(c0, c1)), [0], [0, c1 - c0])
        else:
            ref[T - 1] = E.step(P[c0:c1], list(range(c0, c1)), [0], [0, c1 - c0], False)[0]
    for i in range(T, len(seq)):
        ref[i] = E.step([seq[i]], [i], [0], [0, 1], False)[0]
    mism = [i for i in range(T - 1, len(seq) - 1) if int(ref[i].argmax()) != seq[i + 1]]
    return toks, fin, r, ref, mism


class MTap:
    """The engine as the scheduler sees it (any number of slots): the prompts' last rows and every kept verify row
    against plain decoding's logits (refs: request name → {position: row}; a slot's request by its label), rounds by B,
    prefill chunks that ran while others were decoding, the staggered arrivals (pending: (round, Req), submitted after
    that round). meter: per round (B, R, emitted, accepted, verified, ms verify, ms context rows, ms accept + commit +
    draft, its start)."""

    def __init__(self, refs=None, meter=False):
        self.refs, self.sched, self.pending = refs or {}, None, []
        self.c = collections.defaultdict(lambda: dict(rows=0, bad=0, pf=None))
        self.rounds, self.byB, self.cutin, self.placed = 0, collections.Counter(), 0, {}
        self.meter, self.log = meter, []

    def __getattr__(self, n):
        return getattr(E, n)

    def _name(self, slot):
        r = self.sched.slots[slot].req if self.sched else None
        return r.label if r is not None else None

    def _cut(self):
        if self.sched and any(r.state == "decode" for r in self.sched.running):
            self.cutin += 1

    def fill(self, ids, pos, slot, ss):
        self._cut()
        return E.fill(ids, pos, slot, ss)

    def step(self, ids, pos, slot, ss, all_logits, rows=(), prefill=False):  # (prefill: the step's kind)
        self._cut()
        out = E.step(ids, pos, slot, ss, all_logits, list(rows), prefill)
        nm = self._name(slot[0]) if len(slot) == 1 else None
        if nm in self.refs:
            ref = self.refs[nm].get(pos[-1])
            self.c[nm]["pf"] = ref is not None and bool(torch.equal(out[0], ref))
        return out

    def spec_round(self, slots, draft_next, caps):
        t0 = time.perf_counter()
        P = [E.spec_pos(s) for s in slots]
        names = [self._name(s) for s in slots]
        out = E.spec_round(slots, draft_next, caps)
        ks = list(E.spec_verified())
        if any(n in self.refs for n in names):
            L, o = E.spec_logits(), 0
            for b, nm in enumerate(names):
                if nm in self.refs:
                    c = self.c[nm]
                    for j in range(len(out[b])):
                        ref = self.refs[nm].get(P[b] + j)
                        c["rows"] += 1
                        c["bad"] += ref is None or not torch.equal(L[o + j], ref)
                o += ks[b] + 1
            if o != L.shape[0]:
                say(f"  round {self.rounds}: verify rows {L.shape[0]} but lengths {ks}")
        if self.meter:
            S = E.spec_stats(True)
            self.log.append((len(slots), sum(ks) + len(slots), S["emitted"], S["accepted"], S["verified"],
                             S["ms_verify"], S["ms_ctx"], S["ms_accept"] + S["ms_commit"] + S["ms_draft"], t0))
        self.rounds += 1
        self.byB[len(slots)] += 1
        if self.sched:
            for r in self.sched.running:
                self.placed.setdefault(r.label, self.rounds)
        while self.pending and self.pending[0][0] <= self.rounds:
            self.sched.submit(self.pending.pop(0)[1])
        return out


def multi_specs():
    th, _ = encode("thinking")
    code, _ = encode("code")
    pl, _ = encode("planets")
    st, _ = encode("story")
    D = chat.defaults
    return [dict(name="think", prompt=th, mt=256, thinking=True, budget=96),
            dict(name="code", prompt=code, mt=200, thinking=False),
            dict(name="long", prompt=encode_long(), mt=96, thinking=False),
            dict(name="pen", prompt=pl, mt=160, thinking=False, pen=(1.15, 0.3, 0.4)),
            dict(name="sampled", prompt=st, mt=256, thinking=True, temp=float(D["temperature"]),
                 top_k=D["top_k"] if D["top_k"] > 0 else 0, top_p=float(D["top_p"]), seed=7)]


# multi at --slots > 4: greedy requests of their own (distinct prompts, thinking off) up to slots + 1 (one waits), so
# rounds reach B = slots (verify 8 · B rows at block 7: 64 at 8 slots, 128 at 16)
TOPICS = ["the water cycle", "binary search", "photosynthesis", "a TCP handshake", "hash tables", "black holes",
          "how vaccines work", "the Pythagorean theorem", "what a compiler does", "plate tectonics", "neural networks",
          "public-key cryptography"]


def extra_specs(n):
    out = []
    for k in range(n):
        msgs = [{"role": "user", "content": f"In about 100 words, explain {TOPICS[k % len(TOPICS)]}."}]
        pr = chat.encode_chat(chat.render(normalize_messages(msgs), None, thinking=False, level="high"))
        out.append(dict(name=f"x{k}", prompt=pr, mt=128, thinking=False))
    return out


REFS, PLAIN = {}, {}


def multi():
    specs = multi_specs()
    nx = max(0, a.slots + 1 - len(specs))
    specs += extra_specs(nx)
    at = [0, 3, 6, 10, 16] + [1 + k + k // 3 for k in range(nx)]  # the extras between the first five's arrivals
    for sp in specs:
        if sp.get("temp", 0.0) > 0:
            continue
        pen = sp.get("pen", (1.0, 0.0, 0.0)) != (1.0, 0.0, 0.0)
        toks, fin, r, ref, mism = reference(sp, not pen)
        PLAIN[sp["name"]] = (toks, fin)
        if ref is not None:
            REFS[sp["name"]] = ref
        forced = [i for i in mism if (sp["prompt"] + toks)[i + 1] == chat.think_end]
        say(f"[multi] plain {sp['name']}: prompt {len(sp['prompt'])} · {len(toks)} tokens ({fin}"
            f"{', thinking closed (' + r.forced + ')' if r.forced else ''})"
            + ("" if ref is None else f" · step() argmax = the next token at {len(toks) + 1 - len(mism)}/{len(toks) + 1}"
               f" positions (the rest forced </think>: {len(forced)}/{len(mism)})"))
    tap = MTap(REFS)
    E.spec_stats(True)
    res = drive(tap, specs, a.block, a.verify, at=at)
    S = E.spec_stats(True)
    allok = True
    for sp in specs:
        nm = sp["name"]
        r, got, fin, rd = res[nm]
        if nm in PLAIN:
            same = got == PLAIN[nm][0] and fin == PLAIN[nm][1]
            c = tap.c[nm]
            rows_ok = nm not in REFS or (c["bad"] == 0 and c["pf"])
            allok &= same and rows_ok
            line = (f"[multi] {nm}: {len(got)} tokens ({fin}) · DSpark scheduler tokens "
                    f"{'IDENTICAL to' if same else 'DIFFER from'} its plain decoding alone")
            if nm in REFS:
                line += (f" · prompt's last row {'bitwise equal' if c['pf'] else 'DIFFERS'} · kept rows' logits bitwise "
                         f"equal {c['rows'] - c['bad']}/{c['rows']}")
            else:
                line += " (penalties rep / freq / pres " + " / ".join(str(x) for x in sp["pen"]) + ")"
            say(line + f" · running from round {tap.placed.get(nm, 0)} · ended at round {rd}")
            if not same:
                p = PLAIN[nm][0]
                i = next((j for j in range(min(len(got), len(p))) if got[j] != p[j]), min(len(got), len(p)))
                say(f"  first difference at output token {i}: plain {p[i:i + 4]} · DSpark {got[i:i + 4]}")
        else:
            txt = chat.decode(got)
            distinct = len(set(got)) / max(1, len(got))
            sane = fin in ("stop", "length") and len(got) > 16 and distinct > 0.2 and txt.strip() != ""
            allok &= sane
            say(f"[multi] {nm} (T {sp['temp']} top_p {sp['top_p']}): {len(got)} tokens ({fin}; thinking {r.n_think}) · "
                f"distinct {distinct:.2f} · {'sane' if sane else 'NOT SANE'} · running from round "
                f"{tap.placed.get(nm, 0)} · ended at round {rd} · text: {txt[:200]!r}")
    say(f"[multi] {S['rounds']:.0f} rounds by B {dict(sorted(tap.byB.items()))} · {S['rows'] / max(1, S['rounds']):.2f} "
        f"verify rows · {S['emitted'] / max(1, S['seqs']):.2f} tokens a sequence and round · prefill steps while others "
        f"decoded {tap.cutin} · all: {'PASS' if allok and set(tap.byB) >= {1, 2, 3, 4} and max(tap.byB) == a.slots else 'FAIL'}")
    for i in range(a.slots):
        E.reset(i)


def cost():
    """P143's table: B (--cost-b) slots at ≈ 9.4K (the README's tokens, other offsets), sampled rounds with every draft
    verified up to the cap c (gate 0: R = B·c; c ≤ 8 and B·c ≤ logit rows) → {B: {R: target step ms}}"""
    out = {}
    for B in (int(x) for x in a.cost_b.split(",") if x):
        if B > a.slots:
            say(f"[cost] B {B} > {a.slots} slots: skipped")
            continue
        out[B] = cost_b(B)
    return out


def cost_b(B):
    D = chat.defaults
    temp, top_p, top_k = float(D["temperature"]), float(D["top_p"]), (D["top_k"] if D["top_k"] > 0 else 0)
    text = open(os.path.join(a.model, "README.md"), encoding="utf-8").read()
    toks = chat.encode(text)
    n, slots = 9400, list(range(B))
    toks = toks * ((n + 1000 * B) // len(toks) + 1)
    for s in slots:
        E.reset(s)
        ids = toks[s * 1000:s * 1000 + n]
        for c0 in range(0, n, a.chunk):
            c1 = min(n, c0 + a.chunk)
            if c1 < n:
                E.fill(ids[c0:c1], list(range(c0, c1)), [s], [0, c1 - c0])
            else:
                E.step(ids[c0:c1], list(range(c0, c1)), [s], [0, c1 - c0], False)
        E.spec_begin(s, a.block, temp, SEED + s, 0, top_k, top_p, 0.0, 1.0, 0.0, 0.0, 0.0, True)
    tab, rest = {}, []
    for c in (1, 2, 3, 4, 6, 8):
        if B * c > LR:
            continue
        for _ in range(2):
            E.spec_round(slots, True, [c] * B)
        E.spec_stats(True)
        v, x, other = [], [], []
        for _ in range(8):
            E.spec_round(slots, True, [c] * B)
            S = E.spec_stats(True)
            if S["rows"] != B * c:
                say(f"  cap {c}: {S['rows']:.0f} rows (expected {B * c})")
            v.append(S["ms_verify"]), x.append(S["ms_ctx"])
            other.append(S["ms_accept"] + S["ms_commit"] + S["ms_draft"])
        tab[B * c] = (statistics.median(v), statistics.median(x), statistics.median(other))
        rest += other
    say(f"[cost] B {B} at ≈ " + f"{E.spec_pos(0):,}" + ": verify step ms by total rows (target step + the draft's context "
        "rows; median of 8 rounds): " + " · ".join(f"R {R} {t + c:.1f} ({t:.1f} + {c:.2f})" for R, (t, c, _) in tab.items())
        + f" · the rest of a round (accept + commit + next drafts, depth {a.block}) {statistics.median(rest):.2f} ms")
    for s in slots:
        E.reset(s)
    return {R: t for R, (t, _, _) in tab.items()}


COST = {}  # cost()'s tables for rate: {B: {R: target step ms}}


def rate():
    RC = [c for c in (int(x) for x in a.rate_c.split(",") if x) if c <= a.slots]
    D = chat.defaults
    temp, top_p, top_k = float(D["temperature"]), float(D["top_p"]), (D["top_k"] if D["top_k"] > 0 else 0)
    res = {}
    for mode in ("gate", "cost"):
        if mode == "cost":
            # the verify target step by rows: B = 1 step_prof's (32.0 / 43.3 / 57.1 ms at 1 / 4 / 8 rows, 9.4K), B = 4
            # `cost`'s; the rest of a round (wall start to start less the target step: context rows, accept, commit,
            # next drafts, the scheduler) as the gate runs had it at B = 1 / 4
            rest = [(c, statistics.mean(x["restw"] for k, x in res.items() if k[1] == c)) for c in RC]
            vt = {1: ((1, 32.0), (4, 43.3), (8, 57.1))}
            for B, t in sorted(COST.items()):
                vt[B] = tuple(sorted(t.items()))
            scheduler.ROUND_VERIFY_MS, scheduler.ROUND_REST_MS = vt, tuple(rest)
            say(f"[rate] cost table: verify {vt} · rest " + ", ".join(f"B {c}: {r:.2f}" for c, r in rest))
        for c in RC:
            for name in PROMPTS:
                prompt, th = encode(name)
                specs = [dict(name=f"{name}{i}", prompt=prompt, mt=a.mlen if c == 1 else a.mlen4, thinking=th, temp=temp,
                              top_k=top_k, top_p=top_p, seed=SEED + i) for i in range(c)]
                tap = MTap(meter=True)
                E.spec_stats(True)
                drive(tap, specs, a.block, mode)
                L = tap.log
                idx = [i for i in range(len(L) - 1) if L[i][0] == c]  # rounds with every stream (and a next one)
                wall = [1e3 * (L[i + 1][8] - L[i][8]) for i in idx]  # ms from its start to the next one's
                rr = [L[i] for i in idx]
                n, em = len(rr), sum(x[2] for x in rr)
                x = dict(rounds=n, emitted=em, tps=1e3 * em / max(1e-9, sum(wall)),
                         tps_eng=1e3 * em / max(1e-9, sum(y[5] + y[6] + y[7] for y in rr)),
                         acc=sum(y[3] for y in rr) / max(1, n * c), ver=sum(y[4] for y in rr) / max(1, n * c),
                         rows=sum(y[1] for y in rr) / max(1, n), ms=sum(wall) / max(1, n),
                         restw=sum(w - y[5] for w, y in zip(wall, rr)) / max(1, n),
                         other=sum(y[6] + y[7] for y in rr) / max(1, n))
                res[(mode, c, name)] = x
                say(f"[rate] {mode} · c {c} · {name}: {n} rounds of B {c} · {em:.0f} tokens · derived "
                    f"{x['tps']:.1f} tok/s ({x['ms']:.1f} ms a round, start to start; engine events only "
                    f"{x['tps_eng']:.1f}) · per sequence and round: accepted {x['acc']:.2f} of {x['ver']:.2f} verified · "
                    f"verify rows {x['rows']:.2f} a round · rest of a round {x['restw']:.2f} ms (context rows + accept "
                    f"+ commit + drafts {x['other']:.2f})")
                if mode == "gate" and c == 1:  # the target step by rows at B = 1 (the table's shape)
                    by = collections.defaultdict(list)
                    for y in L:
                        if y[0] == 1:
                            by[y[1]].append(y[5])
                    say("  B 1 target step ms by rows: " + " · ".join(f"R {R} {statistics.median(v):.1f} (×{len(v)})"
                                                                      for R, v in sorted(by.items())))
    for c in RC:
        for name in PROMPTS:
            g, k = res[("gate", c, name)], res[("cost", c, name)]
            say(f"[rate] c {c} · {name}: gate {a.gate} {g['tps']:.1f} → cost {k['tps']:.1f} tok/s derived "
                f"({100 * (k['tps'] / g['tps'] - 1):+.1f} %) · accepted a sequence and round {g['acc']:.2f} → "
                f"{k['acc']:.2f} · verified {g['ver']:.2f} → {k['ver']:.2f} · verify rows {g['rows']:.2f} → "
                f"{k['rows']:.2f}")


def kvwait():
    specs = [sp for sp in multi_specs() if sp["name"] in ("think", "code", "pen")]
    for sp in specs:  # plain decoding of each alone (multi's, or here when it did not run)
        if sp["name"] not in PLAIN:
            pen = sp.get("pen", (1.0, 0.0, 0.0)) != (1.0, 0.0, 0.0)
            toks, fin, _, ref, _ = reference(sp, not pen)
            PLAIN[sp["name"]] = (toks, fin)
            if ref is not None:
                REFS[sp["name"]] = ref
            say(f"[kvwait] plain {sp['name']}: {len(toks)} tokens ({fin})")
    faulthandler.dump_traceback_later(300, exit=True)  # a hang: both ranks' stacks, then out
    total = E.kv_pool_init(15.0, a.kv_gib, False)
    kv = KvGuard(E, a.ctx, 32768, 4.0)
    kv.poll_s = float("inf")  # (both ranks run their own scheduler: no time-driven collective)
    need = [E.kv_need(min(kv.cap, len(sp["prompt"]) + sp["mt"]) + 64) for sp in specs]
    say(f"[kvwait] pool {total} chunks ({a.kv_gib} GiB cap) · the requests reserve {need} chunks")
    tap = MTap(REFS)
    res = drive(tap, specs, a.block, a.verify, kv=kv)
    ok = True
    for sp in specs:
        nm = sp["name"]
        r, got, fin, rd = res[nm]
        same = nm in PLAIN and got == PLAIN[nm][0] and fin == PLAIN[nm][1]
        c = tap.c[nm]
        rows_ok = nm not in REFS or (c["bad"] == 0 and c["pf"])
        ok &= same and rows_ok
        say(f"[kvwait] {nm}: {len(got)} tokens ({fin}) · {'IDENTICAL to' if same else 'DIFFER from'} plain decoding"
            + (f" · kept rows bitwise {c['rows'] - c['bad']}/{c['rows']}" if nm in REFS else "")
            + f" · running from round {tap.placed.get(nm, 0)} · ended at round {rd}")
    first_end = min(res[n][3] for n in ("think", "code"))
    waited = tap.placed.get("pen", 0) >= first_end > 0
    held, free = sum(E.kv_held(i) for i in range(a.slots)), E.kv_free()
    acct = free + held == E.kv_total()
    for i in range(a.slots):
        E.reset(i)
    back = E.kv_free() == E.kv_total()
    say(f"[kvwait] the third request waited for room: {'yes' if waited else 'NO'} (running from round "
        f"{tap.placed.get('pen', 0)}, the first end at round {first_end}) · full ends {kv.full_ends} · engine throws "
        f"{kv.throws} · at the end: held {held} (idle slots) + free {free} = total {E.kv_total()}: "
        f"{'yes' if acct else 'NO'} · after the slots' resets all {E.kv_free()} free: {'yes' if back else 'NO'} · all: "
        f"{'PASS' if ok and waited and acct and back and kv.throws == 0 else 'FAIL'}")
    faulthandler.cancel_dump_traceback_later()


for what in a.do.split(","):
    t = time.time()
    out = globals()[what]()
    if what == "cost":
        COST.update(out)
    say(f"[sched] {what} · {time.time() - t:.0f} s")
E.close()
