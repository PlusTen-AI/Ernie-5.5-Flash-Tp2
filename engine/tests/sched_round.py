#!/usr/bin/env python3
"""sched_round.py — the server scheduler's own cost per DSpark round at B running requests (server/scheduler.py +
kvguard.py + the command link), against a near-free fake engine: no GPU, no model, seconds. Any Python ≥ 3.9:
  python3 engine/tests/sched_round.py [--sched DIR] [--work-us 150] [--gpu-ms 40]
(--sched: import scheduler / kvguard / copy_draft from DIR instead of server/ — an old copy for before → after.)

The requests are what costs the most per round: thinking on with a full loop-guard tail (256 thinking tokens, not
periodic), the KV pool on, decode.verify "cost" (the round table set on the fake), block 7, every round ~2.5 tokens a
sequence (the fake's verify lengths 0 … 7, accepts cycling 0 … 4). Engine calls go through link.Mirror over a socket
(drained by a `cat` process), as on rank 0.
  host    B = 4 / 8 / 16: ms per round of scheduler work alone (wall between two rounds' engine calls, the engine
          ~free); every request's pushed tokens = its emitted ones at the end
  http    B = 4 / 8 / 16 with the HTTP side as the server has it — an asyncio thread with one consumer per request
          (app.tokens' wait_for(q.get()), --work-us of token work: parser, stop strings, json, then SSE bytes to its own
          socket) — and each engine round taking --gpu-ms with the GIL released (the real binding is nogil): the gap
          between one round's return and the next one's start (what the GPU idles), mean / median / p90 / max
  metrics /metrics' speculative counters at B = 4 = the fake engine's own tallies: num_drafts the sequence-rounds that
          verified ≥ 1 draft (some verify 0), num_draft_tokens the verified drafts, accepted the accepted ones
  pieces  server.decode_chunk 2048 (chunk 8192): a 20,000-token prompt arriving while two requests decode is read in
          pieces (each ≥ 512 rows) with rounds between them, every row once and in order, and the slot's prefix
          checkpoints at the same positions as with decode_chunk 0 (the chunks' ends and the prompt's)
  guard   the incremental loop guard (when the scheduler has it) = scheduler.looping / loop_room on random, periodic and
          nearly periodic streams (every token)"""
import argparse
import asyncio
import inspect
import json
import os
import random
import socket
import statistics
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
ap = argparse.ArgumentParser()
ap.add_argument("--sched", default=os.path.join(ROOT, "server"), help="folder with scheduler.py / kvguard.py")
ap.add_argument("--work-us", type=float, default=150.0, help="HTTP token work per request and round (µs)")
ap.add_argument("--gpu-ms", type=float, default=40.0, help="http: ms an engine round takes (GIL released)")
ap.add_argument("--rounds", type=int, default=400)
ap.add_argument("--only", default="host,http,metrics,pieces,guard")
a = ap.parse_args()
sys.path[:0] = [a.sched, os.path.join(ROOT, "server"), os.path.join(ROOT, "tools")]
import qlog  # noqa: E402

qlog.ok = qlog.warn = qlog.info = lambda *x, **k: None
import link  # noqa: E402
import scheduler as S  # noqa: E402
from kvguard import KvGuard  # noqa: E402

EOS, THINK_END = 2, 3
BLOCK = 7
CHUNK = 2 << 20
HAS_DCHUNK = "decode_chunk" in inspect.signature(S.Scheduler.__init__).parameters


def tok(slot, pos):
    return 1000 + (pos * 2654435761 + slot * 40503) % 50000


class Quick:
    """the binding's calls the scheduler makes, each ~free (spec_round sleeps gpu_s with the GIL released); verify
    lengths chosen here (as P143 / P121 would: some 0), the accepted drafts by position"""

    def __init__(self, slots, gpu_s=0.0, ckpt=False):
        self.S, self.gpu, self.ck = slots, gpu_s, ckpt
        self.pos, self.last, self.dr = [0] * slots, [0] * slots, [[] for _ in range(slots)]
        self.depth = [BLOCK] * slots
        self.held, self.total = [0] * slots, 1 << 20
        self.rounds, self.on_round = 0, None
        self.t_ret, self.gaps, self.t_eng = None, [], 0.0
        self.st = {"seqs": 0.0, "verified": 0.0, "accepted": 0.0, "emitted": 0.0, "rounds": 0.0}
        self.ks, self.nd = [], 0  # the last round's verify lengths; sequence-rounds that verified ≥ 1 draft
        self.rows = [[] for _ in range(slots)]  # (first position, rows) of each fill / step
        self.saves = [[] for _ in range(slots)]  # positions of the checkpoints taken

    def _draft(self, s):
        p = self.pos[s]
        self.dr[s] = [tok(s, p + 1 + i) for i in range(self.depth[s])]

    def _rows(self, ids, pos, slot):
        s = slot[0]
        if pos[0] != self.pos[s] or list(pos) != list(range(pos[0], pos[0] + len(ids))):
            raise RuntimeError(f"slot {s}: rows from {pos[0]}, history {self.pos[s]}")
        self.rows[s].append((pos[0], len(ids)))
        self.pos[s] = pos[-1] + 1
        self.held[s] = max(self.held[s], self.kv_need(self.pos[s]))

    # prefill / setup
    def reset(self, s):
        self.pos[s], self.held[s] = 0, 0

    def fill(self, ids, pos, slot, ss):
        self._rows(ids, pos, slot)

    def step(self, ids, pos, slot, ss, all_logits, rows=(), prefill=False):  # (prefill: the step's kind)
        self._rows(ids, pos, slot)

    def spec_begin(self, s, depth, temp, seed, lrow, top_k, top_p, min_p, rep, freq, pres, gate, draft):
        self.depth[s] = depth
        self.last[s] = tok(s, self.pos[s])
        self._draft(s)
        return self.last[s]

    def spec_drafts(self, s):
        return list(self.dr[s])

    def spec_last(self, s, t):
        self.last[s], self.dr[s] = t, []

    def spec_depth(self, s, d):
        self.depth[s] = d

    def spec_draft(self, slots):
        for s in slots:
            self._draft(s)

    def spec_round_cost(self, tab, rmax):
        pass

    def spec_cost(self, tab):
        pass

    def spec_round(self, slots, draft_next, caps):
        t0 = time.perf_counter()
        if self.t_ret is not None:
            self.gaps.append(t0 - self.t_ret)
        if self.gpu:
            time.sleep(self.gpu)
        outs, self.ks = [], []
        for s, c in zip(slots, caps):
            p = self.pos[s]
            k = min(len(self.dr[s]), c - 1, (p // 2) % (BLOCK + 1))  # the verify length (P122: ≤ cap − 1)
            n = min(k, (p // 3) % 5)
            out = self.dr[s][:n] + [tok(s, p + n + 1)]
            self.pos[s] += len(out)
            self.last[s] = out[-1]
            self.held[s] = max(self.held[s], self.kv_need(self.pos[s] + BLOCK + 1))
            self.st["seqs"] += 1
            self.st["verified"] += k
            self.st["accepted"] += n
            self.st["emitted"] += len(out)
            self.nd += k > 0
            self.ks.append(k)
            if draft_next:
                self._draft(s)
            outs.append(out)
        self.rounds += 1
        self.st["rounds"] += 1
        if self.on_round is not None:
            self.on_round(self.rounds)
        self.t_ret = time.perf_counter()
        self.t_eng += self.t_ret - t0
        return outs

    def spec_verified(self):
        return list(self.ks)

    def spec_stats(self, reset):
        return dict(self.st)

    # pool / memory / checkpoints
    def kv_on(self):
        return True

    def kv_total(self):
        return self.total

    def kv_need(self, t):
        return 12 * (-(-724 * (t + 256) // CHUNK) + -(-5 * (t + 256) // CHUNK))

    def kv_held(self, s):
        return self.held[s]

    def kv_free(self):
        return self.total - sum(self.held)

    def mem_free(self):
        return [60.0, 60.0]

    def ckpt_on(self):
        return self.ck

    def ckpt_init(self, n):
        pass

    def ckpt_need(self, q):
        return 1

    def ckpt_save(self, s, k):
        self.saves[s].append(self.pos[s])
        return True

    def ckpt_drop(self, s, k):
        pass


class Link:
    """a socket to a `cat` (rank 1's reader) for link.Mirror"""

    def __init__(self):
        self.a, b = socket.socketpair()
        self.p = subprocess.Popen(["cat"], stdin=b.fileno(), stdout=subprocess.DEVNULL)
        b.close()

    def close(self):
        self.a.close()
        self.p.wait()


def make(eng, B, ln, **kw):
    kv = KvGuard(eng, 1 << 20, 32768, 4.0)
    if not HAS_DCHUNK:
        kw.pop("decode_chunk", None)
    return S.Scheduler(link.Mirror(eng, ln.a), [EOS], B, 1 << 20, 8192, think_end=THINK_END, loop_guard=True,
                       block=BLOCK, kv=kv, verify="cost", copy=True, copy_rows=64, copy_match=8, **kw)


def thinker(i, push, prompt=None):
    r = S.Req(prompt or [1000 + i] * 900, 10 ** 9, 0.0, 0, 1.0, i + 1, push, f"t{i}")
    r.thinking, r.budget = True, None
    rnd = random.Random(i)
    for _ in range(S.LOOP_SPAN):
        t = rnd.randrange(1000, 51000)
        if hasattr(r, "think_tok"):
            r.think_tok(t)
        else:
            r.tail.append(t)
    r.n_think = S.LOOP_SPAN
    return r


def run(B, rounds, gpu_s=0.0, http=None):
    """B requests through one scheduler (Mirror over a socket, KV pool on) for `rounds` rounds → (ms per round outside
    the engine over the last rounds − 50, the engine's gaps, the scheduler, the fake, pushed tokens = emitted)"""
    eng, ln = Quick(B, gpu_s), Link()
    sch = make(eng, B, ln, checkpoints=0)
    marks = {}

    def on_round(n):
        if n == 50:
            marks["t0"], marks["e0"] = time.perf_counter(), eng.t_eng
        if n == rounds:
            sch.drain(0.0)
    eng.on_round = on_round
    got = [0] * B

    def counter(i):
        def push(t, f):
            got[i] += len(t)
        return push
    rs = []
    for i in range(B):
        if http:
            p0, p1 = http.push_of(i), counter(i)
            push = (lambda p0, p1: lambda t, f: (p0(t, f), p1(t, f)))(p0, p1)
        else:
            push = counter(i)
        rs.append(thinker(i, push))
        sch.submit(rs[-1])
    sch.run()
    t1 = eng.t_ret
    ln.close()
    n = rounds - 50
    host = ((t1 - marks["t0"]) - (eng.t_eng - marks.get("e0", 0.0))) / n * 1e3
    same = all(g == r.n_out for g, r in zip(got, rs))
    return host, [g * 1e3 for g in eng.gaps[50:]], sch, eng, same


class Http:
    """the HTTP front's side: an asyncio loop thread, one consumer task per request (as app.py's tokens() + run() +
    sse_events(): wait_for(q.get()), token work, an SSE chunk written to the client's socket)"""

    def __init__(self, B, work_us):
        self.B, self.w = B, work_us / 1e6
        self.loop = asyncio.new_event_loop()
        self.qs = [None] * B
        self.socks = [socket.socketpair() for _ in range(B)]
        self.cats = [subprocess.Popen(["cat"], stdin=b.fileno(), stdout=subprocess.DEVNULL) for _, b in self.socks]
        for _, b in self.socks:
            b.close()
        self.ready = threading.Event()
        self.th = threading.Thread(target=self._main, daemon=True)
        self.th.start()
        self.ready.wait()

    def _main(self):
        asyncio.set_event_loop(self.loop)

        async def consumer(i):
            _, w = await asyncio.open_connection(sock=self.socks[i][0])
            q = self.qs[i]
            while True:
                try:
                    t, f = await asyncio.wait_for(q.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                end = time.perf_counter() + self.w * max(1, len(t)) / 2.5  # token work ~ tokens
                x = 0
                while time.perf_counter() < end:
                    x += 1
                d = {"id": "chatcmpl-x", "object": "chat.completion.chunk", "created": 1, "model": "m",
                     "choices": [{"index": 0, "delta": {"reasoning": "ab" * len(t)}, "finish_reason": f}]}
                w.write(f"data: {json.dumps(d, ensure_ascii=False)}\n\n".encode())
                if f:
                    return

        async def start():
            self.tasks = []
            for i in range(self.B):
                self.qs[i] = asyncio.Queue()
                self.tasks.append(self.loop.create_task(consumer(i)))
            self.ready.set()
        self.loop.run_until_complete(start())
        self.loop.run_forever()

    def push_of(self, i):
        q, loop = self.qs[i], self.loop
        return lambda t, f: loop.call_soon_threadsafe(q.put_nowait, (t, f))

    def close(self):
        """after the scheduler returned: the consumers end on their abort, then the loop stops"""
        async def done():
            await asyncio.wait(self.tasks, timeout=5)
        asyncio.run_coroutine_threadsafe(done(), self.loop).result()
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.th.join(5)
        for s, _ in self.socks:
            s.close()
        for c in self.cats:
            c.wait()


def metrics():
    """/metrics' speculative counters = the fake's tallies (B = 4)"""
    _, _, sch, eng, _ = run(4, 300)
    st, ver = sch.stats, int(eng.st["verified"])
    want = (eng.nd, ver, int(eng.st["accepted"]))
    have = (st["drafts"], st["draft_tokens"], st["accepted"])
    ok = have == want
    print(f"[metrics] {'ok' if ok else 'FAIL'} · num_drafts {have[0]:,} (sequence-rounds with ≥ 1 draft {want[0]:,}, "
          f"all {int(eng.st['seqs']):,}) · num_draft_tokens {have[1]:,} ({want[1]:,}) · accepted {have[2]:,} "
          f"({want[2]:,})")
    return ok


def pieces():
    """a long prompt read in decode_chunk pieces while two requests decode: rows, rounds between, checkpoints"""
    if not HAS_DCHUNK:
        print("[pieces] no server.decode_chunk in this scheduler (skipped)")
        return True
    L = 20000
    prompt = [5000 + (i * 7919) % 40000 for i in range(L)]
    res = {}
    for dch in (0, 2048):
        eng, ln = Quick(3, ckpt=True), Link()
        sch = make(eng, 3, ln, checkpoints=8, checkpoint_every=8192, decode_chunk=dch)
        long = S.Req(prompt, 64, 0.0, 0, 1.0, 9, lambda t, f: None, "long")
        between, state = [], {"fills": 0}

        def on_round(n, sch=sch, eng=eng, long=long, between=between, state=state):
            if n == 5:
                sch.submit(long)
            k = len(eng.rows[2]) if long.slot is not None and long.slot.i == 2 else 0
            if k != state["fills"]:
                between.append(n)
                state["fills"] = k
            if n == 400:
                sch.drain(0.0)
        eng.on_round = on_round
        for i in range(2):
            sch.submit(thinker(i, lambda t, f: None))
        sch.run()
        ln.close()
        s = long.slot.i
        rows = eng.rows[s]
        cover = sum(n for _, n in rows) == L and all(p == q for (p, _), q in
                                                    zip(rows, [0] + [p + n for p, n in rows[:-1]]))
        res[dch] = (rows, cover, eng.saves[s], len(set(between)))
    r0, r1 = res[0], res[2048]
    sizes = [n for _, n in r1[0]]
    ok = (r0[1] and r1[1] and r0[2] == r1[2] and max(sizes) <= 2048 and min(sizes) >= S.MIN_PIECE
          and len(r1[0]) > len(r0[0]))
    print(f"[pieces] {'ok' if ok else 'FAIL'} · decode_chunk 0: {len(r0[0])} steps {[n for _, n in r0[0]]} · 2048: "
          f"{len(r1[0])} steps {sizes} (rows in order, each once: {r0[1]} / {r1[1]}; rounds between the steps "
          f"{r1[3]}) · checkpoints at {r0[2]} / {r1[2]}")
    return ok


def guard():
    """the scheduler's incremental loop guard = the reference functions, every token"""
    if not hasattr(S.Req, "think_tok"):
        print("[guard] no incremental loop guard in this scheduler (skipped)")
        return True
    rnd, ok, n = random.Random(7), True, 0
    for case in range(60):
        kind = case % 4
        per = rnd.randrange(1, 10)
        pat = [rnd.randrange(1000, 1000 + (3 if kind == 3 else 50000)) for _ in range(per)]
        r = S.Req([1], 10, 0.0, 0, 1.0, 1, None)
        ref = []
        for j in range(700):
            if kind == 0:
                t = rnd.randrange(1000, 51000)
            elif kind == 1:
                t = pat[j % per]
            elif kind == 2:  # periodic with a break now and then
                t = pat[j % per] if rnd.random() > 0.01 else 7
            else:  # a tiny alphabet: short periods come and go
                t = pat[rnd.randrange(per)]
            r.think_tok(t)
            ref.append(t)
            tail = ref[-S.LOOP_SPAN:]
            n += 1
            if r.looping() != S.looping(tail) or r.loop_room() != S.loop_room(tail):
                ok = False
                print(f"[guard] FAIL case {case} token {j}: looping {r.looping()} vs {S.looping(tail)}, room "
                      f"{r.loop_room()} vs {S.loop_room(tail)}")
                break
    print(f"[guard] {'ok' if ok else 'FAIL'} · incremental = reference on {n:,} tokens (random, periodic, broken, "
          f"tiny alphabet)")
    return ok


def main():
    only = set(a.only.split(","))
    print(f"[round] scheduler from {a.sched} · Python {sys.version.split()[0]}")
    ok = True
    if "host" in only:
        for B in (4, 8, 16):
            host, _, _, _, same = run(B, a.rounds)
            ok &= same
            print(f"[host] B {B:2d}: {host:.3f} ms a round of scheduler work ({host / B * 1e3:.0f} µs a sequence) · "
                  f"pushed = emitted: {'yes' if same else 'NO'}")
    if "http" in only:
        for B in (4, 8, 16):
            h = Http(B, a.work_us)
            _, g, _, _, same = run(B, max(150, a.rounds // 4), a.gpu_ms / 1e3, h)
            h.close()
            ok &= same
            g.sort()
            print(f"[http] B {B:2d} (engine {a.gpu_ms:g} ms, HTTP {a.work_us:g} µs a request-round): gap "
                  f"{statistics.fmean(g):.3f} ms mean · {statistics.median(g):.3f} median · p90 "
                  f"{g[int(0.9 * len(g))]:.3f} · max {g[-1]:.3f}")
    if "metrics" in only:
        ok &= metrics()
    if "pieces" in only:
        ok &= pieces()
    if "guard" in only:
        ok &= guard()
    print(f"[sched_round] {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
