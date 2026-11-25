#!/usr/bin/env python3
"""kv_admit.py — P125 kv_pool admission, in process (TP2 through tools/tp2.sh, which appends --rank / --head-ip): the
server's Scheduler + kvguard on rank 0 driving the engine through link.Mirror, rank 1 replaying the calls (link.replay),
as server/main.py does — no HTTP front. dspark_init, then the pool as the server takes it (warm-up, then min MemFree
of both nodes − reserve) with the reserve set so that the pool is small (about --pool-gib beyond the warm-up's): then
long prompts fill it in seconds, and MemFree staying ≥ the reserve shows that nothing is allocated after the pool.
Requests (greedy, no EOS: each runs to its max_tokens unless the pool ends it):
  B, C  long prompts that fill the pool beside A (no new chunk within their 2,000 output tokens)
  E     a prompt longer than the pool holds → refused with an error, never placed
  A     placed beside B and C (its reservation fits), needs a new chunk ~16–76 tokens past its reservation while the
        pool has < 6 chunks free → ends with finish "length" (pool full)
  D     needs more than is free while A runs → waits; placed when A has ended (A's idle slot gives its chunks back)
Printed: the pool, each request (finish, tokens, wait), the engine's "KV pool full" throws (should be 0), and the lowest
MemFree of each node during the run vs the reserve."""
import argparse
import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, os.path.join(ROOT, "server"), os.path.join(ROOT, "tools")]
import ext  # noqa: E402
import link  # noqa: E402
import kvguard  # noqa: E402
from scheduler import Req, Scheduler  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, required=True)
ap.add_argument("--head-ip", required=True)
ap.add_argument("--port", type=int, default=29671)
ap.add_argument("--link-port", type=int, default=29675)
ap.add_argument("--slots", type=int, default=4)
ap.add_argument("--ctx", type=int, default=262144)
ap.add_argument("--chunk", type=int, default=8192)
ap.add_argument("--block", type=int, default=7)
ap.add_argument("--pool-gib", type=float, default=0.5, help="room the pool takes beyond the warm-up's (about)")
ap.add_argument("--lazy-gib", type=float, default=1.2, help="the warm-up's lazy allocations (1.19 measured)")
ap.add_argument("--reserve-out", type=int, default=16, help="server.output_reserve")
ap.add_argument("--model", default="/models/ERNIE-5.5-Flash-Preview")
ap.add_argument("--table", default="/work/cache/table.tsv")
ap.add_argument("--prompts", default="/work/results/ref/v3/prompts.json")
ap.add_argument("--log", default="/work/dev/kvpool/build{r}.log")
a = ap.parse_args()
R0 = a.rank == 0

E = ext.load("engine", ext.engine_sources(), log=a.log.format(r=a.rank), ldflags=ext.ENGINE_LDFLAGS)
lr = a.slots * (a.block + 1)
E.init(a.model, a.table, a.rank, a.head_ip, a.slots, a.ctx, max(a.chunk, lr), lr, a.port, True)
D = {"test": "kv_admit"}
if not R0:
    s = link.connect(a.head_ip, a.link_port)
    n = link.replay(E, link.reader(link.hello(s, 1, D)))
    E.close()
    print(f"[kv] rank1 replayed {n:,} calls", flush=True)
    sys.exit(0)

s = link.accept(a.head_ip, a.link_port)
link.hello(s, 0, D)
m = link.Mirror(E, s)
m.dspark_init()
F = m.mem_free()
reserve = round(min(F) - a.lazy_gib - a.pool_gib, 2)
m.kv_pool_init(reserve, 0.0, True)
G = m.mem_free()
print(f"[kv] MemFree after dspark {F[0]:.2f} / {F[1]:.2f} GiB · reserve {reserve:.2f} · after the pool {G[0]:.2f} / "
      f"{G[1]:.2f} GiB", flush=True)


class Guard(kvguard.KvGuard):
    def poll(self, sched):
        super().poll(sched)
        self.minfree = [min(getattr(self, "minfree", [1e9, 1e9])[i], self.free_gib[i]) for i in (0, 1)]


OUT = a.reserve_out
kv = Guard(E, a.ctx, OUT, reserve - 1.0)  # floor 1 GiB under the reserve: it should never trigger
kv.poll_s = 2.0
sched = Scheduler(m, [-1], a.slots, a.ctx, a.chunk, kv=kv)
tot, need, SL = E.kv_total(), E.kv_need, kvguard.SLACK
print(f"[kv] pool {tot:,} chunks ({tot * 2 / 1024:.2f} GiB) · one request ≤ {kv.max_tokens():,} tokens · output_reserve "
      f"{OUT}", flush=True)


def res(L):  # a request's reservation at placing (kvguard.admit)
    return need(min(kv.cap, L + OUT) + SL)


def quiet(L, n):  # no new chunk within n tokens past the reservation
    return need(L + OUT + SL) == need(L + OUT + SL + n)


def fill(budget, lo=8192):  # the longest prompt whose reservation fits, quiet for 480 tokens (B / C: 400 out)
    hi = kv.cap - 1
    while lo < hi:
        x = (lo + hi + 1) // 2
        lo, hi = (x, hi) if res(x) <= budget else (lo, x - 1)
    while lo > 8192 and not quiet(lo, 480):
        lo -= 16
    return lo


# A: a new chunk within 60 tokens past its reservation; B, C: the rest of the pool; fewer than 6 chunks left beside them
# (A's next chunk is 6 MLA chunks)
best = None
for LA in (L for L in range(6000, 60000) if need(L + OUT + SL) < need(L + OUT + SL + 60)):
    rest = tot - res(LA)
    LB = fill(rest // 2)
    LC = fill(rest - res(LB))
    left = rest - res(LB) - res(LC)
    if best is None or left < best[3]:
        best = (LA, LB, LC, left)
    if left < 6:
        break
LA, LB, LC, left = best
LD = LA - 2000
print(f"[kv] A {LA:,} ({res(LA)} chunks) · B {LB:,} ({res(LB)}) · C {LC:,} ({res(LC)}) · free beside them {left} · "
      f"D {LD:,} ({res(LD)}) · E {kv.cap + 100:,}", flush=True)
toks = json.load(open(a.prompts))["doc_long"]
toks = toks * (max(LB, LC, kv.cap + 100) // len(toks) + 2)
out = {}


def req(name, off, L, mt):
    def push(t, f, name=name):
        o = out[name]
        o["n"] += len(t)
        if f:
            o["fin"], o["t_end"] = f, time.time()
    r = Req(toks[off:off + L], mt, 0.0, 0, 1.0, 1, push, name)
    out[name] = {"n": 0, "fin": None, "t0": time.time(), "t_end": None, "r": r}
    sched.submit(r)
    return r


def until(cond, limit=1200):
    t = time.time()
    while not cond():
        if time.time() - t > limit:
            raise TimeoutError("kv_admit: a request did not get there")
        time.sleep(0.05)


def feeder():
    try:
        b, c = req("B", 0, LB, 400), req("C", 7, LC, 400)
        until(lambda: b.state in ("decode", "done") and c.state in ("decode", "done"))
        e = req("E", 3, kv.cap + 100, 16)
        until(lambda: e.state == "done")
        ra = req("A", 11, LA, 1000)
        until(lambda: ra.state in ("decode", "done"))
        req("D", 13, LD, 8)
        until(lambda: all(o["fin"] is not None for o in out.values()))
    finally:
        sched.drain(time.time() + 5)


threading.Thread(target=feeder, daemon=True).start()
t0 = time.time()
sched.run()
H = m.mem_free()
for k in ("B", "C", "E", "A", "D"):
    o = out.get(k)
    if not o:
        continue
    r = o["r"]
    wait = (r.t_place - o["t0"]) if r.t_place else float("nan")
    print(f"[kv] {k}: prompt {len(r.prompt):,} · max_tokens {r.max_tokens:,} · finish {o['fin']} · {r.n_out:,} tokens · "
          f"waited {wait:.1f} s for a slot / memory · reservation {getattr(r, 'kv_tok', 0):,}"
          + (f" · error: {r.error}" if r.error else ""), flush=True)
od = out["D"]["r"]
first = min((o["t_end"] for k, o in out.items() if k in "ABC" and o["fin"] == "length" and o["r"].n_out < o["r"].max_tokens),
            default=None)
print(f"[kv] D placed {od.t_place - first:+.2f} s after the first pool-full end · " if od.t_place and first else "[kv] ",
      end="")
print(f"engine 'KV pool full' throws {kv.throws} · pool-full ends {kv.full_ends} · lowest MemFree during the run rank0 "
      f"{kv.minfree[0]:.2f} · rank1 {kv.minfree[1]:.2f} GiB (reserve {reserve:.2f}) · at the end {H[0]:.2f} / {H[1]:.2f} · "
      f"{time.time() - t0:.0f} s", flush=True)
m.close()
E.close()
