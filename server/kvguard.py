"""Memory defense (rank 0, the scheduler thread): admission by the engine's shared KV pool (P125 kv_pool) and a watch on
both nodes' free memory. The scheduler calls in at four points: admit (placing a request), grow (before a decode step
or forced tokens), call (an engine call that may meet a full pool) and poll (every loop iteration).

Pool (engine kv_*): the slots' growing caches share one pool of 2 MiB chunks taken at start (kv_pool_init: the room left
on the node with less of it after the warm-up, less server.memory_free_gib, at most server.kv_pool_gib), mapped as a
sequence grows. A request is placed only when the pool can hold its prompt + min(max_tokens, server.output_reserve) on
top of what the running requests may still map within their reservations; idle slots (whose caches a later prompt could
continue) give theirs back first, least recently used first, then the busy slots' prefix checkpoints (P146; a checkpoint
is taken only when the pool has its chunks beyond those reservations); else it waits (first come, first served). A request no
pool could ever hold is refused (vLLM's 400: the HTTP front's max_model_len is what one request can have; here too, for
a prompt that reached the scheduler some other way). A reply that grows past its reservation takes another
output_reserve tokens' worth as it goes and ends with finish "length" only when the pool has none left. The engine
throws "KV pool full" only if this accounting was wrong (before anything changed, on both ranks): that slot's request
ends, the server goes on.

Watch: every MEM_POLL_S seconds (idle too: the loop wakes for it) both nodes' MemFree through one tiny collective
(engine mem_free, mirrored to rank 1). Below server.memory_floor_gib on either node no new request is placed (one [mem]
warn line; /health "low memory"); placing resumes when both are above it again. /metrics: pool chunks total / used /
promised, MemFree per node (snapshot taken by the scheduler thread: the HTTP thread never calls the engine)."""
import time

import qlog

SLACK = 64        # rows a request may write past its tokens in one call (DSpark verify rows, forced tokens)
MEM_POLL_S = 5.0  # seconds between MemFree reads


class KvGuard:
    def __init__(self, eng, ctx, output_reserve, floor_gib):
        """eng: rank 0's engine module itself (pool queries: not mirrored); the scheduler's mirror makes the calls that
        change state or run collectives (reset, mem_free)."""
        self.eng, self.ctx, self.reserve, self.floor = eng, ctx, output_reserve, floor_gib
        self.pool = bool(eng.kv_on()) and eng.kv_total() > 0
        self.total = eng.kv_total() if self.pool else 0
        lo, hi = 0, ctx  # the longest single request the whole pool holds
        if self.pool:
            while lo < hi:
                m = (lo + hi + 1) // 2
                lo, hi = (m, hi) if eng.kv_need(m + SLACK) <= self.total else (lo, m - 1)
        self.cap = lo if self.pool else ctx
        self.low, self.free_gib, self.t_poll = False, [float("nan")] * 2, 0.0
        self.full_ends = 0  # replies ended with "length" because the pool was full (/metrics)
        self.throws = 0     # the engine's "KV pool full" (the accounting was wrong; should stay 0)
        self.snap = {}
        self.poll_s = MEM_POLL_S

    def max_tokens(self):
        """prompt + output of one request at most: the context setting, or less if the pool holds fewer"""
        return min(self.ctx, self.cap)

    # ---- accounting (scheduler thread) --------------------------------------------------------------------------
    def need(self, tokens):
        return self.eng.kv_need(tokens) if self.pool else 0

    def promised(self, sched):
        """chunks the running requests may still map within their reservations"""
        return sum(max(0, self.need(x.kv_tok) - self.eng.kv_held(x.slot.i)) for x in sched.running
                   if x.slot is not None and getattr(x, "kv_tok", 0))

    def room(self, sched, need, keep, extra=0, src=None):
        """`need` chunks free beyond the running requests' promises (+ `extra` about to come back): idle slots other than
        `keep` give theirs back, least recently used first, then (P146) the busy slots' checkpoints but `src`'s (the slot
        a request is about to copy one from), the lowest first → whether it fits"""
        while self.eng.kv_free() + extra - self.promised(sched) < need:
            idle = [s for s in sched.slots if s.req is None and s is not keep and
                    (self.eng.kv_held(s.i) > 0 or s.ck)]
            if idle:
                sched._reset(min(idle, key=lambda x: x.used))
                continue
            ck = [(p, s.i, k) for s in sched.slots if s.req is not None and s is not src for k, (p, _) in s.ck.items()]
            if not ck:
                return False
            _, i, k = min(ck)
            sched._ck_drop(sched.slots[i], k)
        return True

    def ckpt_room(self, sched, need):
        """P146: a checkpoint's `need` chunks free beyond the running requests' promises (nothing given back for it)"""
        return not self.pool or self.eng.kv_free() - self.promised(sched) >= need

    def admit(self, sched, r, s, keep, src=None):
        """Before r takes slot s, which keeps the first `keep` positions of its history (all: continuing it; fewer: P146,
        restoring a checkpoint there — the caches above it and the checkpoints above it come back; 0: the slot starts
        over, everything comes back; src: the busy slot whose checkpoint it then copies): False = wait (memory low, or
        no room yet); None = refused (r ended with an error); True = placed, r.kv_tok = its reservation."""
        if self.low:
            return False
        if not self.pool:
            return True
        L = len(r.prompt)
        if L + 1 > self.cap:
            r.error = (f"This model's maximum context length is {self.max_tokens()} tokens. However, your prompt "
                       f"contains {L} input tokens, which exceeds what the server's cache memory holds for one request.")
            r.state = "done"
            r.push([], "error")
            return None
        T = min(self.cap, L + min(r.max_tokens, self.reserve)) + SLACK
        mapped = self.eng.kv_held(s.i)
        if not keep:
            held = 0
        elif keep >= len(s.hist):
            held = mapped
        else:  # P146: the caches above the checkpoint (+ slack) come back
            held = min(mapped, self.need(keep))
        extra = mapped - held + sum(n for p, n in s.ck.values() if not keep or p > keep)  # what comes back
        if not self.room(sched, max(0, self.need(T) - held), s, extra, src):
            return False
        r.kv_tok = T
        return True

    def grow(self, sched, items):
        """items: [(request, rows it is about to run)] before a step. A reply about to pass its reservation takes
        another output_reserve tokens' worth (or just what this step needs), or ends with "length" when the pool has
        nothing left → the requests that go on."""
        if not self.pool:
            return [r for r, _ in items]
        keep = []
        for r, rows in items:
            want = len(r.slot.hist) + rows + SLACK
            if want > r.kv_tok:
                old, r.kv_tok = r.kv_tok, 0  # out of the promises: it asks for all it still has to map
                held = self.eng.kv_held(r.slot.i)
                for T in (max(want, min(self.cap + SLACK, old + self.reserve)), want):
                    if self.room(sched, max(0, self.need(T) - held), r.slot):
                        r.kv_tok = T
                        break
                if not r.kv_tok:
                    r.kv_tok = old
                    self.full_ends += 1
                    qlog.warn("req", f"#{r.id}: the cache memory is full — reply ends at {r.n_out:,} tokens",
                              key=("kv full", r.id))
                    r.push([], "length")
                    sched._finish(r, "length")
                    continue
            keep.append(r)
        return keep

    def call(self, sched, fn, *a):
        """fn(*a); the engine's "KV pool full" (thrown before the call changed anything, on both ranks) ends that slot's
        request (length if it has output, else an error) and empties the slot; the others go on. → fn's result, or None"""
        try:
            return fn(*a)
        except RuntimeError as e:
            msg = str(e)
            if not msg.startswith("KV pool full"):
                raise
            self.throws += 1
            i = int(msg.split("slot ", 1)[1].split()[0])
            r = next((x for x in sched.running if x.slot is not None and x.slot.i == i), None)
            qlog.warn("req", f"#{r.id if r else '?'}: {msg} — the request ends, the server goes on")
            if r is not None:
                fin = "length" if r.n_out else "error"
                r.error = "the server's cache memory is full"
                r.push([], fin)
                sched._finish(r, fin)
            sched._reset(sched.slots[i])
            return None

    # ---- memory watch -------------------------------------------------------------------------------------------
    def poll(self, sched):
        """both nodes' MemFree every MEM_POLL_S s (a collective through the mirror); the floor decides admission"""
        now = time.time()
        if now - self.t_poll >= self.poll_s:
            self.t_poll = now
            f = list(sched.eng.mem_free())
            self.free_gib = f
            low = min(f) < self.floor
            if low and not self.low:
                qlog.warn("mem", f"MemFree rank0 {f[0]:.1f} · rank1 {f[1]:.1f} GiB — below the floor "
                                 f"{self.floor:g} GiB (server.memory_floor_gib): new requests wait", key=("mem low", now))
            elif self.low and not low:
                qlog.ok("mem", f"MemFree rank0 {f[0]:.1f} · rank1 {f[1]:.1f} GiB — above the floor again: requests "
                               "are placed")
            self.low = low
        self.snap = {"total": self.total, "free": self.eng.kv_free() if self.pool else 0,
                     "promised": self.promised(sched) if self.pool else 0, "mem": list(self.free_gib), "low": self.low,
                     "full_ends": self.full_ends, "throws": self.throws}

    # ---- HTTP thread (reads the snapshot only) -----------------------------------------------------------------
    def health(self):
        s = self.snap
        out = {"memory": {"free_gib": [round(x, 2) for x in s.get("mem", self.free_gib)], "floor_gib": self.floor,
                          "admitting": not s.get("low", False)}}
        if self.pool:
            out["kv_pool"] = {"chunks": s.get("total", self.total), "used": s.get("total", 0) - s.get("free", 0),
                              "promised": s.get("promised", 0), "max_tokens": self.max_tokens()}
        return out

    def metrics(self):
        s, rows = self.snap, []
        if self.pool:
            tot, used = s.get("total", self.total), s.get("total", 0) - s.get("free", 0)
            rows += [("gauge", "e55_kv_pool_chunks", tot), ("gauge", "e55_kv_pool_chunks_used", used),
                     ("gauge", "e55_kv_pool_chunks_promised", s.get("promised", 0)),
                     ("counter", "e55_kv_pool_full_ends_total", s.get("full_ends", 0)),
                     ("counter", "e55_kv_pool_full_throws_total", s.get("throws", 0))]
        txt = "".join(f"# TYPE {n.removesuffix('_total')} {t}\n{n} {v}\n" for t, n, v in rows)
        for i, g in enumerate(s.get("mem", self.free_gib)):
            txt += f'e55_mem_free_gib{{rank="{i}"}} {g:.3f}\n'
        txt += f"e55_mem_low {int(s.get('low', False))}\n"
        return txt
