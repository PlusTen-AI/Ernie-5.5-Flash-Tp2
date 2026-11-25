"""Scheduler (rank 0): the only thread that drives the engine. Requests wait for a slot; a slot keeps its sequence's
caches after the request ends so the next request can continue from them (prefix reuse): when the slot's history
(the tokens its caches hold) is a prefix of the new prompt, only the rest is read. The history is the prompt plus every
reply token but the last (the last one is never run). The HCA caches (a 128-token ring window and the compressor's open
group) cannot step back, so a prompt that leaves the history part-way (the client did not send the last reply's
reasoning back, an agent rewrote an earlier message) needs a checkpoint (P146 prefix_ckpt, engine ckpt_*): a copy of
what the slot's caches hold beyond their positional rows at one point of its history, taken at every prompt's end and
after a prefill chunk at least server.checkpoint_every tokens past the slot's last one, at most server.checkpoints a
slot (when full, the one adding the fewest tokens over the one below it goes; 2.9–21 MiB each, KV pool chunks, only
when the pool has them beyond the running requests' reservations). Such a prompt restores the slot's longest checkpoint
at or below the common prefix (the slot's entries above it go) and reads the rest — or, when only a busy slot has one,
an empty slot takes it (that slot's rows below it copied); else the slot idle longest starts over. The same bits as
reading the whole prompt.
Memory (kvguard.KvGuard, P125 kv_pool): the slots' caches come from one shared pool; a request is placed when the pool
can hold its prompt + min(max_tokens, server.output_reserve) beside the running requests' reservations (idle slots give
theirs back first), a reply takes more as it grows, and new requests wait while either node's MemFree is below
server.memory_floor_gib.

Each loop iteration: at most one prefill chunk, then one decode step over every decoding sequence (Engine step +
sampling on the GPU: penalties, temperature, min_p, top-k, top-p — deterministic and batch-invariant; the Philox
counter of a draw is the new token's position, so a token never depends on what else ran with it). Prompts being
read share the chunks so a long one does not starve the others: a prompt whose rest fits in one chunk cuts in (the
fewest tokens left first) — it gets its first token after at most the chunk running now; prompts needing more chunks
take turns, one chunk each (the one served longest ago first). One screen line per finished request.
Thinking: a budget (server setting or the request's) and the loop guard close the thinking by putting
</ernie:think> in place of the next token (it has not run yet, so nothing else changes). tool_choice required / named:
the call's opening tokens follow </ernie:think> at once (thinking on) or end the prompt (thinking off).

DSpark (decode.dspark, the default): the decode step is a speculative round over every decoding sequence — the draft's
proposals for each (made at the end of the previous round) are verified in one target step, and each sequence emits
the tokens the target accepts plus the target's own next one (engine spec_round; the request's own sampling: greedy,
or temperature / top_k / top_p / min_p with speculative acceptance, penalties, its seed). A prompt's last chunk runs
as a plain step whose logits row gives the first token (spec_begin, drafts for the first round). Every rule above
holds token for token because a round never emits past a point where the server could change or end the reply: its
cap (engine: the verify length too, P122) stops at max_tokens, at the thinking budget's and the loop guard's first
possible forced token, and at a draft that is an end token, the one-call stop token or (tool_choice forcing pending)
</ernie:think> — so a forced token is always the round's last, not yet run, and a slot's history is exactly what
plain decoding leaves. How many drafts a round verifies (decode.verify): "cost" (P143) — for all its sequences
together, the counts with the most expected tokens (the draft's confidences) per ms of the round (ROUND_* below);
"gate" (P121) — each sequence's drafts while the product of their confidences stays ≥ decode.gate. Greedy requests
get plain decoding's tokens bit for bit.
Copy drafts (decode.copy, P168; copy_draft.py): a round with one decoding request whose reply repeats text its history
holds verifies the tokens that followed that text there instead (engine spec_copy: up to decode.copy_rows rows, each
drawn as plain decoding draws it) — under the same caps; rounds shared by several requests are the rounds above."""
import collections
import queue
import time

import qlog
from copy_draft import MIN_COPY, Copier

LOOP_SPAN, LOOP_PERIOD = 256, 8  # loop guard: the last 256 thinking tokens repeat a pattern of ≤ 8 tokens
CKPT_MIN = 1024  # P146: no checkpoint below this history (it would save less prefill than its chunks are worth)
# P143 verify_joint (decode.verify = "cost"): ms of a DSpark round, measured 2026-11-25 on TP2 at context ≈ 9.4K: the
# verify target step by sequences B and total rows R — (R, ms) points (B = 1: step_prof --verify; B = 4: sampled rounds,
# sched_dspark.py --do cost), linear in R between them and along the end segments past them — and the rest of a round
# (the draft's context rows, accept, commit, the next drafts at depth 7, the scheduler: start to start less the target
# step, sampled rounds at T 1.0 / top_p 0.95, sched_dspark.py --do rate) by B. A B between two measured ones: linear.
ROUND_VERIFY_MS = {1: ((1, 32.0), (4, 43.3), (8, 57.1)),
                   4: ((4, 48.9), (8, 65.6), (12, 77.9), (16, 90.8), (24, 113.3), (32, 128.5))}
ROUND_REST_MS = ((1, 5.9), (4, 11.0))


def looping(tail):
    """Is the full tail (LOOP_SPAN tokens) periodic with a period of at most LOOP_PERIOD?"""
    if len(tail) < LOOP_SPAN:
        return False
    t = list(tail)
    return any(all(t[i] == t[i + p] for i in range(LOOP_SPAN - p)) for p in range(1, LOOP_PERIOD + 1))


def loop_room(tail):
    """How many more thinking tokens can come before the loop guard could fire, that token included: after j more, the
    full tail keeps the last LOOP_SPAN − j tokens it has now, so it can only be periodic once those already are (the
    longest periodic suffix now, over the periods ≤ LOOP_PERIOD)."""
    t, n, best = list(tail), len(tail), 0
    for p in range(1, LOOP_PERIOD + 1):
        k = min(p, n)
        while k < n and t[n - 1 - k] == t[n - 1 - k + p]:
            k += 1
        best = max(best, k)
    return max(1, LOOP_SPAN - best)


def _lerp(pts, x):
    """y at x on the polyline through pts ((x, y), … by x), extended along its end segments"""
    if len(pts) == 1:
        return pts[0][1]
    i = 1
    while i < len(pts) - 1 and x > pts[i][0]:
        i += 1
    (x0, y0), (x1, y1) = pts[i - 1], pts[i]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def round_cost(slots, rmax, verify=None, rest=None):
    """ms of a DSpark round for B = 1 … slots sequences and R = 0 … rmax verify rows, flattened [B][R] (the engine's
    spec_round_cost; R < B cannot happen: ms(B, B) there)"""
    verify, rest = verify or ROUND_VERIFY_MS, rest or ROUND_REST_MS
    bs, out = sorted(verify), []
    for B in range(1, slots + 1):
        lo = max([b for b in bs if b <= B], default=bs[0])
        hi = min([b for b in bs if b >= B], default=bs[-1])
        w = 0.0 if hi == lo else (B - lo) / (hi - lo)
        e = _lerp(sorted(rest), B)
        for R in range(rmax + 1):
            r = max(R, B)
            out.append((1 - w) * _lerp(verify[lo], r) + w * _lerp(verify[hi], r) + e)
    return out


class Req:
    _n = 0

    def __init__(self, prompt, max_tokens, temp, top_k, top_p, seed, push, label=""):
        Req._n += 1
        self.id, self.label = Req._n, label
        self.prompt, self.max_tokens = prompt, max_tokens
        self.temp, self.top_k, self.top_p, self.seed = temp, top_k, top_p, seed
        self.push = push  # push(tokens: list[int], finish: str | None) from the scheduler thread
        self.cancelled = False
        self.slot = None
        self.state = "wait"  # wait → prefill → decode (⇄ inject) → done
        self.pf = 0  # next prompt position to run
        self.cached = 0
        self.last = None  # last emitted token (not yet run)
        self.n_out = 0
        self.t0, self.t_first = time.time(), None
        self.t_place = self.t_pf = self.t_sse = None  # timing marks (app.timing)
        self.t_chunk = 0.0  # when this request's last prefill chunk ran (prompts needing several chunks take turns)
        self.marks = []  # server.prefill_marks: (prompt position read, time) after each prefill chunk
        # thinking (set by the HTTP front for chat requests that think): budget in thinking tokens (None = none, 0 =
        # answer at once), tokens emitted while thinking, the last LOOP_SPAN of them (loop guard), why the server
        # closed the thinking (if it did)
        self.thinking, self.budget, self.n_think, self.forced = False, None, 0, None
        self.tail = collections.deque(maxlen=LOOP_SPAN)
        # tool_choice required / named: force = the tokens to put right after </ernie:think> (thinking on),
        # prefix_out = the same tokens already appended to the prompt, emitted first (thinking off), stop_tok = end
        # the reply after this token (named, or parallel_tool_calls false)
        self.force, self.prefix_out, self.stop_tok, self.inj = None, None, None, None
        self.min_p, self.pen = 0.0, (1.0, 0.0, 0.0)  # min_p; (repetition, frequency, presence) penalties
        self.n_prompt = len(prompt)  # the prompt without tokens forced onto its end (tool_choice, thinking off)
        self.reuse = "reset"
        self.error = None  # why the server refused it (sent to the client)
        self.cp, self.cp_nodraft = None, False  # P168: copy state (copy_draft.State); a copy went on: no DSpark drafts

    @property
    def penalised(self):
        return self.pen != (1.0, 0.0, 0.0)


class Slot:
    def __init__(self, i):
        self.i, self.hist, self.req, self.used = i, [], None, 0.0  # hist: the tokens the caches hold
        self.ck = {}  # P146: engine entry → (history position, pool chunks) of the slot's checkpoints


def _lcp(a, b):
    """length of the common prefix of two token lists (compared in blocks: list equality runs in C)"""
    n, k, B = min(len(a), len(b)), 0, 4096
    while k < n and a[k:k + B] == b[k:k + B]:
        k += B
    if k >= n:
        return n
    while k < n and a[k] == b[k]:
        k += 1
    return k


class Scheduler:
    """run() is the engine loop; call it on the thread that initialised the engine."""

    def __init__(self, eng, eos, slots, ctx, chunk, think_end=None, loop_guard=True, block=0, gate=0.0, kv=None,
                 checkpoints=8, checkpoint_every=8192, verify="gate", prefill_marks=False, copy=False, copy_rows=64,
                 copy_match=8):
        self.eng, self.eos, self.ctx, self.chunk = eng, set(eos), ctx, chunk
        self.kv = kv  # kvguard.KvGuard: pool admission and the memory watch (None: neither)
        self.think_end, self.loop_guard = think_end, loop_guard
        # DSpark: draft tokens per round (0 = plain decoding; the engine's dspark_init has run), the verify lengths
        # (decode.verify) and P121's gate
        self.block, self.gate, self.verify = block, gate, verify
        if verify not in ("cost", "gate"):
            raise ValueError(f"decode.verify must be \"cost\" or \"gate\", got {verify!r}")
        if block and verify == "cost":  # P143: the round's cost table on both ranks; the sequences' gate < 0 selects it
            rmax = slots * (block + 1)
            tab = round_cost(slots, rmax)
            eng.spec_round_cost(tab, rmax)
            eng.spec_cost(tab[1:block + 2])  # (P143 off: each sequence on its own, B = 1's ms by verified drafts)
            self.gate = -1.0
        self.peek = getattr(eng, "eng", eng)  # reads that change nothing (the drafts) skip the command link
        # P168 copy drafts (decode.copy, with DSpark rounds): the matcher (rank 0 only; its drafts go as arguments)
        self.copy = Copier(copy_rows, copy_match) if copy and block else None
        self.slots = [Slot(i) for i in range(slots)]
        # P146 prefix_ckpt: checkpoints a slot (0 = none: a prompt leaving the history part-way reads from 0), the
        # prefill tokens between two (server.checkpoints / checkpoint_every); the engine's entries made (and emptied)
        on = checkpoints > 0 and bool(getattr(self.peek, "ckpt_on", lambda: False)())
        self.ck_n, self.ck_every = (checkpoints if on else 0), max(1, checkpoint_every)
        self.marks = prefill_marks  # server.prefill_marks: each prefill chunk waited for, its end time kept
        if self.ck_n:
            self.eng.ckpt_init(self.ck_n)
        self.inbox = queue.Queue()
        self.waiting, self.running = [], []
        # running totals (also read by /metrics from the HTTP thread: plain int updates, no lock needed)
        self.stats = {"requests": 0, "prompt": 0, "cached": 0, "out": 0}
        # decode steps by sequences: [steps, ms inside the step, ms from this step's start to the next one's (only
        # when no prefill chunk ran in between), steps counted in that wall total]
        self.rstats = {}
        self._prev = None  # (B, start) of the last decode step, None after a prefill chunk
        self.draining, self.deadline, self.error = False, None, None

    def submit(self, r):
        self.inbox.put(r)

    # ---- stopping (called from other threads) ------------------------------------------------------------------
    def drain(self, deadline):
        """Stop when every request has ended; at `deadline` (time.time()) the rest are cut (finish as aborted). run()
        returns then. The loop is woken from a helper thread (a queue put must not run inside a signal handler)."""
        self.deadline = deadline
        self.draining = True
        self.inbox.put(None)

    def fail_all(self, msg):
        """The engine cannot go on (the other rank is gone): every request ends with an error."""
        self.error = msg
        for r in list(self.waiting) + list(self.running):
            r.push([], "error")
        while True:
            try:
                r = self.inbox.get_nowait()
            except queue.Empty:
                break
            if r is not None:
                r.push([], "error")

    def busy(self):
        return len(self.waiting) + len(self.running)

    def counts(self):
        return len(self.running), len(self.waiting) + self.inbox.qsize()

    def slot_status(self):
        """(slot, busy, tokens held) for /metrics"""
        return [(s.i, s.req is not None, len(s.hist)) for s in self.slots]

    # ---- slot choice and prefix reuse -------------------------------------------------------------------------
    def _place(self, r):
        """A free slot for r, by the most of the prompt its caches already hold: one whose whole history the prompt
        continues, or (P146) one's longest checkpoint at or below the common prefix of its history and the prompt
        (restored; a continuation wins a tie, then the slot idle longest); else a busy slot's such checkpoint, copied
        into the free slot idle longest; else that slot starts over. False: no free slot yet (or no memory: kvguard);
        None: refused (it got its error)."""
        free = [s for s in self.slots if s.req is None]
        if not free:
            return False
        P = r.prompt
        opts = []  # (positions kept, continuation, −used, slot, entry)
        for s in free:
            if s.hist:
                c = _lcp(s.hist, P)
                if c == len(s.hist) < len(P):
                    opts.append((c, 1, -s.used, s.i, None))
                else:
                    q, k = self._ck_at(s, c, len(P))
                    if q:
                        opts.append((q, 0, -s.used, s.i, k))
        n, cont, _, i, k = max(opts) if opts else (0, 0, 0, 0, None)
        best, src = self.slots[i], None
        for s in self.slots:  # P146: a busy slot's checkpoint that holds more, copied
            if s.req is not None and s.ck and max(p for p, _ in s.ck.values()) > n:
                q, kk = self._ck_at(s, _lcp(s.hist, P), len(P))
                if q > n:
                    n, cont, src, k = q, 0, s, kk
        if not n or src is not None:
            best = min(free, key=lambda s: s.used)
        if self.kv is not None:
            ok = self.kv.admit(self, r, best, 0 if src is not None else n, src)
            if not ok:
                return ok
        if cont:
            r.cached, r.reuse = n, "continued"
        elif src is not None and self.eng.ckpt_copy(best.i, src.i, k):
            best.hist, best.ck = src.hist[:n], {}
            r.cached, r.reuse = n, f"via slot {src.i}'s checkpoint"
        elif n and src is None:
            self.eng.ckpt_restore(best.i, k)
            best.hist, best.ck = best.hist[:n], {kk: v for kk, v in best.ck.items() if v[0] <= n}
            r.cached, r.reuse = n, "via checkpoint"
        else:
            self._reset(best)
            r.cached = 0
        s = r.slot = best
        s.req = r
        r.pf = r.cached
        if r.penalised:
            self.eng.pen_begin(s.i, r.prompt[:r.n_prompt])
        r.state, r.t_place = "prefill", time.time()
        return True

    def _reset(self, s):
        """slot s starts over (its caches and checkpoints back to the pool)"""
        self.eng.reset(s.i)
        s.hist, s.ck = [], {}

    # ---- P146 prefix checkpoints --------------------------------------------------------------------------------
    @staticmethod
    def _ck_at(s, c, L):
        """(position, entry) of slot s's longest checkpoint a prompt of L tokens sharing c with its history can start
        from (≤ c, < L: the prompt's last row still runs, for its logits), else (0, None)"""
        q, k = 0, None
        for kk, (p, _) in s.ck.items():
            if q < p <= c and p < L:
                q, k = p, kk
        return q, k

    def _ck_drop(self, s, k):
        self.eng.ckpt_drop(s.i, k)
        del s.ck[k]

    def _ck_take(self, s):
        """A checkpoint of slot s at its history (a prompt's end, a prefill chunk's), if the pool has the chunks beyond
        the running requests' reservations; at the cap first the one adding the fewest positions over the one below"""
        q = len(s.hist)
        if not self.ck_n or q < CKPT_MIN or any(p == q for p, _ in s.ck.values()):
            return
        if len(s.ck) >= self.ck_n:
            ps = sorted((p, k) for k, (p, _) in s.ck.items())
            self._ck_drop(s, min((p - (ps[j - 1][0] if j else 0), k) for j, (p, k) in enumerate(ps))[1])
        need = self.peek.ckpt_need(q)
        if self.kv is not None and not self.kv.ckpt_room(self, need):
            return
        k = next(k for k in range(self.ck_n) if k not in s.ck)
        if self.eng.ckpt_save(s.i, k):
            s.ck[k] = (q, need)

    # ---- prefill ------------------------------------------------------------------------------------------------
    def _next_prefill(self, pf):
        """The prompt whose next chunk runs: one that finishes within a chunk cuts in (fewest left first); else the
        one whose last chunk ran longest ago."""
        short = [r for r in pf if len(r.prompt) - r.pf <= self.chunk]
        if short:
            return min(short, key=lambda r: len(r.prompt) - r.pf)
        return min(pf, key=lambda r: r.t_chunk)

    def _args(self, rs, ctrs):
        """run()'s per-sequence sampling arguments (positional: the command link forwards no keywords)"""
        return ([r.temp for r in rs], [r.top_k for r in rs], [r.top_p for r in rs], [r.min_p for r in rs],
                [r.seed for r in rs], ctrs, [r.pen[0] for r in rs], [r.pen[1] for r in rs], [r.pen[2] for r in rs])

    def _prefill_chunk(self, r):
        s, P = r.slot, r.prompt
        c0, c1 = r.pf, min(len(P), r.pf + self.chunk)
        n = c1 - c0
        if c1 < len(P):
            self.eng.fill(P[c0:c1], list(range(c0, c1)), [s.i], [0, n])
            if self.marks:  # this chunk finished (rank 0's GPU only; the steps are in lockstep), then its time
                self.peek.sync()
                r.marks.append((c1, time.time()))
            s.hist += P[c0:c1]
            r.pf = c1
            if self.copy:
                self._copy_sync(s)
            if self.ck_n and c1 - max((p for p, _ in s.ck.values()), default=0) >= self.ck_every:
                self._ck_take(s)  # P146: every checkpoint_every prefill tokens
            return
        if self.block:  # DSpark: a plain step (its logits row gathered), the first token from it, the first drafts
            self.eng.step(P[c0:c1], list(range(c0, c1)), [s.i], [0, n], False)
            t = self._begin(r, len(P))
        else:
            t = self.eng.run(P[c0:c1], list(range(c0, c1)), [s.i], [0, n], *self._args([r], [len(P)]))[0]
        s.hist += P[c0:c1]
        r.pf = c1
        if self.copy:
            self._copy_sync(s)
        r.t_pf = r.t_first = time.time()
        if self.marks:
            r.marks.append((c1, r.t_pf))
        r.state = "decode"
        self._deliver(r, t)
        if self.block:
            self._redraft(r, t)
        self._ck_take(s)  # P146: the prompt's end (after its first token: the caches change only with the next step)

    # ---- decode -------------------------------------------------------------------------------------------------
    def _round(self, rs):
        t0 = time.perf_counter()
        B = len(rs)
        if self._prev is not None:
            x = self.rstats[self._prev[0]]
            x[2] += 1e3 * (t0 - self._prev[1])
            x[3] += 1
        pos = [len(r.slot.hist) for r in rs]
        toks = self.eng.run([r.last for r in rs], pos, [r.slot.i for r in rs], list(range(B + 1)),
                            *self._args(rs, [p + 1 for p in pos]))
        x = self.rstats.setdefault(B, [0, 0.0, 0.0, 0])
        x[0] += 1
        x[1] += 1e3 * (time.perf_counter() - t0)
        self._prev = (B, t0)
        for r, t in zip(rs, toks):
            r.slot.hist.append(r.last)
            self._deliver(r, t)

    def _think(self, r, t):
        """Thinking budget / loop guard: t, or </ernie:think> in its place."""
        if r.thinking and self.think_end is not None:
            if t == self.think_end:
                r.thinking = False
            else:
                r.n_think += 1
                r.tail.append(t)
                why = ("budget" if r.budget is not None and r.n_think >= r.budget else
                       "loop" if self.loop_guard and looping(r.tail) else None)
                if why and t not in self.eos:
                    t, r.thinking, r.forced = self.think_end, False, why
        return t

    def _deliver(self, r, t):
        """A new token t (not run yet): thinking budget / loop guard, tool_choice forcing, then out."""
        self._take(r, [t])

    def _take(self, r, toks):
        """New tokens, only the last one not run yet (a spec round's cap makes any token the server would change or end
        the reply on its last): thinking budget / loop guard, tool_choice forcing, then out."""
        out = [self._think(r, t) for t in toks]
        r.last = out[-1]
        if r.force and not r.thinking and r.last == self.think_end:  # the call's opening follows </ernie:think> at once
            r.state, r.inj, r.force = "inject", list(r.force), None
        self._emit(r, out)

    # ---- DSpark -----------------------------------------------------------------------------------------------
    def _depth(self, pos):
        """Draft rows for an anchor at pos: the block, less near the slot's end (a running request's last token is
        inside the context, so ≥ 1)."""
        return max(1, min(self.block, self.ctx - pos))

    def _begin(self, r, pos):
        """The last step's logits row 0 → r's first token (its sampling, counter = the token's position pos), then the
        drafts for its first round."""
        if self.copy:
            self.copy.stop(r)
            r.cp_nodraft = False
        return self.eng.spec_begin(r.slot.i, self._depth(pos), r.temp, r.seed, 0, r.top_k, r.top_p, r.min_p, r.pen[0],
                                   r.pen[1], r.pen[2], self.gate, True)

    def _redraft(self, r, t):
        """The server put another token in place of t (a forced </ernie:think>): the engine's last token becomes that
        one, and the drafts are made again from it."""
        if r.state == "decode" and r.last != t:
            self.eng.spec_last(r.slot.i, r.last)
            self.eng.spec_draft([r.slot.i])
            if self.copy:
                self.copy.stop(r)
                r.cp_nodraft = False

    def _cap(self, r, drafts=None):
        """Tokens r's next round may emit: up to max_tokens, to the thinking budget's and the loop guard's first possible
        forced token, and to the first draft that would end the reply (an end token, the one-call stop token) or, with
        tool_choice forcing pending, close the thinking (draft i is emitted as token i + 1). drafts: a copy round's
        (else the DSpark drafts)."""
        cap = max(1, r.max_tokens - r.n_out)
        if r.thinking and self.think_end is not None:
            if r.budget is not None:
                cap = min(cap, max(1, r.budget - r.n_think))
            if self.loop_guard:
                cap = min(cap, loop_room(r.tail))
        end = self.think_end if r.force and r.thinking else None
        for i, d in enumerate((self.peek.spec_drafts(r.slot.i) if drafts is None else drafts)[:cap - 1]):
            if d in self.eos or d == r.stop_tok or d == end:
                return i + 1
        return cap

    def _spec_round(self, rs):
        """One DSpark round over the decoding sequences: verify, accept, roll back, draft the next round. (P168: one
        request alone whose reply repeats its history: a copy round instead.)"""
        t0 = time.perf_counter()
        B = len(rs)
        if self._prev is not None:
            x = self.rstats[self._prev[0]]
            x[2] += 1e3 * (t0 - self._prev[1])
            x[3] += 1
        if self.copy:
            if B == 1 and self._copy_round(rs[0], t0):
                return
            nd = [r for r in rs if r.cp_nodraft]  # a copy that went on made no drafts: DSpark's first
            for r in nd:
                r.cp_nodraft = False
                if len(r.slot.hist) + self.block > self.ctx:
                    self.eng.spec_depth(r.slot.i, self._depth(len(r.slot.hist)))
            if nd:
                self.eng.spec_draft([r.slot.i for r in nd])
        caps = [self._cap(r) for r in rs]
        pos = [len(r.slot.hist) for r in rs]  # the last tokens' positions; after the round ≤ pos + cap
        room = all(p + c + self.block <= self.ctx for p, c in zip(pos, caps))
        slots = [r.slot.i for r in rs]
        outs = self.eng.spec_round(slots, room, caps)
        if not room:  # near a slot's end: the next drafts only as deep as the context lets them
            for r, p, toks in zip(rs, pos, outs):
                self.eng.spec_depth(r.slot.i, self._depth(p + len(toks)))
            self.eng.spec_draft(slots)
        x = self.rstats.setdefault(B, [0, 0.0, 0.0, 0])
        x[0] += 1
        x[1] += 1e3 * (time.perf_counter() - t0)
        self._prev = (B, t0)
        for r, toks in zip(rs, outs):
            r.slot.hist.append(r.last)
            r.slot.hist += toks[:-1]
            self._take(r, toks)
            self._redraft(r, toks[-1])

    def _copy_sync(self, s):
        t = time.perf_counter()
        self.copy.sync(s)
        self.copy.t["chunk"] += time.perf_counter() - t
        self.copy.c["chunk"] += 1

    def _copy_round(self, r, t0):
        """P168: r decodes alone — when its reply repeats its history (copy_draft.Copier), one round verifying the
        source's next tokens (cut to the round's cap and the context) → True; else False (a DSpark round)."""
        cp, t1 = self.copy, time.perf_counter()
        m = cp.propose(r)  # (source, width + 1 tokens)
        if m:
            src, prop = m
            p = len(r.slot.hist)  # the last token's position
            cap = self._cap(r, prop)
            k = min(len(prop) - 1, cap - 1, self.ctx - 1 - p)
        if not m or k < MIN_COPY:
            if m:
                cp.stop(r)
            cp.t["miss"] += time.perf_counter() - t1
            cp.c["miss"] += 1
            return False
        cp.t["hit"] += time.perf_counter() - t1
        cp.c["hit"] += 1
        # follow: the source's token after the copy — when the model's next token is it too, the copy goes on (the
        # engine makes no DSpark drafts); not at the cap (the server may end the reply or change that token there)
        follow = prop[k] if k + 1 < cap else -1
        room = p + k + 1 + self.block <= self.ctx  # (else the next drafts only as deep as the context lets them)
        toks = self.eng.spec_copy(r.slot.i, prop[:k], cap, follow, room)
        go = follow >= 0 and len(toks) == k + 1 and toks[-1] == follow
        if not room and not go:
            self.eng.spec_depth(r.slot.i, self._depth(p + len(toks)))
            self.eng.spec_draft([r.slot.i])
        r.cp_nodraft = go
        x = self.rstats.setdefault(1, [0, 0.0, 0.0, 0])
        x[0] += 1
        x[1] += 1e3 * (time.perf_counter() - t0)
        self._prev = (1, t0)
        r.slot.hist.append(r.last)
        r.slot.hist += toks[:-1]
        cp.after(r, src, k, toks, go)
        self._take(r, toks)
        self._redraft(r, toks[-1])
        return True

    def _inject(self, r):
        """Forced tokens T after the last token: [last] + T[:-1] run like a prompt chunk, T[-1] is the new last.
        DSpark: [last] + T in one step, whose last row gives the token after T (as the first token of a prompt)."""
        s, T = r.slot, r.inj
        p = len(s.hist)
        if self.block:
            ids = [r.last] + T
            if r.penalised:  # a spec sequence's last token is counted with its round: here, with the forced ones
                self.eng.pen_add(s.i, ids)
            self.eng.step(ids, list(range(p, p + len(ids))), [s.i], [0, len(ids)], False)
            s.hist += ids
            t = self._begin(r, p + len(ids))
            r.state, r.inj = "decode", None
            self._emit(r, list(T))
            if r.state == "decode":
                self._deliver(r, t)
                self._redraft(r, t)
            return
        ids = [r.last] + T[:-1]
        self.eng.fill(ids, list(range(p, p + len(ids))), [s.i], [0, len(ids)])
        s.hist += ids
        if r.penalised:  # forced tokens count as generated ones (the sampled ones were counted when drawn)
            self.eng.pen_add(s.i, T)
        r.last, r.state, r.inj = T[-1], "decode", None
        self._emit(r, list(T))

    def _emit(self, r, toks):
        if r.state == "done":
            return
        if r.prefix_out:  # tool_choice forcing, thinking off: the forced tokens ended the prompt; they go out first
            toks, r.prefix_out = list(r.prefix_out) + list(toks), None
        out, fin = [], None
        for t in toks:
            if t in self.eos:  # not sent, but a generated token (usage counts it, as vLLM)
                r.n_out += 1
                fin = "stop"
                break
            out.append(t)
            r.n_out += 1
            if t == r.stop_tok:  # one call, nothing after it
                fin = "stop"
                break
            if r.n_out >= r.max_tokens:
                fin = "length"
                break
        r.push(out, fin)
        if fin:
            self._finish(r, fin)

    def _finish(self, r, why):
        r.state = "done"
        if r in self.running:
            self.running.remove(r)
        r.slot.req, r.slot.used = None, time.time()
        dt = time.time() - (r.t_first or time.time())
        tps = (r.n_out - 1) / dt if r.n_out > 1 and dt > 0 else 0.0
        st = self.stats
        st["requests"] += 1
        st["prompt"] += len(r.prompt)
        st["cached"] += r.cached
        st["out"] += r.n_out
        ttft = (r.t_first - r.t0) if r.t_first else float("nan")
        via = f" {r.reuse}" if r.cached else ""
        line = (f"#{r.id}{' ' + r.label if r.label else ''} · prompt {len(r.prompt):,} (reused {r.cached:,}{via}) · "
                f"first token {ttft:.2f} s · output {r.n_out:,} · {tps:.1f} tok/s · {why}"
                + (f" · thinking closed ({r.forced})" if r.forced else ""))
        (qlog.ok if why in ("stop", "length") else qlog.warn)("req", line)
        if r.marks and r.t_place:  # server.prefill_marks: time to first token of a cold prompt of each length
            want = [L for L in (8192, 32768, 131072, 262144, 524288) if r.cached < L < r.marks[-1][0]] + [r.marks[-1][0]]
            pts = [next(m for m in r.marks if m[0] >= L) for L in want]
            qlog.info("prefill", f"#{r.id} read from {r.cached:,}: " + " · ".join(
                f"{p:,} {t - r.t_place:.1f} s ({(p - r.cached) / max(t - r.t_place, 1e-9):,.0f} tok/s)" for p, t in pts))

    # ---- loop ------------------------------------------------------------------------------------------------
    def run(self):
        """The engine loop. Returns after drain() once no request is left (or the deadline cut them)."""
        while True:
            try:
                block = not (self.waiting or self.running or self.draining)
                while True:  # (kvguard: idle, the loop still wakes for the memory watch)
                    r = self.inbox.get(block=block, timeout=self.kv.poll_s if block and self.kv is not None else None)
                    if r is not None:  # None: a wake-up (drain)
                        self.waiting.append(r)
                    block = False
            except queue.Empty:
                pass
            if self.kv is not None:
                self.kv.poll(self)
            if self.draining and time.time() >= self.deadline:
                for r in list(self.waiting) + list(self.running):
                    r.cancelled = True  # cut below, like a client that went away (finish: abort)
            for r in [r for r in self.waiting + self.running if r.cancelled and r.state != "done"]:
                if r in self.waiting:
                    self.waiting.remove(r)
                    r.state = "done"
                    r.push([], "abort")
                else:
                    r.push([], "abort")
                    self._finish(r, "abort")
            if self.draining and not (self.waiting or self.running):
                return
            while self.waiting:
                got = self._place(self.waiting[0])
                if got is None:  # refused (it got its error)
                    self.waiting.pop(0)
                    continue
                if not got:
                    break
                self.running.append(self.waiting.pop(0))
            pf = [r for r in self.running if r.state == "prefill"]
            if pf:
                r = self._next_prefill(pf)
                r.t_chunk = time.time()
                self._kv_call(self._prefill_chunk, r)
                self._prev = None
            inj = [r for r in self.running if r.state == "inject"]
            # (rows: [last] + T[:-1]; DSpark [last] + T)
            for r in self._kv_grow([(r, len(r.inj) + (1 if self.block else 0)) for r in inj]):
                self._kv_call(self._inject, r)
                self._prev = None
            # (rows: one a sequence; a DSpark round 1 + its drafts, ≤ 1 + block; P168: a copy round ≤ copy_rows)
            dec = [r for r in self.running if r.state == "decode"]
            n = max(1 + self.block, self.copy.wmax + 1) if self.copy and len(dec) == 1 else 1 + self.block
            dec = self._kv_grow([(r, n) for r in dec])
            if dec:
                self._kv_call(self._spec_round if self.block else self._round, dec)

    # ---- memory (kvguard) ---------------------------------------------------------------------------------------
    def _kv_grow(self, items):
        """[(request, rows it is about to run)] → the requests that go on (kvguard: a reply past its reservation takes
        more, or ends with "length" when the pool is full)"""
        return self.kv.grow(self, items) if self.kv is not None else [r for r, _ in items]

    def _kv_call(self, fn, arg):
        """fn(arg); with kvguard the engine's "KV pool full" ends that slot's request instead of the server"""
        return self.kv.call(self, fn, arg) if self.kv is not None else fn(arg)
