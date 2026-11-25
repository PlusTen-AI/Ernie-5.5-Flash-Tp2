"""Copy drafts (decode.copy; engine spec_copy, P168 copy_draft) — the matcher, on rank 0's scheduler only (rank 1
replays the engine calls with the drafts as arguments). After Pico's docs/copy.md (MIT, the design only): when a reply
repeats text its slot's history already holds (a file rewritten with an edit, code echoed back, a tool call quoting
earlier text, the answer restating its reasoning), the tokens that followed that text there are offered as the round's
drafts in place of DSpark's. The engine verifies them like any draft — each verify row drawn as plain decoding draws
it, a copied token kept while the draw equals it — so a wrong copy costs rows, never a token.

- Source: the longest backward match of the reply's tail (its last token included) among the earlier occurrences of
  its last 4 tokens, at least `need` tokens long (decode.copy_match; each copy that kept fewer than MIN_COPY tokens
  doubles it, up to 8×, until one keeps more); the copy's tokens are the source's continuation (overlapping like LZ77:
  a repeating pattern repeats).
- Width (copied tokens a round): WIDTH0 first, 2w + 1 after a copy accepted whole (16 → 32 → 64 rows), at most
  decode.copy_rows − 1, back to WIDTH0 after a rejection. A copy goes on (no new lookup, no DSpark drafts in between)
  while the model's tokens equal the source's; when it stops, the next round is DSpark's (or a new match).
- Only when one request decodes: rounds shared by several requests stay exactly the rounds of before.

Index (per slot, over its history = the tokens its caches hold): table[hash(the 4 tokens ending at p)] = the latest p,
chain[p] = the previous p of that bucket — 4 B a position (array 'i') + a 4 MiB table per slot. Extended as the history
grows (each prefill chunk, each round); a replaced history (reset, a checkpoint restored or copied) is indexed again
from 0, at most `budget` positions a call. Candidates are always checked against the tokens themselves (a bucket mixes
4-grams, an index can lag), and nothing here depends on time: both ranks of a test reach the same decisions."""
from array import array

GRAM = 4
TAB_BITS = 20
TAB_MASK = (1 << TAB_BITS) - 1
MAX_CAND = 32       # candidates a lookup checks at most (the latest first)
WIDTH0 = 7          # copied tokens of a first copy round (8 rows: a wrong first copy costs about a DSpark round)
MIN_COPY = 4        # a copy round verifies at least this many tokens (fewer: DSpark's round)
SYNC_ROUND = 1 << 14   # positions indexed at most per round (a replaced long history catches up over rounds)
SYNC_CHUNK = 1 << 16   # … per prefill chunk


class Index:
    """4-token runs of one slot's history → their end positions (module doc)"""

    def __init__(self):
        self.tab = array("i", [-1]) * (1 << TAB_BITS)
        self.prev = array("i")
        self.ref, self.n = None, 0  # the history list indexed (positions < n)

    def nbytes(self):
        return self.tab.itemsize * len(self.tab) + self.prev.itemsize * len(self.prev)

    def sync(self, hist, budget):
        """index hist[n:] (≤ budget positions); a replaced list starts over"""
        if hist is not self.ref or len(hist) < self.n:
            if self.n:
                self.tab = array("i", [-1]) * (1 << TAB_BITS)
                self.prev = array("i")
            self.ref, self.n = hist, 0
        n, end = self.n, min(len(hist), self.n + budget)
        if end <= n:
            return
        tab, prev = self.tab, self.prev
        for p in range(n, min(end, GRAM - 1)):  # no 4-gram ends here
            prev.append(-1)
        a = max(n, GRAM - 1)
        for p, h in zip(range(a, end), zip(hist[a - 3:end - 3], hist[a - 2:end - 2], hist[a - 1:end - 1], hist[a:end])):
            b = hash(h) & TAB_MASK
            prev.append(tab[b])
            tab[b] = p
        self.n = end

    def lookup(self, hist, last, need, lmax):
        """the stream is hist + [last] (last at H = len(hist)): the end position p < H of the longest backward match
        (≥ need tokens, compared up to lmax) of its 4-token tail, the latest on ties → p or −1"""
        H, n = len(hist), self.n
        if H < GRAM - 1 or n < GRAM:
            return -1
        t3, t2, t1 = hist[H - 3], hist[H - 2], hist[H - 1]
        p = self.tab[hash((t3, t2, t1, last)) & TAB_MASK]
        prev, best, bp, c = self.prev, 0, -1, 0
        while 0 <= p < n and c < MAX_CAND:
            if hist[p] == last and hist[p - 1] == t1 and hist[p - 2] == t2 and hist[p - 3] == t3:
                L = GRAM
                while L < lmax and L <= p and hist[p - L] == hist[H - L]:
                    L += 1
                if L > best:
                    best, bp = L, p
                    if L >= lmax:
                        break
            q = prev[p]
            if q >= p:
                break
            p, c = q, c + 1
        return bp if best >= need else -1


class State:
    """a request's copy state: the width, the match a new copy needs, the source position of `last` while a copy goes
    on (else None)"""

    def __init__(self, need):
        self.width, self.need, self.src = WIDTH0, need, None


class Copier:
    def __init__(self, rows=64, match=8):
        if not 2 <= rows <= 64:
            raise ValueError(f"decode.copy_rows must be 2 … 64, got {rows}")
        if match < GRAM:
            raise ValueError(f"decode.copy_match must be ≥ {GRAM}, got {match}")
        self.wmax, self.match = rows - 1, match
        self.ix = {}  # slot index → Index
        # host time (s) and counts: lookups that found nothing / something, prefill-chunk syncs (tests, logs)
        self.t = {"miss": 0.0, "hit": 0.0, "chunk": 0.0}
        self.c = {"miss": 0, "hit": 0, "chunk": 0, "rounds": 0, "whole": 0}

    def index(self, slot):
        x = self.ix.get(slot.i)
        if x is None:
            x = self.ix[slot.i] = Index()
        return x

    def nbytes(self):
        return sum(x.nbytes() for x in self.ix.values())

    def sync(self, slot):
        """after a prefill chunk: index the slot's new history"""
        self.index(slot).sync(slot.hist, SYNC_CHUNK)

    def propose(self, r):
        """r decodes alone: (source position of r.last, up to width + 1 tokens that followed it there) or None"""
        st = r.cp
        if st is None:
            st = r.cp = State(self.match)
        hist, last = r.slot.hist, r.last
        H = len(hist)
        src = st.src
        if src is not None and not (0 <= src < H and hist[src] == last):
            src = st.src = None
        if src is None:
            x = self.index(r.slot)
            x.sync(hist, SYNC_ROUND)
            src = x.lookup(hist, last, st.need, max(2 * st.need, 32))
            if src < 0:
                return None
        out = []
        for i in range(min(st.width, self.wmax) + 1):  # positions H + 1 + i ← source src + 1 + i (LZ77 overlap)
            j = src + 1 + i
            out.append(hist[j] if j < H else last if j == H else out[j - H - 1])
        return src, out

    def after(self, r, src, k, toks, go):
        """a copy round of k tokens from source src emitted toks; go: the copy goes on (the next token was the source's).
        (The index takes the round's tokens now, so a later round without a copy only looks up.)"""
        st = r.cp
        self.c["rounds"] += 1
        self.index(r.slot).sync(r.slot.hist, SYNC_ROUND)
        if len(toks) == k + 1:  # accepted whole
            self.c["whole"] += 1
            st.width, st.need = min(2 * st.width + 1, self.wmax), self.match
            st.src = src + k + 1 if go else None
        else:  # back to the first width; a copy that gave fewer than MIN_COPY tokens doubles the match the next needs
            st.width, st.src = WIDTH0, None
            st.need = self.match if len(toks) - 1 >= MIN_COPY else min(2 * st.need, 8 * self.match)

    def stop(self, r):
        """the server changed r's last token (or started its drafts again): no copy goes on"""
        if r.cp is not None:
            r.cp.src = None
