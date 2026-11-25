#!/usr/bin/env python3
"""xfer_trace.py — reads an E55_COMM_TRACE file (comm.h; written by xfer_prof.py --trace) and prints per traced window:
the wall, the comm stream's busy time, per transfer class (kind + bytes: the P094 / P100 lane collectives of an
8192-row chunk) count · MiB · time on the stream · GB/s · queued behind others · the time s_ waited in its joins, the
waits by the phase of the layer where s_ joined, and the bytes moved under the MoE experts (marks X → Y).
  python3 xfer_trace.py trace.rank0 [--layers] [--window LABEL]"""
import argparse
import collections

ap = argparse.ArgumentParser()
ap.add_argument("file")
ap.add_argument("--window", default="")
ap.add_argument("--layers", action="store_true", help="also per layer: phase spans and waits")
a = ap.parse_args()

NAMES = {("G", 7340032): "gather attn (HCA)", ("G", 3670016): "gather attn (DSA)",
         ("G", 33554432): "o exchange (HCA)", ("G", 16777216): "o exchange (DSA)",
         ("Z", 33554432): "o exchange (HCA)", ("Z", 16777216): "o exchange (DSA)",  # (P176 puts)
         ("G", 6471680): "gather ffn", ("G", 2244608): "gather ffn (P136)", ("G", 4227072): "gather ffn (dense)",
         ("R", 8388608): "RS shared / dense", ("R", 4194304): "RS latent"}
recs = []
for ln in open(a.file):
    k, lab, what, v, t0, t1, t2 = ln.rstrip("\n").split("\t")
    recs.append((k, lab, what, int(v), float(t0), float(t1), float(t2)))
labels = [x for x in dict.fromkeys(r[1] for r in recs) if not a.window or x == a.window]


def phase_of(t, marks):
    """the phase of s_ at time t from the marks [(time, what, layer)]"""
    ph = "?"
    for tm, w, L in marks:
        if tm > t:
            break
        ph = {"L": "fronts+cores", "C": "own side", "F": "ffn tp", "X": "experts", "Y": "after experts"}[w] + f" L{L}"
    return ph


for lab in labels:
    R = [(i, r) for i, r in enumerate(recs) if r[1] == lab]
    calls = {i: r for i, r in R if r[0] in "AFRGZ"}
    joins = [r for i, r in R if r[0] == "J"]
    marks = sorted((r[4], r[2], r[3]) for i, r in R if r[0] == "M")
    t_beg = min(r[4] for i, r in R)
    t_end = max(max(r[4], r[5], r[6]) for i, r in R)
    busy = 0.0
    last = -1e30
    for c in sorted(calls.values(), key=lambda c: c[5]):
        s, e = max(c[5], last), c[6]
        if e > s:
            busy += e - s
        last = max(last, e)
    ex = []  # experts spans
    xs = None
    for tm, w, L in marks:
        if w == "X":
            xs = tm
        elif w == "Y" and xs is not None:
            ex.append((xs, tm))
            xs = None
    print(f"== {lab}: wall {t_end - t_beg:.1f} ms · comm stream busy {busy:.1f} ms · {len(calls)} calls · "
          f"{sum(c[3] for c in calls.values()) / 2**20:.0f} MiB · experts {sum(e - s for s, e in ex):.1f} ms")
    cls = collections.OrderedDict()
    for i, c in sorted(calls.items()):
        n = NAMES.get((c[0], c[3]), f"{c[0]} {c[3] / 2**20:.2f} MiB")
        d = cls.setdefault(n, dict(n=0, b=0, dur=0.0, q=0.0, wait=0.0, under=0.0, nj=0))
        d["n"] += 1
        d["b"] += c[3]
        d["dur"] += c[6] - c[5]
        d["q"] += c[5] - c[4]
        d["under"] += sum(max(0.0, min(c[6], e) - max(c[5], s)) for s, e in ex)
    wph = collections.Counter()
    for j in joins:
        c = calls.get(j[3])
        if c is None:
            continue
        n = NAMES.get((c[0], c[3]), f"{c[0]} {c[3] / 2**20:.2f} MiB")
        w = j[5] - j[4]
        cls[n]["wait"] += w
        cls[n]["nj"] += 1
        wph[(n, phase_of(j[4], marks).split(" L")[0])] += w
    print(f"  {'class':22s} {'n':>4s} {'MiB':>6s} {'stream ms':>9s} {'GB/s':>6s} {'queued':>8s} {'s_ wait':>8s}"
          f" {'under ex':>8s}")
    for n, d in cls.items():
        gbs = d["b"] / d["dur"] / 1e6 if d["dur"] > 0 else 0
        print(f"  {n:22s} {d['n']:4d} {d['b'] / 2**20:6.0f} {d['dur']:9.1f} {gbs:6.1f} {d['q']:8.1f} {d['wait']:8.1f}"
              f" {d['under']:8.1f}")
    print(f"  s_ waits total {sum(d['wait'] for d in cls.values()):.1f} ms; by class · phase of the join:")
    for (n, ph), w in sorted(wph.items(), key=lambda x: -x[1]):
        if w >= 0.05:
            print(f"    {w:7.1f} ms  {n} · {ph}")
    if a.layers:
        lm = collections.defaultdict(dict)
        for tm, w, L in marks:
            lm[L][w] = tm
        Ls = sorted(lm)
        print("  layer: L→C (tails, fronts, cores) · C→F (own side, ffn fronts) · F→X · X→Y (experts) · Y→next L")
        for L in Ls:
            m = lm[L]
            nxt = lm.get(L + 1, {}).get("L", t_end)
            seg = [m.get("C", 0) - m.get("L", 0), m.get("F", 0) - m.get("C", 0), m.get("X", 0) - m.get("F", 0),
                   m.get("Y", 0) - m.get("X", 0), nxt - m.get("Y", 0)]
            print(f"   L{L:2d} " + " · ".join(f"{x:6.2f}" for x in seg) + f" = {nxt - m.get('L', 0):6.2f}")
