#!/usr/bin/env python3
"""test_hca.py — the HCA attention chain (kernels/attn/hca.cu: q prep, KV rows, compressor, attention, ring, VHA) base vs
patches, on one real HCA layer's weights (VHA U/V, sink, compressor ape/norm, rank 0) and realistic random q/kv/compressor
rows at the real shapes; caches filled through the base kernels.
  python3 engine/tests/test_hca.py --model /models/ERNIE-5.5-Flash-Preview --table /work/cache/table.tsv
Per shape: every output of the chain (prepped q, KV rows, compressed entries, state, ring, o, VHA partial, wo_a input)
compared bit for bit, then ms per stage (CUDA events, median; decode: L2 flushed before every run, then the stage's
activations read back — KV caches cold, activations L2-hot as after the previous kernel of a real step)."""
import argparse
import json
import os
import statistics
import sys

import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ext  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--table", default="")
ap.add_argument("--layer", type=int, default=3)
ap.add_argument("--only", default="", help="comma list of case names")
ap.add_argument("--runs", type=int, default=25)
ap.add_argument("--base-only", action="store_true")
ap.add_argument("--det", action="store_true", help="determinism: big prefill calls repeated and against 4096-row calls")
ap.add_argument("--prefill", action="store_true",
                help="P088 / P089: M 1024 / 4096 × position 0 / 16384, current default vs the patches: every output bitwise, ms per kernel")
ap.add_argument("--stages", default="pre,attn_fused,post", help="--prefill: the stages timed")
ap.add_argument("--shapes", default="", help="--prefill: subset, e.g. 1024@0,4096@16384")
ap.add_argument("--lanes", action="store_true", help="--prefill: also the model.cc lane convention (4 × 1024 rows) with the patches")
ap.add_argument("--long", action="store_true",
                help="P150 / P172: attn_fused, --rows rows of one sequence at --positions, current vs patch (--long-patch: "
                     "the patch switched; current = every other patch at its default): o / zp bitwise, ms (CUDA events, "
                     "median, current / patch interleaved), TFLOPS vs the bf16 MMA floor; then a mixed-sequence call and 4 "
                     "lanes (model.cc convention) at 262144 bitwise")
ap.add_argument("--long-patch", default="hca_attn_long2",
                help="--long: the patch compared (hca_attn_long: P150 vs P088; hca_attn_long3: P180 vs the default P172)")
ap.add_argument("--positions", default="0,16384,131072,524288,1040000")
ap.add_argument("--rows", type=int, default=2048)
ap.add_argument("--G", type=int, default=0, help="--long: P150 chunks per work item (0: the default)")
ap.add_argument("--minch", type=int, default=0, help="--long: P150 from rows of this many chunks (0: the default)")
ap.add_argument("--no-extra", action="store_true", help="--long: skip the mixed / lanes checks")
a = ap.parse_args()
m = ext.load("hca", ["src/kernels/attn/hca.cu", "src/model/patches.cc"])
PATCHES = ["hca_vha_row", "hca_attn_tc", "hca_attn_dec", "hca_compress_dec", "hca_attn_ws"]  # drop-in patches (same stage list); hca_fuse has its own stage list

wm = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))["weight_map"]


def get(name):
    with safe_open(os.path.join(a.model, wm[name]), framework="pt", device="cpu") as f:
        return f.get_tensor(name).cuda()


pre = f"model.layers.{a.layer}.self_attn."
W = dict(U=get(pre + "vha_postmix_U").float()[:32].contiguous(), V=get(pre + "vha_postmix_V").float()[:32].contiguous(),
         sink=get(pre + "core_attention.attn_sink").float()[:32].contiguous(),
         ape=get(pre + "core_attention.compressor.ape").float().contiguous(),
         nw=get(pre + "core_attention.compressor.norm.weight").bfloat16().contiguous())
INV = m.inv_freq()
dev = "cuda"


def meta_of(seqs):
    """seqs: [(slot, p0, n)] → RowMeta int32 [M, 5] (pos, slot, r0, p0, pend)"""
    rows, r0 = [], 0
    for s, p0, n in seqs:
        rows += [(p0 + i, s, r0, p0, p0 + n - 1) for i in range(n)]
        r0 += n
    return torch.tensor(rows, dtype=torch.int32, device=dev)


class Case:
    def __init__(self, name, seqs, seed=0):
        self.name, self.seqs = name, seqs
        g = torch.Generator(device=dev).manual_seed(seed)
        S = max(s for s, _, _ in seqs) + 1
        ctx = max(p0 + n for _, p0, n in seqs)
        maxc = ctx // 128 + 1
        self.meta = meta_of(seqs)
        M = self.meta.shape[0]
        self.M = M
        self.max_keys = max((p + 1) // 128 + min(p + 1, 128) for s, p0, n in seqs for p in range(p0, p0 + n))
        # caches before the step: compressed entries 0 … p0/128 − 1, ring = the last 128 positions, state = the open group
        self.ring = torch.zeros(S, 128, 584, dtype=torch.uint8, device=dev)
        self.comp = torch.zeros(S, maxc, 584, dtype=torch.uint8, device=dev)
        self.state = torch.zeros(S, 128, 1024, dtype=torch.float32, device=dev)
        for s, p0, n in seqs:
            nc = p0 // 128
            if nc:
                cm = meta_of([(s, 0, 1)]).repeat(nc, 1)
                cm[:, 0] = torch.arange(nc, device=dev, dtype=torch.int32) * 128
                kv = torch.randn(nc, 512, generator=g, device=dev).bfloat16()
                m.kv_rows(kv, cm, INV, self.comp[s, :nc])
            lo = max(0, p0 - 128)
            if p0 > lo:
                pm = meta_of([(s, lo, p0 - lo)])
                kv = torch.randn(p0 - lo, 512, generator=g, device=dev).bfloat16()
                rr = torch.empty(p0 - lo, 584, dtype=torch.uint8, device=dev)
                m.kv_rows(kv, pm, INV, rr)
                self.ring[s, torch.arange(lo, p0, device=dev) % 128] = rr
            g0 = p0 // 128 * 128
            if p0 > g0:
                st = torch.randn(p0 - g0, 1024, generator=g, device=dev)
                st[:, 512:] = st[:, 512:] * 2 + W["ape"][: p0 - g0]
                self.state[s, : p0 - g0] = st
        self.q = (torch.randn(M, 32 * 512, generator=g, device=dev) * 0.7).bfloat16()
        self.kv = torch.randn(M, 512, generator=g, device=dev).bfloat16()
        self.ckv = torch.randn(M, 1024, generator=g, device=dev)
        self.ckv[:, 512:] *= 2
        self.other = torch.randn(M, 16, 512, generator=g, device=dev) * 0.05  # the other rank's VHA partial

    def fresh(self):
        """per-run tensors: inputs that are written in place, caches, outputs"""
        M = self.M
        nan = float("nan")  # outputs and the workspace poisoned: a stage that skips a write or reads stale data shows
        t = dict(q=self.q.clone(), ring=self.ring.clone(), comp=self.comp.clone(), state=self.state.clone(),
                 rows=torch.full((M, 584), 0xA5, dtype=torch.uint8, device=dev),
                 o=torch.full((M, 32 * 512), nan, dtype=torch.bfloat16, device=dev),
                 zp=torch.full((M, 16, 512), nan, dtype=torch.float32, device=dev),
                 q8=torch.full((4, M, 4096), 0xA5, dtype=torch.uint8, device=dev),
                 sc=torch.full((4, M, 32), 0xA5, dtype=torch.uint8, device=dev))
        WS[: min(WS.numel(), 1 << 26)].fill_(nan)
        return t


WS = torch.empty(1 << 28, dtype=torch.float32, device=dev)  # 1 GiB, the engine's attention workspace


def stages_base(c, t):
    """attn_hca's HCA kernels as model.cc calls them (base): name → callable"""
    cache = (t["ring"], t["comp"], t["state"])
    return [
        ("q_prep", lambda: m.q_prep(t["q"], c.meta, INV)),
        ("kv_rows", lambda: m.kv_rows(c.kv, c.meta, INV, t["rows"])),
        ("compress", lambda: m.compress(c.ckv, W["ape"], W["nw"], c.meta, INV, *cache)),
        ("attn", lambda: m.attn(t["q"], t["rows"], c.meta, *cache, W["sink"], c.max_keys, t["o"], WS)),
        ("ring_store", lambda: m.ring_store(t["rows"], c.meta, *cache)),
        ("vha_partial", lambda: m.vha_partial(t["o"], 512, W["U"], t["zp"])),
        ("allreduce", lambda: t.__setitem__("z", t["zp"] + c.other)),
        ("vha_out", lambda: m.vha_out(t["o"], t["z"], W["V"], c.meta, INV, t["q8"], t["sc"])),
    ]


def stages_fused(c, t):
    """P042 hca_fuse: the three launches model.cc makes when it is on (q is read raw, not prepped in place)"""
    cache = (t["ring"], t["comp"], t["state"])
    return [
        ("pre", lambda: m.pre(c.kv, c.ckv, W["ape"], W["nw"], c.meta, INV, t["rows"], *cache, WS)),
        ("attn_fused", lambda: m.attn_fused(t["q"], t["rows"], c.meta, *cache, W["sink"], INV, W["U"], c.max_keys,
                                            t["o"], t["zp"], WS)),
        ("allreduce", lambda: t.__setitem__("z", t["zp"] + c.other)),
        ("post", lambda: m.post(t["o"], t["z"], W["V"], c.meta, INV, t["rows"], c.ckv, W["ape"], *cache, t["q8"],
                                t["sc"])),
    ]


def setp(on):
    for n in PATCHES:
        m.patch(n, on)


def compare(tb, tp, keys):
    bad = []
    for k in keys:
        if not torch.equal(tb[k].view(torch.uint8) if tb[k].dtype != torch.uint8 else tb[k],
                           tp[k].view(torch.uint8) if tp[k].dtype != torch.uint8 else tp[k]):
            x, y = tb[k].float(), tp[k].float()
            d = (x - y).abs()
            bad.append(f"{k} differs ({(d > 0).sum().item()} el, max {d.max().item():.3g})")
    return bad


def run(stages):
    for _, f in stages:
        f()


def timeit(f, cold, runs, warm=()):
    """median ms; cold: L2 flushed, then the activations (warm: written by the previous kernels in a real step) read
    back into L2 — the caches stay cold"""
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(runs):
        if cold:
            m.flush()
            for x in warm:
                x.view(torch.uint8).max()
        torch.cuda._sleep(200_000)  # let the host queue the timed work
        e0.record()
        f()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    return statistics.median(ts)


def outputs(t):
    return {k: t[k] for k in ("q", "rows", "comp", "state", "ring", "o", "zp", "q8", "sc")}


CASES = [
    Case("decode M=1 ctx 64", [(0, 64, 1)]),
    Case("decode M=1 ctx 9400", [(0, 9400, 1)]),
    Case("decode M=1 ctx 16511 (closes a group)", [(0, 16511, 1)]),
    Case("decode M=4 ctx ~9400", [(0, 9400, 1), (1, 9401, 1), (2, 9531, 1), (3, 9450, 1)]),
    Case("decode M=4 two close a group", [(0, 9471, 1), (1, 383, 1), (2, 9531, 1), (3, 127, 1)]),
    Case("draft M=8 closes a group mid-block", [(0, 9468, 8)]),
    Case("draft M=4 one sequence ctx 9400", [(0, 9400, 4)]),
    Case("decode M=8 ctx ~9400", [(i, 9400 + 37 * i, 1) for i in range(8)]),
    Case("draft M=8 one sequence ctx 9400", [(0, 9400, 8)]),
    Case("draft M=16 two sequences ctx ~9400", [(0, 9400, 8), (1, 9500, 8)]),
    Case("decode M=1 ctx 40000 (2 chunks)", [(0, 40000, 1)]),
    Case("decode M=2 ctx ~100000 (4 chunks)", [(0, 100000, 1), (1, 99990, 1)]),
    Case("mixed M=302 (decode + chunked prefill)", [(0, 9400, 1), (1, 40000, 1), (2, 5000, 300)]),
    Case("prefill 4096 at 0", [(0, 0, 4096)]),
    Case("prefill 4096 at 16384", [(0, 16384, 4096)]),
]
def det_check(p0, n, split=4096, reps=3, ws=False):
    """one n-row fused chain at p0, `reps` times (bitwise equal?) and as consecutive calls of ≤ split rows (equal?)"""
    c = Case(f"det {n} at {p0}", [(0, p0, n)], seed=11)
    keys = ["rows", "comp", "state", "ring", "o", "zp", "q8", "sc"]
    setp(True)
    m.patch("hca_attn_ws", ws)
    outs = []
    for _ in range(reps):
        t = c.fresh()
        run(stages_fused(c, t))
        torch.cuda.synchronize()
        outs.append(t)
    rep = [compare(outs[0], o, keys) for o in outs[1:]]
    t = c.fresh()  # the same rows as consecutive calls (caches carried over)
    parts = {k: [] for k in ("rows", "o", "zp", "q8", "sc")}
    for r0 in range(0, n, split):
        m_ = min(split, n - r0)
        sub = object.__new__(Case)
        sub.M, sub.meta = m_, meta_of([(0, p0 + r0, m_)])
        sub.max_keys = max((p + 1) // 128 + min(p + 1, 128) for p in range(p0 + r0, p0 + r0 + m_))
        sub.kv, sub.ckv, sub.other = c.kv[r0:r0 + m_], c.ckv[r0:r0 + m_], c.other[r0:r0 + m_]
        ts = dict(q=c.q[r0:r0 + m_].clone(), ring=t["ring"], comp=t["comp"], state=t["state"],
                  rows=torch.full((m_, 584), 0xA5, dtype=torch.uint8, device=dev),
                  o=torch.full((m_, 32 * 512), float("nan"), dtype=torch.bfloat16, device=dev),
                  zp=torch.full((m_, 16, 512), float("nan"), dtype=torch.float32, device=dev),
                  q8=torch.full((4, m_, 4096), 0xA5, dtype=torch.uint8, device=dev),
                  sc=torch.full((4, m_, 32), 0xA5, dtype=torch.uint8, device=dev))
        run(stages_fused(sub, ts))
        torch.cuda.synchronize()
        for k in parts:
            parts[k].append(ts[k])
    sp = dict(ring=t["ring"], comp=t["comp"], state=t["state"], rows=torch.cat(parts["rows"]), o=torch.cat(parts["o"]),
              zp=torch.cat(parts["zp"]), q8=torch.cat(parts["q8"], 1), sc=torch.cat(parts["sc"], 1))
    spl = compare(outs[0], sp, keys)
    good = not any(rep) and not spl
    print(f"[det {n} rows at {p0}{' P070' if ws else ''}] {reps} runs: " + ("bitwise equal" if not any(rep) else "; ".join(sum(rep, []))) +
          f" · vs calls of ≤ {split} rows: " + ("bitwise equal" if not spl else "; ".join(spl)), flush=True)
    return good


def lanes_check(p0, n, rebase, lane=4096):
    """one step's rows as row lanes (pointer views at the lane's first row, launches in order) against one call"""
    c = Case(f"lanes {n} at {p0}", [(0, p0, n)], seed=11)
    setp(True)
    full = c.fresh()
    run(stages_fused(c, full))
    t = c.fresh()
    for r0 in range(0, n, lane):
        m_ = min(lane, n - r0)
        sub = object.__new__(Case)
        sub.M, sub.max_keys = m_, c.max_keys
        sub.meta = c.meta[r0:r0 + m_].clone()
        if rebase:
            sub.meta[:, 2] -= r0  # RowMeta.r0: the sequence's first row relative to this lane's pointers
        sub.kv, sub.ckv, sub.other = c.kv[r0:r0 + m_], c.ckv[r0:r0 + m_], c.other[r0:r0 + m_]
        ts = dict(q=t["q"][r0:r0 + m_], ring=t["ring"], comp=t["comp"], state=t["state"], rows=t["rows"][r0:r0 + m_],
                  o=t["o"][r0:r0 + m_], zp=t["zp"][r0:r0 + m_],
                  q8=torch.empty((4, m_, 4096), dtype=torch.uint8, device=dev),
                  sc=torch.empty((4, m_, 32), dtype=torch.uint8, device=dev))
        run(stages_fused(sub, ts))
    torch.cuda.synchronize()
    bad = compare(full, t, ["rows", "comp", "state", "ring", "o", "zp"])
    print(f"[lanes {n} rows at {p0}, lanes of {lane}, RowMeta.r0 {'rebased to the lane' if rebase else 'as the step'}] vs one call: "
          + ("bitwise equal" if not bad else "; ".join(bad)), flush=True)
    return not bad


def lanes_modelcc(p0, n, lane=4096, ws=False):
    """model.cc's P075 lane convention: per lane meta / q / kv / ckv / o / zp at the lane's row, hca_pre and the attention
    reading the step's KV / compressor rows through the base pointers (RowMeta.r0 absolute), hca_post on the lane's rows;
    order: every lane's pre + attention, then every lane's post (the all-reduce between)"""
    c = Case(f"lanes {n} at {p0}", [(0, p0, n)], seed=11)
    setp(True)
    m.patch("hca_attn_ws", ws)
    res = []
    for _ in range(2):
        t = c.fresh()
        ckv = torch.full_like(c.ckv, float("nan"))
        L = [(r0, min(lane, n - r0)) for r0 in range(0, n, lane)]
        for r0, m_ in L:
            ckv[r0:r0 + m_] = c.ckv[r0:r0 + m_]  # the lane's compressor rows (lin_forward_f32 at row o)
            meta = c.meta[r0:r0 + m_]
            m.pre(c.kv[r0:r0 + m_], ckv, W["ape"], W["nw"], meta, INV, t["rows"][r0:r0 + m_], t["ring"], t["comp"],
                  t["state"], WS)
            m.attn_fused(t["q"][r0:r0 + m_], t["rows"], meta, t["ring"], t["comp"], t["state"], W["sink"], INV, W["U"],
                         c.max_keys, t["o"][r0:r0 + m_], t["zp"][r0:r0 + m_], WS)
        z = t["zp"] + c.other
        for r0, m_ in L:
            m.post(t["o"][r0:r0 + m_], z[r0:r0 + m_], W["V"], c.meta[r0:r0 + m_], INV, t["rows"][r0:r0 + m_],
                   ckv[r0:r0 + m_], W["ape"], t["ring"], t["comp"], t["state"], torch.empty((4, m_, 4096), dtype=torch.uint8, device=dev),
                   torch.empty((4, m_, 32), dtype=torch.uint8, device=dev))
        torch.cuda.synchronize()
        res.append(t)
    full = c.fresh()
    run(stages_fused(c, full))
    torch.cuda.synchronize()
    keys = ["rows", "comp", "state", "ring", "o", "zp"]
    b1, b2 = compare(full, res[0], keys), compare(res[0], res[1], keys)
    print(f"[lanes {n} rows at {p0}{' P070' if ws else ''}, model.cc convention] vs one call: " + ("bitwise equal" if not b1 else "; ".join(b1)) +
          " · run vs run: " + ("bitwise equal" if not b2 else "; ".join(b2)), flush=True)
    return not b1 and not b2


NEW = ["hca_attn_pf", "hca_pre_pf"]  # P088, P089


def prefill_check():
    """the prefill shapes (one sequence, caches holding p0 positions of history): the fused chain with every patch at its
    default except P088 / P089 (current) against the same with them on (patch), bit for bit; then ms per kernel"""
    shapes = [(1024, 0), (1024, 16384), (4096, 0), (4096, 16384)]
    if a.shapes:
        shapes = [tuple(int(x) for x in sh.split("@")) for sh in a.shapes.split(",")]
    good = True
    for M, p0 in shapes:
        c = Case(f"prefill {M} at {p0}", [(0, p0, M)])
        keys = ["rows", "comp", "state", "ring", "o", "zp", "q8", "sc"]
        tt = {}
        for nm, on in (("current", False), ("patch", True)):
            setp(True)
            for n in NEW:
                m.patch(n, on)
            t = c.fresh()
            run(stages_fused(c, t))
            torch.cuda.synchronize()
            tt[nm] = t
        bad = compare(tt["current"], tt["patch"], keys)
        good &= not bad
        ts = {nm: {k: [] for k in a.stages.split(",")} for nm in tt}  # current / patch interleaved run by run
        for _ in range(a.runs):
            for nm, on in (("current", False), ("patch", True)):
                setp(True)
                for n in NEW:
                    m.patch(n, on)
                st = dict(stages_fused(c, tt[nm]))
                for k in ts[nm]:
                    ts[nm][k].append(timeit(st[k], False, 1))
        ms = {nm: {k: statistics.median(v) for k, v in d.items()} for nm, d in ts.items()}
        print(f"[{c.name}] bitwise: " + ("yes (" + ", ".join(keys) + ")" if not bad else "NO: " + "; ".join(bad)) + " · ms " +
              " · ".join(f"{k} {ms['current'][k]:.3f} → {ms['patch'][k]:.3f}" for k in ms["current"]), flush=True)
    if a.lanes:
        for n in NEW:
            m.patch(n, True)
        for p0 in (0, 16384):
            good &= lanes_modelcc(p0, 4096, lane=1024, ws=True)
    setp(True)
    for n in NEW:
        m.patch(n, False)
    return good


PEAK = 94e12  # bf16 MMA, GB10


def long_check():
    """--long-patch (P172 / P150) against the current default without it on attn_fused: o and zp bit for bit, ms per
    call, TFLOPS"""
    m.long_tune(a.G, a.minch)
    LP = a.long_patch
    if LP == "hca_attn_long":  # P172 (default, ≥ 5 chunks) would take both sides
        m.patch("hca_attn_long2", False)
    good = True
    for p0 in [int(x) for x in a.positions.split(",")]:
        c = Case(f"attn_fused {a.rows} rows at {p0}", [(0, p0, a.rows)], seed=5)
        setp(True)
        m.patch(LP, False)
        t = c.fresh()
        st = dict(stages_fused(c, t))
        st["pre"]()  # the step's KV rows and closing groups' compressed entries (the default path)
        keys = (c.meta[:, 0].long() + 1) // 128 + torch.clamp(c.meta[:, 0].long() + 1, max=128)
        flops = keys.sum().item() * 32 * 1024 * 2
        out = {}
        for nm, on in (("current", False), ("patch", True), ("patch2", True)):
            m.patch(LP, on)
            t["o"].fill_(float("nan")), t["zp"].fill_(float("nan"))
            st["attn_fused"]()
            torch.cuda.synchronize()
            out[nm] = dict(o=t["o"].clone(), zp=t["zp"].clone())
        bad = compare(out["current"], out["patch"], ["o", "zp"]) + [f"rerun: {x}" for x in compare(out["patch"], out["patch2"], ["o", "zp"])]
        good &= not bad
        ts = {"current": [], "patch": []}
        for _ in range(a.runs):
            for nm, on in (("current", False), ("patch", True)):
                m.patch(LP, on)
                ts[nm].append(timeit(st["attn_fused"], False, 1))
        ms = {nm: statistics.median(v) for nm, v in ts.items()}
        fl = flops / PEAK * 1e3
        print(f"[{c.name}] bitwise: " + ("yes (o, zp; rerun equal)" if not bad else "NO: " + "; ".join(bad)) +
              f" · ms {ms['current']:.3f} → {ms['patch']:.3f} · TFLOPS {flops / ms['current'] / 1e9:.1f} → "
              f"{flops / ms['patch'] / 1e9:.1f} · floor {fl:.3f} ms ({ms['patch'] / fl:.2f}× floor)", flush=True)
        del t, st, out
    if not a.no_extra:
        # several sequences in one call (rows of 1 / 2 / 5 / 33 chunks: single- and multi-item rows, empty groups)
        c = Case("mixed", [(0, 131072, 700), (1, 3000, 200), (2, 40000, 300), (3, 1040000, 64)], seed=7)
        setp(True)
        t = c.fresh()
        st = dict(stages_fused(c, t))
        st["pre"]()
        out = {}
        for nm, on in (("current", False), ("patch", True)):
            m.patch(LP, on)
            t["o"].fill_(float("nan")), t["zp"].fill_(float("nan"))
            st["attn_fused"]()
            torch.cuda.synchronize()
            out[nm] = dict(o=t["o"].clone(), zp=t["zp"].clone())
        bad = compare(out["current"], out["patch"], ["o", "zp"])
        good &= not bad
        print(f"[mixed 4 sequences, M={c.M}] bitwise: " + ("yes (o, zp)" if not bad else "NO: " + "; ".join(bad)), flush=True)
        m.patch(LP, True)
        good &= lanes_modelcc(262144, 4096, lane=1024)  # 9 chunks: the long-context kernel
    setp(True)
    m.patch(LP, LP != "hca_attn_long3"), m.patch("hca_attn_long2", True)
    return good


only = [s.strip() for s in a.only.split(",") if s.strip()]
ok = True
if a.long:
    ok = long_check()
    print("[hca]", "all OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)
if a.prefill:
    ok = prefill_check()
    print("[hca]", "all OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)
if a.det:
    for p0, n in ((8192, 8192), (8192, 6000), (0, 8192), (5000, 16384)):
        for ws in (False, True):
            ok &= lanes_modelcc(p0, n, ws=ws)
    lanes_check(8192, 8192, False)  # expected to differ: the kernels index rows / ckv by RowMeta.r0
    ok &= lanes_check(8192, 8192, True)
    ok &= lanes_check(8192, 6000, True)
    for p0, n in ((8192, 8192), (8192, 6000), (0, 8192), (16384, 8192), (5000, 16384)):
        for ws in (False, True):
            ok &= det_check(p0, n, ws=ws)
    m.patch("hca_attn_ws", True)
    print("[hca]", "all OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)
for c in CASES:
    if only and not any(o in c.name for o in only):
        continue
    cold = c.M <= 16
    setp(False)
    tb = c.fresh()
    run(stages_base(c, tb))
    torch.cuda.synchronize()
    res = {}
    if not a.base_only:
        setp(True)
        tp = c.fresh()
        run(stages_base(c, tp))
        tf = c.fresh()
        run(stages_fused(c, tf))
        torch.cuda.synchronize()
        everything = ["q", "rows", "comp", "state", "ring", "o", "zp", "q8", "sc"]
        res["patch"] = compare(tb, tp, everything)
        res["fused"] = compare(tb, tf, everything[1:])
        if c.M <= 10:  # the P041 decode slices and the base pre stage (P043, P044 off)
            m.patch("hca_attn_dec", False), m.patch("hca_compress_dec", False)
            tq, tg = c.fresh(), c.fresh()
            run(stages_base(c, tq))
            run(stages_fused(c, tg))
            torch.cuda.synchronize()
            res["patch w/o P043"] = compare(tb, tq, everything)
            res["fused w/o P043/44"] = compare(tb, tg, everything[1:])
            setp(True)
        for k, v in res.items():
            ok &= not v
            print(f"[{c.name}] {k}: " + ("bit-identical (" + ", ".join(everything if k.startswith("patch") else everything[1:]) + ")"
                                        if not v else "; ".join(v)))
    chains = [("base", False, stages_base(c, tb))]
    if not a.base_only:
        chains += [("patch", True, stages_base(c, tp)), ("fused", True, stages_fused(c, tf))]
    for nm, on, st in chains:  # timed on the state after one full chain
        setp(on)
        tt = tb if nm == "base" else tp if nm == "patch" else tf
        warm = [tt["q"], c.kv, c.ckv, tt["rows"], tt["o"], tt["zp"], tt["z"]]
        times = {k: timeit(f, cold, a.runs, warm) for k, f in st if k != "allreduce"}
        tot = sum(times.values())
        print(f"[{c.name}] {nm:5s} ms: " + " · ".join(f"{k} {v:.3f}" for k, v in times.items()) +
              f" · chain {tot:.3f}")
print("[hca]", "all OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
