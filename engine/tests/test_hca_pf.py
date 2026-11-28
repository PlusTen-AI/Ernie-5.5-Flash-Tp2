#!/usr/bin/env python3
"""test_hca_pf.py — P210 hca_attn_1p (prefill rule 11-28: one online softmax over a row's keys in 16-key tiles, its own
accumulation order) against the current prefill kernel of hca_attn_fused (P088 / P150 / P172 by the call's key count),
on one real HCA layer's VHA U / sink / compressor (rank 0) and random q / kv / compressor rows (test_hca.py's cases).
  python3 engine/tests/test_hca_pf.py --model /models/ERNIE-5.5-Flash-Preview [--do gate,noise,det,chunk,time]
  gate   patch on but a decode / verify step (step kind off): bitwise the default kernel (M 1 / 4 / 128 at 131K)
  noise  o and zp of both kernels vs an fp64 reference (sampled rows: the reference's rounding points q bf16, K = the
         cache values, o bf16), and P210 vs current over all rows in bf16 ulps — P210 must sit at the current's noise
  det    P210 twice: o, zp bitwise
  chunk  the fused chain (pre → attention → all-reduce stand-in → post, caches carried) of one 8192-row prompt piece
         as one call, as 4 × 2048, as 1 / 3 / 60 / 1984 / 2048 / 4096 rows and as model.cc's 4 lanes of 2048 rows:
         every row's o / zp (and KV rows, wo_a input, caches) bitwise
  time   2048 rows (an 8192-row chunk's lane) at --positions: ms (CUDA events, median, current / P210 interleaved),
         TFLOPS, × the bf16 MMA floor"""
import argparse
import json
import math
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
ap.add_argument("--layer", type=int, default=3)
ap.add_argument("--do", default="gate,noise,det,chunk,time")
ap.add_argument("--positions", default="0,131072,524288", help="noise / det / time: the sequence's first position")
ap.add_argument("--chunk-positions", default="0,131072,524288")
ap.add_argument("--rows", type=int, default=2048, help="noise / det / time: rows of the call")
ap.add_argument("--kscale", default="1,4", help="noise: key scales (sharper softmax at larger values)")
ap.add_argument("--sample", type=int, default=16, help="noise: rows of the fp64 reference at the start / middle / end")
ap.add_argument("--runs", type=int, default=25)
a = ap.parse_args()
m = ext.load("hca_pf", ["src/kernels/attn/hca.cu", "src/model/patches.cc"])
dev = "cuda"
P210 = "hca_attn_1p"

wm = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))["weight_map"]


def get(name):
    with safe_open(os.path.join(a.model, wm[name]), framework="pt", device="cpu") as f:
        return f.get_tensor(name).cuda()


pre_ = f"model.layers.{a.layer}.self_attn."
W = dict(U=get(pre_ + "vha_postmix_U").float()[:32].contiguous(), V=get(pre_ + "vha_postmix_V").float()[:32].contiguous(),
         sink=get(pre_ + "core_attention.attn_sink").float()[:32].contiguous(),
         ape=get(pre_ + "core_attention.compressor.ape").float().contiguous(),
         nw=get(pre_ + "core_attention.compressor.norm.weight").bfloat16().contiguous())
INV = m.inv_freq()
WS = torch.empty(1 << 28, dtype=torch.float32, device=dev)  # 1 GiB, the engine's attention workspace
PEAK = 94e12  # bf16 MMA, GB10 (test_hca.py)
DRAM = 250e9  # achievable DRAM read+write, B/s (est.)
L2BW = 1.6e12  # L2 gather (P172 notes: 1.5–1.7 TB/s)


def kernel(which):
    """which: 'cur' — a decode-kind call sees the default dispatch; 'p210' — a prefill step with P210 on"""
    m.set_prefill(which == "p210")
    m.patch(P210, which == "p210")


def meta_of(seqs):
    """seqs: [(slot, p0, n)] → RowMeta int32 [M, 5] (pos, slot, r0, p0, pend)"""
    rows, r0 = [], 0
    for s, p0, n in seqs:
        rows += [(p0 + i, s, r0, p0, p0 + n - 1) for i in range(n)]
        r0 += n
    return torch.tensor(rows, dtype=torch.int32, device=dev)


def nkeys(p):
    return (p + 1) // 128 + min(p + 1, 128)


class Case:
    """one sequence (slot 0), rows p0 … p0 + n − 1; caches as after p0 positions (test_hca.py's Case)"""

    def __init__(self, p0, n, seed=0, kscale=1.0):
        self.p0, self.M = p0, n
        g = torch.Generator(device=dev).manual_seed(seed)
        maxc = (p0 + n) // 128 + 1
        self.meta = meta_of([(0, p0, n)])
        self.max_keys = max(nkeys(p) for p in range(p0, p0 + n))
        self.ring = torch.zeros(1, 128, 584, dtype=torch.uint8, device=dev)
        self.comp = torch.zeros(1, maxc, 584, dtype=torch.uint8, device=dev)
        self.state = torch.zeros(1, 128, 1024, dtype=torch.float32, device=dev)
        nc = p0 // 128
        if nc:
            cm = meta_of([(0, 0, 1)]).repeat(nc, 1)
            cm[:, 0] = torch.arange(nc, device=dev, dtype=torch.int32) * 128
            kv = (torch.randn(nc, 512, generator=g, device=dev) * kscale).bfloat16()
            m.kv_rows(kv, cm, INV, self.comp[0, :nc])
        lo = max(0, p0 - 128)
        if p0 > lo:
            pm = meta_of([(0, lo, p0 - lo)])
            kv = (torch.randn(p0 - lo, 512, generator=g, device=dev) * kscale).bfloat16()
            rr = torch.empty(p0 - lo, 584, dtype=torch.uint8, device=dev)
            m.kv_rows(kv, pm, INV, rr)
            self.ring[0, torch.arange(lo, p0, device=dev) % 128] = rr
        g0 = p0 // 128 * 128
        if p0 > g0:
            st = torch.randn(p0 - g0, 1024, generator=g, device=dev)
            st[:, 512:] = st[:, 512:] * 2 + W["ape"][: p0 - g0]
            self.state[0, : p0 - g0] = st
        self.q = (torch.randn(n, 32 * 512, generator=g, device=dev) * 0.7).bfloat16()
        self.kv = (torch.randn(n, 512, generator=g, device=dev) * kscale).bfloat16()
        self.ckv = torch.randn(n, 1024, generator=g, device=dev)
        self.ckv[:, 512:] *= 2
        self.other = torch.randn(n, 16, 512, generator=g, device=dev) * 0.05  # the other rank's VHA partial

    def fresh(self):
        n, nan = self.M, float("nan")
        return dict(ring=self.ring.clone(), comp=self.comp.clone(), state=self.state.clone(),
                    rows=torch.full((n, 584), 0xA5, dtype=torch.uint8, device=dev),
                    o=torch.full((n, 32 * 512), nan, dtype=torch.bfloat16, device=dev),
                    zp=torch.full((n, 16, 512), nan, dtype=torch.float32, device=dev),
                    q8=torch.full((4, n, 4096), 0xA5, dtype=torch.uint8, device=dev),
                    sc=torch.full((4, n, 32), 0xA5, dtype=torch.uint8, device=dev))


def do_pre(c, t):
    m.pre(c.kv, c.ckv, W["ape"], W["nw"], c.meta, INV, t["rows"], t["ring"], t["comp"], t["state"], WS)


def do_attn(c, t, which, rows=None):
    kernel(which)
    sl = slice(None) if rows is None else rows
    m.attn_fused(c.q[sl], t["rows"], c.meta[sl].contiguous(), t["ring"], t["comp"], t["state"], W["sink"], INV, W["U"],
                 c.max_keys, t["o"][sl], t["zp"][sl], WS)


def bits_equal(x, y):
    return torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))


def diff(tb, tp, keys):
    bad = []
    for k in keys:
        if not bits_equal(tb[k], tp[k]):
            d = (tb[k].float() - tp[k].float()).abs()
            bad.append(f"{k} differs ({(d > 0).sum().item()} el, max {d.nan_to_num(float('inf')).max().item():.3g})")
    return bad


def timeit(f):
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda._sleep(200_000)
    e0.record()
    f()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1)


# ---- fp64 reference of one row (ernie_ref.py hca(): q bf16 after prep, K = the cache values, softmax with the sink) --
def dq_rows(r8):
    """[n, 584] uint8 KV rows → [n, 512] float64 (nope e4m3 · 2^(e − 127) per 64 dims, rope bf16)"""
    n = r8.shape[0]
    nope = r8[:, :448].contiguous().view(torch.float8_e4m3fn).double().view(n, 7, 64)
    sc = torch.pow(2.0, r8[:, 576:583].double() - 127.0)
    rope = r8[:, 448:576].contiguous().view(torch.bfloat16).double()
    return torch.cat([(nope * sc[:, :, None]).view(n, 448), rope], 1)


def keys_of(c, t, r):
    p = c.p0 + r
    ncomp, nwin = (p + 1) // 128, min(p + 1, 128)
    win = []
    for tt in range(p - nwin + 1, p + 1):
        win.append(t["rows"][tt - c.p0] if tt >= c.p0 else t["ring"][0, tt % 128])
    return dq_rows(torch.cat([t["comp"][0, :ncomp], torch.stack(win)], 0))


def q_ref(c, rs):
    """the prepped q of rows rs (fp32 as the reference, rounded to bf16)"""
    x = c.q[rs].float().view(len(rs), 32, 512)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    pos = (c.p0 + torch.tensor(rs, device=dev)).float()
    ang = pos[:, None] * INV[None, :]  # fp32 products: fl(pos · inv[i])
    cs, sn = torch.cos(ang)[:, None, :], torch.sin(ang)[:, None, :]
    e, o = x[..., 448::2], x[..., 449::2]
    x = x.clone()
    x[..., 448::2], x[..., 449::2] = e * cs - o * sn, o * cs + e * sn
    return x.bfloat16()


def o_ref(c, t, rs):
    qb = q_ref(c, rs).double()
    sink = W["sink"].double()
    out = []
    for i, r in enumerate(rs):
        K = keys_of(c, t, r)
        s = qb[i] @ K.t() / math.sqrt(512)  # [32, nk]
        mx = torch.maximum(s.amax(-1), sink)
        e = torch.exp(s - mx[:, None])
        den = e.sum(-1) + torch.exp(sink - mx)
        out.append((e @ K) / den[:, None])
    return torch.stack(out)  # [n, 32, 512]


def ulps(x, y):
    """bf16 ulp distance (ordered integers of the bit patterns)"""
    def od(v):
        i = v.contiguous().view(torch.int16).int()
        return torch.where(i < 0, -(i & 0x7FFF), i)
    return (od(x) - od(y)).abs()


# ---- checks ---------------------------------------------------------------------------------------------------------
def gate_check():
    """P210 on in a decode / verify / draft step (kind off): the default kernel, bitwise"""
    good = True
    for n in (1, 4, 128):
        c = Case(131072, n, seed=3)
        t = c.fresh()
        do_pre(c, t)
        outs = {}
        for nm, on in (("default", False), ("patch on, decode kind", True)):
            m.set_prefill(False)
            m.patch(P210, on)
            t["o"].fill_(float("nan")), t["zp"].fill_(float("nan"))
            m.attn_fused(c.q, t["rows"], c.meta, t["ring"], t["comp"], t["state"], W["sink"], INV, W["U"], c.max_keys,
                         t["o"], t["zp"], WS)
            torch.cuda.synchronize()
            outs[nm] = dict(o=t["o"].clone(), zp=t["zp"].clone())
        bad = diff(outs["default"], outs["patch on, decode kind"], ["o", "zp"])
        good &= not bad
        print(f"[gate M={n} at 131072] patch on, step kind decode vs default: " + ("bitwise equal" if not bad else "; ".join(bad)),
              flush=True)
    m.patch(P210, False)
    return good


def noise_check():
    good = True
    for ks in [float(x) for x in a.kscale.split(",")]:
        for p0 in [int(x) for x in a.positions.split(",")]:
            n = a.rows
            c = Case(p0, n, seed=5, kscale=ks)
            t = c.fresh()
            do_pre(c, t)
            outs = {}
            for which in ("cur", "p210"):
                t["o"].fill_(float("nan")), t["zp"].fill_(float("nan"))
                do_attn(c, t, which)
                torch.cuda.synchronize()
                outs[which] = dict(o=t["o"].clone().view(n, 32, 512), zp=t["zp"].clone())
            S = a.sample
            rs = sorted(set(list(range(S)) + list(range(n // 2 - S // 2, n // 2 + S // 2)) + list(range(n - S, n))))
            ref = o_ref(c, t, rs)
            U = W["U"].double()
            zref = torch.einsum("thd,hr->trd", ref, U)
            st = {}
            for which in ("cur", "p210"):
                eo = outs[which]["o"][rs].double() - ref
                ez = outs[which]["zp"][rs].double() - zref
                st[which] = dict(omax=eo.abs().max().item(), orms=eo.pow(2).mean().sqrt().item(),
                                 zmax=ez.abs().max().item(), zrms=ez.pow(2).mean().sqrt().item(),
                                 nan=bool(torch.isnan(outs[which]["o"].float()).any() or torch.isnan(outs[which]["zp"]).any()))
            u = ulps(outs["cur"]["o"], outs["p210"]["o"])
            tot = u.numel()
            zd = (outs["cur"]["zp"] - outs["p210"]["zp"]).abs()
            orms = ref.pow(2).mean().sqrt().item()
            ok = (not st["p210"]["nan"] and st["p210"]["orms"] <= 1.1 * st["cur"]["orms"] + 1e-7 * orms and
                  st["p210"]["omax"] <= 1.5 * st["cur"]["omax"] + 1e-7)
            good &= ok
            print(f"[noise {n} rows at {p0}, kscale {ks:g}] vs fp64 ({len(rs)} rows, rms |o_ref| {orms:.3g}): o max / rms "
                  f"current {st['cur']['omax']:.3g} / {st['cur']['orms']:.3g} · P210 {st['p210']['omax']:.3g} / "
                  f"{st['p210']['orms']:.3g}; zp max / rms {st['cur']['zmax']:.3g} / {st['cur']['zrms']:.3g} · "
                  f"{st['p210']['zmax']:.3g} / {st['p210']['zrms']:.3g} · P210 vs current (all rows): o bf16 ulps = 0 "
                  f"{(u == 0).sum().item() / tot:.4f}, ≤ 1 {(u <= 1).sum().item() / tot:.5f}, ≤ 2 "
                  f"{(u <= 2).sum().item() / tot:.6f}, max {u.max().item()}; zp max |Δ| {zd.max().item():.3g} · "
                  + ("OK" if ok else "NOT at the noise floor"), flush=True)
            del c, t, outs
            torch.cuda.empty_cache()
    return good


def det_check():
    good = True
    for p0 in [int(x) for x in a.positions.split(",")]:
        c = Case(p0, a.rows, seed=9)
        t = c.fresh()
        do_pre(c, t)
        outs = []
        for _ in range(2):
            t["o"].fill_(float("nan")), t["zp"].fill_(float("nan"))
            do_attn(c, t, "p210")
            torch.cuda.synchronize()
            outs.append(dict(o=t["o"].clone(), zp=t["zp"].clone()))
        bad = diff(outs[0], outs[1], ["o", "zp"])
        # the same rows in reverse order in one call (rows are independent of their neighbours and of their CTA)
        rv = torch.arange(c.M - 1, -1, -1, device=dev)
        o2 = torch.full_like(t["o"], float("nan")), torch.full_like(t["zp"], float("nan"))
        kernel("p210")
        m.attn_fused(c.q[rv].contiguous(), t["rows"], c.meta[rv].contiguous(), t["ring"], t["comp"], t["state"],
                     W["sink"], INV, W["U"], c.max_keys, o2[0], o2[1], WS)
        torch.cuda.synchronize()
        bad += [f"reversed rows: {x}" for x in diff(outs[0], dict(o=o2[0][rv], zp=o2[1][rv]), ["o", "zp"])]
        good &= not bad
        print(f"[det {c.M} rows at {p0}] P210 run vs run, and the rows reversed: " + ("bitwise equal" if not bad else "; ".join(bad)),
              flush=True)
    return good


def chain(c, t, pieces):
    """the fused chain over consecutive pieces of c's rows, each its own step (caches carried): o, zp, rows, q8, sc"""
    parts = {k: [] for k in ("rows", "o", "zp", "q8", "sc")}
    r0 = 0
    for n in pieces:
        sub = object.__new__(Case)
        sub.p0, sub.M = c.p0 + r0, n
        sub.meta = meta_of([(0, c.p0 + r0, n)])
        sub.max_keys = max(nkeys(p) for p in range(c.p0 + r0, c.p0 + r0 + n))
        sub.q, sub.kv, sub.ckv = c.q[r0:r0 + n], c.kv[r0:r0 + n], c.ckv[r0:r0 + n]
        ts = dict(ring=t["ring"], comp=t["comp"], state=t["state"],
                  rows=torch.full((n, 584), 0xA5, dtype=torch.uint8, device=dev),
                  o=torch.full((n, 32 * 512), float("nan"), dtype=torch.bfloat16, device=dev),
                  zp=torch.full((n, 16, 512), float("nan"), dtype=torch.float32, device=dev),
                  q8=torch.full((4, n, 4096), 0xA5, dtype=torch.uint8, device=dev),
                  sc=torch.full((4, n, 32), 0xA5, dtype=torch.uint8, device=dev))
        do_pre(sub, ts)
        do_attn(sub, ts, "p210")
        z = ts["zp"] + c.other[r0:r0 + n]
        m.post(ts["o"], z, W["V"], sub.meta, INV, ts["rows"], sub.ckv, W["ape"], ts["ring"], ts["comp"], ts["state"],
               ts["q8"], ts["sc"])
        for k in parts:
            parts[k].append(ts[k])
        r0 += n
    torch.cuda.synchronize()
    out = {k: torch.cat(v, 1 if k in ("q8", "sc") else 0) for k, v in parts.items()}
    out.update(ring=t["ring"], comp=t["comp"], state=t["state"])
    return out


def lanes(c, t, lane):
    """model.cc's lane convention in one step: per lane meta / q / kv / ckv / o / zp at the lane's row (RowMeta.r0
    absolute: the attention reads the step's KV rows through the base pointer), every lane's pre + attention, then every
    lane's post"""
    ckv = torch.full_like(c.ckv, float("nan"))
    L = [(r0, min(lane, c.M - r0)) for r0 in range(0, c.M, lane)]
    kernel("p210")
    for r0, n in L:
        ckv[r0:r0 + n] = c.ckv[r0:r0 + n]
        meta = c.meta[r0:r0 + n]
        m.pre(c.kv[r0:r0 + n], ckv, W["ape"], W["nw"], meta, INV, t["rows"][r0:r0 + n], t["ring"], t["comp"],
              t["state"], WS)
        m.attn_fused(c.q[r0:r0 + n], t["rows"], meta, t["ring"], t["comp"], t["state"], W["sink"], INV, W["U"],
                     c.max_keys, t["o"][r0:r0 + n], t["zp"][r0:r0 + n], WS)
    z = t["zp"] + c.other
    for r0, n in L:
        m.post(t["o"][r0:r0 + n], z[r0:r0 + n], W["V"], c.meta[r0:r0 + n], INV, t["rows"][r0:r0 + n],
               ckv[r0:r0 + n], W["ape"], t["ring"], t["comp"], t["state"],
               torch.empty((4, n, 4096), dtype=torch.uint8, device=dev), torch.empty((4, n, 32), dtype=torch.uint8, device=dev))
    torch.cuda.synchronize()
    return t


def live_state(t, end):
    """the compressor's open-group state rows that are live after a step ending at position `end` (end % 128 of them);
    the rest of the 128-row buffer is dead (never read) and holds whatever the last group left there"""
    out = dict(t)
    out["state"] = t["state"][:, : end % 128]
    return out


def chunk_check():
    good = True
    keys = ["rows", "o", "zp", "q8", "sc", "ring", "comp", "state"]
    for p0 in [int(x) for x in a.chunk_positions.split(",")]:
        c = Case(p0, 8192, seed=11)
        ref = live_state(chain(c, c.fresh(), [8192]), p0 + 8192)
        for pieces in ([2048] * 4, [1, 3, 60, 1984, 2048, 4096], [1, 3, 60, 1984, 2048, 4000]):
            if sum(pieces) != 8192:  # a chain that ends inside a group: compare the live state of its own one-call run
                c2 = Case(p0, sum(pieces), seed=11)
                r2 = live_state(chain(c2, c2.fresh(), [sum(pieces)]), p0 + sum(pieces))
                g2 = live_state(chain(c2, c2.fresh(), pieces), p0 + sum(pieces))
                bad = diff(r2, g2, keys)
                good &= not bad
                print(f"[chunk {sum(pieces)} rows at {p0}] one call vs pieces {pieces}: "
                      + ("bitwise equal" if not bad else "; ".join(bad)), flush=True)
                del c2, r2, g2
                continue
            got = live_state(chain(c, c.fresh(), pieces), p0 + 8192)
            bad = diff(ref, got, keys)
            good &= not bad
            print(f"[chunk 8192 rows at {p0}] one call vs pieces {pieces}: " + ("bitwise equal" if not bad else "; ".join(bad)),
                  flush=True)
        got = live_state(lanes(c, c.fresh(), 2048), p0 + 8192)  # (the lanes' wo_a inputs are lane-major: not compared)
        bad = diff(ref, {k: got[k] for k in ("rows", "o", "zp", "ring", "comp", "state")}, ["rows", "o", "zp", "ring", "comp", "state"])
        good &= not bad
        print(f"[chunk 8192 rows at {p0}] one call vs model.cc lanes of 2048: " + ("bitwise equal" if not bad else "; ".join(bad)),
              flush=True)
        del c, ref
        torch.cuda.empty_cache()
    return good


def time_check():
    for p0 in [int(x) for x in a.positions.split(",")]:
        c = Case(p0, a.rows, seed=5)
        t = c.fresh()
        do_pre(c, t)
        nk = sum(nkeys(p) for p in range(p0, p0 + c.M))
        flops = nk * 32 * 512 * 4
        fl = flops / PEAK * 1e3
        dram = c.M * (32 * 512 * 2 * 2 + 16 * 512 * 4) / DRAM * 1e3  # q in, o + zp out
        l2 = nk * 584 / L2BW * 1e3  # every row's keys from L2
        ts = {"cur": [], "p210": []}
        for _ in range(a.runs):
            for which in ("cur", "p210"):
                ts[which].append(timeit(lambda: do_attn(c, t, which)))
        ms = {k: statistics.median(v) for k, v in ts.items()}
        print(f"[time {c.M} rows at {p0}: {nk / c.M:.0f} keys a row] ms current {ms['cur']:.3f} → P210 {ms['p210']:.3f} · "
              f"TFLOPS {flops / ms['cur'] / 1e9:.1f} → {flops / ms['p210'] / 1e9:.1f} · floors: MMA {fl:.3f} ms, DRAM "
              f"(q, o, zp) {dram:.3f}, L2 keys {l2:.3f} → P210 at {ms['p210'] / max(fl, dram, l2):.2f}× the bound",
              flush=True)
        del c, t
        torch.cuda.empty_cache()
    return True


ok = True
todo = [x.strip() for x in a.do.split(",") if x.strip()]
for name, f in (("gate", gate_check), ("noise", noise_check), ("det", det_check), ("chunk", chunk_check), ("time", time_check)):
    if name in todo:
        ok &= f()
m.set_prefill(False)
m.patch(P210, False)
print("[hca_pf]", "all OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
