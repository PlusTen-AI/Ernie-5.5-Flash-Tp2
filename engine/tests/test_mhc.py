#!/usr/bin/env python3
"""test_mhc.py — mHC (kernels/mhc): the base against an fp64 rendering of its math, the patches (P050 mhc_dec,
P051 mhc_big, P072 mhc_fuse, P106 mhc_prefill, P107 mhc_rows_pdl, P164 mhc_verify, P192 mhc_verify64 (--v64)) against
the base bit for bit.
  python3 engine/tests/test_mhc.py --model /models/ERNIE-5.5-Flash-Preview --table /work/cache/table.tsv [--all]
Real mixers (mapping_proj fp32, bias, alphas, the fused RMSNorm weights) of layers 0–12; R bf16 with a realistic spread
(per-dim log-normal scale, per-stream scale, outlier dims); post / comb from a real pre.
Default (P164): decode / verify rows M = 1 … 32
  a. P164 vs the current default (P050 / P051 / P107; P164 off) bitwise — R', post, comb, inp — for pre, post + pre and
     layer-0 pre at M = 1, 2, 3, 5, 8, 9, 15, 16, 17, 24, 31, 32 (3 mixers), and with every forced R (rows a warp)
  b. µs per post + pre call at M = 1, 5, 8, 16, 32, current → P164 (24 mixers in turn: 38 MB of fp32 fn > L2; one event
     pair around the 24 calls, median of 9 rounds): each behind a 10 µs programmatic predecessor (lindec_spin: the
     call's fn copy and prologue run under it, as under the comm kernel before every mHC call in the engine; the spins
     subtracted) and back to back; × 86 a step. --tune: also each R (rows a warp chain)
--all: also
0. fn_b (layer 0's broadcast mixer) == torch's fn.view(24, 4, 4096).sum(1) on the GPU (vLLM-ERNIE), bit for bit
1. base vs fp64 math: post / comb close, R' and inp equal but for rare bf16 ties
2. patches vs base: R', post, comb, inp bitwise, M = 1, 2, 4, 8, 16, 17, 64, 65, 300, 512, 1024, 2048, 4096; pre,
   post + pre, layer-0 pre
3. time (post + pre, median per call, CUDA events): M = 512, 1024, 2048 current default → P106 (4 mixers in turn);
   M = 1, 4, 8 current default → P107 (24 mixers in turn: cold, 38 MB of fp32 mixers > L2)"""
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
ap.add_argument("--table", required=True)
ap.add_argument("--quick", action="store_true")
ap.add_argument("--tune", action="store_true", help="P164: also every R (rows a warp chain), behind")
ap.add_argument("--all", action="store_true", help="also the older sections (fn_b, fp64, every patch vs base, P106 / P107 times)")
ap.add_argument("--v64", action="store_true", help="P192 mhc_verify64: 33 … 64 rows vs P051, bitwise, and µs per call")
a = ap.parse_args()
m = ext.load("mhc", ["src/kernels/mhc/mhc.cu", "src/model/patches.cc", "src/core/weights.cc", "tests/bind/lindec_spin.cu"])
m.open(a.model, a.table)
sms, l2, smem_sm, smem_blk, clk = m.device()
print(f"[device] {sms} SMs · L2 {l2 >> 20} MB · smem {smem_sm >> 10} KB/SM, {smem_blk >> 10} KB/block · {clk / 1e6:.2f} GHz")
wm = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))["weight_map"]
files = {}


def get(name):
    f = wm[name]
    if f not in files:
        files[f] = safe_open(os.path.join(a.model, f), framework="pt", device="cpu")
    return files[f].get_tensor(name).cuda()


PATCHES = ("mhc_dec", "mhc_big", "mhc_fuse", "mhc_prefill", "mhc_rows_pdl", "mhc_verify", "mhc_verify64")
DEFAULT = ("mhc_dec", "mhc_big", "mhc_fuse")  # the state before P106 / P107
CURRENT = ("mhc_dec", "mhc_big", "mhc_fuse", "mhc_prefill", "mhc_rows_pdl")  # the state before P164


def only(*ps):
    for p in PATCHES:
        m.set_patch(p, p in ps)


layers = list(range(1, 13)) if not a.quick else [1, 2]
H = {(0, 0): m.upload(0, 0)}
for L in layers:
    for k in (0, 1):
        H[(L, k)] = m.upload(L, k)
hs = [H[(L, k)] for L in layers for k in (0, 1)]


def params(L, k):
    p = f"model.layers.{L}." + ("self_attention_hyper_connection." if k == 0 else "mlp_hyper_connection.")
    nw = get(f"model.layers.{L}." + ("input_layernorm.weight" if k == 0 else "post_attention_layernorm.weight"))
    return (get(p + "mapping_proj.weight").float(), get(p + "bias").float(),
            [get(p + f"alpha_{x}").float()[0] for x in ("pre", "post", "res")], nw)


def same(x, y):
    if x.dtype == torch.bfloat16:
        return torch.equal(x.view(torch.int16), y.view(torch.int16))
    return torch.equal(x.view(torch.int32), y.view(torch.int32))


def make_R(M, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    col = torch.exp(torch.randn(4096, device="cuda", generator=g) * 0.8)
    col[torch.randint(0, 4096, (12,), device="cuda", generator=g)] *= 30.0
    st = torch.tensor([0.6, 1.0, 1.7, 3.0], device="cuda")[:, None]
    return (torch.randn(M, 4, 4096, device="cuda", generator=g) * col * st * 0.05).bfloat16().reshape(M, 16384)


def make_out(M, seed):
    g = torch.Generator(device="cuda").manual_seed(seed + 7)
    return (torch.randn(M, 4096, device="cuda", generator=g) * 0.3).bfloat16()


def make_e(M, seed):  # layer 0: R = [e, e, e, e]
    g = torch.Generator(device="cuda").manual_seed(seed + 3)
    e = (torch.randn(M, 4096, device="cuda", generator=g) * 0.02).bfloat16()
    return e[:, None, :].expand(M, 4, 4096).reshape(M, 16384).contiguous()


def ref(what, L, k, out, R, post, comb):
    """fp64 rendering of mhc.h (what 0 pre, 1 post + pre, 2 layer-0 pre)."""
    fn, b, sc, nw = params(L, k)
    M = R.shape[0]
    Rd = R.double().view(M, 4, 4096)
    if what == 1:
        c = comb.double().view(M, 4, 4)
        r = post.double()[:, :, None] * out.double()[:, None, :] + torch.einsum("tij,tih->tjh", c, Rd)
        Rs = r.float().bfloat16()
    else:
        r, Rs = Rd, R.view(M, 4, 4096)
    if what == 2:
        y = r[:, 0] @ fn.double().view(4, 4096, 24).sum(0)
        f = 1 / (torch.sqrt(r[:, 0].pow(2).sum(-1) / 4096) + 1e-6)
    else:
        y = r.reshape(M, 16384) @ fn.double()
        f = 1 / (torch.sqrt(r.reshape(M, 16384).pow(2).sum(-1) / 16384) + 1e-6)
    mx = y * f[:, None]
    bd = b.double()
    pre = torch.sigmoid(mx[:, :4] * sc[0].double() + bd[:4]) + 1e-6
    po = 2 * torch.sigmoid(mx[:, 4:8] * sc[1].double() + bd[4:8])
    cm = torch.softmax((mx[:, 8:] * sc[2].double() + bd[8:]).view(M, 4, 4), -1) + 1e-6
    cm = cm / (cm.sum(-2, keepdim=True) + 1e-6)
    for _ in range(19):
        cm = cm / (cm.sum(-1, keepdim=True) + 1e-6)
        cm = cm / (cm.sum(-2, keepdim=True) + 1e-6)
    ol = (pre[:, :, None] * Rs.double()).sum(1)
    inp = (ol.float().bfloat16().double() * torch.rsqrt(ol.pow(2).mean(-1, keepdim=True) + 1e-6) * nw.double()).float().bfloat16()
    return Rs.reshape(M, 16384), po.float(), cm.reshape(M, 16).float(), inp


ok = True


def p164_section():
    """a. P164 vs the current default bitwise; b. µs per post + pre call, current → P164."""
    global ok
    for M in (1, 2, 3, 5, 8, 9, 15, 16, 17, 24, 31, 32):
        R = make_R(M, M)
        out = make_out(M, M)
        E = make_e(M, M)
        z4, z16 = torch.zeros(M, 4, device="cuda"), torch.zeros(M, 16, device="cuda")
        res = []
        for (L, k) in [(1, 0), (7, 1), (12, 0)]:
            h = H[(L, k)]
            only()
            _, p0, c0, _ = m.run(0, H[(L, 1 - k)], out, R, z4, z16)
            cases = [(0, h, R, p0, c0), (1, h, R, p0, c0)] + ([(2, H[(0, 0)], E, z4, z16)] if (L, k) == (1, 0) else [])
            for what, hh, RR, pp, cc in cases:
                only(*CURRENT)
                b = m.run(what, hh, out, RR, pp, cc)
                for rr in (0, 1, 2, 4, 8):
                    if rr and rr * 8 < min(M, 16):
                        continue
                    m.p164_rows(rr)
                    only(*CURRENT, "mhc_verify")
                    q = m.run(what, hh, out, RR, pp, cc)
                    exact = all(same(x, y) for x, y in zip(b, q))
                    ok &= exact
                    if not exact:
                        res.append(f"L{L}{'af'[k]} {('pre', 'post+pre', 'pre0')[what]} R={rr} DIFFERS {[same(x, y) for x, y in zip(b, q)]}")
                m.p164_rows(0)
        print(f"[P164 M={M}] pre, post+pre, layer-0 pre × 3 mixers × forced R: " + ("bit-identical" if not res else "; ".join(res)))
    print("[time] post + pre, µs per call (median of 9 rounds of 24 mixers in turn, one event pair a round), current →"
          " P164: behind a 10 µs predecessor (spins subtracted) · back to back; per step (× 86)")
    for M in (1, 5, 8, 16, 32):
        R = make_R(M, 100 + M)
        out = make_out(M, 100 + M)
        only()
        _, p0, c0, _ = m.run(0, hs[1], out, make_R(M, 200 + M), torch.zeros(M, 4, device="cuda"), torch.zeros(M, 16, device="cuda"))

        def tm(spin):
            return statistics.median(m.timed_batch(1, hs, out, R, p0, c0, 9, spin))
        only(*CURRENT)
        cb, cn = tm(10000), tm(0)
        only(*CURRENT, "mhc_verify")
        pb, pn = tm(10000), tm(0)
        tune = []
        if a.tune:
            for rr in (1, 2, 4, 8):
                if rr * 8 < min(M, 16):
                    continue
                m.p164_rows(rr)
                tune.append(f"R{rr} {tm(10000):.2f}")
            m.p164_rows(0)
        print(f"  M={M:<3d} behind: {cb:6.2f} → {pb:6.2f} µs ({cb * 86e-3:.2f} → {pb * 86e-3:.2f} ms a step) · back to back:"
              f" {cn:6.2f} → {pn:6.2f} µs ({cn * 86e-3:.2f} → {pn * 86e-3:.2f} ms)" + (" · tune behind: " + ", ".join(tune) if tune else ""))


def p192_section():
    """P192 mhc_verify64 (with P164) vs the default (P051 at 33 … 64 rows) bitwise — R', post, comb, inp — for pre,
    post + pre and layer-0 pre, 3 mixers, every forced R; then µs per post + pre call as p164_section."""
    global ok
    DEF = (*CURRENT, "mhc_verify")
    for M in (33, 40, 47, 48, 56, 63, 64):
        R = make_R(M, M)
        out = make_out(M, M)
        E = make_e(M, M)
        z4, z16 = torch.zeros(M, 4, device="cuda"), torch.zeros(M, 16, device="cuda")
        res = []
        for (L, k) in [(1, 0), (7, 1), (12, 0)]:
            h = H[(L, k)]
            only()
            _, p0, c0, _ = m.run(0, H[(L, 1 - k)], out, R, z4, z16)
            cases = [(0, h, R, p0, c0), (1, h, R, p0, c0)] + ([(2, H[(0, 0)], E, z4, z16)] if (L, k) == (1, 0) else [])
            for what, hh, RR, pp, cc in cases:
                only(*DEF)
                b = m.run(what, hh, out, RR, pp, cc)
                for rr in (0, 2, 4, 8):
                    if rr and rr * 8 < 16:
                        continue
                    m.p164_rows(rr)
                    only(*DEF, "mhc_verify64")
                    q = m.run(what, hh, out, RR, pp, cc)
                    exact = all(same(x, y) for x, y in zip(b, q))
                    ok &= exact
                    if not exact:
                        res.append(f"L{L}{'af'[k]} {('pre', 'post+pre', 'pre0')[what]} R={rr} DIFFERS {[same(x, y) for x, y in zip(b, q)]}")
                m.p164_rows(0)
        print(f"[P192 M={M}] pre, post+pre, layer-0 pre × 3 mixers × forced R: " + ("bit-identical" if not res else "; ".join(res)))
    print("[time] post + pre, µs per call (as P164's), default (P051) → P192: behind a 10 µs predecessor · back to back")
    for M in (40, 48, 56, 64):
        R = make_R(M, 100 + M)
        out = make_out(M, 100 + M)
        only()
        _, p0, c0, _ = m.run(0, hs[1], out, make_R(M, 200 + M), torch.zeros(M, 4, device="cuda"), torch.zeros(M, 16, device="cuda"))

        def tm(spin):
            return statistics.median(m.timed_batch(1, hs, out, R, p0, c0, 9, spin))
        only(*DEF)
        cb, cn = tm(10000), tm(0)
        only(*DEF, "mhc_verify64")
        pb, pn = tm(10000), tm(0)
        print(f"  M={M:<3d} behind: {cb:6.2f} → {pb:6.2f} µs ({cb * 86e-3:.2f} → {pb * 86e-3:.2f} ms a step) · back to back:"
              f" {cn:6.2f} → {pn:6.2f} µs ({cn * 86e-3:.2f} → {pn * 86e-3:.2f} ms)")


def old_sections():
    global ok
    # 0. broadcast mixer
    w0 = get("model.layers.0.self_attention_hyper_connection.mapping_proj.weight").float()  # [16384, 24]
    tb = w0.t().contiguous().view(24, 4, 4096).sum(dim=1)
    mb = m.fnb(H[(0, 0)], w0).t()
    r0 = same(tb.contiguous(), mb.contiguous())
    ok &= r0
    print(f"[fn_b] torch .view(24,4,4096).sum(1) vs ours: {'bit-identical' if r0 else 'DIFFERS: ' + str((tb - mb).abs().max().item())}")

    # 1–2. exactness
    Ms = (1, 2, 4, 8, 16, 17, 64, 65, 300, 512, 1024, 2048, 4096) if not a.quick else (1, 4, 300, 512)
    for M in Ms:
        R = make_R(M, M)
        out = make_out(M, M)
        E = make_e(M, M)
        z4, z16 = torch.zeros(M, 4, device="cuda"), torch.zeros(M, 16, device="cuda")
        for (L, k) in ([(1, 0), (7, 1), (12, 0)] if not a.quick else [(1, 0), (2, 1)]):
            h = H[(L, k)]
            only()
            _, p0, c0, _ = m.run(0, H[(L, 1 - k)], out, R, z4, z16)  # realistic post / comb (the previous pre)
            cases = [(0, h, R, p0, c0), (1, h, R, p0, c0)] + ([(2, H[(0, 0)], E, z4, z16)] if (L, k) == (1, 0) else [])
            for what, hh, RR, pp, cc in cases:
                only()
                b = m.run(what, hh, out, RR, pp, cc)
                only(*PATCHES)
                q = m.run(what, hh, out, RR, pp, cc)
                exact = all(same(x, y) for x, y in zip(b, q))
                ok &= exact
                line = f"[M={M} L{0 if what == 2 else L}{'af'[0 if what == 2 else k]} {('pre', 'post+pre', 'pre0')[what]}] " \
                       f"{'P164' if M <= 32 else 'P106' if M >= 256 and what == 1 else 'P072'} {'bit-identical' if exact else 'DIFFERS ' + str([same(x, y) for x, y in zip(b, q)])}"
                if M <= 300:  # base vs fp64 math (sanity of the definition)
                    Rr, pr, cr, ir = ref(what, 0 if what == 2 else L, 0 if what == 2 else k, out, RR, pp, cc)
                    dp = (b[1] - pr).abs().max().item()
                    dc = (b[2] - cr).abs().max().item()
                    nR = (b[0] != Rr).float().mean().item()
                    ni = (b[3] != ir).float().mean().item()
                    good = dp < 1e-5 and dc < 1e-5 and nR < 1e-3 and ni < 2e-3  # bf16 ties: fp64 vs the fp32 chain
                    ok &= good
                    line += f" · vs fp64: |Δpost| {dp:.1e} |Δcomb| {dc:.1e} R'≠ {nR:.1e} inp≠ {ni:.1e}{'' if good else ' FAIL'}"
                print(line)


    # 3. timing ---------------------------------------------------------------------------------------------------------
    def med(what, hl, M, reps, *patches):
        R = make_e(M, 100 + M) if what == 2 else make_R(M, 100 + M)
        out = make_out(M, 100 + M)
        only()
        _, p0, c0, _ = m.run(0, hs[1], out, make_R(M, 100 + M), torch.zeros(M, 4, device="cuda"), torch.zeros(M, 16, device="cuda"))
        only(*patches)
        return statistics.median(m.bench(what, hl, out, R, p0, c0, reps))


    print("[time] post + pre, median µs per call, CUDA events: current default → patch")
    for M, patch, hl in [(M, "mhc_prefill", hs[:4]) for M in (512, 1024, 2048)] + [(M, "mhc_rows_pdl", hs) for M in (1, 4, 8)]:
        reps = 25 if M <= 512 else 10
        b = med(1, hl, M, reps, *DEFAULT)
        q = med(1, hl, M, reps, *DEFAULT, patch)
        print(f"  M={M:<5d} {patch:12s} {b * 1e3:8.1f} → {q * 1e3:8.1f} µs ({(q - b) * 1e3:+.1f})  [{len(hl)} mixers in turn]")


if a.v64:
    p192_section()
else:
    p164_section()
if a.all:
    old_sections()
only()
print("[mhc]", "all OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
