#!/usr/bin/env python3
"""test_moe.py — the W4A8 MoE (kernels/moe/moe.cu) on a real layer, both ranks.
  python3 engine/tests/test_moe.py --model /models/ERNIE-5.5-Flash-Preview --table /work/cache/table.tsv [--layer 5]
     [--skip-ref] [--only moe_decode,…]
1. routing == vLLM-ERNIE's routing kernel, bitwise: ids in torch.topk(sorted=False) slot order, the 16-slot tree sum
2. rank 0 + rank 1 partials == the reference's expert sum (dequantized weights, MXFP8 activations, fp32) to fp32 order
3. a row's partial is bitwise the same at every M
4. patches (moe.cu P035–P037, P065–P069, P086) against the base, same inputs: bitwise equal; time per call (CUDA events,
   median), decode cold (L2 flushed before every call, a different routing each call). P066 (moe_fused: routing +
   latent quantization + experts) against route + act_quant (tests/bind/moe_aq.cu, the engine's kernel) + experts, timed
   on the reference dumps' real routing (/work/results/ref/v3) as DSpark-like blocks of 8 consecutive tokens. P086
   (moe_pf_dn) against the default prefill path (P037 + P065) at M = 4096 / 8192 on layer L's real routing (the
   reference's router on real hidden states, ref v3 dumps: doc_long + en_tools + ko_short = 4575 tokens, 8192 rows
   repeat doc_long), both ranks, bf16 and fp32 outputs bitwise; ms per call (median of 12, warm: the 1.7 GB expert
   set does not fit L2). P190 (moe_fused64, with P066) as P066 at 33 … 64 rows: bitwise vs the base, timed on real
   routing as 5–8 DSpark-like blocks (--only moe_fused64). P194 (moe_fused128, with P066) at 65 … 128 rows: bitwise vs
   the base and, per row, vs P066 on the rows' 8-row pieces (each sequence's block alone); timed against today's
   default at those M (route + act_quant + P037 / P065 / P086) on real routing as 9–16 blocks (--only moe_fused128).
   The timed parts read the ref v3 dumps' routing (--ref-dir, default /work/results/ref/v3: en_tools, doc_long)"""
import argparse
import json
import os
import sys

import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools", "ref"))
import ext  # noqa: E402
import ernie_ref as R  # noqa: E402

PATCHES = ["moe_decode", "route_fast", "moe_prefill", "moe_combine", "moe_fused", "moe_pf_dn",
           "moe_fused64", "moe_fused128"]  # this family's patches
REF = {"moe_combine": ["moe_prefill"], "moe_fused": ["moe_decode", "route_fast"],  # timed against the base with these on
       "moe_pf_dn": ["moe_prefill", "moe_combine"],  # (bitwise against the plain base; P086 against this default)
       "moe_fused64": ["moe_decode", "route_fast"],
       "moe_fused128": ["moe_decode", "route_fast", "moe_prefill", "moe_combine", "moe_pf_dn"]}  # today's M > 64
# P190 moe_fused64 (33 … 64 rows) runs with P066 on: the patches each one switches on (default: itself)
WITH = {"moe_fused64": ["moe_fused", "moe_fused64"], "moe_fused128": ["moe_fused", "moe_fused128"]}
# the fused patches' bitwise sizes and timed sizes (P066 ≤ 32, P190 33 … 64: DSpark verify of 5–8 sequences × 8 rows)
FUSED = {"moe_fused": ((1, 2, 3, 4, 5, 8, 9, 16, 17, 24, 32), (1, 4, 8, 16, 32)),
         "moe_fused64": ((33, 40, 47, 48, 56, 63, 64), (40, 48, 56, 64)),
         "moe_fused128": ((65, 72, 96, 112, 127, 128), (65, 96, 112, 128))}  # 9–16 sequences × 7 / 8 rows
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--table", required=True)
ap.add_argument("--layer", type=int, default=5)
ap.add_argument("--skip-ref", action="store_true", help="only the base-vs-patch part (4)")
ap.add_argument("--only", default="", help="patches to compare (default: all of the family)")
ap.add_argument("--ref-dir", default="/work/results/ref/v3", help="ref dumps with the real routing (timed parts)")
a = ap.parse_args()
m = ext.load("moe", ["src/kernels/moe/moe.cu", "tests/bind/moe_aq.cu", "src/core/weights.cc", "src/model/patches.cc"])
m.open(a.model, a.table)
wm = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))["weight_map"]
files = {}


def get(name):
    f = wm[name]
    if f not in files:
        files[f] = safe_open(os.path.join(a.model, f), framework="pt", device="cpu")
    return files[f].get_tensor(name).cuda()


def quant(x, g=32):
    """MXFP8 per g (R.act_q8's rule) → e4m3 bytes [M, K], UE8M0 bytes [M, K/g] (the kernels' input format)."""
    M_, K_ = x.shape
    xf = x.float().reshape(M_, K_ // g, g)
    e = torch.ceil(torch.log2(xf.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 448.0))
    q = R.e4m3(xf / torch.exp2(e)).to(torch.float8_e4m3fn).view(torch.uint8).reshape(M_, K_)
    return q.contiguous(), (e + 127).to(torch.uint8).reshape(M_, K_ // g).contiguous()


def route_ref(lg, bias):
    """vLLM-ERNIE _ernie5_5_routing_kernel (ernie5_5/nvidia/moe.py), exact: top 10 of s + bias (ties → lower id), slots
    in torch.topk(sorted=False) order (above the 10th value by id, then equal to it by id), tl.sum's tree over 16
    zero-padded slots, IEEE division."""
    s = torch.sigmoid(lg[:, :512]) + torch.sigmoid(lg[:, 512:])
    v, i = (s + bias).sort(dim=-1, descending=True, stable=True)
    ids = i[:, :10]
    ids = ids.gather(1, torch.where(v[:, :10] > v[:, 9:10], ids, ids + 512).argsort(-1))
    w = s.gather(1, ids)
    t = torch.nn.functional.pad(w, (0, 6))
    t = t[:, :8] + t[:, 8:]
    t = t[:, :4] + t[:, 4:]
    t = t[:, :2] + t[:, 2:]
    return ids.int(), w / (t[:, :1] + t[:, 1:]).clamp_min(1e-20) * scale[ids]


def base():
    for p in PATCHES:
        m.set_patch(p, False)


L = a.layer
mp = f"model.layers.{L}.block_sparse_moe."
h = [m.upload(r, L) for r in (0, 1)]
base()
torch.manual_seed(1)
wr = torch.cat([get(mp + "gate.weight").float(), get(mp + "gate.weight_1").float()], 0)
bias, scale = get(mp + "e_score_correction_bias").float(), get(mp + "gate.routed_scaling_factor_param").float()


def routed(MM, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    xx = (torch.randn(MM, 4096, device="cuda", generator=g) * 0.3).bfloat16()
    ids, w = m.route(xx.float() @ wr.t(), bias, scale)
    q, s = quant((torch.randn(MM, 2048, device="cuda", generator=g) * 0.5).bfloat16(), 32)
    return ids, w, q, s


REAL = {}


def real_logits(MM, i):
    """Router logits whose top 10 (zero bias) are the reference's real routing of layer L (results/ref dumps): DSpark-
    like verify blocks — streams of 8 consecutive tokens (M = 16: 2 streams, 32: 4 from en_tools / doc_long), block i
    shifted by 8·i tokens."""
    if not REAL:
        from safetensors import safe_open as so
        for n in ("en_tools", "doc_long"):
            fn = os.path.join(a.ref_dir, f"{n}.safetensors")
            if not os.path.exists(fn):
                sys.exit(f"[moe] {fn} missing (the timed parts' real routing): copy results/ref/v3 from A, or --ref-dir")
            with so(fn, framework="pt", device="cpu") as f:
                REAL[n] = f.get_tensor("route")[L].long()
    starts = [("en_tools", 100), ("doc_long", 1000), ("en_tools", 500), ("doc_long", 3000), ("en_tools", 300),
              ("doc_long", 1500), ("doc_long", 2000), ("doc_long", 2500),  # (8 streams: M ≤ 64)
              ("doc_long", 100), ("en_tools", 200), ("doc_long", 400), ("en_tools", 0), ("doc_long", 700),
              ("doc_long", 1250), ("doc_long", 1750), ("doc_long", 2250)]  # (16 streams: M ≤ 128)
    rows = []
    for k in range(max(1, -(-MM // 8))):  # (ceil: an M that is not a multiple of 8 gets its last stream cut)
        n, s0 = starts[k]
        s0 = min(s0, REAL[n].shape[0] - 8 * 8 - 8)  # a short dump: the block's 8 shifts stay inside it
        rows.append(REAL[n][s0 + 8 * i: s0 + 8 * i + min(8, MM)])
    ids_ = torch.cat(rows)[:MM]
    lg = torch.full((MM, 1024), -6.0)
    for r in range(MM):
        for j, e in enumerate(ids_[r].tolist()):
            lg[r, e] = lg[r, 512 + e] = 4.0 + 0.05 * j
    return lg.cuda()


ok = True
if not a.skip_ref:
    M = 300
    x = (torch.randn(M, 4096, device="cuda") * 0.3).bfloat16()
    lg = x.float() @ wr.t()
    ids, w = m.route(lg, bias, scale)
    rids, rw = route_ref(lg, bias)
    same = torch.equal(ids, rids)
    wd = (w - rw).abs().max().item()
    lt = (torch.randint(-3, 4, (M, 1024), device="cuda") * 0.5).float()  # many ties at the 10th value
    zb = torch.zeros_like(bias)
    tids, tw = m.route(lt, zb, scale)
    t_ids, t_w = route_ref(lt, zb)
    ties = torch.equal(tids, t_ids) and torch.equal(tw, t_w)
    print(f"[route] ids (slot order) equal {same} · weights bitwise {torch.equal(w, rw)} (max diff {wd:.1e})"
          f" · tied scores: {'equal' if ties else 'DIFFERENT'}")
    ok = same and torch.equal(w, rw) and ties

    lat = (torch.randn(M, 2048, device="cuda") * 0.5).bfloat16()
    q, sc = quant(lat, 32)
    latq = R.act_q8(lat, 32)
    ref = torch.zeros(M, 2048, device="cuda")
    for e in ids.unique().tolist():
        rows, slot = torch.where(ids == e)
        ep = f"{mp}experts.{e}."
        w1 = R.dq_mxfp4(get(ep + "w1.weight"), get(ep + "w1.weight_scale_inv"))
        w3 = R.dq_mxfp4(get(ep + "w3.weight"), get(ep + "w3.weight_scale_inv"))
        w2 = R.dq_mxfp4(get(ep + "w2.weight"), get(ep + "w2.weight_scale_inv"))
        for r in (0, 1):  # the TP split: each rank's half of the inter dims, its own MXFP8 groups
            sl = slice(r * 1024, (r + 1) * 1024)
            aa = latq[rows]
            hq = R.act_q8(R.situ(aa @ w1[sl].t(), aa @ w3[sl].t()), 32)
            ref.index_add_(0, rows, w[rows, slot, None] * (hq @ w2[:, sl].t()))
    parts = [m.experts(h[r], q, sc, ids, w) for r in (0, 1)]
    tot = parts[0] + parts[1]
    rel = ((tot - ref).abs().max() / ref.abs().max()).item()
    err = (tot - ref).abs() / ref.abs().max()
    frac = (err > 1e-5).float().mean().item()
    bad_rows = (err > 1e-5).any(1).sum().item()
    print(f"    error: median {err.median().item():.1e} · 99.9% {err.flatten().kthvalue(int(err.numel() * 0.999)).values.item():.1e}"
          f" · elements > 1e-5: {frac:.1e} · rows with one: {bad_rows}/{M}")
    inv = True
    for MM in (1, 5, 64):
        for r0 in (0, 200):
            p2 = m.experts(h[0], q[r0:r0 + MM].contiguous(), sc[r0:r0 + MM].contiguous(), ids[r0:r0 + MM].contiguous(),
                           w[r0:r0 + MM].contiguous())
            inv &= torch.equal(p2, parts[0][r0:r0 + MM])
    good = inv and (rel < 1e-5 or bad_rows <= max(1, M // 50))  # isolated MXFP8 boundary flips of h (fp32 sum order of g, u)
    ok &= good
    print(f"[experts L{L}] rank sum vs reference rel {rel:.2e} · batch-invariant {inv} · {'OK' if good else 'FAIL'}")

# ---- 4. patches vs base ------------------------------------------------------------------------------------------
only = [p for p in a.only.split(",") if p] or PATCHES
SIZES = {"moe_decode": (1, 2, 4, 8, 9), "moe_prefill": (65, 300, 4096), "moe_combine": (65, 300, 4096)}
TIMED = {"moe_decode": (1, 4, 8), "moe_prefill": (128, 4096), "moe_combine": (128, 4096)}
for p in only:
    good = True
    if p in FUSED:  # route + act_quant + experts (base, P035 / P036 on) vs one launch: ids, w, part bitwise
        sizes, timed = FUSED[p]

        def on():
            for q in WITH.get(p, [p]):
                m.set_patch(q, True)
        for MM in sizes:
            for seed in range(3):
                g = torch.Generator(device="cuda").manual_seed(500 + 10 * MM + seed)
                tie = seed == 2
                lt = (torch.randint(-3, 4, (MM, 1024), device="cuda", generator=g) * 0.5).float() if tie else \
                    (torch.randn(MM, 4096, device="cuda", generator=g) * 0.3).bfloat16().float() @ wr.t()
                bb = torch.zeros_like(bias) if tie else bias
                lat = (torch.randn(MM, 2048, device="cuda", generator=g) * 0.5).bfloat16()
                lat[0, :32] = 0  # an all-zero group (the 1e-10 floor)
                for r in (0, 1):
                    base()
                    i0, w0 = m.route(lt, bb, scale)
                    q0, s0 = m.act_q32(lat)
                    y0, b0 = m.experts(h[r], q0, s0, i0, w0), m.experts(h[r], q0, s0, i0, w0, True)
                    on()
                    for preq in (False, True):  # the latent bf16 (quantized inside) or already MXFP8
                        i1, w1, y1 = m.fused(h[r], lt, bb, scale, lat, False, preq)
                        _, _, b1 = m.fused(h[r], lt, bb, scale, lat, True, preq)
                        if not (torch.equal(i0, i1) and torch.equal(w0, w1)):
                            print(f"    M={MM} seed {seed} rank {r} preq {preq}: ids / w differ")
                            good = False
                        if not (torch.equal(y0, y1) and torch.equal(b0, b1)):
                            print(f"    M={MM} seed {seed} rank {r} preq {preq}: NOT bitwise equal (fp32 max diff "
                                  f"{(y0 - y1).abs().max().item():.3e})")
                            good = False
                    base()
        if p == "moe_fused128":  # per row vs the decode path of one sequence: P066 on each 8-row piece alone
            for MM in (112, 128):
                g = torch.Generator(device="cuda").manual_seed(900 + MM)
                lt = (torch.randn(MM, 4096, device="cuda", generator=g) * 0.3).bfloat16().float() @ wr.t()
                lat = (torch.randn(MM, 2048, device="cuda", generator=g) * 0.5).bfloat16()
                for r in (0, 1):
                    base()
                    on()
                    for preq in (False, True):
                        i1, w1, y1 = m.fused(h[r], lt, bias, scale, lat, False, preq)
                        pc = [m.fused(h[r], lt[k:k + 8].contiguous(), bias, scale, lat[k:k + 8].contiguous(), False,
                                      preq) for k in range(0, MM, 8)]
                        same = all(torch.equal(x, torch.cat([q[j] for q in pc])) for j, x in enumerate((i1, w1, y1)))
                        if not same:
                            d = (y1 - torch.cat([q[2] for q in pc])).abs().max().item()
                            print(f"    M={MM} rank {r} preq {preq}: rows differ from P066 on 8-row pieces "
                                  f"(fp32 max diff {d:.3e})")
                            good = False
                    base()
        for MM in timed:
            lg = torch.stack([real_logits(MM, i) for i in range(8)])
            lat = (torch.randn(MM, 2048, device="cuda") * 0.5).bfloat16()
            zb = torch.zeros_like(bias)
            nexp = sum(m.route(lg[i], zb, scale)[0].unique().numel() for i in range(8)) / 8
            base()
            for rp in REF[p]:
                m.set_patch(rp, True)
            preq = MM <= 8  # model.cc at M ≤ 8 (P060): fc1 writes the latent's MXFP8 copy; above, act_quant runs
            t0 = m.bench_dec(h[0], lg, zb, scale, lat, 30, preq)
            on()
            t1 = m.bench_dec(h[0], lg, zb, scale, lat, 30, preq)
            base()
            gbs = nexp * 3.19e6 / 1e6
            print(f"    {p} M={MM}: {t0 * 1e3:.1f} µs → {t1 * 1e3:.1f} µs (×{t0 / t1:.2f}) · {nexp:.0f} experts · "
                  f"{gbs / t0:.0f} → {gbs / t1:.0f} GB/s cold (route + {'' if preq else 'act_quant + '}experts; real "
                  f"routing)")
        print(f"[{p}] {'bitwise equal to base' if good else 'FAIL'}")
        ok &= good
        continue
    if p == "moe_pf_dn":  # against the default prefill path, real routing (ref v3)
        V3 = []
        for n in ("doc_long", "en_tools", "ko_short"):
            with safe_open(f"/work/results/ref/v3/{n}.safetensors", framework="pt", device="cpu") as f:
                V3.append(f.get_tensor("route")[L].long())
        V3 = torch.cat(V3)
        for MM in (4096, 8192):
            rid = torch.cat([V3, V3[:max(0, MM - V3.shape[0])]])[:MM]
            lg = torch.full((MM, 1024), -6.0)  # router logits whose top 10 (zero bias) are the real experts
            lg.scatter_(1, rid, (4.0 + 0.05 * torch.arange(10.0)).expand(MM, 10).contiguous())
            lg[:, 512:] = lg[:, :512]
            ids, w = m.route(lg.cuda(), torch.zeros_like(bias), scale)
            g = torch.Generator(device="cuda").manual_seed(7)
            q, sc = quant((torch.randn(MM, 2048, device="cuda", generator=g) * 0.5).bfloat16(), 32)
            nexp = ids.unique().numel()
            for r in (0, 1):
                base()
                for rp in REF[p]:
                    m.set_patch(rp, True)
                y0, b0 = m.experts(h[r], q, sc, ids, w), m.experts(h[r], q, sc, ids, w, True)
                m.set_patch(p, True)
                y1, b1 = m.experts(h[r], q, sc, ids, w), m.experts(h[r], q, sc, ids, w, True)
                if not (torch.equal(y0, y1) and torch.equal(b0, b1)):
                    print(f"    M={MM} rank {r}: NOT bitwise equal (fp32 max diff {(y0 - y1).abs().max().item():.3e})")
                    good = False
            base()
            for rp in REF[p]:
                m.set_patch(rp, True)
            t0 = m.bench_cold(h[0], q, sc, ids[None], w[None], 12, False)
            m.set_patch(p, True)
            t1 = m.bench_cold(h[0], q, sc, ids[None], w[None], 12, False)
            base()
            print(f"    {p} M={MM}: {t0:.2f} ms → {t1:.2f} ms per call (P037 + P065 → P086; layer {L}, {nexp} experts, "
                  f"real routing)")
        print(f"[{p}] {'bitwise equal to the default prefill path' if good else 'FAIL'}")
        ok &= good
        continue
    if p == "route_fast":
        for MM in (1, 4, 64, 4096):
            for tie in (False, True):
                g = torch.Generator(device="cuda").manual_seed(MM)
                lt = (torch.randint(-3, 4, (MM, 1024), device="cuda", generator=g) * 0.5).float() if tie else \
                    (torch.randn(MM, 4096, device="cuda", generator=g) * 0.3).bfloat16().float() @ wr.t()
                bb = torch.zeros_like(bias) if tie else bias
                base()
                i0, w0 = m.route(lt, bb, scale)
                m.set_patch(p, True)
                i1, w1 = m.route(lt, bb, scale)
                if not (torch.equal(i0, i1) and torch.equal(w0, w1)):
                    print(f"    route M={MM}{' (ties)' if tie else ''}: NOT bitwise equal")
                    good = False
            base()
            t0 = m.bench_route(lt, bb, scale, 50)
            m.set_patch(p, True)
            t1 = m.bench_route(lt, bb, scale, 50)
            base()
            print(f"    route_fast M={MM}: {t0 * 1e3:.1f} µs → {t1 * 1e3:.1f} µs (×{t0 / t1:.2f}) (an empty launch: "
                  f"{m.bench_null(50) * 1e3:.1f} µs)")
        print(f"[{p}] {'bitwise equal to base' if good else 'FAIL'}")
        ok &= good
        continue
    for MM in SIZES[p]:
        ids, w, q, s = routed(MM, 100 + MM)
        for r in (0, 1):
            base()
            y0, b0 = m.experts(h[r], q, s, ids, w), m.experts(h[r], q, s, ids, w, True)
            m.set_patch(p, True)
            y1, b1 = m.experts(h[r], q, s, ids, w), m.experts(h[r], q, s, ids, w, True)
            if not (torch.equal(b0, y0.bfloat16()) and torch.equal(b1, b0)):
                print(f"    M={MM} rank {r}: bf16 output is not bf16(fp32 output) or differs from the base")
                good = False
            if not torch.equal(y0, y1):
                d = (y0 - y1).abs().max().item()
                print(f"    M={MM} rank {r}: NOT bitwise equal (max abs diff {d:.3e})")
                good = False
    base()
    for MM in TIMED[p]:
        sets = [routed(MM, 1000 + 7 * i) for i in range(8 if MM < 100 else 2)]
        ids = torch.stack([x[0] for x in sets])
        w = torch.stack([x[1] for x in sets])
        q, s = sets[0][2], sets[0][3]
        nexp = sum(x[0].unique().numel() for x in sets) / len(sets)
        it = 30 if MM < 100 else 5
        base()
        for rp in REF.get(p, []):
            m.set_patch(rp, True)
        t0 = m.bench_cold(h[0], q, s, ids, w, it, True)
        m.set_patch(p, True)
        t1 = m.bench_cold(h[0], q, s, ids, w, it, True)
        base()
        gbs = nexp * 3.19e6 / 1e6
        if MM < 100:
            print(f"    {p} M={MM}: {t0 * 1e3:.1f} µs → {t1 * 1e3:.1f} µs (×{t0 / t1:.2f}) · {nexp:.0f} experts · "
                  f"{gbs / t0:.0f} → {gbs / t1:.0f} GB/s cold")
        else:
            print(f"    {p} M={MM}: {t0:.2f} ms → {t1:.2f} ms (×{t0 / t1:.2f}) per layer · ×42 layers: "
                  f"{42 * t0:.0f} → {42 * t1:.0f} ms")
            if p == "moe_combine":
                print(f"      plan alone: {m.bench_plan(ids[0], False, 20) * 1e3:.0f} µs (base: k_plan ×2, k_scan, k_tiles37)"
                      f" → {m.bench_plan(ids[0], True, 20) * 1e3:.0f} µs (k_plan65a / b)")
    print(f"[{p}] {'bitwise equal to base' if good else 'FAIL'}")
    ok &= good
print("[moe]", "all OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
