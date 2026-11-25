#!/usr/bin/env python3
"""test_overlap.py — the overlap gate (one GPU): does long-context HCA attention run beside the MoE prefill experts?
  python3 engine/tests/test_overlap.py --model /models/ERNIE-5.5-Flash-Preview --table /work/cache/table.tsv
A = one HCA layer's attention for an 8192-row chunk as model.cc's lanes run it: 4 hca_attn_fused calls of 2048 rows ×
32 heads (rows at --positions + 0 … 8191 of one sequence, the caches holding the history; default dispatch).
B = one MoE layer's experts for the same 8192 rows: moe_experts_bf16 (default prefill path, rank 0's experts of
--moe-layer) on the reference's real routing (ref v3 dumps, as test_moe's P086 case) and a random MXFP8 latent.
Timed (CUDA events, median of --runs, interleaved): t(A) and t(B) alone, t(A ‖ B) with A on one stream and B on
another, both released by one event after a spin (the host has queued everything); A's persistent attention CTAs fill
whole SMs (512 threads × 128 registers), so B also runs against A capped to --caps CTAs (P172's test-only cap).
Saving = 1 − t(A ‖ B) / (t(A) + t(B)), ceiling min(tA, tB) / (tA + tB). The concurrent outputs (o, zp, the MoE
partial) are compared bit for bit with the solo ones."""
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
ap.add_argument("--hca-layer", type=int, default=3)
ap.add_argument("--moe-layer", type=int, default=5)
ap.add_argument("--positions", default="131072,524288")
ap.add_argument("--caps", default="16,24,32")
ap.add_argument("--runs", type=int, default=7)
a = ap.parse_args()
m = ext.load("overlap", ["src/kernels/attn/hca.cu", "src/kernels/moe/moe.cu", "tests/bind/moe_aq.cu",
                         "src/core/weights.cc", "src/model/patches.cc"])
wm = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))["weight_map"]
dev = "cuda"
ROWS, LANE = 8192, 2048


def get(name):
    with safe_open(os.path.join(a.model, wm[name]), framework="pt", device="cpu") as f:
        return f.get_tensor(name).to(dev)


hp = f"model.layers.{a.hca_layer}.self_attn."
U = get(hp + "vha_postmix_U").float()[:32].contiguous()
SINK = get(hp + "core_attention.attn_sink").float()[:32].contiguous()
APE = get(hp + "core_attention.compressor.ape").float().contiguous()
NW = get(hp + "core_attention.compressor.norm.weight").bfloat16().contiguous()
INV = m.inv_freq()
WS = torch.empty(1 << 28, dtype=torch.float32, device=dev)  # 1 GiB, the engine's attention workspace


def meta_of(p0, n):
    return torch.tensor([(p0 + i, 0, 0, p0, p0 + n - 1) for i in range(n)], dtype=torch.int32, device=dev)


def hca_case(p0, seed=5):
    """the caches holding positions 0 … p0 − 1 (compressed entries, ring), then hca_pre of the chunk's 8192 rows"""
    g = torch.Generator(device=dev).manual_seed(seed)
    t = dict(ring=torch.zeros(1, 128, 584, dtype=torch.uint8, device=dev),
             comp=torch.zeros(1, (p0 + ROWS) // 128 + 1, 584, dtype=torch.uint8, device=dev),
             state=torch.zeros(1, 128, 1024, dtype=torch.float32, device=dev))
    nc = p0 // 128
    cm = meta_of(0, 1).repeat(nc, 1)
    cm[:, 0] = torch.arange(nc, device=dev, dtype=torch.int32) * 128
    m.kv_rows(torch.randn(nc, 512, generator=g, device=dev).bfloat16(), cm, INV, t["comp"][0, :nc])
    rr = torch.empty(128, 584, dtype=torch.uint8, device=dev)
    m.kv_rows(torch.randn(128, 512, generator=g, device=dev).bfloat16(), meta_of(p0 - 128, 128), INV, rr)
    t["ring"][0, torch.arange(p0 - 128, p0, device=dev) % 128] = rr
    t["meta"] = meta_of(p0, ROWS)
    t["q"] = (torch.randn(ROWS, 32 * 512, generator=g, device=dev) * 0.7).bfloat16()
    kv = torch.randn(ROWS, 512, generator=g, device=dev).bfloat16()
    ckv = torch.randn(ROWS, 1024, generator=g, device=dev)
    ckv[:, 512:] *= 2
    t["rows"] = torch.empty(ROWS, 584, dtype=torch.uint8, device=dev)
    m.pre(kv, ckv, APE, NW, t["meta"], INV, t["rows"], t["ring"], t["comp"], t["state"], WS)
    t["o"] = torch.empty(ROWS, 32 * 512, dtype=torch.bfloat16, device=dev)
    t["zp"] = torch.empty(ROWS, 16, 512, dtype=torch.float32, device=dev)
    # model.cc's lanes: rows [l·2048, +2048) per call, the lane's max_keys, the step's keys through the whole-step rows
    t["lanes"] = [(r0, max((p + 1) // 128 + min(p + 1, 128) for p in range(p0 + r0, p0 + r0 + LANE)))
                  for r0 in range(0, ROWS, LANE)]
    return t


def run_a(t):
    for r0, mk in t["lanes"]:
        s = slice(r0, r0 + LANE)
        m.attn_fused(t["q"][s], t["rows"], t["meta"][s], t["ring"], t["comp"], t["state"], SINK, INV, U, mk, t["o"][s],
                     t["zp"][s], WS)


# ---- B: the MoE experts on real routing
m.moe_open(a.model, a.table, a.moe_layer)
mp = f"model.layers.{a.moe_layer}.block_sparse_moe."
scale = get(mp + "gate.routed_scaling_factor_param").float()
V3 = []
for n in ("doc_long", "en_tools", "ko_short"):
    with safe_open(f"/work/results/ref/v3/{n}.safetensors", framework="pt", device="cpu") as f:
        V3.append(f.get_tensor("route")[a.moe_layer].long())
V3 = torch.cat(V3)
rid = torch.cat([V3, V3[:max(0, ROWS - V3.shape[0])]])[:ROWS]
lg = torch.full((ROWS, 1024), -6.0)  # router logits whose top 10 (zero bias) are the real experts
lg.scatter_(1, rid, (4.0 + 0.05 * torch.arange(10.0)).expand(ROWS, 10).contiguous())
lg[:, 512:] = lg[:, :512]
IDS, WT = m.route(lg.to(dev), torch.zeros(512, device=dev), scale)
g = torch.Generator(device=dev).manual_seed(7)
LQ, LS = m.act_q32((torch.randn(ROWS, 2048, device=dev, generator=g) * 0.5).bfloat16())
PART = torch.empty(ROWS, 2048, dtype=torch.bfloat16, device=dev)


def run_b():
    m.experts(LQ, LS, IDS, WT, PART)


SA, SB = torch.cuda.Stream(), torch.cuda.Stream()


def timed(fa, fb):
    """ms from the release to the last end: fa on stream A, fb on stream B (either may be None)"""
    e0 = torch.cuda.Event(enable_timing=True)
    ea, eb = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda._sleep(300_000)
    e0.record()
    SA.wait_event(e0), SB.wait_event(e0)
    if fa:
        with torch.cuda.stream(SA):
            fa()
            ea.record()
    if fb:
        with torch.cuda.stream(SB):
            fb()
            eb.record()
    torch.cuda.synchronize()
    ta = e0.elapsed_time(ea) if fa else 0.0
    tb = e0.elapsed_time(eb) if fb else 0.0
    return max(ta, tb), ta, tb


caps = [int(x) for x in a.caps.split(",") if x]
for p0 in [int(x) for x in a.positions.split(",")]:
    t = hca_case(p0)
    nch = (t["lanes"][-1][1] + 255) // 256
    m.hca_cap(0)
    run_a(t), run_b()
    torch.cuda.synchronize()
    ref = dict(o=t["o"].clone(), zp=t["zp"].clone(), part=PART.clone())
    variants = [("A", lambda: timed(lambda: run_a(t), None), 0), ("B", lambda: timed(None, run_b), 0),
                ("A||B", lambda: timed(lambda: run_a(t), run_b), 0)]
    variants += [(f"A||B cap {c}", lambda: timed(lambda: run_a(t), run_b), c) for c in caps]
    variants += [(f"A cap {c}", lambda: timed(lambda: run_a(t), None), c) for c in caps]
    ts = {k: [] for k, _, _ in variants}
    ends = {k: [] for k, _, _ in variants}
    bad = []
    for it in range(a.runs + 1):
        for k, f, c in variants:
            m.hca_cap(c)
            t["o"].fill_(float("nan")), t["zp"].fill_(float("nan")), PART.fill_(float("nan"))
            tot, ta, tb = f()
            if k.startswith("A||B") and it == 0:  # the concurrent outputs vs the solo ones
                for nm, x in (("o", t["o"]), ("zp", t["zp"]), ("part", PART)):
                    if not torch.equal(x.view(torch.uint8), ref[nm].view(torch.uint8)):
                        bad.append(f"{k}: {nm}")
            if it:
                ts[k].append(tot), ends[k].append((ta, tb))
    m.hca_cap(0)
    med = {k: statistics.median(v) for k, v in ts.items()}
    tA, tB = med["A"], med["B"]
    print(f"[gate {p0}, {nch} chunks] t(A) 4 × 2048-row attention {tA:.2f} ms · t(B) MoE 8192 rows {tB:.2f} ms · "
          f"ceiling {min(tA, tB) / (tA + tB) * 100:.1f} %", flush=True)
    for k, _, c in variants[2:]:
        if not k.startswith("A||B"):
            continue
        ea = statistics.median(x for x, _ in ends[k])
        eb = statistics.median(y for _, y in ends[k])
        line = f"  {k}: {med[k]:.2f} ms (A ends {ea:.2f}, B ends {eb:.2f}) · saving {(1 - med[k] / (tA + tB)) * 100:.1f} %"
        if c:  # A capped all along; estimate: A's work left when B ends (from A's capped solo rate) at the full-GPU rate
            tc = med[f"A cap {c}"]
            left = min(1.0, max(0.0, (ea - eb) / tc))
            est = eb + left * tA
            line += (f" · A cap {c} alone {tc:.2f} ms · estimate with A back on every SM once B ends: {est:.2f} ms, saving "
                     f"{(1 - est / (tA + tB)) * 100:.1f} %")
        print(line, flush=True)
    print("  outputs (o, zp, MoE partial) concurrent vs solo: " + ("bitwise equal" if not bad else "DIFFER: " + "; ".join(bad)),
          flush=True)
    del t
m.moe_close()
