#!/usr/bin/env python3
"""test_fuse.py — decode-path fusions around the GEMMs, on real checkpoint weights (tests/bind/fuse.cpp).
  python3 engine/tests/test_fuse.py --model /models/ERNIE-5.5-Flash-Preview --table /work/cache/table.tsv [--no-time]
1. P060 f8_fuse_ops: model.cc's decode sequences (HCA qkv → norms → q_up + gate → σ … wo_a → z ⊙ gate → o_proj; DSA
   qkv → norms → q_b + gate + iq → σ; MoE fc1 + shared gate/up → latent MXFP8 · SiTU → down · latent_norm → fc2 + shared;
   dense gate/up → SiTU → down) base (separate misc kernels) vs fused: every output bitwise equal, M = 1 … 8, both ranks,
   several random inputs; then ms per layer back to back over several layers (cold weights), base → fused.
2. P061 lin_pdl / P062 lin12: router and compressor linears (fp32 out) bitwise equal to the base path; ms per call
   back to back (after an FP8 GEMM, as in the step), base → patch.
3. P063 sparse_wire: sampling from the two vocab halves' top-64 records (sample_cands per half → sample_rows_sparse)
   gives sample_rows' tokens and probabilities on the full rows, bitwise, for the sampling settings it takes (greedy,
   top_k ≤ 64, top_p ≤ 0; with / without min_p); this rank's ms after multimax, full → sparse (all-gathers not
   included: 2 × R × 402 KB full rows vs 2 × R × 520 B records)."""
import argparse
import os
import statistics
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ext  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--table", required=True)
ap.add_argument("--no-time", action="store_true")
ap.add_argument("--only", default="hca,dsa,moe,dense,ops,lin,samp")
ap.add_argument("--layers", type=int, default=4, help="layers per timed run")
a = ap.parse_args()
m = ext.load("fuse", ["src/kernels/gemm/fp8.cu", "src/kernels/misc/misc.cu", "src/kernels/linear/linear.cu",
                      "src/kernels/sample/sample.cu", "src/kernels/sample/penalty.cu", "src/core/weights.cc",
                      "src/model/patches.cc"])
m.open(a.model, a.table)
only = a.only.split(",")
med = statistics.median


def L(i):
    return f"model.layers.{i}."


def hca_w(rank, i):
    p = L(i) + "self_attn."
    h = [m.f8(rank, [p + "linear_q_down_proj", p + "linear_kv_proj"], 1), m.f8(rank, [p + "linear_q_up_proj"], 1),
         m.f8(rank, [p + "gate_proj"], 1), m.f8(rank, [p + "linear_o_group_proj"], 4), m.f8(rank, [p + "o_proj"], 1)]
    return h, [m.vec(rank, p + "q_layernorm.weight"), m.vec(rank, p + "kv_layernorm.weight")]


def dsa_w(rank, i):
    p = L(i) + "self_attn."
    h = [m.f8(rank, [p + "q_a_proj", p + "kv_a_proj_with_mqa"], 1), m.f8(rank, [p + "q_b_proj"], 1),
         m.f8(rank, [p + "gate_proj"], 1), m.f8(rank, [p + "core_attention.indexer.wq_b"], 1)]
    return h, [m.vec(rank, p + "q_a_layernorm.weight"), m.vec(rank, p + "kv_a_layernorm.weight")]


def moe_w(rank, i):
    p = L(i) + "block_sparse_moe."
    h = [m.f8(rank, [p + "fc1_latent_proj"], 1), m.f8(rank, [p + "shared_experts.w1", p + "shared_experts.w3"], 1),
         m.f8(rank, [p + "shared_experts.w2"], 1), m.f8(rank, [p + "fc2_latent_proj"], 1)]
    return h, [m.vec(rank, p + "latent_norm.weight")]


def dense_w(rank, i):
    return [m.f8(rank, [L(0) + "mlp.gate_proj", L(0) + "mlp.up_proj"], 1), m.f8(rank, [L(0) + "mlp.down_proj"], 1)], []


KINDS = {"hca": (hca_w, [1, 2, 3, 4, 5, 6]), "dsa": (dsa_w, [7, 14, 21, 28, 35, 42]), "moe": (moe_w, [1, 2, 3, 4, 5, 6]),
         "dense": (dense_w, [0])}
OUT = {"hca": ["t1", "q", "gate σ", "kv norm", "wo_a", "o_proj"], "dsa": ["t1", "q_b", "gate σ", "iq", "kv norm"],
       "moe": ["fc1", "latent q", "latent s", "gate/up", "pb", "fc2 + shared"], "dense": ["down"]}


def inputs(M, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    inp = (torch.randn(M, 4096, device="cuda", generator=g) * (0.3 + seed % 3)).bfloat16()
    inp[0, :64] *= 40  # a few outliers
    oq, os_ = m.quant((torch.randn(4 * M, 4096, device="cuda", generator=g) * 0.7).bfloat16(), 128)
    lat = (torch.randn(M, 2048, device="cuda", generator=g) * (0.05 + seed % 4)).bfloat16()
    return inp, oq, os_, lat


ok = True
for kind in ("hca", "dsa", "moe", "dense"):
    if kind not in only:
        continue
    up, layers = KINDS[kind]
    for rank in (0, 1):
        h, v = up(rank, layers[0])
        same, n = True, 0
        for M in ((1, 2, 3, 4, 8, 13, 16, 32, 300) if kind != "dense" else (1, 2, 3, 4, 8)):
            for seed in range(4 if M <= 32 else 1):
                inp, oq, os_, lat = inputs(M, seed * 10 + M)
                m.set_patch("f8_fuse_ops", False)
                base = m.run(kind, h, v, inp, oq, os_, lat, M, False)
                m.set_patch("f8_fuse_ops", True)
                assert m.fuse_ok(M)
                fz = m.run(kind, h, v, inp, oq, os_, lat, M, True)
                for name, x, y in zip(OUT[kind], base, fz):
                    if not torch.equal(x, y):
                        if same:
                            d = (x.float() - y.float()).abs()
                            print(f"    {kind} r{rank} M={M} seed {seed}: {name} differs at {int((x != y).sum())} of "
                                  f"{x.numel()} (max |d| {d.max().item():.3g})")
                        same = False
                n += 1
        fin = all(bool(t.float().isfinite().all()) for t in base if t.dtype == torch.bfloat16)
        ok &= same
        print(f"[{kind} r{rank}] layer {layers[0]}: fused == base, every output bitwise ({n} runs, M 1–300): {same}"
              f" · finite {fin}")
    if a.no_time:
        continue
    hs, vs = [], []
    for i in layers[:a.layers]:
        hh, vv = up(0, i)
        hs.append(hh)
        vs += vv
    if kind == "dense":
        continue  # (model.cc keeps the dense FFN unfused: SiTU over k = 7168 per CTA costs more than it saves at M ≥ 4)
    print(f"    {kind} × {len(hs)} layers back to back, µs per layer: base → P064 → P060 + P064")
    for M in (1, 4, 8, 16, 32, 4096):
        inp, oq, os_, lat = inputs(M, 1)
        t = []
        for fused, pf in ((False, False), (False, True), (True, True)):
            m.set_patch("f8_fuse_ops", fused)
            m.set_patch("f8_pf_next", pf)
            torch.cuda._sleep(20_000_000)
            t.append(med(m.bench(kind, hs, vs, inp, oq, os_, M, fused, 30 if M <= 32 else 5)) / len(hs) * 1e3)
        print(f"      M={M}: {t[0]:.1f} → {t[1]:.1f} → {t[2]:.1f} µs (−{t[0] - t[2]:.1f}, ×{t[0] / t[2]:.3f})")
    m.set_patch("f8_pf_next", True)
m.set_patch("f8_fuse_ops", False)

# each fused op's own cost (GEMM with the op vs the same GEMM on a plain deferred activation; the misc launch it
# replaces is not counted): what grows with M
if "ops" in only and not a.no_time:
    m.set_patch("f8_fuse_ops", True)
    for kind, ops in (("hca", ("rmsq", "mul")), ("moe", ("q32", "situ", "rms"))):
        up, layers = KINDS[kind]
        hs, ws_ = [], []
        for i in layers[:a.layers]:
            hh, vv = up(0, i)
            hs.append(hh)
            ws_.append(vv[-1])  # HCA: kv norm weight (rmsq uses it for both), MoE: latent norm
        for op in ops:
            line = f"    op {op} (× {len(hs)} layers): GEMM alone → with the op, µs per layer:"
            for M in (1, 4, 8, 16, 32):
                inp, oq, os_, lat = inputs(M, 1)
                t = []
                for on in (False, True):
                    torch.cuda._sleep(20_000_000)
                    t.append(med(m.bench("op:" + op, hs, ws_, inp, oq, os_, M, on, 30)) / len(hs) * 1e3)
                line += f" M={M} {t[0]:.1f} → {t[1]:.1f} (+{t[1] - t[0]:.1f}) ·"
            print(line.rstrip(" ·"))

# 2. router / compressor linears: bf16 copy (base) vs P013's 12-bit copy (P062), each with / without P061 PDL
if "lin" in only:
    LIN = {"router": lambda i: [L(i) + "block_sparse_moe.gate.weight", L(i) + "block_sparse_moe.gate.weight_1"],
           "compressor": lambda i: [L(i) + "self_attn.core_attention.compressor.linear_wkv.weight",
                                    L(i) + "self_attn.core_attention.compressor.linear_wgate.weight"]}
    for tag, names in LIN.items():
        for rank in (0, 1):
            hb, hp = m.lin(rank, names(1), False), m.lin(rank, names(1), True)
            n, k, sp, pk, nx = m.lin_info(hp)
            same = True
            for M in (1, 2, 4, 8, 13, 64):
                x = (torch.randn(M, k, device="cuda") * (1 + M % 3)).bfloat16()
                m.set_patch("lin_pdl", False)
                ref = m.lin_fwd(hb, x)
                for pdl in (False, True):
                    m.set_patch("lin_pdl", pdl)
                    for h in (hb, hp):
                        same &= torch.equal(m.lin_fwd(h, x), ref)
            m.set_patch("lin_pdl", False)
            ok &= same
            print(f"[{tag} r{rank}] {n}×{k} split {sp}: 12-bit copy exceptions {nx} ({nx / (n * k) * 100:.4f} %) · "
                  f"12-bit and PDL == bf16 base, fp32 bitwise (M 1–64): {same}")
        if a.no_time:
            continue
        lay = [1, 2, 3, 4, 5, 6]
        hb = [m.lin(0, names(i), False) for i in lay]
        hp = [m.lin(0, names(i), True) for i in lay]
        before = [hca_w(0, i)[0][1] for i in lay]  # an FP8 GEMM before each (q_up, as the compressor's in the step)
        line = f"    {tag}: q_up GEMM → linear, × {len(lay)} back to back, µs per pair:"
        for M in (1, 4, 8, 16, 32):
            x = torch.randn(M, 4096, device="cuda").bfloat16()
            t = []
            for hs, pdl in ((hb, False), (hb, True), (hp, False), (hp, True)):
                m.set_patch("lin_pdl", pdl)
                torch.cuda._sleep(20_000_000)
                t.append(med(m.bench_lin(hs, before, x, 30)) / len(lay) * 1e3)
            line += f"\n      M={M}: bf16 {t[0]:.1f} · +PDL {t[1]:.1f} · 12-bit {t[2]:.1f} · 12-bit+PDL {t[3]:.1f} (−{t[0] - t[3]:.1f})"
        m.set_patch("lin_pdl", False)
        print(line)
        line = f"    {tag} alone × {len(lay)} back to back, µs per call, bf16 → 12-bit (P061 on):"
        m.set_patch("lin_pdl", True)
        for M in (8, 16, 32, 64, 128, 4096):
            x = torch.randn(M, 4096, device="cuda").bfloat16()
            t = []
            for hs in (hb, hp):
                torch.cuda._sleep(20_000_000)
                t.append(med(m.bench_lin(hs, [], x, 20)) / len(lay) * 1e3)
            line += f" M={M} {t[0]:.1f} → {t[1]:.1f} ·"
        m.set_patch("lin_pdl", False)
        print(line.rstrip(" ·"))

if "samp" in only:
    V = 201216
    SETS = [(0.0, 0, 1.0, 0.0), (0.7, 20, 0.95, 0.0), (1.0, 64, 1.0, 0.05), (0.6, 50, 0.8, 0.1), (1.3, 1, 1.0, 0.0),
            (0.8, 0, 0.0, 0.0), (1.0, 40, 1.0, 0.0), (0.0, 0, 0.9, 0.2)]
    same, n, nd = True, 0, 0
    for R in (1, 3, 8):
        for seed in range(6):
            g = torch.Generator(device="cuda").manual_seed(1000 + 17 * R + seed)
            lg = torch.randn(R, V, device="cuda", generator=g) * (1.5 + seed % 3)
            for r in range(R):  # a peaked head, as real logits; ties in some rows
                k = torch.randint(0, V, (40,), device="cuda", generator=g)
                lg[r, k] += torch.linspace(14, 4, 40, device="cuda") * (0.5 + 0.2 * seed)
                if seed % 2:
                    lg[r, k[5]] = lg[r, k[4]]
            for st in SETS:
                if not m.sparse_ok(st[0], st[1], st[2]):
                    continue
                f = lambda v, dt: torch.tensor([v] * R, dtype=dt, device="cuda")
                pr = (f(st[0], torch.float32), f(st[1], torch.int32), f(st[2], torch.float32), f(st[3], torch.float32),
                      torch.arange(R, dtype=torch.int64, device="cuda") + 77 * seed,
                      torch.arange(R, dtype=torch.int64, device="cuda") + 1000)
                tf, pf, ts, ps = m.samp_cmp(lg, *pr)
                eq = torch.equal(tf, ts) and torch.equal(pf, ps)
                nd += int((ts == -2).sum())
                same &= eq
                n += 1
    ok &= same
    print(f"[samp] sparse records == full rows (token, prob) bitwise: {same} ({n} cases, R 1–8; dense fallbacks {nd})")
    if not a.no_time:
        line = "    this rank after multimax, ms per step:"
        for R in (1, 4, 8):
            lg = torch.randn(R, V, device="cuda") * 2
            pr = (torch.full((R,), 0.7, device="cuda"), torch.full((R,), 20, dtype=torch.int32, device="cuda"),
                  torch.full((R,), 0.95, device="cuda"), torch.zeros(R, device="cuda"),
                  torch.arange(R, dtype=torch.int64, device="cuda"), torch.arange(R, dtype=torch.int64, device="cuda"))
            t = [med(m.samp_bench(lg, *pr, sp, 30)) * 1e3 for sp in (False, True)]
            line += f" R={R} {t[0]:.1f} → {t[1]:.1f} µs ·"
        print(line.rstrip(" ·"))
print("[fuse]", "all OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
