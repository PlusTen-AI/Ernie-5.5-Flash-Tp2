#!/usr/bin/env python3
"""test_gemm.py — the FP8 block GEMM (kernels/gemm/fp8.cu) on real checkpoint weights, both ranks' slices.
  python3 engine/tests/test_gemm.py --model /models/ERNIE-5.5-Flash-Preview --table /work/cache/table.tsv [--quick]
  … --sites all [--variants cur,patch,c3,c6s] [--sm 300,1000]: the prefill call sites of model.cc (8192-row chunk),
  current (P133 / P134 off) → patch, bitwise and ms (P134 f8_situ_epi: site moe.gd)
1. act_quant == the reference's activation quantization (tools/ref/ernie_ref.py act_q8), bit for bit
2. base: y == fp32 matmul of the dequantized operands (the reference's math) to fp32 summation order; bf16 outputs
   equal but for rare last-place ties; a row's output is bitwise the same at every M (fixed K splits)
3. patches P030 f8_decode / P031 f8_prefill / P032 f8_fuse_act (… P103–P105, P133 f8_tile_seq): bitwise equal to the
   base at every M (bf16 and fp32 outputs; P032 from the bf16 activation)
4. time per call (median of CUDA-event timed calls): decode M = 1 / 4 cold (weights rotated over copies ≫ L2) as
   GB/s of weights, with and without the activation quantization, and 8 calls back to back (base → P030 → + P033
   PDL); prefill M = 4096 as TFLOPS — base → patch"""
import argparse
import json
import os
import statistics
import sys

import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools", "ref"))
import ext  # noqa: E402
import ernie_ref as R  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--table", required=True)
ap.add_argument("--quick", action="store_true", help="rank 0 only, fewer M values")
ap.add_argument("--only", default="", help="comma list of case tags to run")
ap.add_argument("--no-time", action="store_true")
ap.add_argument("--time", default="decode,prefill", help="which timings: decode, prefill")
ap.add_argument("--sites", default="", help="comma list of prefill call sites (or 'all'): bitwise + ms current → "
                "patch at the site's rows (or --sm), then exit")
ap.add_argument("--sm", default="", help="rows for --sites (default: the site's rows in the 8192-row chunk)")
ap.add_argument("--variants", default="", help="--sites: comma list of cur / patch / cN[s] (P133 config N forced, "
                "'s': + P134); default cur,patch")
ap.add_argument("--runs", type=int, default=20)
a = ap.parse_args()
m = ext.load("gemm", ["src/kernels/gemm/fp8.cu", "src/core/weights.cc", "src/model/patches.cc"])
m.open(a.model, a.table)
wm = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))["weight_map"]
files = {}
PATCHES = ("f8_decode", "f8_prefill", "f8_fuse_act", "f8_pdl", "f8_multi", "f8_tile", "f8_tile_epi", "f8_tile_chain",
           "f8_tile_seq", "f8_situ_epi")


def patches(on):
    for p in PATCHES:
        m.set_patch(p, on)


def get(name):
    f = wm[name]
    if f not in files:
        files[f] = safe_open(os.path.join(a.model, f), framework="pt", device="cpu")
    return files[f].get_tensor(name).cuda()


def ref_w(name, rank, split):
    bare = name.endswith("linear_o_group_proj")
    w = get(name if bare else name + ".weight")
    s = get(name + "_scale_inv" if bare else name + ".weight_scale_inv")
    d = R.dq_block(w, s)
    if split == "col":
        h = d.shape[0] // 2
        d = d[rank * h:(rank + 1) * h]
    elif split == "row":
        h = d.shape[1] // 2
        d = d[:, rank * h:(rank + 1) * h]
    return d


def med(v):
    return statistics.median(v)


ok = True
H, D, F = "model.layers.3.self_attn.", "model.layers.7.self_attn.", "model.layers.5.block_sparse_moe."

# ---- the prefill FP8 call sites of model.cc as they run now (P060 on, every patch at its default; P100 a2a: the gates,
# wo_a and o_proj on the row owner with the peer's halves = rank 1's slices here): `current` (P133 f8_tile_seq off) →
# the variants (--variants dev: P133 with each tile config forced; default: P133 at its measured choice), each bitwise
# against current on every output, then ms per call (CUDA events, median of --runs, weights rotated over copies ≫ L2).
# Rows and calls per 8192-row chunk from the step profile (results/runs/20261125-session5/step2/m2-prof.log, prefill
# 8192 rows at 0: lanes of 2048 rows, the P094 halves 1024, the FFN part 8192; 37 HCA / 6 DSA / 42 MoE layers × 4 lanes).
SITES = {  # tag: (model.cc line, [(rank, names, groups, epilogue)], rows: lane / half / part, calls per 8192-row chunk)
    "hca.qkv": ("1173 attn_front q_down+kv", [(0, [H + "linear_q_down_proj", H + "linear_kv_proj"], 1, 0)], "half", 148),
    "dsa.qkv": ("1179 attn_front q_a+kv_a", [(0, [D + "q_a_proj", D + "kv_a_proj_with_mqa"], 1, 0)], "half", 24),
    "hca.qup": ("747 q norm → q_up, kv norm aside", [(0, [H + "linear_q_up_proj"], 1, 0)], "lane", 148),
    "dsa.qbi": ("886 q_a norm → q_b + iq, kv_a norm aside",
                [(0, [D + "q_b_proj"], 1, 0), (0, [D + "core_attention.indexer.wq_b"], 1, 0)], "lane", 24),
    "hca.gate": ("1323 attn_gate q norm → gate σ ×2", [(0, [H + "gate_proj"], 1, 1), (1, [H + "gate_proj"], 1, 1)],
                 "half", 148),
    "dsa.gate": ("1323 attn_gate q_a norm → gate σ ×2", [(0, [D + "gate_proj"], 1, 1), (1, [D + "gate_proj"], 1, 1)],
                 "half", 24),
    "hca.woa": ("1346 attn_own wo_a ×2", [(0, [H + "linear_o_group_proj"], 4, 0), (1, [H + "linear_o_group_proj"], 4, 0)],
                "half", 148),
    "hca.oproj": ("1354 z⊙gate → o_proj ×2 (+ partial)", [(0, [H + "o_proj"], 1, 0), (1, [H + "o_proj"], 1, 2)],
                  "half", 148),
    "dsa.oproj": ("1372 act → o_proj ×2 (+ partial)", [(0, [D + "o_proj"], 1, 0), (1, [D + "o_proj"], 1, 2)], "half", 24),
    "moe.fc1": ("1211 ffn_front fc1 + MXFP8", [(0, [F + "fc1_latent_proj"], 1, 3)], "half", 168),
    "moe.gu": ("1261 shared gate/up", [(0, [F + "shared_experts.w1", F + "shared_experts.w3"], 1, 0)], "lane", 168),
    "moe.down": ("1269 SiTU → shared down", [(0, [F + "shared_experts.w2"], 1, 0)], "lane", 168),
    "moe.gd": ("1261 + 1269 shared gate/up → SiTU → down",
               [(0, [F + "shared_experts.w1", F + "shared_experts.w3"], 1, 5), (0, [F + "shared_experts.w2"], 1, 0)],
               "lane", 168),
    "moe.fc2": ("1116 latent_norm → fc2 + shared", [(0, [F + "fc2_latent_proj"], 1, 2)], "half", 168),
    "dense.gu": ("1241 dense gate/up", [(0, ["model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj"], 1, 0)],
                 "part", 1),
    "dense.down": ("1241 dense down", [(0, ["model.layers.0.mlp.down_proj"], 1, 0)], "part", 1),
}
for t_ in ("moe.down", "moe.fc2", "hca.gate", "hca.oproj"):  # the GEMM (+ epilogue) alone, the activation given (dev)
    SITES[t_ + ".g"] = (SITES[t_][0] + " (GEMM alone)",) + SITES[t_][1:]
if a.sites:
    def bf(*sh):
        return torch.zeros(*sh, dtype=torch.bfloat16, device="cuda")

    def u8(*sh):
        return torch.zeros(*sh, dtype=torch.uint8, device="cuda")

    def rnd(M, k, sc):
        return (torch.randn(M, k, device="cuda") * sc).bfloat16()

    def qa(M, k):
        return m.quant(rnd(M, k, 0.5), 128)

    def gate(M, n):
        return torch.sigmoid(torch.randn(M, n, device="cuda") * 2).bfloat16()

    def tensors(tag, M, v):
        """(tensors, indices of the outputs compared bitwise)"""
        tag = tag.removesuffix(".g")
        if tag in ("hca.qkv", "dsa.qkv", "moe.gu", "dense.gu", "dense.down"):
            k = {"dense.down": 7168}.get(tag, 4096)
            n = {"hca.qkv": 1536, "dsa.qkv": 1600, "moe.gu": 2048, "dense.gu": 14336}.get(tag, 4096)
            return [*qa(M, k), bf(M, n)], [2]
        if tag == "hca.qup":
            return [rnd(M, 1536, 1.0), v[0], v[1], u8(M, 1024), u8(M, 8), bf(M, 16384), bf(M, 512)], [5, 6]
        if tag == "dsa.qbi":
            return [rnd(M, 1600, 1.0), v[0], v[1], u8(M, 1024), u8(M, 8), bf(M, 8192), bf(M, 8192), bf(M, 512)], [5, 6, 7]
        if tag in ("hca.gate", "dsa.gate"):
            w, n = (1536, 4096) if tag == "hca.gate" else (1600, 8192)
            return [rnd(M, w, 1.0), v[0], u8(M, 1024), u8(M, 8), bf(2 * M, n)], [4]
        if tag == "hca.woa":
            return [*qa(4 * M, 4096), *qa(4 * M, 4096), bf(M, 4096), bf(M, 4096)], [4, 5]
        if tag == "hca.oproj":
            return ([rnd(M, 4096, 0.5), rnd(M, 4096, 0.5), gate(M, 4096), gate(M, 4096), u8(M, 4096), u8(M, 32),
                     bf(M, 4096), bf(M, 4096)], [6, 7])
        if tag == "dsa.oproj":
            return [rnd(M, 8192, 0.5), rnd(M, 8192, 0.5), u8(M, 8192), u8(M, 64), bf(M, 4096), bf(M, 4096)], [4, 5]
        if tag == "moe.fc1":
            return [*qa(M, 4096), bf(M, 2048), u8(M, 2048), u8(M, 64)], [2, 3, 4]
        if tag == "moe.down":
            return [rnd(M, 2048, 1.0), u8(M, 1024), u8(M, 8), bf(M, 4096)], [3]
        if tag == "moe.gd":
            return [*qa(M, 4096), bf(M, 2048), u8(M, 1024), u8(M, 8), bf(M, 4096)], [3, 4, 5]
        if tag == "moe.fc2":
            return [rnd(M, 2048, 0.3), v[0], rnd(M, 4096, 0.5), u8(M, 2048), u8(M, 16), bf(M, 4096)], [5]
        raise ValueError(tag)

    VECS = {"hca.qup": [H + "q_layernorm.weight", H + "kv_layernorm.weight"],
            "dsa.qbi": [D + "q_a_layernorm.weight", D + "kv_a_layernorm.weight"],
            "hca.gate": [H + "q_layernorm.weight"], "dsa.gate": [D + "q_a_layernorm.weight"],
            "moe.fc2": [F + "latent_norm.weight"]}
    ROWS = {"lane": 2048, "half": 1024, "part": 8192}
    for p_ in ("f8_decode", "f8_prefill", "f8_fuse_act", "f8_pdl", "f8_multi", "f8_fuse_ops", "f8_tile", "f8_tile_epi",
               "f8_tile_chain"):
        m.set_patch(p_, True)
    # variants: (label, P133 on, its forced config or −1, P134 on); --variants cur,patch,c3,c6s: current (P133 / P134
    # off), patch (on, the measured configs), cN = P133 with config N forced (+ P134 with 's')
    def var(t):
        if t == "cur":
            return ("current", False, -1, False)
        if t == "patch":
            return ("patch", True, -1, True)
        return (f"P133 c{t[1:].rstrip('s')}" + ("+P134" if t.endswith("s") else ""), True, int(t[1:].rstrip("s")),
                t.endswith("s"))
    VAR = [var(t) for t in (a.variants or "cur,patch").split(",")]
    tags = list(SITES) if a.sites == "all" else a.sites.split(",")
    save = {}
    for tag in tags:
        line, ws_, rows, calls = SITES[tag]
        hs = [m.upload(rk, nm, gr) for rk, nm, gr, _ in ws_]
        info = [m.info(h) for h in hs]
        v = [m.vec(0, nm) for nm in VECS.get(tag.removesuffix(".g"), [])]
        nbytes = sum(i[2] * i[1] for i in info)
        copies = [hs] + [[m.clone(h) for h in hs] for _ in range(max(1, -(-(160 << 20) // nbytes) - 1))]
        shape = " + ".join(f"{i[0]}×{i[1]}" + (f"/s{i[3]}" if i[3] > 1 else "") + (f"/g{i[4]}" if i[4] > 1 else "")
                           for i in info)
        Ms = [int(x) for x in a.sm.split(",")] if a.sm else [ROWS[rows]]
        print(f"[site {tag}] {line} · {shape} · M = {'/'.join(map(str, Ms))} ({rows}) · {calls} calls / 8192-row chunk")
        for M in Ms:
            T, outs = tensors(tag, M, v)

            def variant(nv):
                m.set_patch("f8_tile_seq", VAR[nv][1])
                m.set_patch("f8_situ_epi", VAR[nv][3])
                m.test_seq(VAR[nv][2])
            paths = []
            for nv in range(len(VAR)):
                variant(nv)
                paths.append(",".join(m.path(h, M, ws_[j][3]) for j, h in enumerate(hs)))
            res, smp = [], [[] for _ in VAR]
            for nv in range(len(VAR)):
                variant(nv)
                for i in outs:
                    T[i].zero_()
                m.site_run(tag, hs, T, M)
                res.append([T[i].clone() for i in outs])
            for _ in range(-(-a.runs // 5)):  # variants interleaved, 5 calls each a round
                for nv in range(len(VAR)):
                    variant(nv)
                    torch.cuda._sleep(5_000_000)
                    smp[nv] += m.site_bench(tag, copies, T, M, 5)
            t = [med(x) for x in smp]
            same = [all(torch.equal(x, y) for x, y in zip(res[0], r)) for r in res]
            ok &= all(same)
            for nv, r in enumerate(res[1:], 1):
                for j, (x, y) in enumerate(zip(res[0], r)):
                    if not torch.equal(x, y):
                        print(f"    M={M} {VAR[nv][0]} output {outs[j]}: {int((x != y).sum())} of {x.numel()} differ")
            fl = sum(2.0 * M * i[0] * i[1] for i in info)
            save[(tag, M)] = t
            print(f"    M={M}: " + " → ".join(f"{VAR[nv][0]} [{paths[nv]}] {t[nv]:.4f} ms {fl / t[nv] / 1e9:.0f} TFLOPS"
                                          + ("" if nv == 0 else f" bitwise {same[nv]}") for nv in range(len(VAR))))
            del T, res
        for c in copies[1:]:
            for h in c:
                m.release(h)
        for h in hs:
            m.release(h)
        torch.cuda.empty_cache()
    # per-chunk estimate from the profile's call counts (8192-row chunk: lane 2048, half 1024, part 8192); a combined
    # site replaces its parts, the GEMM-alone (.g) sites are not counted
    PARTS = {"moe.gd": ("moe.gu", "moe.down")}
    skip = {p_ for c_, ps_ in PARTS.items() if c_ in tags for p_ in ps_} | {t_ for t_ in tags if t_.endswith(".g")}
    tot = [0.0] * len(VAR)
    miss = []
    for tag in [t_ for t_ in tags if t_ not in skip]:
        k = (tag, ROWS[SITES[tag][2]])
        if k in save:
            for nv in range(len(VAR)):
                tot[nv] += save[k][nv] * SITES[tag][3]
        else:
            miss.append(tag)
    print("[sites] 8192-row chunk (estimate: call counts × isolated ms): " +
          " → ".join(f"{VAR[nv][0]} {tot[nv]:.1f}" for nv in range(len(VAR))) +
          f" ms (best −{tot[0] - min(tot):.1f})" + (f" · not counted: {','.join(miss)}" if miss else ""))
    for p_ in ("f8_tile_seq", "f8_situ_epi"):
        m.set_patch(p_, True)
    m.test_seq(-1)
    print("[gemm sites]", "all bitwise" if ok else "FAILED")
    sys.exit(0 if ok else 1)
# 1. activation quantization
torch.manual_seed(0)
for g in (128, 32):
    x = (torch.randn(37, 4096, device="cuda") * torch.logspace(-3, 2, 37, device="cuda")[:, None]).bfloat16()
    x[3, :256] = 0
    q, s = m.quant(x, g)
    deq = q.view(torch.float8_e4m3fn).float().view(37, -1, g) * torch.exp2(s.float() - 127)[..., None]
    ref = R.act_q8(x, g).view(37, -1, g)
    same = torch.equal(deq, ref)
    ok &= same
    print(f"[act_quant g{g}] {'bit-exact' if same else 'DIFFERS: max ' + str((deq - ref).abs().max().item())}")
if not a.no_time:
    for MM, k in ((1, 4096), (4, 4096), (4096, 4096)):
        torch.cuda._sleep(20_000_000)
        ms = med(m.bench_act(torch.randn(MM, k, device="cuda").bfloat16(), 128, 50))
        print(f"    act_quant M={MM} k={k}: {ms * 1e3:.1f} µs · {MM * k * 3 / ms / 1e6:.0f} GB/s")

cases = [  # (tag, names, split per name, groups)
    ("hca.qkv", [H + "linear_q_down_proj", H + "linear_kv_proj"], "rep", 1),
    ("hca.q_up", [H + "linear_q_up_proj"], "col", 1),
    ("hca.gate", [H + "gate_proj"], "col", 1),
    ("hca.wo_a", [H + "linear_o_group_proj"], "col", 4),
    ("hca.o_proj", [H + "o_proj"], "row", 1),
    ("dsa.qkv", [D + "q_a_proj", D + "kv_a_proj_with_mqa"], "rep", 1),
    ("dsa.q_b", [D + "q_b_proj"], "col", 1),
    ("dsa.gate", [D + "gate_proj"], "col", 1),
    ("dsa.iq", [D + "core_attention.indexer.wq_b"], "rep", 1),
    ("dsa.o_proj", [D + "o_proj"], "row", 1),
    ("moe.fc1", [F + "fc1_latent_proj"], "rep", 1),
    ("moe.sh_gu", [F + "shared_experts.w1", F + "shared_experts.w3"], "col", 1),
    ("moe.sh_down", [F + "shared_experts.w2"], "row", 1),
    ("moe.fc2", [F + "fc2_latent_proj"], "rep", 1),
    ("dense.gu", ["model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj"], "col", 1),
    ("dense.down", ["model.layers.0.mlp.down_proj"], "row", 1),
]
if a.only:
    cases = [c for c in cases if c[0] in a.only.split(",")]
EXACT_M = (1, 4, 8, 13, 32, 33, 300) if not a.quick else (1, 4, 32, 300)


def acts(X, M, groups, k):
    """(q, s, gx, gs) for M rows per group of X [groups·M0, k] (first M rows of each group block)"""
    M0 = X.shape[0] // groups
    Xm = torch.cat([X[j * M0:j * M0 + M] for j in range(groups)]) if groups > 1 else X[:M].contiguous()
    q, s = m.quant(Xm, 128)
    return Xm, q, s, (M * k, M * (k // 128)) if groups > 1 else (0, 0)


for tag, names, split, groups in cases:
    for rank in ((0,) if a.quick else (0, 1)):
        patches(False)
        h = m.upload(rank, names, groups)
        n, k, npad, sp, gr = m.info(h)
        W = torch.cat([ref_w(nm, rank, split) for nm in names], 0)
        M = 300
        X = (torch.randn(groups * 4096, k, device="cuda") * 0.5).bfloat16()
        Xm, q, s, (gx, gs) = acts(X, M, groups, k)
        y = m.forward(h, q, s, M, True, gx, gs)
        xd = R.act_q8(Xm, 128)
        if groups > 1:
            ng = n // groups
            ref = torch.cat([xd[j * M:(j + 1) * M] @ W[j * ng:(j + 1) * ng].t() for j in range(groups)], 1)
        else:
            ref = xd @ W.t()
        del W
        rel = ((y - ref).abs().max() / ref.abs().max()).item()
        yb = m.forward(h, q, s, M, False, gx, gs)
        neq = (yb != ref.bfloat16()).float().mean().item()
        # base batch invariance: rows 0 and 150 alone vs inside M = 300 / 17 / 64
        inv = True
        for MM in (1, 17, 64):
            for r0 in (0, 150):
                if groups > 1:
                    qq = torch.cat([q[j * M + r0:j * M + r0 + MM] for j in range(groups)])
                    ss = torch.cat([s[j * M + r0:j * M + r0 + MM] for j in range(groups)])
                    y2 = m.forward(h, qq, ss, MM, True, MM * k, MM * (k // 128))
                else:
                    y2 = m.forward(h, q[r0:r0 + MM].contiguous(), s[r0:r0 + MM].contiguous(), MM, True, 0, 0)
                inv &= torch.equal(y2, y[r0:r0 + MM])
        good = rel < 2e-6 and neq < 1e-3 and inv
        # patch vs base, bitwise, at every M (bf16 and fp32 out); P032 from the bf16 activation at M ≤ 8
        same, fused = True, True
        for MM in EXACT_M + ((4096,) if rank == 0 else ()):
            Xm, q, s, (gx, gs) = acts(X, MM, groups, k)
            outs = []
            for on in (False, True):
                patches(on)
                outs.append([m.forward(h, q, s, MM, f32, gx, gs) for f32 in (True, False)])
            same &= all(torch.equal(u, v) for u, v in zip(*outs))
            if groups == 1 and MM <= 8:
                patches(True)
                assert m.defer_act(MM)
                q2, s2 = torch.empty_like(q), torch.empty_like(s)
                fz = [m.forward(h, q2, s2, MM, f32, 0, 0, Xm) for f32 in (True, False)]
                fused &= all(torch.equal(u, v) for u, v in zip(fz, outs[0]))
        patches(False)
        good &= same and fused
        ok &= good
        print(f"[{tag} r{rank}] {n}×{k} split {sp} groups {gr} · base: rel err {rel:.2e} · bf16 ≠ ref {neq:.1e} · "
              f"batch-invariant {inv} · patch == base {same} · fused act == base {fused if groups == 1 else '-'} · "
              f"{'OK' if good else 'FAIL'}")
        if rank == 0 and not a.no_time:
            row = [tag, n, k, sp]
            for MM in ((1, 4) if "decode" in a.time else ()):
                Xm, q, s, (gx, gs) = acts(X, MM, groups, k)
                t = []
                for on in (False, True):
                    patches(on)
                    torch.cuda._sleep(20_000_000)
                    t.append(med(m.bench(h, q, s, MM, 40, True, gx, gs)))
                gb = [n * k / v / 1e6 for v in t]
                line = (f"    decode M={MM} cold: {t[0] * 1e3:.1f} → {t[1] * 1e3:.1f} µs (×{t[0] / t[1]:.2f}) · "
                        f"{gb[0]:.0f} → {gb[1]:.0f} GB/s")
                row += t
                if groups == 1:  # with the activation quantization: act_quant + GEMM → fused (P032)
                    tf = []
                    for on in (False, True):
                        patches(on)
                        torch.cuda._sleep(20_000_000)
                        tf.append(med(m.bench(h, q, s, MM, 40, True, 0, 0, Xm, on)))
                    line += f" · +act: {tf[0] * 1e3:.1f} → {tf[1] * 1e3:.1f} µs (×{tf[0] / tf[1]:.2f})"
                    row += tf
                # 8 calls back to back (no event between): P033 lets each call start under the previous one
                tc = []
                for on, pdl in ((False, False), (True, False), (True, True)):
                    patches(on)
                    m.set_patch("f8_pdl", pdl)
                    torch.cuda._sleep(20_000_000)
                    tc.append(med(m.bench(h, q, s, MM, 20, True, gx, gs, chain=8)))
                line += f" · chained: {tc[0] * 1e3:.1f} → {tc[1] * 1e3:.1f} → PDL {tc[2] * 1e3:.1f} µs ({n * k / tc[2] / 1e6:.0f} GB/s)"
                print(line)
            if "mid" in a.time:  # chunk / verify sizes between decode and prefill, cold
                line = "    mid M:"
                for MM in (16, 32, 48, 64, 128, 256, 512):
                    Xm, q, s, (gx, gs) = acts(X, MM, groups, k)
                    t = []
                    for on in (False, True):
                        patches(on)
                        torch.cuda._sleep(20_000_000)
                        t.append(med(m.bench(h, q, s, MM, 20, True, gx, gs)))
                    line += f" {MM}: {t[0] * 1e3:.0f} → {t[1] * 1e3:.0f} µs ·"
                print(line.rstrip(" ·"))
            MM = 4096
            Xm, q, s, (gx, gs) = acts(X, MM, groups, k)
            t = [1, 1] if "prefill" not in a.time else []
            for on in ((False, True) if "prefill" in a.time else ()):
                patches(on)
                torch.cuda._sleep(20_000_000)
                t.append(med(m.bench(h, q, s, MM, 20, False, gx, gs)))
            tf = [2 * n * k * MM / v / 1e9 for v in t]
            if "prefill" in a.time:
                print(f"    prefill M=4096: {t[0]:.3f} → {t[1]:.3f} ms (×{t[0] / t[1]:.2f}) · {tf[0]:.0f} → {tf[1]:.0f} TFLOPS")
            patches(False)
        m.release(h)
        torch.cuda.empty_cache()
# P034: weights on one activation as one launch — bitwise vs separate base calls; time base → P030 separate → P034
multi = [("hca q_up+gate", [[H + "linear_q_up_proj"], [H + "gate_proj"]]),
         ("dsa q_b+gate+iq", [[D + "q_b_proj"], [D + "gate_proj"], [D + "core_attention.indexer.wq_b"]]),
         ("moe fc1+sh_gu", [[F + "fc1_latent_proj"], [F + "shared_experts.w1", F + "shared_experts.w3"]])]
for tag, parts in multi:
    if a.only and not any(tag.split()[0] + "." in c for c in a.only.split(",")) and "multi" not in a.only:
        continue
    for rank in ((0,) if a.quick else (0, 1)):
        patches(False)
        hs = [m.upload(rank, nm, 1) for nm in parts]
        k = m.info(hs[0])[1]
        X = (torch.randn(32, k, device="cuda") * 0.5).bfloat16()
        same = True
        for MM in (1, 4, 8, 13, 32):
            q, s = m.quant(X[:MM].contiguous(), 128)
            patches(False)
            ref = [m.forward(h, q, s, MM, False, 0, 0) for h in hs]
            patches(True)
            ys = m.forward_multi(hs, q, s, MM)
            same &= all(torch.equal(u, v) for u, v in zip(ys, ref))
            if MM <= 8:
                ys = m.forward_multi(hs, torch.empty_like(q), torch.empty_like(s), MM, X[:MM].contiguous())
                same &= all(torch.equal(u, v) for u, v in zip(ys, ref))
        patches(False)
        ok &= same
        print(f"[multi {tag} r{rank}] one launch == separate base calls (M 1–32, fused act M ≤ 8): {same}")
        if rank == 0 and not a.no_time and "decode" in a.time:
            nbytes = sum(m.info(h)[0] * m.info(h)[1] for h in hs)
            for MM in (1, 4):
                Xm = X[:MM].contiguous()
                q, s = m.quant(Xm, 128)
                t = []
                for cfg in ((False, False), (True, False), (True, True)):
                    patches(cfg[0])
                    m.set_patch("f8_multi", cfg[1])
                    torch.cuda._sleep(20_000_000)
                    t.append(med(m.bench_multi(hs, q, s, MM, 40)))
                tf = []
                for on in (False, True):  # + activation quantization: act_quant + separate → fused (P032) one launch
                    patches(on)
                    torch.cuda._sleep(20_000_000)
                    tf.append(med(m.bench_multi(hs, q, s, MM, 40, Xm, quant_first=not on)))
                torch.cuda._sleep(20_000_000)
                tc = med(m.bench_multi(hs, q, s, MM, 20, Xm, chain=8))  # all patches on, back to back
                patches(False)
                print(f"    decode M={MM} cold: {t[0] * 1e3:.1f} → {t[1] * 1e3:.1f} → one launch {t[2] * 1e3:.1f} µs "
                      f"({nbytes / t[2] / 1e6:.0f} GB/s) · +act: {tf[0] * 1e3:.1f} → {tf[1] * 1e3:.1f} µs · "
                      f"chained all on {tc * 1e3:.1f} µs ({nbytes / tc / 1e6:.0f} GB/s)")
        for h in hs:
            m.release(h)
        torch.cuda.empty_cache()

# P033 / P032 in a chain: x → [act_quant →] o_proj → [act_quant →] o_proj … 6 times, every launch queued back to back
# behind a GPU sleep; with P032 each GEMM reads the previous GEMM's output directly (the previous one triggers its
# dependents early: a kernel that read its input before the previous one finished would change the result)
if not a.only or "hca.o_proj" in a.only:
    patches(False)
    h = m.upload(0, [H + "o_proj"], 1)
    for MM in (1, 4, 64):
        X = (torch.randn(MM, 4096, device="cuda") * 0.5).bfloat16()
        outs = []
        for on, fuse in ((False, False), (True, False), (True, True)):
            patches(on)
            m.set_patch("f8_fuse_act", fuse)
            y = X
            torch.cuda._sleep(50_000_000)
            for _ in range(6):
                if fuse and m.defer_act(MM):
                    y = m.forward(h, torch.empty(MM, 4096, dtype=torch.uint8, device="cuda"),
                                  torch.empty(MM, 32, dtype=torch.uint8, device="cuda"), MM, False, 0, 0, y)
                else:
                    q, s = m.quant(y, 128)
                    y = m.forward(h, q, s, MM, False, 0, 0)
            outs.append(y)
        same = all(torch.equal(outs[0], o) for o in outs[1:]) and bool(outs[0].float().isfinite().all())
        ok &= same
        print(f"[chain M={MM}] 6 × (act → o_proj) queued back to back: base == P030+P033 == +P032 {same} "
              f"(|y| max {outs[0].float().abs().max().item():.3g})")
    patches(False)
    m.release(h)
print("[gemm]", "all OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
