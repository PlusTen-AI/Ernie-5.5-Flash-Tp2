#!/usr/bin/env python3
"""test_gemm_mid.py — P200 f8_mid (kernels/gemm/p200_f8_mid.cuh): the FP8 block GEMM at 33 … 128 rows (verify / draft
steps of 4–16 sequences) on real checkpoint weights, both ranks' slices, against the decode path.
  python3 engine/tests/test_gemm_mid.py --model /models/ERNIE-5.5-Flash-Preview --table /work/cache/table.tsv [--quick]
  [--only hca.qkv,moe.fc1] [--no-time] [--sweep] [--runs 20]
1. every ERNIE FP8 linear (test_gemm.py's cases), M = 33 / 48 / 64 / 96 / 128, bf16 and fp32 outputs, bitwise:
   P200 at every token tile (MT 32 / 64 / 128, forced) and with its measured rule == the decode path in 32-row pieces (P030, or the base kernel where P030 cannot take the shape: every row's
   decode chain) == today's path at M (P200 off: base
   k_f8<64 / 128> + k_reduce, or P031) == the base (every patch off)
2. the call sites with the GEMM's epilogue / several weights in one launch (P034 + P060 σ, + e, MXFP8 copy; the
   q norm → q_up / q_b + iq sites through p060::prep): P200 off vs on, bitwise on every output
3. µs per call, cold (weights rotated over copies ≫ L2): today → the rule (P200 where it is faster) at M = 33 / 64 / 128
   (the sites too), the byte floor (weights + activation + bf16 output once at 230 GB/s) and the best's share of it;
   --sweep: today and P200 forced at MT 32 / 64 / 128 (the numbers the rule is read from)
--mid32: P220 f8_mid32 on with P200 (P200 from 17 rows where P030 cannot take the shape); with --ms 17,24,28,32
   --tm 24,32 the same checks and timings at the B = 3–4 verify / draft rows"""
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
ap.add_argument("--quick", action="store_true", help="rank 0 only")
ap.add_argument("--only", default="", help="comma list of case / site tags")
ap.add_argument("--no-time", action="store_true")
ap.add_argument("--sweep", action="store_true", help="time P200 at every token tile (forced)")
ap.add_argument("--runs", type=int, default=20)
ap.add_argument("--mid32", action="store_true", help="P220 f8_mid32 on wherever P200 is (rule / forced modes)")
ap.add_argument("--ms", default="", help="M values of the bit checks (default 33,48,64,96,128)")
ap.add_argument("--tm", default="", help="M values timed (default 33,64,128)")
a = ap.parse_args()
m = ext.load("gemm", ["src/kernels/gemm/fp8.cu", "src/core/weights.cc", "src/model/patches.cc"])
m.open(a.model, a.table)
# the FP8 patches the engine runs by default (all on); P200 switched on its own
PATCHES = ("f8_decode", "f8_prefill", "f8_fuse_act", "f8_pdl", "f8_multi", "f8_fuse_ops", "f8_pf_next", "f8_tile",
           "f8_tile_epi", "f8_tile_chain", "f8_tile_seq", "f8_situ_epi")
BW = 230e3  # bytes per µs: GB10's measured streaming rate (~230 GB/s, limit ≈ 237)
MS = tuple(int(x) for x in a.ms.split(",")) if a.ms else (33, 48, 64, 96, 128)
TM = tuple(int(x) for x in a.tm.split(",")) if a.tm else (33, 64, 128)


MTS = (32, 64, 128)


def mode(name):
    """base: every patch off · today: the engine's defaults, P200 off · rule: defaults + P200 (its measured rule) ·
    mt32 / mt64 / mt128: defaults + P200 forced at that token tile wherever it can run"""
    for p in PATCHES:
        m.set_patch(p, name != "base")
    m.set_patch("f8_mid", name not in ("base", "today"))
    m.set_patch("f8_mid32", a.mid32 and name not in ("base", "today"))
    m.test_mid(int(name[2:]) if name.startswith("mt") else 0)


FORCED = tuple(f"mt{t}" for t in MTS)
TIMED = ("today",) + FORCED if a.sweep else ("today", "rule")


def med(v):
    return statistics.median(v)


ok = True
H, D, F = "model.layers.3.self_attn.", "model.layers.7.self_attn.", "model.layers.5.block_sparse_moe."
cases = [  # (tag, names, groups) — as test_gemm.py
    ("hca.qkv", [H + "linear_q_down_proj", H + "linear_kv_proj"], 1),
    ("hca.q_up", [H + "linear_q_up_proj"], 1),
    ("hca.gate", [H + "gate_proj"], 1),
    ("hca.wo_a", [H + "linear_o_group_proj"], 4),
    ("hca.o_proj", [H + "o_proj"], 1),
    ("dsa.qkv", [D + "q_a_proj", D + "kv_a_proj_with_mqa"], 1),
    ("dsa.q_b", [D + "q_b_proj"], 1),
    ("dsa.gate", [D + "gate_proj"], 1),
    ("dsa.iq", [D + "core_attention.indexer.wq_b"], 1),
    ("dsa.o_proj", [D + "o_proj"], 1),
    ("moe.fc1", [F + "fc1_latent_proj"], 1),
    ("moe.sh_gu", [F + "shared_experts.w1", F + "shared_experts.w3"], 1),
    ("moe.sh_down", [F + "shared_experts.w2"], 1),
    ("moe.fc2", [F + "fc2_latent_proj"], 1),
    ("dense.gu", ["model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj"], 1),
    ("dense.down", ["model.layers.0.mlp.down_proj"], 1),
]
only = set(a.only.split(",")) if a.only else None


def acts(X, M, groups, k, i=0, mm=None):
    """(q, s, gx, gs): rows i … i + mm of the first M rows of each group block of X [groups · 4096, k]"""
    mm = M if mm is None else mm
    rows = torch.cat([X[j * 4096 + i:j * 4096 + i + mm] for j in range(groups)]) if groups > 1 else X[i:i + mm]
    q, s = m.quant(rows.contiguous(), 128)
    return q, s, (mm * k, mm * (k // 128)) if groups > 1 else (0, 0)


# ---- 1 + 3: every linear
for tag, names, groups in cases:
    if only and tag not in only:
        continue
    for rank in ((0,) if a.quick else (0, 1)):
        mode("base")
        h = m.upload(rank, names, groups)
        n, k, npad, sp, gr = m.info(h)
        X = (torch.randn(groups * 4096, k, device="cuda") * 0.5).bfloat16()
        good, paths = True, {}
        for M in MS:
            q, s, (gx, gs) = acts(X, M, groups, k)
            res = {}
            for md in ("base", "today", "rule") + FORCED:
                mode(md)
                paths[(md, M)] = m.path(h, M, 0)
                res[md] = [m.forward(h, q, s, M, f32, gx, gs) for f32 in (True, False)]
            mode("today")  # the decode path: P030 on 32-row pieces
            dec = [[], []]
            for i in range(0, M, 32):
                mm = min(32, M - i)
                qq, ss, (gx2, gs2) = acts(X, M, groups, k, i, mm)
                for j, f32 in enumerate((True, False)):
                    dec[j].append(m.forward(h, qq, ss, mm, f32, gx2, gs2))
            res["decode"] = [torch.cat(d) for d in dec]
            for md in ("today", "base", "rule") + FORCED:
                if not all(torch.equal(u, v) for u, v in zip(res["decode"], res[md])):
                    good = False
                    nd = int((res["decode"][1] != res[md][1]).sum())
                    print(f"    M={M}: {md} ≠ decode pieces ({nd} of {res[md][1].numel()} bf16 outputs differ)")
            for md in FORCED:
                if paths[(md, M)] != "P200":
                    good = False
                    print(f"    M={M}: P200 forced ({md}) takes {paths[(md, M)]}")
        mode("today")
        ok &= good
        today = "/".join(sorted({paths[("today", M)] for M in MS}))
        rule = "/".join(paths[("rule", M)] for M in MS)
        print(f"[{tag} r{rank}] {n}×{k} split {sp} groups {gr} · today {today} · rule {rule} · P200 (MT 32/64/128, "
              f"rule) == decode pieces == today == base at M {'/'.join(map(str, MS))} (bf16 + fp32): "
              f"{'OK' if good else 'FAIL'}")
        if rank == 0 and not a.no_time:
            line = f"    cold µs {' / '.join(TIMED)} (floor µs · share of the best):"
            for M in TM:
                q, s, (gx, gs) = acts(X, M, groups, k)
                t = {}
                for md in TIMED:
                    mode(md)
                    torch.cuda._sleep(20_000_000)
                    t[md] = m.bench(h, q, s, M, a.runs, True, gx, gs)
                us = {md: med(v) * 1e3 for md, v in t.items()}
                fl = (n * k + groups * M * k + M * n * 2) / BW
                best = min(us.values())
                line += (f" M={M} [{paths[('today', M)]}→{paths[('rule', M)]}] "
                         f"{' / '.join(f'{us[md]:.1f}' for md in TIMED)} ({fl:.1f} · {fl / best * 100:.0f} %) ·")
            mode("today")
            print(line.rstrip(" ·"))
        m.release(h)
        torch.cuda.empty_cache()

# ---- 2 + 3: the call sites with epilogues in the store / several weights in one launch (tests/bind/gemm.cpp site())
SITES = {  # tag: [(rank, names[, groups])], norm vectors
    # the verify-step calls (model.cc at 33 … 128 rows, P078 halves off)
    "hca.qg": ([(0, [H + "linear_q_up_proj"]), (0, [H + "gate_proj"])], [H + "q_layernorm.weight", H + "kv_layernorm.weight"]),
    "dsa.qgi": ([(0, [D + "q_b_proj"]), (0, [D + "gate_proj"]), (0, [D + "core_attention.indexer.wq_b"])],
                [D + "q_a_layernorm.weight", D + "kv_a_layernorm.weight"]),
    "hca.wo": ([(0, [H + "linear_o_group_proj"], 4), (0, [H + "o_proj"])], []),  # wo_a → z ⊙ gate → o_proj
    "moe.front": ([(0, [F + "fc1_latent_proj"]), (0, [F + "shared_experts.w1", F + "shared_experts.w3"])], []),
    "moe.fc2": ([(0, [F + "fc2_latent_proj"])], [F + "latent_norm.weight"]),  # latent_norm → fc2 + shared
    # other sites
    "hca.gate.g": ([(0, [H + "gate_proj"]), (1, [H + "gate_proj"])], []),  # 2 weights, σ each
    "hca.oproj.g": ([(0, [H + "o_proj"]), (1, [H + "o_proj"])], []),  # plain, then + partial (kEpiAdd)
    "moe.fc1": ([(0, [F + "fc1_latent_proj"])], []),  # + MXFP8 copy (kEpiQ32, cluster path)
    "moe.fc2.g": ([(0, [F + "fc2_latent_proj"])], []),  # + shared (kEpiAdd)
    "hca.qup": ([(0, [H + "linear_q_up_proj"])], [H + "q_layernorm.weight", H + "kv_layernorm.weight"]),  # prep → P200
    "dsa.qbi": ([(0, [D + "q_b_proj"]), (0, [D + "core_attention.indexer.wq_b"])],
                [D + "q_a_layernorm.weight", D + "kv_a_layernorm.weight"]),  # prep → one P200 launch for both
}


def tensors(tag, M, v):
    """(site tensors, indices of the outputs compared)"""
    def bf(*sh):
        return torch.zeros(*sh, dtype=torch.bfloat16, device="cuda")

    def u8(*sh):
        return torch.zeros(*sh, dtype=torch.uint8, device="cuda")

    def rnd(r, c, sc):
        return (torch.randn(r, c, device="cuda") * sc).bfloat16()

    def qa(r, c):
        return m.quant(rnd(r, c, 0.5), 128)
    if tag == "hca.qg":
        return [rnd(M, 1536, 1.0), v[0], v[1], u8(M, 1024), u8(M, 8), bf(M, 16384), bf(M, 4096), bf(M, 512)], [5, 6, 7]
    if tag == "dsa.qgi":
        return ([rnd(M, 1600, 1.0), v[0], v[1], u8(M, 1024), u8(M, 8), bf(M, 8192), bf(M, 8192), bf(M, 8192),
                 bf(M, 512)], [5, 6, 7, 8])
    if tag == "hca.wo":
        return [*qa(4 * M, 4096), rnd(M, 4096, 1.0), bf(M, 4096), u8(M, 4096), u8(M, 32), bf(M, 4096)], [3, 6]
    if tag == "moe.front":
        return [*qa(M, 4096), bf(M, 2048), u8(M, 2048), u8(M, 64), bf(M, 2048)], [2, 3, 4, 5]
    if tag == "moe.fc2":
        return [rnd(M, 2048, 1.0), v[0], rnd(M, 4096, 0.5), u8(M, 2048), u8(M, 16), bf(M, 4096)], [5]
    if tag == "hca.gate.g":
        return [rnd(M, 1536, 1.0), bf(1), *qa(M, 1024), bf(2 * M, 4096)], [4]
    if tag == "hca.oproj.g":
        return [bf(1), bf(1), bf(1), bf(1), *qa(M, 4096), bf(M, 4096), bf(M, 4096)], [6, 7]
    if tag == "moe.fc1":
        return [*qa(M, 4096), bf(M, 2048), u8(M, 2048), u8(M, 64)], [2, 3, 4]
    if tag == "moe.fc2.g":
        return [bf(1), bf(1), rnd(M, 4096, 0.5), *qa(M, 2048), bf(M, 4096)], [5]
    if tag == "hca.qup":
        return [rnd(M, 1536, 1.0), v[0], v[1], u8(M, 1024), u8(M, 8), bf(M, 16384), bf(M, 512)], [5, 6]
    if tag == "dsa.qbi":
        return [rnd(M, 1600, 1.0), v[0], v[1], u8(M, 1024), u8(M, 8), bf(M, 8192), bf(M, 8192), bf(M, 512)], [5, 6, 7]
    raise ValueError(tag)


for tag, (ws_, vecs) in SITES.items():
    if only and tag not in only:
        continue
    mode("today")
    hs = [m.upload(w[0], w[1], w[2] if len(w) > 2 else 1) for w in ws_]
    v = [m.vec(0, nm) for nm in vecs]
    good = True
    for M in MS:
        T, outs = tensors(tag, M, v)
        res = []
        for md in ("today", "rule") + FORCED:
            mode(md)
            for i in outs:
                T[i].zero_()
            m.site_run(tag, hs, T, M)
            res.append([T[i].clone() for i in outs])
        for md, r_ in zip(("rule",) + FORCED, res[1:]):
            for j, (x, y) in enumerate(zip(res[0], r_)):
                if not torch.equal(x, y):
                    good = False
                    print(f"    M={M} {md} output {outs[j]}: {int((x != y).sum())} of {x.numel()} differ")
    ok &= good
    print(f"[site {tag}] P200 off == rule == MT 32/64/128 at M {'/'.join(map(str, MS))}: {'OK' if good else 'FAIL'}")
    if not a.no_time:
        info = [m.info(h) for h in hs]
        nbytes = sum(i[0] * i[1] for i in info)
        copies = [hs] + [[m.clone(h) for h in hs] for _ in range(max(1, -(-(160 << 20) // nbytes) - 1))]
        line = f"    cold µs {' / '.join(TIMED)} (floor µs):"
        for M in TM:
            T, _ = tensors(tag, M, v)
            t = {md: [] for md in TIMED}
            for _ in range(-(-a.runs // 5)):
                for md in t:
                    mode(md)
                    torch.cuda._sleep(5_000_000)
                    t[md] += m.site_bench(tag, copies, T, M, 5)
            us = {md: med(x) * 1e3 for md, x in t.items()}
            fl = sum(i[0] * i[1] + M * i[1] + M * i[0] * 2 for i in info) / BW
            line += f" M={M} {' / '.join(f'{us[md]:.1f}' for md in TIMED)} ({fl:.1f}) ·"
        print(line.rstrip(" ·"))
        for c in copies[1:]:
            for h in c:
                m.release(h)
    for h in hs:
        m.release(h)
    torch.cuda.empty_cache()
mode("today")
print("[gemm mid]", "all OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
