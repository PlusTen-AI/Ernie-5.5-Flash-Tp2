#!/usr/bin/env python3
"""test_lindec.py — P114 lin_pair / P115 lin_small / P165 lin_small2 (kernels/linear p114_lin_pair.cuh), decode / verify
rows (rank 0):
the router (gate.weight ‖ gate.weight_1) and HCA compressor (linear_wkv ‖ linear_wgate) as the engine holds them (12-bit
copy only, P062 + P098) and their P078 column halves (both ranks' views), the DSA indexer k ‖ weights_proj (bf16),
lm_head (12-bit only).
1. bitwise (fp32 / bf16 bits): P114, P115, both (and both with 2- / 4-warp CTAs forced), both + P165 == P114 / P115 /
   P165 off — the halves pairs, the whole router / compressor, the indexer — at M = 1, 4, 8 (+ 16, 32: NT 2 / 4)
2. P165: µs per call, 16 layers in turn (48 / 96 MB of weights > L2: cold), one event pair a round (median of 9), each
   call behind a 10 µs programmatic predecessor (spins subtracted) and back to back: the router / compressor as the
   engine calls them — P078 halves pairs at M = 1, 5, 8, 16, the whole Lins at M = 32 — P165 off → on; × 42 / 37 a step
3. (--all) µs per call, cold (256 MB read before every call, x re-read: it is hot in the engine), median of 24 (lm_head 20):
   router pair, compressor pair (8 layers each), indexer k (the 6 DSA layers), lm_head at M = 1, 4, 8 — current →
   P114 · P115 · both; the pairs also behind a 10 µs predecessor (lindec_spin.cu: the call a programmatic dependent
   whose prologue runs under it, as in the engine; the 10 µs subtracted)
  tools/agent.sh lindec --gpu python3 engine/tests/test_lindec.py --model /models/ERNIE-5.5-Flash-Preview --table /work/cache/table.tsv"""
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
ap.add_argument("--log", default="/tmp/build-lindec.log")
ap.add_argument("--all", action="store_true", help="also section 3 (P114 / P115 per-call times)")
a = ap.parse_args()
m = ext.load("lindec", ["src/kernels/linear/linear.cu", "src/core/weights.cc", "src/model/patches.cc",
                        "tests/bind/lindec_spin.cu"], log=a.log)
m.open(a.model, a.table, 0)

MOE = list(range(1, 17))
HCA = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18]
DSA = [7, 14, 21, 28, 35, 42]
pre = "model.layers.{}."
router = [m.lin([pre.format(i) + "block_sparse_moe.gate.weight", pre.format(i) + "block_sparse_moe.gate.weight_1"], 2)
          for i in MOE]
comp = [m.lin([pre.format(i) + "self_attn.core_attention.compressor.linear_wkv.weight",
               pre.format(i) + "self_attn.core_attention.compressor.linear_wgate.weight"], 2) for i in HCA]
ikw = [m.lin([pre.format(i) + "self_attn.core_attention.indexer.wk.weight",
              pre.format(i) + "self_attn.core_attention.indexer.weights_proj.weight"], 0) for i in DSA]
lm = m.lin(["lm_head.weight"], 2)


def halves(h, r):  # model.cc: lin_rows(L, r·256, 256), lin_rows(L, 512 + r·256, 256)
    return m.rows(h, r * 256, 256), m.rows(h, 512 + r * 256, 256)


pairs = {"router": [[halves(h, r) for h in router] for r in (0, 1)],
         "compressor": [[halves(h, r) for h in comp] for r in (0, 1)]}
for tag, hs in (("router", router), ("compressor", comp), ("indexer k", ikw), ("lm_head", [lm])):
    n, k, sp, hw, hk, nx = m.lin_info(hs[0])
    print(f"[lindec] {tag}: {n}×{k}, {sp} K splits, bf16 copy {hw}, 12-bit copy {hk}")

CFG = {"current": (False, False, 0, False), "P114": (True, False, 0, False), "P115": (False, True, 0, False),
       "both": (True, True, 0, False), "both W2": (True, True, 2, False), "both W4": (True, True, 4, False),
       "both+P165": (True, True, 0, True)}


def setp(c):
    m.set_patch("lin_pair", CFG[c][0])
    m.set_patch("lin_small", CFG[c][1])
    m.warps(CFG[c][2])
    m.set_patch("lin_small2", CFG[c][3])


def bits(t):
    return t.view(torch.int32) if t.dtype == torch.float32 else t.view(torch.int16)


def outs(x):
    o = []
    for tag in pairs:
        for r in (0, 1):
            for a_, b_ in pairs[tag][r]:
                o += [bits(t).clone() for t in m.pair(a_, b_, x)]
    o += [bits(m.fwd(h, x)).clone() for h in router + comp]
    o += [bits(m.fwd_bf16(h, x)).clone() for h in ikw]
    return o


ok = True
torch.manual_seed(0)
for M in (1, 4, 8, 16, 32):
    x = torch.randn(M, 4096, device="cuda").bfloat16()
    setp("current")
    ref = outs(x)
    res = []
    for c in ("P114", "P115", "both", "both W2", "both W4", "both+P165"):
        setp(c)
        same = all(torch.equal(u, v) for u, v in zip(outs(x), ref))
        ok &= same
        res.append(f"{c} {'identical' if same else 'DIFFERENT'}")
    print(f"[lindec] bitwise M={M}: " + " · ".join(res))

md = statistics.median
print("[lindec] P165: µs per call (16 layers in turn, median of 9 rounds), P165 off → on: behind a 10 µs predecessor ·"
      " back to back; ms a step (router × 42, compressor × 37)")
for M in (1, 5, 8, 16, 32):
    x = torch.randn(M, 4096, device="cuda").bfloat16()
    line, step = [], {}
    for tag, calls in (("router", 42), ("compressor", 37)):
        kind, hs = (2, [h for p in pairs[tag][0] for h in p]) if M <= 16 else (0, router if tag == "router" else comp)
        t = {}
        for c in ("both", "both+P165"):
            setp(c)
            t[c, 1] = md(m.timed_batch(kind, hs, x, 9, 10000))
            t[c, 0] = md(m.timed_batch(kind, hs, x, 9, 0))
        for c in t:
            step[c] = step.get(c, 0.0) + t[c] * calls / 1000
        line.append(f"{tag} {'pair' if M <= 16 else 'whole'} {t['both', 1]:.2f} → {t['both+P165', 1]:.2f} · "
                    f"{t['both', 0]:.2f} → {t['both+P165', 0]:.2f}")
    print(f"[lindec] M={M}: " + " | ".join(line) + f" · step behind {step['both', 1]:.2f} → {step['both+P165', 1]:.2f} ms,"
          f" back to back {step['both', 0]:.2f} → {step['both+P165', 0]:.2f} ms")

if a.all:
    flush = torch.empty(64 << 20, device="cuda", dtype=torch.float32).fill_(1.0)  # 256 MB
    print(f"[lindec] µs per call, cold (L2 {torch.cuda.get_device_properties(0).L2_cache_size >> 20} MB; 256 MB read before "
          "each call), median; [behind a 10 µs predecessor]")
    md = statistics.median
    for M in (1, 4, 8):
        x = torch.randn(M, 4096, device="cuda").bfloat16()
        line = []
        for tag in pairs:
            flat = [h for p in pairs[tag][0] for h in p]
            t, t10 = {}, {}
            for c in ("current", "P114", "P115", "both"):
                setp(c)
                t[c] = md(m.timed(2, flat, x, flush, 3, 0))
                t10[c] = md(m.timed(2, flat, x, flush, 3, 10000))
            line.append(f"{tag} pair {t['current']:.1f} → P114 {t['P114']:.1f} · P115 {t['P115']:.1f} · both {t['both']:.1f} "
                        f"[{t10['current']:.1f} → {t10['P114']:.1f} · {t10['P115']:.1f} · {t10['both']:.1f}]")
        t = {}
        for c in ("current", "both"):
            setp(c)
            t[c] = md(m.timed(1, ikw, x, flush, 4, 0))
        line.append(f"indexer k {t['current']:.1f} → both {t['both']:.1f}")
        for c in ("current", "both"):
            setp(c)
            t[c] = md(m.timed(0, [lm], x, flush, 20, 0))
        line.append(f"lm_head {t['current']:.0f} → both {t['both']:.0f}")
        print(f"[lindec] M={M}: " + " | ".join(line))


setp("current")
m.free_all()
print("[lindec] " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
