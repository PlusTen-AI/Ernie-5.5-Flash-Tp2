#!/usr/bin/env python3
"""test_dsa.py — the DSA attention chain (kernels/attn/dsa.cu: MLA cache rows, indexer prep, q_pe, W_UK / W_UV batched
GEMMs, top-k selection, sparse MLA attention, VHA + gate) base vs patches P045–P048, on one real DSA layer's weights
(kv_b_proj → W_UK / W_UV, softmax_offset sink, indexer k_norm, kv_a_layernorm, VHA V; rank 0) and realistic random rows
at the real shapes; the caches' history filled through the base kernels.
  python3 engine/tests/test_dsa.py --model /models/ERNIE-5.5-Flash-Preview --table /work/cache/table.tsv
First the DSA RoPE against vLLM's bf16 cos_sin_cache formula. Per shape: every output of the chain (caches, q576,
selected key lists + counts, attention latent, W_UV output, gated output) compared bit for bit, then ms per stage
base → patch (CUDA events, median; decode cold: L2 flushed before every run). Extra cases: ties (7 distinct indexer
keys: the top-2048 boundary inside a run of equal logits) and wide key-scale spreads (P046's k32 fallback path)."""
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
ap.add_argument("--layer", type=int, default=7)
ap.add_argument("--only", default="", help="comma list of case-name substrings")
ap.add_argument("--runs", type=int, default=21)
ap.add_argument("--base-only", action="store_true")
ap.add_argument("--no-time", action="store_true")
ap.add_argument("--prefill", action="store_true",
                help="session-5 prefill round, step 2: dsa_attn at M ∈ {1024, 2048} × chunk position ∈ {0, 16384, 49152}, "
                     "current default → + P110: bitwise check, then ms")
ap.add_argument("--shapes", default="", help="--prefill subset, e.g. 1024@16384,2048@0")
ap.add_argument("--wide", action="store_true",
                help="session 6: P142 dsa_select_wide — dsa_select at ≥ 65536 candidates, P046 (P142 off) vs P091, "
                     "bitwise (random rows; one equal key everywhere and three distinct keys: the 32-bit count path), "
                     "then ms of 4096 rows at 131072")
ap.add_argument("--split", action="store_true",
                help="session 6: P140 / P141 as model.cc runs them on a P094 lane, both ranks emulated on one GPU (own "
                     "rows' query side and select, the peer's first 2048 list entries copied, dsa_select_tail, the "
                     "attention own rows then the peer's) against the whole lane, bitwise")
ap.add_argument("--long", action="store_true",
                help="session 6: P153 dsa_select_long — the k64 FP4 MMA's peak (pure MMA loop, dsa_peak.cu), then "
                     "dsa_select of one 1024-row call at --sizes candidates (P153 off → on: ms, TFLOPS and ps a "
                     "row·candidate of the logits work, lists + counts bitwise), dsa_attn of one 2048-row call at "
                     "131072 / 1040000 (current), and P153's correctness cases (ties, wide spreads, two slots, row "
                     "passes of a small workspace)")
ap.add_argument("--verify", action="store_true",
                help="P221 dsa_sel_v32 — dsa_select of verify steps (B sequences × 8 rows at ~9.6K, B = 3 / 4 / 8): "
                     "today (P091 / P046 rpc 8 at ≥ 32 rows) vs P221 and P046's grid forced (rows a CTA × keys a "
                     "split): lists + counts bitwise, then cold ms")
ap.add_argument("--sizes", default="16384,131072,524288,1040000", help="--long: candidates of the first row")
ap.add_argument("--no-peak", action="store_true")
ap.add_argument("--no-attn", action="store_true")
ap.add_argument("--no-cases", action="store_true")
a = ap.parse_args()
m = ext.load("dsa", ["src/kernels/attn/dsa.cu", "src/model/patches.cc", "tests/bind/dsa_peak.cu"],
             log=os.environ.get("E55_BUILD_LOG"))
# patch name → the stages it changes (timed base → patch)
PATCHES = {} if a.base_only else {"dsa_attn_fast": ("attn",), "dsa_select_fast": ("select",),
                                       "dsa_vha_fast": ("vha_gate",), "dsa_bmm_dec": ("bmm_uk", "bmm_uv"),
                                       "dsa_attn_ws": ("attn",), "dsa_bmm_f8": ("bmm_uk", "bmm_uv"),
                                       "dsa_select_pf": ("select",), "dsa_attn_pf": ("attn",),
                                       "dsa_bmm_pf": ("bmm_uk", "bmm_uv"), "dsa_attn_ep": ("attn",),
                                       "dsa_select_long": ("select",)}
# batch-2 patches: also timed against the batch-1 state (every other patch on), base-state numbers in the first line
NEW = {} if a.base_only else {"dsa_attn_ws": ("attn",), "dsa_bmm_f8": ("bmm_uk", "bmm_uv")}
dev = "cuda"
kWin = 128  # the DSA window: a row at p has p − 127 indexer candidates

wm = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))["weight_map"]


def get(name):
    with safe_open(os.path.join(a.model, wm[name]), framework="pt", device="cpu") as f:
        return f.get_tensor(name).cuda()


pre = f"model.layers.{a.layer}.self_attn."
ix = pre + "core_attention.indexer."
kvb8 = get(pre + "kv_b_proj.weight").view(torch.uint8)[:32 * 448].contiguous()  # rank 0's rows (P073)
kvs = get(pre + "kv_b_proj.weight_scale_inv").float()
kvb = kvb8.view(torch.float8_e4m3fn).float()
kvd = kvb * kvs.repeat_interleave(128, 0)[:kvb.shape[0]].repeat_interleave(128, 1)[:, :kvb.shape[1]]
kvd = kvd[:32 * 448].view(32, 448, 512)  # rank 0: heads 0 … 31
W = dict(
    wuk=kvd[:, :192, :].transpose(1, 2).contiguous().bfloat16(),  # [32][512][192]
    wuv=kvd[:, 192:, :].contiguous().bfloat16(),                  # [32][256][512]
    sink=get(pre + "core_attention.softmax_offset").float()[:32].contiguous(),
    kvan=get(pre + "kv_a_layernorm.weight").float(),
    knw=get(ix + "k_norm.weight").bfloat16().contiguous(), knb=get(ix + "k_norm.bias").bfloat16().contiguous(),
    V=get(pre + "vha_postmix_V").float()[:32].contiguous(),
    uk8=kvb8.view(32, 448, 512)[:, :192, :].transpose(1, 2).contiguous(),  # P073: [h][k_lat][i] e4m3 bytes
    uv8=kvb8.view(32, 448, 512)[:, 192:, :].contiguous(),                  # [h][n][k_lat]
    kvs=kvs[:32 * 448 // 128].contiguous())
assert torch.equal(W["wuk"].float(), kvd[:, :192, :].transpose(1, 2)), "kv_b_proj not exact in bf16"
del kvb, kvs, kvd, kvb8
INV = m.inv8()


def meta_of(seqs):
    """seqs: [(slot, p0, n)] → RowMeta int32 [M, 5] (pos, slot, r0, p0, pend)"""
    rows, r0 = [], 0
    for s, p0, n in seqs:
        rows += [(p0 + i, s, r0, p0, p0 + n - 1) for i in range(n)]
        r0 += n
    return torch.tensor(rows, dtype=torch.int32, device=dev)


def rand_rows(M, g):
    """realistic random rows of one step: kv_c after kv_a_layernorm, k_pe, q (q_b_proj), indexer k ‖ weights, indexer q"""
    return dict(kvn=(torch.randn(M, 512, generator=g, device=dev) * W["kvan"]).bfloat16(),
                kpe=(torch.randn(M, 64, generator=g, device=dev) * 0.8).bfloat16(),
                q=torch.randn(M, 32 * 256, generator=g, device=dev).bfloat16(),
                kw=torch.cat([torch.randn(M, 128, generator=g, device=dev) * 1.5 + 0.3,
                              torch.randn(M, 64, generator=g, device=dev) * 2.0], 1).bfloat16(),
                qi=torch.randn(M, 64 * 128, generator=g, device=dev).bfloat16())


class Case:
    def __init__(self, name, seqs, seed=0, ties=0, kspread=0):
        self.name, self.seqs = name, seqs
        g = torch.Generator(device=dev).manual_seed(seed)
        S = max(s for s, _, _ in seqs) + 1
        ctx = max(p0 + n for _, p0, n in seqs)
        self.meta = meta_of(seqs)
        self.M = M = self.meta.shape[0]
        self.max_pos = max(p0 + n - 1 for _, p0, n in seqs)
        self.max_keys = max(min(p + 1, 2176) for _, p0, n in seqs for p in range(p0, p0 + n))
        self.mla = torch.zeros(S, ctx, 656, dtype=torch.uint8, device=dev)
        self.idx = torch.zeros(S, ctx, 68, dtype=torch.uint8, device=dev)
        for s, p0, n in seqs:  # history 0 … p0−1 through the base kernels
            for h0 in range(0, p0, 4096):
                hn = min(4096, p0 - h0)
                hm = meta_of([(s, h0, hn)])
                r = rand_rows(hn, g)
                m.mla_rows(r["kvn"], r["kpe"], hm, INV, self.mla, self.idx)
                m.idx_prep(r["kw"], r["qi"], W["knw"], W["knb"], hm, INV, self.mla, self.idx,
                           torch.empty(hn, 4096, dtype=torch.uint8, device=dev),
                           torch.empty(hn, 256, dtype=torch.uint8, device=dev),
                           torch.empty(hn, 64, dtype=torch.float32, device=dev))
            if ties and p0 > ties:  # few distinct indexer keys: ties at the top-k boundary
                src = self.idx[s, torch.arange(p0, device=dev) % ties].clone()
                self.idx[s, :p0] = src
            if kspread and p0:  # wide UE8M0 exponent spreads within key rows
                d = torch.randint(-kspread, kspread + 1, (p0, 4), generator=g, device=dev)
                self.idx[s, :p0, 64:] = (self.idx[s, :p0, 64:].int() + d).clamp(1, 254).to(torch.uint8)
        self.r = rand_rows(M, g)
        self.z = torch.randn(M, 16, 256, generator=g, device=dev) * 0.05
        self.gate = torch.sigmoid(torch.randn(M, 32 * 256, generator=g, device=dev)).bfloat16()

    def fresh(self):
        M = self.M
        return dict(mla=self.mla.clone(), idx=self.idx.clone(),
                    q576=torch.zeros(M, 32 * 576, dtype=torch.bfloat16, device=dev),
                    qf4=torch.zeros(M, 4096, dtype=torch.uint8, device=dev),
                    qs=torch.zeros(M, 256, dtype=torch.uint8, device=dev),
                    wts=torch.zeros(M, 64, dtype=torch.float32, device=dev),
                    list=torch.full((M, 2176), -1, dtype=torch.int32, device=dev),
                    cnt=torch.zeros(M, dtype=torch.int32, device=dev),
                    olat=torch.zeros(M, 32 * 512, dtype=torch.bfloat16, device=dev),
                    o=torch.zeros(M, 32 * 256, dtype=torch.bfloat16, device=dev),
                    y=torch.zeros(M, 32 * 256, dtype=torch.bfloat16, device=dev))


WS = torch.empty(1 << 28, dtype=torch.float32, device=dev)  # 1 GiB, the engine's attention workspace


def stages(c, t):
    """attn_dsa's DSA kernels as model.cc calls them: name → callable"""
    r, M = c.r, c.M
    return [
        ("mla_rows", lambda: m.mla_rows(r["kvn"], r["kpe"], c.meta, INV, t["mla"], t["idx"])),
        ("idx_prep", lambda: m.idx_prep(r["kw"], r["qi"], W["knw"], W["knb"], c.meta, INV, t["mla"], t["idx"],
                                        t["qf4"], t["qs"], t["wts"])),
        ("q_pe", lambda: m.q_pe(r["q"], c.meta, INV, t["q576"])),
        ("bmm_uk", lambda: m.bmm_f8(r["q"], 0, 8192, 256, W["uk8"], W["kvs"], 0, 32, 512, 192, M, t["q576"], 0, 32 * 576, 576)
                 or m.bmm(r["q"], 0, 8192, 256, W["wuk"], 32, 512, 192, M, t["q576"], 0, 32 * 576, 576)),
        ("select", lambda: m.select(t["qf4"], t["qs"], t["wts"], c.meta, t["mla"], t["idx"], c.max_pos, t["list"],
                                    t["cnt"], WS)),
        ("attn", lambda: m.attn(t["q576"], t["list"], t["cnt"], c.meta, t["mla"], t["idx"], W["sink"], c.max_keys,
                                t["olat"], WS)),
        ("bmm_uv", lambda: m.bmm_f8(t["olat"], 0, 32 * 512, 512, W["uv8"], W["kvs"], 1, 32, 256, 512, M, t["o"], 0, 8192, 256)
                 or m.bmm(t["olat"], 0, 32 * 512, 512, W["wuv"], 32, 256, 512, M, t["o"], 0, 8192, 256)),
        ("vha_gate", lambda: m.vha_gate(t["o"], c.z, W["V"], c.gate, t["y"])),
    ]


def run(st):
    for _, f in st:
        f()


def timeit(f, cold, runs):
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(runs):
        if cold:
            m.flush()
        torch.cuda._sleep(200_000)  # let the host queue the timed work
        e0.record()
        f()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    return statistics.median(ts)


def set_patches(on):
    for p in PATCHES:
        m.patch(p, on)


def rope_check():
    """DSA RoPE == vLLM get_rope default (cos_sin_cache built in fp32 on the GPU, stored bf16; fp32 rotation, bf16 out):
    q_pe at random positions up to 1M, k_pe (MLA cache rows) up to 32K, bit for bit."""
    inv = 1.0 / (8000000.0 ** (torch.arange(0, 64, 2, dtype=torch.float, device=dev) / 64))
    t = torch.arange(1 << 20, dtype=torch.float, device=dev)
    freqs = torch.einsum("i,j -> ij", t, inv)
    cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(torch.bfloat16)

    def rot(x, pos):  # x [N, H, 64] bf16, interleaved pairs
        cs = cache[pos.long()].float()
        c, s = cs[:, None, :32], cs[:, None, 32:]
        e, o = x[..., 0::2].float(), x[..., 1::2].float()
        return torch.stack((e * c - o * s, o * c + e * s), -1).flatten(-2).bfloat16()

    g = torch.Generator(device=dev).manual_seed(9)
    pos = torch.cat([torch.randint(0, 1 << 20, (4000,), generator=g, device=dev),
                     torch.tensor([0, 1, 127, 128, 9400, 16383, 65535, (1 << 20) - 1], device=dev)]).int()
    M = pos.numel()
    meta = torch.zeros(M, 5, dtype=torch.int32, device=dev)
    meta[:, 0] = pos
    q = torch.randn(M, 32 * 256, generator=g, device=dev).bfloat16()
    q576 = torch.zeros(M, 32 * 576, dtype=torch.bfloat16, device=dev)
    m.q_pe(q, meta, INV, q576)
    okq = torch.equal(q576.view(M, 32, 576)[..., 512:], rot(q.view(M, 32, 256)[..., 192:], pos))
    pk = torch.randperm(32768, generator=g, device=dev)[:2048].int()
    mk = torch.zeros(2048, 5, dtype=torch.int32, device=dev)
    mk[:, 0] = pk
    mla = torch.zeros(1, 32768, 656, dtype=torch.uint8, device=dev)
    idx = torch.zeros(1, 32768, 68, dtype=torch.uint8, device=dev)
    kvn = torch.randn(2048, 512, generator=g, device=dev).bfloat16()
    kpe = torch.randn(2048, 64, generator=g, device=dev).bfloat16()
    m.mla_rows(kvn, kpe, mk, INV, mla, idx)
    got = mla[0, pk.long(), 528:].contiguous().view(torch.bfloat16)
    okk = torch.equal(got, rot(kpe[:, None, :], pk)[:, 0])
    print(f"[rope] q_pe (4008 positions < 1M) {'bit-identical' if okq else 'DIFFERS'} · k_pe (2048 < 32K) "
          f"{'bit-identical' if okk else 'DIFFERS'} to vLLM's bf16 cos_sin_cache rotation", flush=True)
    return okq and okk


OUTS = ("mla", "idx", "q576", "qf4", "qs", "wts", "list", "cnt", "olat", "o", "y")
CASES = [  # built lazily (history fill)
    ("decode M=1 ctx 64", [(0, 64, 1)], {}),
    ("decode M=1 ctx 9400", [(0, 9400, 1)], {}),
    ("decode M=4 ctx ~9400", [(0, 9400, 1), (1, 9401, 1), (2, 9531, 1), (3, 9450, 1)], dict(seed=1)),
    ("ties M=4 ctx ~5000", [(0, 5000, 1), (1, 5001, 1), (2, 2300, 1), (3, 4100, 1)], dict(seed=2, ties=7)),
    ("spread M=4 ctx ~6000", [(0, 6000, 1), (1, 6001, 1), (2, 3000, 1), (3, 7000, 1)], dict(seed=5, kspread=4)),
    ("spread prefill 512 at 8192", [(0, 8192, 512)], dict(seed=6, kspread=3)),
    ("ties prefill 512 at 8192", [(0, 8192, 512)], dict(seed=7, ties=600)),  # P091: equal keys inside its kept-aside set
    ("ties7 prefill 256 at 4096", [(0, 4096, 256)], dict(seed=8, ties=7)),   # P091: too many equal keys → third pass
    ("prefill 4096 at 0", [(0, 0, 4096)], dict(seed=3)),
    ("prefill 4096 at 16384", [(0, 16384, 4096)], dict(seed=4)),
]
if a.prefill:  # session 5 step 2: P110 against the current default (every older patch at its default)
    NEWP = {"dsa_attn_ep": ("attn",)}
    TIMED = ("attn",)

    def set_new(on):
        for p in NEWP:
            m.patch(p, on)

    shapes = [(M, P) for M in (1024, 2048) for P in (0, 16384, 49152)]
    if a.shapes:
        want = {tuple(int(v) for v in s.split("@")) for s in a.shapes.split(",")}
        shapes = [s for s in shapes if s in want]
    ok = True
    for M, P in shapes:
        c = Case(f"prefill {M} at {P}", [(0, P, M)], seed=11 + P // 4096 + M // 1024)
        set_new(False)
        tb = c.fresh()
        run(stages(c, tb))
        set_new(True)
        tp = c.fresh()
        run(stages(c, tp))
        torch.cuda.synchronize()
        bad = [k for k in OUTS if not torch.equal(tb[k], tp[k])]
        ok &= not bad
        print(f"[{c.name}] max_keys {c.max_keys} · " + ("bit-identical (all outputs)" if not bad else
                                                        "DIFFERS: " + ", ".join(bad)), flush=True)
        if a.no_time:
            continue
        for name, f in stages(c, tb):
            if name not in TIMED:
                continue
            set_new(False)
            t0 = timeit(f, False, a.runs)
            set_new(True)
            t1 = timeit(f, False, a.runs)
            print(f"    {name:9s} {t0:8.3f} → {t1:8.3f} ms", flush=True)
    print("[dsa prefill]", "all OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)

if a.wide:  # session 6: P142 dsa_select_wide (every other patch at its default; P153 off: it would take these calls)
    m.patch("dsa_select_long", False)
    ok = True
    for name, seqs, kw, timed in [("prefill 4096 at 131072", [(0, 131072, 4096)], dict(seed=31), True),
                                  ("ties1 prefill 256 at 131072", [(0, 131072, 256)], dict(seed=32, ties=1), False),
                                  ("ties3 prefill 256 at 100000", [(0, 100000, 256)], dict(seed=33, ties=3), False)]:
        c = Case(name, seqs, **kw)
        t = c.fresh()
        st = dict(stages(c, t))
        st["mla_rows"](), st["idx_prep"]()
        res = {}
        for on in (False, True):
            m.patch("dsa_select_wide", on)
            t["list"].fill_(-1), t["cnt"].zero_()
            st["select"]()
            torch.cuda.synchronize()
            res[on] = (t["list"].clone(), t["cnt"].clone())
        same = torch.equal(res[0][0], res[1][0]) and torch.equal(res[0][1], res[1][1])
        ok &= same
        line = f"[wide {name}] lists + counts " + ("bit-identical to P046's" if same else "DIFFER")
        if timed and not a.no_time:
            ms = {}
            for on in (False, True):
                m.patch("dsa_select_wide", on)
                ms[on] = timeit(st["select"], False, a.runs)
            line += f" · dsa_select {ms[False]:.2f} → {ms[True]:.2f} ms"
        m.patch("dsa_select_wide", True)
        print(line, flush=True)
    print("[dsa wide]", "all OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)

if a.verify:  # P221 dsa_sel_v32 (every other patch at its default)
    CFG = [("today", False, 0, 0), ("P221", True, 0, 0), ("rpc1 kr512", True, 1, 512), ("rpc1 kr1024", True, 1, 1024),
           ("rpc8 kr512", True, 8, 512), ("rpc8 kr1024", True, 8, 1024)]
    ok = True
    for B in (3, 4, 8):
        c = Case(f"verify {B}×8", [(s, 9400 + 97 * s, 8) for s in range(B)], seed=20 + B)
        t = c.fresh()
        st = dict(stages(c, t))
        st["mla_rows"](), st["idx_prep"]()
        lists, ms = {}, {}
        for name, on, rpc, kr in CFG:
            m.patch("dsa_sel_v32", on)
            m.select_cfg(rpc, kr)
            t["list"].fill_(-1), t["cnt"].zero_()
            st["select"]()
            torch.cuda.synchronize()
            lists[name] = (t["list"].clone(), t["cnt"].clone())
            if not a.no_time:
                ms[name] = timeit(st["select"], True, a.runs)
        m.patch("dsa_sel_v32", False)
        m.select_cfg(0, 0)
        same = {n: torch.equal(lists[n][0], lists["today"][0]) and torch.equal(lists[n][1], lists["today"][1])
                for n in lists}
        ok &= all(same.values())
        print(f"[verify {B}×8 = {c.M} rows, ≤ {c.max_pos + 1} positions] lists + counts == today: "
              + " · ".join(f"{n} {'OK' if v else 'DIFFER'}" for n, v in same.items()), flush=True)
        if ms:
            print("    dsa_select cold ms: " + " · ".join(f"{n} {v:.3f}" for n, v in ms.items()), flush=True)
    print("[dsa verify]", "all OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)

if a.split:  # session 6: P140 dsa_sel_split + P141 dsa_q_own (every patch at its default)
    ok = True
    for name, seqs in [("lane 2048 at 49152", [(0, 49152, 2048)]), ("odd lane 2047 at 1100", [(0, 1100, 2047)]),
                       ("two sequences 1024 at 3000 + 1024 at 0", [(0, 3000, 1024), (1, 0, 1024)])]:
        c = Case(name, seqs, seed=21)
        M, r = c.M, c.r
        ref = c.fresh()
        run(stages(c, ref))
        h = (M + 1) // 2
        own = [(0, h), (h, M - h)]  # rank → its rows (a, n); rank 1's half of an odd lane ends in the scratch row
        t = [c.fresh(), c.fresh()]
        for k in (0, 1):  # each rank: caches, its rows' query side (P141), q576, its rows' select
            x, (o, n) = t[k], own[k]
            m.mla_rows(r["kvn"], r["kpe"], c.meta, INV, x["mla"], x["idx"])
            for p0, pn in ((0, o), (o + n, M - o - n)):
                if pn:
                    m.idx_prep_k(r["kw"][p0:p0 + pn], W["knw"], W["knb"], c.meta[p0:p0 + pn], INV, x["mla"], x["idx"],
                                 x["wts"][p0:p0 + pn])
            m.idx_prep(r["kw"][o:o + n], r["qi"][o:o + n], W["knw"], W["knb"], c.meta[o:o + n], INV, x["mla"], x["idx"],
                       x["qf4"][o:o + n], x["qs"][o:o + n], x["wts"][o:o + n])
            m.q_pe(r["q"], c.meta, INV, x["q576"])
            m.bmm_f8(r["q"], 0, 8192, 256, W["uk8"], W["kvs"], 0, 32, 512, 192, M, x["q576"], 0, 32 * 576, 576) or \
                m.bmm(r["q"], 0, 8192, 256, W["wuk"], 32, 512, 192, M, x["q576"], 0, 32 * 576, 576)
            m.select(x["qf4"][o:o + n], x["qs"][o:o + n], x["wts"][o:o + n], c.meta[o:o + n], x["mla"], x["idx"],
                     c.max_pos, x["list"][o:o + n], x["cnt"][o:o + n], WS)
        for k in (0, 1):  # the exchange, then the tail and the attention (own rows, then the peer's)
            x, (o, n), (po, pn) = t[k], own[k], own[1 - k]
            x["list"][po:po + pn, :2048] = t[1 - k]["list"][po:po + pn, :2048]
            m.sel_tail(c.meta[po:po + pn], x["list"][po:po + pn], x["cnt"][po:po + pn])
            for ro, rn in ((o, n), (po, pn)):
                m.attn(x["q576"][ro:ro + rn], x["list"][ro:ro + rn], x["cnt"][ro:ro + rn], c.meta[ro:ro + rn], x["mla"],
                       x["idx"], W["sink"], c.max_keys, x["olat"][ro:ro + rn], WS)
        torch.cuda.synchronize()
        for k in (0, 1):
            x, (o, n) = t[k], own[k]
            bad = [f for f in ("mla", "idx", "q576", "wts", "list", "cnt", "olat") if not torch.equal(ref[f], x[f])]
            bad += [f + " (own rows)" for f in ("qf4", "qs") if not torch.equal(ref[f][o:o + n], x[f][o:o + n])]
            ok &= not bad
            print(f"[split {name}] rank {k}: " + ("caches, lists, counts, attention bit-identical to the whole lane"
                                                  if not bad else "DIFFERS: " + ", ".join(bad)), flush=True)
    print("[dsa split]", "all OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)

if a.long:  # session 6: P153 dsa_select_long (every other patch at its default)
    ok = True
    PN = "dsa_select_long"
    if not a.no_peak:  # the k64 mxf4 MMA's peak: pure register loop, all SMs, a few occupancies / chain counts
        nsm = torch.cuda.get_device_properties(0).multi_processor_count
        best = (0.0, None)
        for cps, ch in [(1, 4), (1, 8), (2, 2), (2, 4), (2, 8), (4, 4)]:  # CTAs of 8 warps per SM, chains a warp
            it = int(4e7 / (8 * cps * ch))
            ms, cyc, n = m.mma_peak(nsm * cps, 256, ch, it)
            tf = n * 8192 * 2 / (ms * 1e-3) / 1e12
            print(f"[peak] {8 * cps:2d} warps/SM × {ch} chains: {tf:6.1f} TFLOPS · {cyc / (ms * 1e3):5.0f} MHz · "
                  f"{n / nsm * 8192 / cyc:6.0f} MAC/clk/SM", flush=True)
            if tf > best[0]:
                best = (tf, f"{8 * cps} warps/SM × {ch} chains")
        PEAK = best[0]
        print(f"[peak] k64 e2m1 MMA peak {PEAK:.1f} TFLOPS ({best[1]})", flush=True)
    else:
        PEAK = 0.0

    def sel_call(c, t, n, ws=None):
        mp = int(c.meta[:n, 0].max())
        return lambda: m.select(t["qf4"][:n], t["qs"][:n], t["wts"][:n], c.meta[:n], t["mla"], t["idx"], mp,
                                t["list"][:n], t["cnt"][:n], WS if ws is None else ws)

    def both(c, t, n, ws=None):
        res = {}
        f = sel_call(c, t, n, ws)
        for on in (False, True):
            m.patch(PN, on)
            t["list"].fill_(-1), t["cnt"].zero_()
            f()
            torch.cuda.synchronize()
            res[on] = (t["list"][:n].clone(), t["cnt"][:n].clone())
        m.patch(PN, True)
        same = torch.equal(res[0][0], res[1][0]) and torch.equal(res[0][1], res[1][1])
        if not same:  # which rows, and how
            bad = ((res[0][0] != res[1][0]).any(1) | (res[0][1] != res[1][1])).nonzero().flatten().tolist()
            r = bad[0]
            d = (res[0][0][r] != res[1][0][r]).nonzero().flatten().tolist()
            print(f"    {len(bad)} rows differ (first {bad[:8]}); row {r}: pos {int(c.meta[r, 0])} cnt "
                  f"{int(res[0][1][r])} / {int(res[1][1][r])}, {len(d)} entries from {d[:1]}: "
                  f"{res[0][0][r][d[:6]].tolist()} vs {res[1][0][r][d[:6]].tolist()}", flush=True)
        return same

    sizes = [int(v) for v in a.sizes.split(",") if v]
    for N in sizes:
        attn = not a.no_attn and N in (131072, 1040000)
        rows = 2048 if attn else 1024
        c = Case(f"long {N}", [(0, N + kWin - 1, rows)], seed=50 + N % 97)
        t = c.fresh()
        st = dict(stages(c, t))
        st["mla_rows"](), st["idx_prep"]()
        torch.cuda.synchronize()
        same = both(c, t, 1024)
        ok &= same
        nc = (c.meta[:1024, 0] - kWin + 1).double()
        flop = float(nc.sum()) * 64 * 128 * 2
        runs = a.runs if N <= 200000 else min(a.runs, 7)
        f = sel_call(c, t, 1024)
        ms = {}
        for on in (False, True):
            m.patch(PN, on)
            ms[on] = timeit(f, False, runs)
        m.patch(PN, True)
        tf = {k: flop / (v * 1e-3) / 1e12 for k, v in ms.items()}
        pk = f" ({tf[False] / PEAK * 100:.0f} → {tf[True] / PEAK * 100:.0f} % of peak)" if PEAK else ""
        ps = {k: v * 1e9 / float(nc.sum()) for k, v in ms.items()}  # ps a row·candidate
        print(f"[long {N}] dsa_select 1024 rows: {ms[False]:8.2f} → {ms[True]:8.2f} ms · {tf[False]:5.1f} → "
              f"{tf[True]:5.1f} TFLOPS{pk} · {ps[False]:5.1f} → {ps[True]:5.1f} ps a row·candidate · lists + counts "
              + ("bit-identical" if same else "DIFFER"), flush=True)
        if attn:  # dsa_attn of the whole 2048-row lane (current path): its lists from the current select
            st["q_pe"](), st["bmm_uk"]()
            m.patch(PN, False)
            st["select"]()
            m.patch(PN, True)
            ta = timeit(st["attn"], False, runs)
            print(f"[long {N}] dsa_attn 2048 rows (top-2048 + window): {ta:8.2f} ms", flush=True)
        del c, t, st
        torch.cuda.empty_cache()
    if not a.no_cases:  # P153's correctness cases: lists + counts P153 off vs on
        for name, seqs, kw, wsz in [
                ("ties1 256 at 131072", [(0, 131072, 256)], dict(seed=61, ties=1), None),
                ("ties3 256 at 140000", [(0, 140000, 256)], dict(seed=62, ties=3), None),
                ("ties600 256 at 140000", [(0, 140000, 256)], dict(seed=63, ties=600), None),
                ("spread 256 at 140000", [(0, 140000, 256)], dict(seed=64, kspread=3), None),
                ("two slots 300 at 150000 + 212 at 70000", [(0, 150000, 300), (1, 70000, 212)], dict(seed=65), None),
                ("mixed 200 at 150000 + 100 at 0 + 60 at 2140", [(0, 150000, 200), (1, 0, 100), (2, 2140, 60)],
                 dict(seed=67), None),
                ("small ws: 512 at 131072, 40 MiB", [(0, 131072, 512)], dict(seed=66), 10 << 20)]:
            c = Case(name, seqs, **kw)
            t = c.fresh()
            st = dict(stages(c, t))
            st["mla_rows"](), st["idx_prep"]()
            same = both(c, t, c.M, None if wsz is None else WS[:wsz])
            ok &= same
            print(f"[long case {name}] lists + counts " + ("bit-identical" if same else "DIFFER"), flush=True)
            del c, t, st
            torch.cuda.empty_cache()
    print("[dsa long]", "all OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)

only = [s.strip() for s in a.only.split(",") if s.strip()]
ok = rope_check()
for cname, cseqs, ckw in CASES:
    if only and not any(o in cname for o in only):
        continue
    c = Case(cname, cseqs, **ckw)
    cold = c.M <= 16
    set_patches(False)
    tb = c.fresh()
    run(stages(c, tb))
    torch.cuda.synchronize()
    nsel = int((tb["cnt"] > 0).sum())
    line = f"[{c.name}] M {c.M} · max_keys {c.max_keys}"
    if PATCHES:
        set_patches(True)
        tp = c.fresh()
        run(stages(c, tp))
        torch.cuda.synchronize()
        bad = [k for k in OUTS if not torch.equal(tb[k], tp[k])]
        if bad:
            ok = False
            line += " · DIFFERS: " + ", ".join(bad)
            if "olat" in bad:
                d = (tb["olat"].float() - tp["olat"].float()).abs()
                line += f" (olat max abs {d.max().item():.3e}, n {int((d > 0).sum())})"
        else:
            line += " · all outputs bit-identical"
    print(line, flush=True)
    if a.no_time:
        continue
    for name, f in stages(c, tb):
        pats = [p for p, sts in PATCHES.items() if name in sts]
        set_patches(False)
        tbase = timeit(f, cold, a.runs)
        if pats:
            set_patches(True)
            tpat = timeit(f, cold, a.runs)
            print(f"    {name:9s} {tbase:9.3f} → {tpat:8.3f} ms (×{tbase / tpat:.1f})", flush=True)
            for p in [p for p, sts in NEW.items() if name in sts]:
                m.patch(p, False)
                tprev = timeit(f, cold, a.runs)
                m.patch(p, True)
                print(f"    {name:9s} {tprev:9.3f} → {tpat:8.3f} ms (×{tprev / tpat:.2f}) {p} vs the other patches", flush=True)
        else:
            print(f"    {name:9s} {tbase:9.3f} ms", flush=True)
print("[dsa]", "all OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
