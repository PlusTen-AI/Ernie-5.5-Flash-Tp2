#!/usr/bin/env python3
"""test_comm.py — collectives micro test on both ranks (TP=2; through tools/tp2.sh, which appends --rank/--head-ip):
  E55_GPU_LOCK=1 E55_TREE=comm tools/tp2.sh comm python3 /work/dev/comm/src/engine/tests/test_comm.py [--parts …]
Parts:
  bits     all-reduce bf16 / fp32 (LL, fused, large paths; P077 on/off) and row-piece all-gathers (P076, LL / fused /
           large, plain and async P075) against NCCL and the expected values — bitwise
  dec      the decode / verify step's collectives at M = 1 / 4 / 8: µs per call back-to-back and behind a weight-stream
           stand-in, the transport's phase times (RdmaAR::probe), the step totals; --dec-variants to compare
  lat      decode sizes: µs per call back-to-back, and the cost added behind a weight-stream stand-in kernel (the way
           a collective sits between two GEMMs), P077 off / on; all-gathers of the candidate split points; logits
  bw       large all-reduces (8 / 50 / 134 MB): GB/s with the large-call tuning variants (--variants chunkKB/copy/add/
           ahead/xfer), the transport alone (xfer = 1) and NCCL; the staging size per call is E55_RDMA_STAGING_MB (--env)
  overlap  a prefill-size all-reduce on the comm stream (P075) under a compute-bound GEMM loop on the compute stream
Rank 0 prints one line per measurement."""
import argparse
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ext  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, required=True)
ap.add_argument("--head-ip", required=True)
ap.add_argument("--port", type=int, default=29651)
ap.add_argument("--parts", default="bits,lat,bw,overlap")
ap.add_argument("--iters", type=int, default=200)
ap.add_argument("--reps", type=int, default=21)
ap.add_argument("--env", default="", help="K=V,… set before init (E55_RDMA_LL_MAX, E55_RDMA_STAGING_MB, …)")
ap.add_argument("--variants", default="", help="bw: chunkKB/copy/add/ahead/xfer list instead of the built-in one")
ap.add_argument("--dec-variants", default="", help="dec: variants after the base (dec_variants())")
ap.add_argument("--rounds", type=int, default=3, help="dec: rounds over the variants (best kept)")
ap.add_argument("--patches", default="", help="patches switched on at the start (comma list), e.g. for bits")
a = ap.parse_args()
for kv in filter(None, a.env.split(",")):
    os.environ[kv.split("=")[0]] = kv.split("=", 1)[1]
os.environ.setdefault("E55_RDMA_TIMEOUT_S", "30")  # a lost peer traps instead of hanging
R0 = a.rank == 0
PARTS = set(a.parts.split(","))


def say(*x):
    if R0:
        print(*x, flush=True)


m = ext.load("commtest", ["src/core/comm.cc", "src/kernels/comm/rdma_ar.cu", "src/model/patches.cc", "tests/comm_k.cu"],
             log=f"/tmp/comm_build{a.rank}.log", ldflags=["-lnccl", "-libverbs"])
m.init(a.rank, a.head_ip, a.port)
for p in ("rdma_ar", "rdma_gather", "ar_pdl", "ar_lanes") + tuple(filter(None, a.patches.split(","))):
    m.set_patch(p, True)
dev = "cuda"


def gen(n, r, j, dt):
    """rank r's input, variant j — mixed magnitudes, the same on both nodes"""
    g = torch.Generator(device=dev).manual_seed(7919 * n + 31 * j + r)
    x = torch.randn(n, generator=g, device=dev)
    e = torch.randint(-12, 12, (n,), generator=g, device=dev).float()
    return (x * torch.exp2(e)).to(dt)


def same(x, y):
    return torch.equal(x.view(torch.int16) if x.dtype == torch.bfloat16 else x.view(torch.int32),
                       y.view(torch.int16) if y.dtype == torch.bfloat16 else y.view(torch.int32))


fails = 0


def check(ok, what):
    global fails
    if not ok:
        fails += 1
        say(f"[bits] FAIL {what}")


if "bits" in PARTS:
    # all-reduce: rows × width so every path is hit (LL ≤ 16 KiB, fused ≤ 1 MiB, large); odd sizes too
    cases = [(torch.bfloat16, n) for n in (4096, 6144, 4 * 6144, 64 * 4096, 512 * 6144, 4096 * 4096 + 64, 999)] + \
            [(torch.float32, n) for n in (4096, 8192, 4 * 8192, 64 * 8192, 2048 * 8192, 1001)]
    for dt, n in cases:
        x0, x1 = gen(n, 0, 1, dt), gen(n, 1, 1, dt)
        exp = (x0.float() + x1.float()).to(dt)
        mine = x0 if a.rank == 0 else x1
        res = {}
        for name, pat in (("nccl", {"rdma_ar": False}), ("rdma", {"rdma_ar": True, "ar_pdl": False}),
                          ("rdma+pdl", {"rdma_ar": True, "ar_pdl": True})):
            for k, v in pat.items():
                m.set_patch(k, v)
            t = mine.clone()
            m.ar(t)
            torch.cuda.synchronize()
            res[name] = t
            check(same(t, exp), f"all-reduce {dt} n={n} {name}")
        t = mine.clone()
        tk = m.ar_async(t)  # P075 ticket
        m.join(tk)
        torch.cuda.synchronize()
        check(same(t, exp), f"all-reduce async {dt} n={n}")
    m.set_patch("rdma_ar", True)
    # row-piece all-gathers: 3 pieces into 2 buffers (one shared), bf16 / fp32 / raw bytes; sentinel elsewhere
    for M, widths in ((1, (512, 2048, 1024)), (4, (512, 1024, 1024)), (64, (2048, 2048, 512)), (300, (4096, 1024, 2048)),
                      (2048, (2048, 2048, 1024))):
        def run(name, async_=False):
            src = [gen(M * w // 4, a.rank, 3 + j, torch.float32).view(torch.uint8).view(M, w) for j, w in enumerate(widths)]
            # piece 0 → buffer A at [r·w0], pieces 1, 2 → buffer B (row = [p1 r0 | p1 r1 | pad 16 | p2 r0 | p2 r1])
            wa, wb = 2 * widths[0] + 32, 2 * widths[1] + 16 + 2 * widths[2]
            A = torch.full((M, wa), 0xA5, dtype=torch.uint8, device=dev)
            B = torch.full((M, wb), 0x5A, dtype=torch.uint8, device=dev)
            offs = [16, 0, 2 * widths[1] + 16]
            tk = m.ag([src[0], src[1], src[2]], [A, B, B], offs, [widths[0], widths[1], widths[2]], M, async_)
            m.join(tk)
            torch.cuda.synchronize()
            # expected
            peer = [gen(M * w // 4, 1 - a.rank, 3 + j, torch.float32).view(torch.uint8).view(M, w)
                    for j, w in enumerate(widths)]
            ex = {0: src, 1: peer} if a.rank == 0 else {0: peer, 1: src}
            EA = torch.full_like(A, 0xA5)
            EB = torch.full_like(B, 0x5A)
            for r in (0, 1):
                EA[:, 16 + r * widths[0]:16 + (r + 1) * widths[0]] = ex[r][0]
                EB[:, r * widths[1]:(r + 1) * widths[1]] = ex[r][1]
                o = offs[2] + r * widths[2]
                EB[:, o:o + widths[2]] = ex[r][2]
            check(torch.equal(A, EA) and torch.equal(B, EB), f"all-gather M={M} widths={widths} {name}")
        for name, pat in (("nccl", {"rdma_gather": False}), ("rdma", {"rdma_gather": True, "ar_pdl": False}),
                          ("rdma+pdl", {"rdma_gather": True, "ar_pdl": True})):
            for k, v in pat.items():
                m.set_patch(k, v)
            run(name)
        run("async", True)
    # flat all-gather (logits rows: R × 100608 fp32 per rank; P063's 520-byte candidate rows: 8-byte aligned, LL)
    for R, w in ((1, 100608), (4, 100608), (33, 100608), (1, 130), (3, 130), (16, 130)):
        x = gen(R * w, a.rank, 9, torch.float32)
        outs = []
        for on in (False, True):
            m.set_patch("rdma_gather", on)
            y = torch.full((2 * R * w,), -1.0, device=dev)
            m.ag_flat(x, y)
            torch.cuda.synchronize()
            outs.append(y)
        x0, x1 = gen(R * w, 0, 9, torch.float32), gen(R * w, 1, 9, torch.float32)
        check(same(outs[0], torch.cat([x0, x1])) and same(outs[1], outs[0]), f"flat all-gather R={R} × {w * 4} B")
    m.set_patch("rdma_gather", True)
    say(f"[bits] {'all equal (NCCL = RDMA = expected, PDL on/off, async)' if not fails else f'{fails} FAILED'}")

if "lat" in PARTS:
    it, rp = a.iters, a.reps
    PRE = 8 << 20  # stand-in: 8 MB of weights (a rotation of 8 buffers, 64 MB > L2) before each collective
    base = m.time_pre(PRE, it, rp)[0]
    say(f"[lat] stand-in alone (8 MB read, PDL launch): {base:.1f} µs")
    rows = [("bf16 o_proj / dense", 4096, False), ("bf16 MoE [lat‖shared]", 6144, False),
            ("fp32 HCA VHA z", 8192, True), ("fp32 DSA VHA z", 4096, True)]
    for M in (1, 4):
        for name, w, f32 in rows:
            n = M * w
            out = []
            for pdl in (False, True):
                m.set_patch("ar_pdl", pdl)
                b2b = m.time_ar(n, f32, it, rp, 0)[0]
                ctx = m.time_ar(n, f32, it, rp, PRE)[0] - base
                out.append(f"b2b {b2b:5.1f} · after GEMM +{ctx:5.1f}")
            say(f"[lat] AR {name:22s} M={M} {n * (4 if f32 else 2) / 1024:6.1f} KiB · PDL off: {out[0]} · on: {out[1]} µs")
    m.set_patch("ar_pdl", True)
    # all-gathers: candidate split points (bytes per row per rank, pieces)
    ags = [("HCA front kv‖ckv", (512, 1024, 1024)), ("MoE front router‖fc1", (2048, 2048)), ("MoE tail fout", (4096,)),
           ("one bf16 row 4096", (8192,))]
    for M in (1, 4):
        for name, ws in ags:
            out = []
            for pdl in (False, True):
                m.set_patch("ar_pdl", pdl)
                b2b = m.time_ag(list(ws), M, it, rp, 0)[0]
                ctx = m.time_ag(list(ws), M, it, rp, PRE)[0] - base
                out.append(f"b2b {b2b:5.1f} · after GEMM +{ctx:5.1f}")
            say(f"[lat] AG {name:22s} M={M} {M * sum(ws) / 1024:5.1f} KiB/rank · PDL off: {out[0]} · on: {out[1]} µs")
    m.set_patch("ar_pdl", True)
    for R, w in ((1, 130), (4, 130)):  # P063's candidate rows (520 B)
        x = torch.zeros(R * w, device=dev)
        y = torch.zeros(2 * R * w, device=dev)
        res = []
        for on in (False, True):
            m.set_patch("rdma_gather", on)
            for _ in range(3):
                m.ag_flat(x, y)
            m.barrier()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(200):
                m.ag_flat(x, y)
            e1.record()
            torch.cuda.synchronize()
            res.append(e0.elapsed_time(e1) * 1000 / 200)
        say(f"[lat] candidate all-gather R={R} ({R * w * 4} B/rank): NCCL {res[0]:.1f} µs → RDMA {res[1]:.1f} µs")
    for R in (1, 4, 16):
        n = R * 100608
        x = torch.zeros(n, device=dev)
        y = torch.zeros(2 * n, device=dev)
        res = []
        for on in (False, True):
            m.set_patch("rdma_gather", on)
            for _ in range(3):
                m.ag_flat(x, y)
            m.barrier()
            torch.cuda.synchronize()
            t = time.perf_counter()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(50):
                m.ag_flat(x, y)
            e1.record()
            torch.cuda.synchronize()
            res.append(e0.elapsed_time(e1) * 1000 / 50)
        say(f"[lat] logits all-gather R={R} ({n * 4 / 1024:.0f} KiB/rank): NCCL {res[0]:.1f} µs → RDMA {res[1]:.1f} µs")
    m.set_patch("rdma_gather", True)

# the decode / verify step's collectives (one lane, P078 halves): (name, kind, bf16/fp32 elements or piece bytes per row,
# calls a step). HCA ×37, DSA ×6, MoE ×42, dense FFN ×1, embedding (P095) ×1.
DEC = [("HCA kv|ckv AG", "ag", (512, 1024, 1024), 37), ("HCA z f32 AR", "f32", 8192, 37),
       ("o_proj bf16 AR", "bf16", 4096, 44), ("DSA z f32 AR", "f32", 4096, 6),
       ("MoE router|fc1 AG", "ag", (1024, 1024, 1024, 32), 42), ("MoE lat|sh AR", "bf16", 6144, 42),
       ("MoE fc2 AG", "ag", (4096,), 42), ("embed AG", "ag", (8192,), 1)]


def tune_base():
    m.tune_small(-2, -2, 0, 4096, -2, -2)  # P120's plan (LL limits, block sizes)


def dec_variants():
    """(name, tokens) — the base first. A variant is tokens joined by "+": llA/B (LL limits, KiB, all-reduce /
    all-gather), fcN (the fused all-reduce in at most N chunks), lsN (one-block LL calls on two rails above N bytes),
    lbN (LL KiB per block), fbN (fused all-reduce KiB per block), patch names"""
    return [("base", [])] + [(v, v.split("+")) for v in filter(None, a.dec_variants.split(","))]


def dec_setup(tokens, extra):
    for n in extra:
        m.set_patch(n, False)
    tune_base()
    for t in tokens:
        if t.startswith("ll"):
            x, y = (int(u) for u in t[2:].split("/"))
            m.tune_small(x << 10, y << 10, -1, -1, -1, -1)
        elif t.startswith("fc"):
            m.tune_small(-1, -1, int(t[2:]), -1, -1, -1)
        elif t.startswith("ls"):
            m.tune_small(-1, -1, -1, int(t[2:]), -1, -1)
        elif t.startswith("lb"):
            m.tune_small(-1, -1, -1, -1, int(t[2:]) << 10, -1)
        elif t.startswith("fb"):
            m.tune_small(-1, -1, -1, -1, -1, int(t[2:]) << 10)
        else:
            m.set_patch(t, True)


def dec_call(kind, w, M, probe=False):
    """b2b µs (median of reps) of one decode collective, or its phase stamps"""
    if kind == "ag":
        return m.probe_ag(list(w), M, a.iters) if probe else m.time_ag(list(w), M, a.iters, a.reps, 0)[0]
    return m.probe_ar(M * w, kind == "f32", a.iters) if probe else m.time_ar(M * w, kind == "f32", a.iters, a.reps, 0)[0]


if "dec" in PARTS:
    # back to back, every variant of a line measured in turn, --rounds times (best kept: the nodes run other work)
    say(f"[dec] GPU <-> CPU round trip through pinned memory: {m.ping(20000) / 1e3:.2f} µs")
    VAR = dec_variants()
    extra = sorted({t for _, tk in VAR for t in tk if not t.startswith(("ll", "fc", "ls", "lb", "fb"))})
    keep = {n: m.patch_on(n) for n in extra}  # the base runs with these off; restored after
    for M in (1, 4, 8):
        tot = {v: 0.0 for v, _ in VAR}
        for name, kind, w, cnt in DEC:
            best = {v: 1e9 for v, _ in VAR}
            for _ in range(a.rounds):
                for v, tk in VAR:
                    dec_setup(tk, extra)
                    best[v] = min(best[v], dec_call(kind, w, M))
            kb = M * (sum(w) if kind == "ag" else w * (4 if kind == "f32" else 2)) / 1024
            for v, _ in VAR:
                tot[v] += cnt * best[v]
            say(f"[dec] M={M} {name:18s} {kb:6.1f} KiB · b2b µs " + " · ".join(f"{v} {best[v]:.1f}" for v, _ in VAR))
            for v, tk in (VAR[0], VAR[-1]) if len(VAR) > 1 else VAR[:1]:
                dec_setup(tk, extra)
                pr = dec_call(kind, w, M, True)
                say(f"[dec]      phases {v}: start>ring {pr[1] / 1e3:.1f} ring>data {pr[2] / 1e3:.1f} data>end "
                    f"{pr[3] / 1e3:.1f} gap {pr[4] / 1e3:.1f} · posts/call {pr[5] / max(pr[0], 1):.1f} × {pr[6] / 1e3:.2f} µs")
        say(f"[dec] M={M} step total (calls × b2b µs): " + " · ".join(f"{v} {tot[v] / 1e3:.2f} ms" for v, _ in VAR))
    for n in extra:
        m.set_patch(n, keep[n])
    tune_base()


if "bw" in PARTS:
    # large all-reduces: GB/s = bytes per rank / time; base tuning, variants, the transport alone, NCCL
    rp = 5
    sizes = [(8 << 20, True), (50 << 20, True), (134 << 20, True), (50 << 20, False)]
    variants = [("base", (1024, 2, 4, 1, 0))] + [(v, t) for v, t in (
        ("chunk 512K", (512, 2, 4, 1, 0)), ("chunk 2M", (2048, 2, 4, 1, 0)), ("chunk 4M", (4096, 2, 4, 1, 0)),
        ("copy 4", (1024, 4, 4, 1, 0)), ("copy 8", (1024, 8, 4, 1, 0)), ("add 8", (1024, 2, 8, 1, 0)),
        ("add 16", (1024, 2, 16, 1, 0)), ("ahead 2", (1024, 2, 4, 2, 0)), ("ahead 4", (1024, 2, 4, 4, 0)),
        ("ahead 2 · 4/8 blk", (1024, 4, 8, 2, 0)), ("transport only", (1024, 2, 4, 1, 1)))
        if t != (1024, 2, 4, 1, 0)]
    if a.variants:
        variants = [(n, tuple(int(x) for x in n.split("/"))) for n in a.variants.split(",")]
    for b, f32 in sizes:
        out = []
        for name, t in variants:
            m.tune(*t)
            ms = m.time_big(b, f32, rp)
            out.append(f"{name} {b / ms / 1e6:.1f}")
        m.tune(1024, 2, 4, 1, 0)
        m.set_patch("rdma_ar", False)
        ms = m.time_big(b, f32, 3)
        m.set_patch("rdma_ar", True)
        out.append(f"NCCL {b / ms / 1e6:.1f}")
        say(f"[bw] {b >> 20} MB {'fp32' if f32 else 'bf16'} GB/s: " + " · ".join(out))

if "overlap" in PARTS:
    # prefill: the half-chunk VHA z all-reduce (2048 rows × 16 × 512 fp32 = 67 MB) and o_proj (2048 × 4096 bf16)
    # alone, a bf16 GEMM loop alone, and both at once (the all-reduce on the comm stream)
    A = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
    Bm = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
    C = torch.empty(4096, 4096, device=dev, dtype=torch.bfloat16)

    def ev():
        return torch.cuda.Event(enable_timing=True)

    def gemms(k):
        for _ in range(k):
            torch.matmul(A, Bm, out=C)

    for name, n, dt in (("z fp32 67 MB", 2048 * 16 * 512, torch.float32), ("o bf16 17 MB", 2048 * 4096, torch.bfloat16)):
        z = torch.zeros(n, device=dev, dtype=dt)
        for _ in range(2):
            m.ar(z)
        gemms(3)
        m.barrier()
        torch.cuda.synchronize()
        ts = {}
        for mode in ("ar", "gemm", "both"):
            m.barrier()
            torch.cuda.synchronize()
            e0, e1 = ev(), ev()
            e0.record()
            if mode == "ar":
                m.ar(z)
            elif mode == "gemm":
                gemms(4)
            else:
                tk = m.ar_async(z)
                gemms(4)
                m.join(tk)
            e1.record()
            torch.cuda.synchronize()
            ts[mode] = e0.elapsed_time(e1)
        say(f"[overlap] {name}: all-reduce {ts['ar']:.2f} ms · 4 GEMMs {ts['gemm']:.2f} ms · together {ts['both']:.2f} ms "
            f"(serial {ts['ar'] + ts['gemm']:.2f}; hidden {100 * (ts['ar'] + ts['gemm'] - ts['both']) / ts['ar']:.0f}% of the all-reduce)")

m.barrier()
m.close()
say(f"[comm] done · {'OK' if not fails else f'{fails} failures'}")
