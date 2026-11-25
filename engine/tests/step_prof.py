#!/usr/bin/env python3
"""step_prof.py — where a decode step's time goes (TP=2, both ranks; through tools/tp2.sh, which appends --rank and
--head-ip):  tools/tp2.sh prof python3 /work/src/engine/tests/step_prof.py [--ctx 64,9400] [--steps 32]
For each context length: one sequence prefilled to that length (doc_long's tokens repeated), then
  wall   --steps greedy decode steps as the server runs them (step + sampling), nothing synchronized in between
  stages the same number of steps with every stage synchronized and timed (model.cc chk, sync mode 2): ms per step of
         each stage, grouped (attention, FP8 / bf16 GEMMs, MoE, collectives, mHC and the rest), then the top lines.
Synchronizing every stage adds its own latency (~tens of µs a stage), so the staged total is above the wall time;
the shares show where the time goes. Rank 0 prints.
--nsys: instead, one window with NVTX ranges "decode" (--steps steps at the first --ctx) and "prefill" (one 4096-row
chunk, then the same chunk staged), synchronized at each end (nsys on GB10 recorded no kernels — hardware trace).
--prefill-at P,…: one 4096-row chunk at each position, wall and staged.
--verify N,…: at each --ctx, also an N-row step on the sequence (all logits, rewound to the same position each time:
a DSpark verify's target step without the draft's context rows), wall and staged.
--lanes L,…: the --prefill-at chunks also with E55_LANES=L (P075 lanes per part; 1 = one lane, no overlap)."""
import argparse
import collections
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ext  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ctx", default="64,9400")
ap.add_argument("--steps", type=int, default=32)
ap.add_argument("--model", default="/models/ERNIE-5.5-Flash-Preview")
ap.add_argument("--table", default="/work/cache/table.tsv")
ap.add_argument("--prompts", default="/work/results/ref/v3/prompts.json")
ap.add_argument("--rank", type=int, required=True)
ap.add_argument("--head-ip", required=True)
ap.add_argument("--port", type=int, default=29611)
ap.add_argument("--nsys", action="store_true")
ap.add_argument("--prefill-at", default="", help="also time one 4096-row prefill chunk starting at these positions")
ap.add_argument("--full", action="store_true", help="every stage line (name:line ms calls), not only the top ones")
ap.add_argument("--rows", default="4096", help="--prefill-at chunk sizes (rows), e.g. 4096,8192,16384")
ap.add_argument("--verify", default="", help="also N-row verify-shaped steps at each --ctx, e.g. 4,8")
ap.add_argument("--lanes", default="", help="--prefill-at chunks also with E55_LANES=L, e.g. 1,2")
ap.add_argument("--lanes-env", default="E55_LANES", help="the variable --lanes sets (e.g. E55_A2A_AHEAD)")
a = ap.parse_args()
R0 = a.rank == 0


def say(*x):
    if R0:
        print(*x, flush=True)


ctxs = [int(x) for x in a.ctx.split(",")]
E = ext.load("engine", ext.engine_sources(), log=f"/tmp/build{a.rank}.log", ldflags=ext.ENGINE_LDFLAGS)
PF = [int(x) for x in a.prefill_at.split(",") if x]
ROWS = [int(x) for x in a.rows.split(",")]
VR = [int(x) for x in a.verify.split(",") if x]
LANES = [int(x) for x in a.lanes.split(",") if x]
E.init(a.model, a.table, a.rank, a.head_ip, 1,
       max(ctxs + [x + max(ROWS) for x in PF]) + 2 * a.steps + 8 + max(VR + [0]) + (8192 if a.nsys else 0),
       max(ROWS + [4096]), max(VR + [1]), a.port)
base = json.load(open(a.prompts))["doc_long"]

GROUPS = [("attention", ("hca_", "dsa_", "vha_", "mul_bf16", "sigmoid")), ("FP8 GEMM", ("f8_forward", "act")),
          ("bf16 GEMM", ("lin_forward",)), ("MoE", ("moe_",)), ("collectives", ("comm_",)),
          ("mHC", ("mhc_",)), ("norms / FFN act", ("rmsnorm", "situ", "add_bf16")),
          ("head / logits", ("hc_head", "lm_head", "logits"))]


def group(name):
    return next((g for g, pre in GROUPS if name.startswith(pre)), "other")


if a.nsys:
    import torch
    ctx = ctxs[0]
    ids = (base * ((ctx + 12288) // len(base) + 1))[:ctx + 12288]
    E.reset(0)
    for c0 in range(0, ctx - 1, 4096):
        c1 = min(ctx - 1, c0 + 4096)
        E.fill(ids[c0:c1], list(range(c0, c1)), [0], [0, c1 - c0])
    p, tok = ctx - 1, ids[ctx - 1]
    for _ in range(4):  # warm-up
        tok = E.run([tok], [p], [0], [0, 1], [0.0], [0], [1.0], [0.0], [1], [p + 1], [1.0], [0.0], [0.0])[0]
        p += 1
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push("decode")
    t = time.time()
    for _ in range(a.steps):
        tok = E.run([tok], [p], [0], [0, 1], [0.0], [0], [1.0], [0.0], [1], [p + 1], [1.0], [0.0], [0.0])[0]
        p += 1
    torch.cuda.synchronize()
    dt = time.time() - t
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("prefill")
    t = time.time()
    E.fill(ids[p:p + 4096], list(range(p, p + 4096)), [0], [0, 4096])
    torch.cuda.synchronize()
    dp = time.time() - t
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    say(f"[prof] nsys window: decode {1e3 * dt / a.steps:.1f} ms/step at {ctx:,} · prefill 4096 rows {1e3 * dp:.0f} ms "
        f"({4096 / dp:,.0f} tok/s)")
    # the same prefill chunk again, every stage synchronized and timed
    E.sync_mode(2)
    E.prof()
    p += 4096
    E.fill(ids[p:p + 4096], list(range(p, p + 4096)), [0], [0, 4096])
    rows = E.prof()
    E.sync_mode(0)
    g = collections.Counter()
    for name, line, ms, n in rows:
        g[group(name)] += ms
    tot = sum(g.values())
    say(f"[prof] prefill 4096 rows staged {tot:.0f} ms: " + " · ".join(f"{k} {v:.0f} ms ({100 * v / tot:.0f}%)"
                                                                    for k, v in g.most_common()))
    top = sorted(rows, key=lambda x: -x[2])[:16]
    say("  top: " + " · ".join(f"{n}:{ln} {ms:.0f}" for n, ln, ms, _ in top))
    E.close()
    sys.exit(0)

for ctx in ctxs:
    ids = (base * (ctx // len(base) + 1))[:ctx]
    E.reset(0)
    for c0 in range(0, ctx - 1, 4096):
        c1 = min(ctx - 1, c0 + 4096)
        E.fill(ids[c0:c1], list(range(c0, c1)), [0], [0, c1 - c0])
    p = ctx - 1
    tok = ids[p]

    def decode(n):
        global p, tok
        for _ in range(n):
            tok = E.run([tok], [p], [0], [0, 1], [0.0], [0], [1.0], [0.0], [1], [p + 1], [1.0], [0.0], [0.0])[0]
            p += 1

    decode(2)  # warm-up
    t = time.time()
    decode(a.steps)
    wall = 1e3 * (time.time() - t) / a.steps
    E.sync_mode(2)
    E.prof()
    t = time.time()
    decode(a.steps)
    staged = 1e3 * (time.time() - t) / a.steps
    rows = E.prof()
    E.sync_mode(0)
    say(f"[prof] context {ctx:,}: wall {wall:.1f} ms/step ({1e3 / wall:.1f} tok/s) · staged {staged:.1f} ms/step")
    g = collections.Counter()
    for name, line, ms, n in rows:
        g[group(name)] += ms / a.steps
    tot = sum(g.values())
    say("  " + " · ".join(f"{k} {v:.1f} ms ({100 * v / tot:.0f}%)" for k, v in g.most_common()))
    top = sorted(rows, key=lambda x: -x[2])[:None if a.full else 14]
    say("  top: " + " · ".join(f"{n}:{ln} {ms / a.steps:.3f}" + (f"×{c // a.steps}" if a.full else "") for n, ln, ms, c in top))
    vids = (base * ((p + max(VR + [0])) // len(base) + 1))[p:p + max(VR + [0])]
    for nv in VR:
        def vstep(k):
            for _ in range(k):
                E.set_history(0, p)
                E.step(vids[:nv], list(range(p, p + nv)), [0], [0, nv], True)
        vstep(2)
        t = time.time()
        vstep(a.steps)
        wall = 1e3 * (time.time() - t) / a.steps
        E.sync_mode(2)
        E.prof()
        t = time.time()
        vstep(a.steps)
        staged = 1e3 * (time.time() - t) / a.steps
        rows = E.prof()
        E.sync_mode(0)
        say(f"[prof] verify {nv} rows at {p:,}: wall {wall:.1f} ms/step · staged {staged:.1f} ms/step")
        g = collections.Counter()
        for name, line, ms, n in rows:
            g[group(name)] += ms / a.steps
        tot = sum(g.values())
        say("  " + " · ".join(f"{k} {v:.1f} ms ({100 * v / tot:.0f}%)" for k, v in g.most_common()))
        top = sorted(rows, key=lambda x: -x[2])[:None if a.full else 14]
        say("  top: " + " · ".join(f"{n}:{ln} {ms / a.steps:.3f}" + (f"×{c // a.steps}" if a.full else "")
                                   for n, ln, ms, c in top))
    E.set_history(0, p)

for start, nr, lanes in [(p, r, L) for L in [0] + LANES for p in (PF if not L else PF[:1]) for r in ROWS]:
    import torch
    if lanes:
        os.environ[a.lanes_env] = str(lanes)
    else:
        os.environ.pop(a.lanes_env, None)
    lt = f" · {a.lanes_env} {lanes}" if lanes else ""
    ids = (base * ((start + nr) // len(base) + 1))[:start + nr]
    for staged in (False, True):
        E.reset(0)
        for c0 in range(0, start, 4096):
            c1 = min(start, c0 + 4096)
            E.fill(ids[c0:c1], list(range(c0, c1)), [0], [0, c1 - c0])
        torch.cuda.synchronize()
        if staged:
            E.sync_mode(2)
            E.prof()
        t = time.time()
        E.fill(ids[start:start + nr], list(range(start, start + nr)), [0], [0, nr])
        torch.cuda.synchronize()
        dt = 1e3 * (time.time() - t)
        if not staged:
            wall = dt
            continue
        rows = E.prof()
        E.sync_mode(0)
        g = collections.Counter()
        for name, line, ms, n in rows:
            g[group(name)] += ms
        say(f"[prof] prefill {nr} rows at {start:,}{lt}: wall {wall:.0f} ms ({nr * 1e3 / wall:,.0f} tok/s) · staged {dt:.0f} ms")
        say("  " + " · ".join(f"{k} {v:.0f} ms ({100 * v / dt:.0f}%)" for k, v in g.most_common()))
        top = sorted(rows, key=lambda x: -x[2])[:None if a.full else 16]
        say("  top: " + " · ".join(f"{n}:{ln} {ms:.1f}" + (f"×{c}" if a.full else "") for n, ln, ms, c in top))
E.close()
