#!/usr/bin/env python3
"""test_dspark.py — DSpark speculative decoding in the engine (model/dspark.cc): the draft against the reference, plain
vs speculative decoding token for token and logits bit for bit, and the speed (ms / tok/s, our own measurements).

  ref   (one GPU, node B, no engine): tools/agent.sh dspark --gpu python3 engine/tests/test_dspark.py ref
        the reference draft (tools/ref/dspark_ref.py) at every anchor of the stored target states (<data>/<name>.pt,
        from engine/tests/dspark_accept.py gen) for N = 7 and 4: drafts + the top-2 margin of each Markov step
        → <out>/ref_drafts.pt (copy it to the other node's <out> for the TP=2 run)
  tp2   (both ranks): E55_GPU_LOCK=1 E55_TREE=dspark tools/tp2.sh dsp python3 /work/dev/dspark/src/engine/tests/
        test_dspark.py tp2 --do agree,lossless,curve,batch2,c4,gate,sampled
        agree     the engine's draft on the stored states (their aux features fed straight into the context rows) vs
                  the reference drafts; the confidence head against the drafts' real acceptance
        lossless  per prompt plain greedy decoding (thinking ≤ --think-cap, then </think> forced, answer ≤ --answer)
                  vs greedy DSpark at N = --ns: identical tokens, and every kept verify row's logits bitwise equal
                  to plain decoding's at that position (row 0 after a rejection: the rollback); tok/s per content
                  type (thinking / the four answers) and the ms split
        curve     the target step's ms by rows (1 … 8) on one sequence (verify cost per N) + distinct experts
        batch2    two sequences per verify step (tokens checked against plain decoding)
        c4        four sequences per verify step, N = 1 … 4: four content types, and four code prompts
        gate      confidence-gated verify length (draft block 7, verify while Π σ(conf) ≥ gate) vs fixed N
        sampled   temperature 1.0, top_p 0.95 (generation_config): acceptance and tok/s
        check     short losslessness check at the default config (--depth, --gate-def): per prompt (--names, default
                  ko_prose, en_prose, py_code) plain greedy decoding vs greedy DSpark — identical tokens, kept verify
                  rows' logits bitwise plain decoding's
        draftms   one draft pass (spec_draft, wall ms incl. its sync) at --dms-depths, the current default vs
                  --dms-patches on (drafts / confidences compared bit for bit), and the per-stage split (E55_DSP_PROF)"""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "server"))
sys.path.insert(0, os.path.join(ROOT, "tools", "ref"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

ap = argparse.ArgumentParser()
ap.add_argument("phase", choices=("ref", "tp2"))
ap.add_argument("--model", default="/models/ERNIE-5.5-Flash-Preview")
ap.add_argument("--table", default="/work/dev/dspark/table.tsv")
ap.add_argument("--data", default="/work/dev/dspark/data")
ap.add_argument("--out", default="/work/dev/dspark/out")
ap.add_argument("--rank", type=int, default=0)
ap.add_argument("--head-ip", default="")
ap.add_argument("--port", type=int, default=29611)
ap.add_argument("--do", default="agree,lossless,curve,batch2,c4,gate,sampled")
ap.add_argument("--ns", default="3,4,7")
ap.add_argument("--think-cap", type=int, default=192)
ap.add_argument("--answer", type=int, default=256)
ap.add_argument("--limit", type=int, default=0, help="agree / ref: first N anchors per segment")
ap.add_argument("--names", default="")
ap.add_argument("--gates", default="0.3,-1", help="gate: thresholds (> 0) or -1 (expected tokens per ms)")
ap.add_argument("--depth", type=int, default=7, help="check: the default config's draft block")
ap.add_argument("--gate-def", type=float, default=0.3, help="check: the default config's gate")
ap.add_argument("--dms-depths", default="4,7", help="draftms: draft depths timed")
ap.add_argument("--dms-patches", default="", help="draftms: patches compared against the current default (off → on)")
ap.add_argument("--dms-sync", action="store_true", help="draftms: also one synchronized depth-7 pass (sync mode 2)")
ap.add_argument("--perturb", action="store_true", help="ref: the attention probabilities rounded to bf16 before P·V "
                "(the engine's rounding) → ref_drafts_p.pt, and its agreement with the plain reference")
a = ap.parse_args()
R0 = a.rank == 0
NAMES = ["ko_prose", "en_prose", "py_code", "tool_call"]
SEED = 1234


def say(*x):
    if R0:
        print(*x, flush=True)


# ---------------------------------------------------------------------------------------------------- prompts
TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "Current weather and a 3-day forecast for a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "search_flights", "description": "Search one-way flights.",
        "parameters": {"type": "object", "properties": {
            "origin": {"type": "string", "description": "IATA code"},
            "destination": {"type": "string", "description": "IATA code"},
            "date": {"type": "string", "description": "YYYY-MM-DD"},
            "passengers": {"type": "integer"},
            "cabin": {"type": "string", "enum": ["economy", "premium", "business"]}},
            "required": ["origin", "destination", "date"]}}},
    {"type": "function", "function": {
        "name": "search_hotels", "description": "Search hotels in a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}, "check_in": {"type": "string", "description": "YYYY-MM-DD"},
            "check_out": {"type": "string", "description": "YYYY-MM-DD"}, "guests": {"type": "integer"},
            "max_price_eur": {"type": "number"}}, "required": ["city", "check_in", "check_out"]}}},
]
PROMPTS = {  # engine/tests/dspark_accept.py's
    "ko_prose": dict(messages=[{"role": "user", "content":
                                "조선 시대의 과거 제도가 어떻게 운영되었고, 조선 사회에 어떤 영향을 미쳤는지 자세히 설명해 줘."}]),
    "en_prose": dict(messages=[{"role": "user", "content":
                                "Explain in detail how the printing press changed European society in the 15th and "
                                "16th centuries."}]),
    "py_code": dict(messages=[{"role": "user", "content":
                               "Write a Python module that implements a thread-safe LRU cache with a per-entry TTL, "
                               "with type hints and docstrings, followed by pytest unit tests. Output only the code."}]),
    "tool_call": dict(messages=[
        {"role": "system", "content": "You are a travel assistant. Use the tools; call every tool you need at once."},
        {"role": "user", "content": "Plan my trip: fly from Seoul (ICN) to Paris (CDG) on 2026-12-10 for 2 adults in "
                                    "economy, then Paris to Rome (FCO) on 2026-12-14, then Rome back to Seoul on "
                                    "2026-12-18. Check the weather in Paris and Rome in celsius, and find hotels in "
                                    "both cities for 2 guests under 250 EUR a night for those dates."}],
        tools=TOOLS),
}
CODE4 = {  # four code prompts (c4, same content type)
    "code_lru": PROMPTS["py_code"],
    "code_json": dict(messages=[{"role": "user", "content":
                                 "Write a TypeScript module with a small recursive-descent JSON parser (no JSON.parse) "
                                 "that reports line and column on errors, with JSDoc comments. Output only the code."}]),
    "code_sql": dict(messages=[{"role": "user", "content":
                                "Write a Go package implementing an in-memory key-value store with snapshots and a "
                                "write-ahead log, with doc comments and table-driven tests. Output only the code."}]),
    "code_rust": dict(messages=[{"role": "user", "content":
                                 "Write a Rust library crate with a bounded multi-producer multi-consumer queue built "
                                 "on Mutex and Condvar, with docs and unit tests. Output only the code."}]),
}


# ---------------------------------------------------------------------------------------------------- ref phase
def draft_margin(D, atok, apos, ctx, cok, N):
    """dspark_ref.Drafter.draft with the top-2 margin of each Markov step."""
    import dspark_ref as dsr
    B = atok.numel()
    tok = torch.full((B, N), dsr.NOISE, dtype=torch.long, device=dsr.DEV)
    tok[:, 0] = atok
    pos = apos[:, None] + torch.arange(N, device=dsr.DEV)[None, :]
    h, _ = D.forward(tok.reshape(-1), pos.reshape(-1), ctx, cok, N)
    h = h.view(B, N, -1)
    prev = atok.clone()
    out = torch.empty(B, N, dtype=torch.long, device=dsr.DEV)
    mg = torch.empty(B, N, device=dsr.DEV)
    for i in range(N):
        lg = dsr.multimax(h[:, i].float() @ D.head.t() + D.mk1[prev].float() @ D.mk2f.t(), D.mm_r, D.mm_t)
        t2 = lg.topk(2, -1)
        prev = lg.argmax(-1)
        out[:, i] = prev
        mg[:, i] = t2.values[:, 0] - t2.values[:, 1]
    return out, mg


@torch.no_grad()
def ref_phase():
    import dspark_ref as dsr
    import ernie_ref as ref
    t0 = time.time()
    if a.perturb:  # a rounding-level change: P in bf16 for P·V, as the engine's attention
        sm = ref.softmax_sink
        dsr.ref.softmax_sink = lambda s, sink: sm(s, sink).to(torch.bfloat16).float()
    D = dsr.Drafter(ref.Ckpt(a.model))
    print(f"[ref] drafter loaded · {time.time() - t0:.0f} s", flush=True)
    res = {}
    for name in (a.names.split(",") if a.names else NAMES):
        S = torch.load(os.path.join(a.data, f"{name}.pt"), weights_only=False)
        seq, rows, segs = S["seq"], S["rows"], S["segs"]
        n = len(rows)
        pos2row = torch.full((len(seq) + 1,), -1, dtype=torch.long)
        pos2row[torch.tensor(rows)] = torch.arange(n)
        sel = [S["aux_layers"].index(L) for L in dsr.TARGET_LAYERS]
        aux = S["aux"][sel].to(ref.DEV).permute(1, 0, 2).reshape(n, 3 * dsr.H)
        main_x = D.features(aux)
        ckv = D.context_kv(main_x, torch.tensor(rows, device=ref.DEV))
        anchors = [p for _, s0, s1, _ in segs for p in range(s0, s1 if not a.limit else min(s1, s0 + a.limit))]
        ap_ = torch.tensor(anchors, device=ref.DEV)
        atok = torch.tensor([seq[p] for p in anchors], device=ref.DEV)
        cpos = ap_[:, None] - 128 + torch.arange(128, device=ref.DEV)[None, :]
        cidx = pos2row.to(ref.DEV)[cpos.clamp(min=0)]
        cok = (cpos >= 0) & (cidx >= 0)
        cidx = cidx.clamp(min=0)
        r = {"anchors": anchors}
        for N in (7, 4):
            dr, mg = [], []
            for c0 in range(0, len(anchors), 128):
                sl = slice(c0, c0 + 128)
                ctx = [c[cidx[sl]] for c in ckv]
                d_, m_ = draft_margin(D, atok[sl], ap_[sl], ctx, cok[sl], N)
                dr.append(d_.cpu()), mg.append(m_.cpu())
            r[f"d{N}"], r[f"m{N}"] = torch.cat(dr), torch.cat(mg)
        res[name] = r
        print(f"[ref] {name}: {len(anchors)} anchors · {time.time() - t0:.0f} s", flush=True)
    os.makedirs(a.out, exist_ok=True)
    fn = os.path.join(a.out, "ref_drafts_p.pt" if a.perturb else "ref_drafts.pt")
    torch.save(res, fn)
    print(f"[ref] → {fn}", flush=True)
    if a.perturb:  # how far a rounding-level change moves the reference's own drafts
        R0_ = torch.load(os.path.join(a.out, "ref_drafts.pt"), weights_only=False)
        for N in (7, 4):
            n = same = first = 0
            for name, r in res.items():
                x, y = r[f"d{N}"], R0_[name][f"d{N}"]
                n += x.shape[0]
                same += int((x == y).all(1).sum())
                first += int((x[:, 0] == y[:, 0]).sum())
            print(f"[ref] reference vs reference with bf16 P, N={N}: blocks identical {same}/{n} "
                  f"({100 * same / n:.1f} %) · first draft {first}/{n} ({100 * first / n:.1f} %)", flush=True)


# ---------------------------------------------------------------------------------------------------- engine
E = chat = None
RES = {}
MAXR = 1024


def prefill(slot, ids):
    """The prompt in chunks of MAXR; returns the last row's logits (fp32 [V])."""
    E.reset(slot)
    T = len(ids)
    for c0 in range(0, T, MAXR):
        c1 = min(T, c0 + MAXR)
        if c1 < T:
            E.fill(ids[c0:c1], list(range(c0, c1)), [slot], [0, c1 - c0])
        else:
            lg = E.step(ids[c0:c1], list(range(c0, c1)), [slot], [0, c1 - c0], False)
    return lg[0]


def encode(kw):
    from chat import normalize_tools
    return chat.encode_chat(chat.render(kw["messages"], normalize_tools(kw.get("tools")), thinking=True, level="high"))


DRAFT_PATCHES = ("draft_lm8", "draft_lm4", "markov8", "markov_topk")


def agree():
    """The base draft (draft patches off) against the reference, then the default draft patches' drafts."""
    dflt = {p: E.patch_on(p) for p in DRAFT_PATCHES}
    for p in DRAFT_PATCHES:
        E.patch(p, False)
    say("[agree] base draft (P080–P083 off)")
    agree_run("base")
    if any(dflt.values()):
        for p, v in dflt.items():
            E.patch(p, v)
        say("[agree] with the default draft patches: " + ", ".join(p for p, v in dflt.items() if v))
        agree_run("patched")
    for p, v in dflt.items():
        E.patch(p, v)


def agree_run(tag):
    import dspark_ref as dsr
    R = torch.load(os.path.join(a.out, "ref_drafts.pt"), weights_only=False)
    fp = os.path.join(a.out, "ref_drafts_p.pt")
    RP = torch.load(fp, weights_only=False) if os.path.exists(fp) else None
    tot = {}
    for name in (a.names.split(",") if a.names else NAMES):
        S = torch.load(os.path.join(a.data, f"{name}.pt"), weights_only=False)
        seq, rows = S["seq"], S["rows"]
        sel = [S["aux_layers"].index(L) for L in dsr.TARGET_LAYERS]
        n = len(rows)
        aux = S["aux"][sel].permute(1, 0, 2).reshape(n, 3 * 4096).contiguous().cuda()
        p2r = {p: i for i, p in enumerate(rows)}
        r = R[name]
        rp = RP[name] if RP else None
        anchors = r["anchors"]
        E.reset(0)
        fed = -1
        st = {N: dict(n=0, same=0, first=0, tie=0, dpos=0, mism=[], same_p=0) for N in (7, 4)}
        cal = [[0.0, 0, 0] for _ in range(7)]  # per draft position: Σ σ(conf), reached, accepted
        for ai, P in enumerate(anchors):
            want = [p for p in range(max(0, P - 128), P) if p > fed]
            if want:
                idx = torch.tensor([p2r[p] for p in want], device="cuda")
                E.dspark_ctx(0, aux[idx].contiguous(), want)
                fed = P - 1
            for N in (7, 4):
                E.set_history(0, P)
                dr, cf = E.draft_at(0, seq[P], P, N)
                ref_d = r[f"d{N}"][ai].tolist()
                s = st[N]
                s["n"] += 1
                s["same"] += dr == ref_d
                s["first"] += dr[0] == ref_d[0]
                if rp is not None:
                    s["same_p"] += dr == rp[f"d{N}"][ai].tolist()
                if dr != ref_d:
                    i = next(j for j in range(N) if dr[j] != ref_d[j])
                    m = float(r[f"m{N}"][ai][i])
                    s["mism"].append(m)
                    s["dpos"] += i
                if N == 7:  # confidence vs the real continuation (engine drafts)
                    for j in range(N):
                        if P + 1 + j >= len(seq):
                            break
                        cal[j][0] += 1 / (1 + torch.exp(torch.tensor(-cf[j])).item())
                        cal[j][1] += 1
                        if dr[j] != seq[P + 1 + j]:
                            break
                        cal[j][2] += 1
        for N in (7, 4):
            s = st[N]
            mm = sorted(s["mism"])
            say(f"[agree] {name} N={N}: blocks identical {s['same']}/{s['n']} ({100 * s['same'] / s['n']:.1f} %) · first "
                f"draft {s['first']}/{s['n']} · at the first difference the reference's top-2 margin: "
                + (f"median {mm[len(mm) // 2]:.4f}, max {mm[-1]:.4f}, ≤ 0.05: {sum(x <= 0.05 for x in mm)}/{len(mm)}"
                   if mm else "—"))
            t = tot.setdefault(N, dict(n=0, same=0, first=0, mism=[], same_p=0))
            for k in ("n", "same", "first", "same_p"):
                t[k] += s[k]
            t["mism"] += s["mism"]
        say(f"[agree] {name} confidence (engine N=7): mean σ(conf) vs measured P(accept | prefix accepted) by position: "
            + " ".join(f"{c[0] / max(1, c[1]):.2f}/{c[2] / max(1, c[1]):.2f}" for c in cal))
        RES.setdefault("agree_" + tag, {})[name] = {str(N): {k: v for k, v in st[N].items() if k != "mism"} for N in st}
        RES["agree_" + tag][name]["cal"] = cal
    for N, t in tot.items():
        mm = t["mism"]
        say(f"[agree] all N={N}: blocks identical {t['same']}/{t['n']} ({100 * t['same'] / max(1, t['n']):.1f} %) · first "
            f"draft {t['first']}/{t['n']} ({100 * t['first'] / max(1, t['n']):.1f} %) · differing blocks whose first "
            f"difference has a reference top-2 margin ≤ 0.05: {sum(x <= 0.05 for x in mm)}/{len(mm)}, ≤ 0.25: "
            f"{sum(x <= 0.25 for x in mm)}/{len(mm)}"
            + (f" · vs the reference with bf16 attention probabilities: blocks identical {t['same_p']}/{t['n']} "
               f"({100 * t['same_p'] / max(1, t['n']):.1f} %)" if RP else ""))
    E.reset(0)


def plain(slot, ids, think_cap, answer):
    """Greedy plain decoding (as dspark_accept gen) → (seq, T, te, forced position or -1, {pos: logits}, step ms)."""
    T = len(ids)
    lg = prefill(slot, ids)
    seq, te, forced, lgs = list(ids), -1, -1, {T - 1: lg}
    torch.cuda.synchronize()
    t = time.time()
    steps = 0
    while True:
        nxt = int(lg.argmax())
        if te < 0 and len(seq) - T >= think_cap and nxt != chat.think_end:
            nxt, forced = chat.think_end, len(seq)
        seq.append(nxt)
        if nxt == chat.think_end and te < 0:
            te = len(seq) - 1
        if nxt in chat.eos or (te >= 0 and len(seq) - 1 - te >= answer) or len(seq) >= T + think_cap + answer + 8:
            break
        lg = E.step([nxt], [len(seq) - 1], [slot], [0, 1], False)[0]
        lgs[len(seq) - 1] = lg
        steps += 1
    dt = (time.time() - t) / max(1, steps)
    return seq, T, te, forced, lgs, dt * 1e3


def spec_single(name, ids, ref, N, gate=0.0, temp=0.0, top_p=1.0, check=True):
    """One sequence with DSpark (slot 0). ref = plain's (seq, T, te, forced, lgs): the same forced </think> and length;
    check: tokens and logits against it. Returns stats per segment (thinking / answer)."""
    seq_p, T, te, forced, lgs, _ = ref
    lg0 = prefill(0, ids)
    if check:
        assert torch.equal(lg0, lgs[T - 1]), "prefill logits differ"
    tok = E.spec_begin(0, N, temp, SEED, 0, gate=gate, top_p=top_p)
    out = list(ids) + [tok]
    st = {"think": [0, 0.0, 0], "answer": [0, 0.0, 0]}  # tokens, s, rounds
    cmp = dict(rows=0, bad=0, after_rej=0, after_rej_bad=0, cross_rej=0)
    prev_rej = False
    end = len(seq_p)
    E.spec_stats(True)
    ok = True
    te_s = -1  # this run's </think> position
    while len(out) < end:
        P = E.spec_pos(0)
        cap = end - 1 - P
        if forced > P and te_s < 0:
            cap = min(cap, forced - P)
        E.spec_cap(0, cap)
        torch.cuda.synchronize()
        t = time.time()
        toks = E.spec_round([0])[0]
        dt = time.time() - t
        s = st["think" if te_s < 0 else "answer"]
        s[0] += len(toks)
        s[1] += dt
        s[2] += 1
        if check:
            L = E.spec_logits()
            for j in range(len(toks)):
                same = torch.equal(L[j], lgs[P + j])
                cmp["rows"] += 1
                cmp["bad"] += not same
                if j == 0 and prev_rej:
                    cmp["after_rej"] += 1
                    cmp["after_rej_bad"] += not same
            ver = L.shape[0] - 1
            rej = len(toks) < ver + 1  # rows dropped (rejected or past the cap): their ring rows were put back
            cmp["cross_rej"] += rej and any((p + 1) % 128 == 0 for p in range(P + len(toks), P + ver + 1))
            prev_rej = rej
        out += toks
        if forced == len(out) - 1 and te_s < 0:
            E.spec_last(0, chat.think_end)
            out[-1] = chat.think_end
        new = out[len(out) - len(toks):]
        if te_s < 0 and chat.think_end in new:
            te_s = len(out) - len(toks) + new.index(chat.think_end)
        if check and out != seq_p[:len(out)]:
            i = next(j for j in range(len(out)) if out[j] != seq_p[j])
            say(f"  [lossless] {name} N={N}: TOKEN MISMATCH at position {i} (plain {seq_p[i]}, spec {out[i]})")
            ok = False
            break
        if not check and (out[-1] in chat.eos):
            break
    S = E.spec_stats(True)
    return st, cmp, S, ok, out


def fmt_split(S):
    r = max(1, S["rounds"])
    return (f"verify {S['ms_verify'] / r:.1f} + ctx {S['ms_ctx'] / r:.2f} · accept {S['ms_accept'] / r:.2f} · commit "
            f"{S['ms_commit'] / r:.2f} · draft {S['ms_draft'] / r:.1f} ms/round · {S['emitted'] / r:.2f} tok/round "
            f"(accepted {S['accepted'] / max(1, S['verified']):.2f} of verified)")


PLAIN = {}


def lossless():
    ns = [int(x) for x in a.ns.split(",")]
    agg = {}  # (type, N) → [tokens, s, rounds]
    plain_ms = []
    for name in (a.names.split(",") if a.names else NAMES):
        ids = encode(PROMPTS[name])
        ref = plain(0, ids, a.think_cap, a.answer)
        PLAIN[name] = ref
        seq, T, te, forced, lgs, ms = ref
        plain_ms.append(ms)
        say(f"[lossless] {name}: prompt {T} · thinking {te - T if te > 0 else len(seq) - T}"
            f"{' (forced end)' if forced >= 0 else ''} · answer {len(seq) - te - 1 if te > 0 else 0} · plain "
            f"{ms:.1f} ms/token")
        for N in ns:
            st, cmp, S, ok, _ = spec_single(name, ids, ref, N)
            say(f"  N={N}: tokens {'IDENTICAL' if ok else 'DIFFER'} ({len(seq) - T}) · logits rows bitwise equal "
                f"{cmp['rows'] - cmp['bad']}/{cmp['rows']} · row 0 after a rejection {cmp['after_rej'] - cmp['after_rej_bad']}"
                f"/{cmp['after_rej']} · rejections across a compressor group end {cmp['cross_rej']}")
            say(f"        {fmt_split(S)}")
            for seg, ty in (("think", "thinking"), ("answer", name)):
                s = st[seg]
                if s[0]:
                    say(f"        {ty}: {s[0]} tokens in {s[2]} rounds · {s[0] / s[1]:.1f} tok/s")
                    g = agg.setdefault((ty, N), [0, 0.0, 0])
                    g[0] += s[0]
                    g[1] += s[1]
                    g[2] += s[2]
            RES.setdefault("lossless", {}).setdefault(name, {})[str(N)] = dict(ok=ok, cmp=cmp, split=S, seg=st)
        del lgs, ref
        PLAIN[name] = (seq, T, te, forced, None, ms)
        torch.cuda.empty_cache()
    say(f"[speed] single stream, greedy · plain decode {sum(plain_ms) / len(plain_ms):.1f} ms/token "
        f"({1e3 / (sum(plain_ms) / len(plain_ms)):.1f} tok/s)")
    for ty in ["ko_prose", "en_prose", "py_code", "tool_call", "thinking"]:
        line = []
        best = None
        for N in ns:
            g = agg.get((ty, N))
            if g:
                ts = g[0] / g[1]
                line.append(f"N={N} {ts:.1f} tok/s ({g[0] / g[2]:.2f} tok/round)")
                best = max(best or (0, 0), (ts, N))
        say(f"  {ty:10s} " + " · ".join(line) + (f" · best N={best[1]}" if best else ""))
    RES["speed1"] = {f"{k[0]}/{k[1]}": v for k, v in agg.items()}


def curve():
    """Target step ms by rows on one sequence (the verify cost per N) + distinct experts per layer."""
    name = "py_code"
    seq, T, te, forced, _, _ = PLAIN.get(name) or plain(0, encode(PROMPTS[name]), 64, 64)
    P0 = te + 20 if te > 0 else T + 20
    prefill(0, seq[:P0])
    res = {}
    for n in range(1, 9):
        ms = []
        E.expert_count(True)
        for rep in range(6):
            E.set_history(0, P0)
            torch.cuda.synchronize()
            t = time.time()
            E.step(seq[P0:P0 + n], list(range(P0, P0 + n)), [0], [0, n], True)
            torch.cuda.synchronize()
            ms.append((time.time() - t) * 1e3)
        ex, steps = E.expert_take()
        ms = sorted(ms)[1:-1]
        res[n] = (sum(ms) / len(ms), sum(ex[1:]) / 42)
    say("[curve] target step on one sequence (all-logits verify step, incl. the draft's context rows): rows → ms "
        "(distinct experts / layer): " + " · ".join(f"{n}: {v[0]:.1f} ({v[1]:.1f})" for n, v in res.items()))
    RES["curve"] = res


def variants():
    """The draft patches (P080–P083) one by one: single stream, greedy, N=4, every prompt (tokens equal plain's by
    construction — greedy verification). The fastest set stays on for the later phases."""
    cfgs = [("base", ()), ("lm8", ("draft_lm8",)), ("lm4", ("draft_lm4",)), ("markov8", ("markov8",)),
            ("topk", ("markov_topk",)), ("lm4+markov8", ("draft_lm4", "markov8"))]
    best = None
    for tag, on in cfgs:
        for p in DRAFT_PATCHES:
            E.patch(p, p in on)
        tok, sec, rounds, dms, acc, ver = 0, 0.0, 0, 0.0, 0, 0
        for name in NAMES:
            ref = PLAIN.get(name)
            st, _, S, _, _ = spec_single(name, encode(PROMPTS[name]), ref, 4, check=False)
            for seg in st.values():
                tok += seg[0]
                sec += seg[1]
                rounds += seg[2]
            dms += S["ms_draft"]
            acc += S["accepted"]
            ver += S["verified"]
        ts = tok / sec
        say(f"[variants] {tag:12s}: {ts:.1f} tok/s · {tok / rounds:.2f} tok/round · accepted {acc / max(1, ver):.3f} of "
            f"verified · draft {dms / rounds:.2f} ms/round")
        RES.setdefault("variants", {})[tag] = dict(tok_s=ts, tok_round=tok / rounds, draft_ms=dms / rounds)
        best = max(best or (0, "", ()), (ts, tag, on))
    for p in DRAFT_PATCHES:
        E.patch(p, p in best[2])
    RES["draft_ms"] = RES["variants"][best[1]]["draft_ms"]
    say(f"[variants] kept for the next phases: {best[1]}")


def multi(names, kws, N, think_cap, answer, gate=0.0, temp=0.0, top_p=1.0, plains=None, label=""):
    """B sequences in slots 0 … B−1 decoding together with DSpark: thinking (≤ think_cap, then </think> forced) and
    answers (≤ answer). Phases timed apart: 'think' rounds while every sequence still thinks, 'answer' rounds while
    every sequence answers (sequences that reach a phase end early wait). plains: per sequence the plain greedy tokens
    (checked)."""
    B = len(names)
    ids = [encode(k) for k in kws]
    first = []
    for b in range(B):
        prefill(b, ids[b])
        first.append(E.spec_begin(b, N, temp, SEED + b, 0, gate=gate, top_p=top_p, draft=False))
    E.spec_draft(list(range(B)))
    out = [list(ids[b]) + [first[b]] for b in range(B)]
    T = [len(x) for x in ids]
    te = [-1] * B
    done = [False] * B
    res = {}
    ok = True
    for phase in ("think", "answer"):
        E.spec_stats(True)
        E.expert_count(True)
        tok, tsum, rounds = 0, 0.0, 0
        while True:
            act = []
            for b in range(B):
                if done[b]:
                    continue
                if phase == "think" and te[b] >= 0:
                    continue
                act.append(b)
            if not act:
                break
            for b in act:
                P = E.spec_pos(b)
                if phase == "think":
                    E.spec_cap(b, max(1, T[b] + think_cap - P))  # positions ≤ T + think_cap (plain's forced one)
                else:
                    E.spec_cap(b, max(1, te[b] + answer - P))
            torch.cuda.synchronize()
            t = time.time()
            r = E.spec_round(act)
            tsum += time.time() - t
            rounds += 1
            for b, toks in zip(act, r):
                out[b] += toks
                tok += len(toks)
                if phase == "think":
                    if chat.think_end in toks:
                        te[b] = len(out[b]) - len(toks) + toks.index(chat.think_end)
                        # tokens after </think> in the same round belong to the answer (kept)
                    elif len(out[b]) - T[b] >= think_cap + 1:
                        out[b][-1] = chat.think_end
                        E.spec_last(b, chat.think_end)
                        te[b] = len(out[b]) - 1
                if any(x in chat.eos for x in toks) or (te[b] >= 0 and len(out[b]) - 1 - te[b] >= answer):
                    done[b] = True
                if plains and plains[b] is not None:
                    p = plains[b]
                    n = min(len(out[b]), len(p))
                    if out[b][:n] != p[:n]:
                        ok = False
        S = E.spec_stats(True)
        ex, steps = E.expert_take()
        res[phase] = dict(tokens=tok, s=tsum, rounds=rounds, split=S, experts=sum(ex[1:]) / 42 if steps else 0)
        say(f"  {label} N={N}{f' gate {gate}' if gate else ''}{f' T={temp}' if temp else ''} {phase}: {tok} tokens, "
            f"{rounds} rounds · {tok / max(tsum, 1e-9):.1f} tok/s aggregate · distinct experts/layer in verify steps "
            f"{res[phase]['experts']:.1f} · {fmt_split(S)}")
    for b in range(B):
        E.reset(b)
    return res, ok, out


def batch2():
    names = ["py_code", "tool_call"]
    pl = [PLAIN[n][0] if n in PLAIN else None for n in names]
    res, ok, out = multi(names, [PROMPTS[n] for n in names], 4, a.think_cap, a.answer, plains=pl, label="[batch2]")
    # the plain runs forced </think> at think_cap like multi(): their tokens must match where both exist
    say(f"[batch2] tokens vs plain decoding: {'IDENTICAL' if ok else 'DIFFER'}")
    RES["batch2"] = dict(res=res, ok=ok)


def c4():
    groups = [("mixed", NAMES, [PROMPTS[n] for n in NAMES]), ("code4", list(CODE4), list(CODE4.values()))]
    # plain c=4 decode, for reference
    for gname, names, kws in groups:
        ids = [encode(k) for k in kws]
        pos = [len(x) for x in ids]
        n = 24
        # plain batched decode: the 4 sequences stepped together from their first token on
        lgs = [prefill(b, ids[b]) for b in range(4)]
        toks = [int(l.argmax()) for l in lgs]
        torch.cuda.synchronize()
        t = time.time()
        for i in range(n):
            lg = E.step(toks, [p + i for p in pos], [0, 1, 2, 3], [0, 1, 2, 3, 4], False)
            toks = [int(lg[b].argmax()) for b in range(4)]
        torch.cuda.synchronize()
        dt = (time.time() - t) / n
        say(f"[c4] {gname} plain decode (thinking): {dt * 1e3:.1f} ms/step · {4 / dt:.1f} tok/s aggregate")
        for b in range(4):
            E.reset(b)
        for N in (1, 2, 3, 4):
            res, _, _ = multi(names, kws, N, 96, 192, label=f"[c4] {gname}")
            RES.setdefault("c4", {})[f"{gname}/{N}"] = res


def c4gate():
    """c4 with the confidence gate (verify while Π σ(conf) ≥ 0.3), draft blocks 4 and 7."""
    groups = [("mixed", NAMES, [PROMPTS[n] for n in NAMES]), ("code4", list(CODE4), list(CODE4.values()))]
    for gname, names, kws in groups:
        for N, g in ((4, 0.3), (7, 0.3), (7, 0.5)):
            res, _, _ = multi(names, kws, N, 96, 192, gate=g, label=f"[c4gate] {gname}")
            RES.setdefault("c4gate", {})[f"{gname}/{N}/{g}"] = res


# The ratio policy's cost table: ms of a round by verified drafts = the target step by rows (curve, measured on
# rank 0: 1 … 8 rows) + the draft (P081 + P082, N = 7). Fixed numbers: both ranks must decide the same verify count.
CURVE_MS = [36.9, 40.9, 44.7, 48.0, 51.3, 56.2, 59.1, 61.1]
DRAFT_MS = 5.3


def gate():
    cost = [c + DRAFT_MS for c in CURVE_MS]
    E.spec_cost(cost)
    say("[gate] ratio policy cost table (ms by verified drafts 0 … 7): " + " ".join(f"{c:.1f}" for c in cost))
    for g in [float(x) for x in a.gates.split(",")]:
        agg = {}
        for name in NAMES:
            ref = PLAIN.get(name)
            if ref is None:
                continue
            st, _, S, _, _ = spec_single(name, encode(PROMPTS[name]), ref, 7, gate=g, check=False)
            for seg, ty in (("think", "thinking"), ("answer", name)):
                s = st[seg]
                gg = agg.setdefault(ty, [0, 0.0, 0])
                gg[0] += s[0]
                gg[1] += s[1]
                gg[2] += s[2]
            say(f"  [gate {g}] {name}: {fmt_split(S)}")
        say(f"[gate] block 7, " + (f"verify while Π σ(conf) ≥ {g}: " if g > 0 else "verify the count with the best "
            "expected tokens per ms: ") + " · ".join(
            f"{ty} {v[0] / max(v[1], 1e-9):.1f} tok/s ({v[0] / max(1, v[2]):.2f} tok/round)" for ty, v in agg.items()))
        RES.setdefault("gate", {})[str(g)] = agg


def sampled():
    for N, g in ((3, 0.0), (4, 0.0), (7, 0.3)):
        agg = {}
        for name in NAMES:
            ref = PLAIN.get(name)
            if ref is None:
                continue
            st, _, S, _, _ = spec_single(name, encode(PROMPTS[name]), ref, N, gate=g, temp=1.0, top_p=0.95,
                                         check=False)
            for seg, ty in (("think", "thinking"), ("answer", name)):
                s = st[seg]
                gg = agg.setdefault(ty, [0, 0.0, 0])
                gg[0] += s[0]
                gg[1] += s[1]
                gg[2] += s[2]
            say(f"  [T=1 top_p 0.95] {name} N={N}{f' gate {g}' if g else ''}: {fmt_split(S)}")
        say(f"[sampled] T=1.0 top_p 0.95, N={N}{f' gate {g}' if g else ''}: " + " · ".join(
            f"{ty} {v[0] / max(v[1], 1e-9):.1f} tok/s ({v[0] / max(1, v[2]):.2f} tok/round)" for ty, v in agg.items()))
        RES.setdefault("sampled", {})[f"{N}/{g}"] = agg


def check():
    """The losslessness check, once, at the default config: plain greedy decoding vs greedy DSpark per prompt."""
    allp = {**PROMPTS, **CODE4}
    names = a.names.split(",") if a.names else ["ko_prose", "en_prose", "py_code"]
    N, g = a.depth, a.gate_def
    tot = dict(rows=0, bad=0, after_rej=0, after_rej_bad=0, cross_rej=0)
    allok = True
    for name in names:
        ids = encode(allp[name])
        ref = plain(0, ids, a.think_cap, a.answer)
        seq, T, te, forced, lgs, ms = ref
        st, cmp, S, ok, _ = spec_single(name, ids, ref, N, gate=g)
        allok &= ok
        for k in tot:
            tot[k] += cmp[k]
        say(f"[check] {name}: prompt {T} · {len(seq) - T} tokens (thinking {te - T if te > 0 else len(seq) - T}"
            f"{' forced end' if forced >= 0 else ''}) · block {N} gate {g}: tokens {'IDENTICAL' if ok else 'DIFFER'} · "
            f"kept rows' logits bitwise equal {cmp['rows'] - cmp['bad']}/{cmp['rows']} · row 0 after a rejection "
            f"{cmp['after_rej'] - cmp['after_rej_bad']}/{cmp['after_rej']} · rejections across a compressor group end "
            f"{cmp['cross_rej']}")
        say(f"        {fmt_split(S)}")
        RES.setdefault("check", {})[name] = dict(ok=ok, cmp=cmp, split=S)
        PLAIN[name] = (seq, T, te, forced, None, ms)
        del lgs, ref
        torch.cuda.empty_cache()
    say(f"[check] all: tokens {'IDENTICAL' if allok else 'DIFFER'} · logits rows bitwise equal "
        f"{tot['rows'] - tot['bad']}/{tot['rows']} · row 0 after a rejection {tot['after_rej'] - tot['after_rej_bad']}/"
        f"{tot['after_rej']} · rejections across a compressor group end {tot['cross_rej']}")


def draftms():
    """The draft pass (spec_draft wall ms incl. its sync; 8 passes, the middle 4 averaged) at depths --dms-depths, in
    the current default and with --dms-patches on; both states' drafts and confidences compared bit for bit (greedy at
    4 anchors × depths 4 and 7, and one sampled draw: T 1.0, top_p 0.95, depth 7); then the per-stage split of the last
    state (E55_DSP_PROF: CUDA events at the stage boundaries, rank 0's engine log, 2 passes a depth). --dms-sync: one
    depth-7 pass with every model.cc stage synchronized and timed (sync mode 2: the FFN's lines)."""
    name = "py_code"
    seq, T, te, forced, _, _ = plain(0, encode(PROMPTS[name]), 64, 64)  # its decode steps write every context row
    P0 = len(seq) - 1
    prefill(0, seq[:P0])
    anchors = [P0 - 120, P0 - 80, P0 - 40, P0]
    depths = [int(x) for x in a.dms_depths.split(",")]
    pats = [p for p in a.dms_patches.split(",") if p]
    dflt = {p: E.patch_on(p) for p in pats}
    states = [("current", False)] + ([("patch (" + ", ".join(pats) + ")", True)] if pats else [])
    got = []
    for tag, on in states:
        for p in pats:
            E.patch(p, on)
        E.set_history(0, P0)
        res = {}
        for N in depths:
            ms = []
            for _ in range(8):
                torch.cuda.synchronize()
                t = time.time()
                E.draft_at(0, seq[P0], P0, N)
                ms.append((time.time() - t) * 1e3)
            ms = sorted(ms)[2:-2]
            res[N] = sum(ms) / len(ms)
        say(f"[draftms] {tag}: depth → ms per draft pass: " + " · ".join(f"{n}: {v:.3f}" for n, v in res.items()))
        RES.setdefault("draftms", {})[tag] = res
        g = {}
        for P in anchors:
            for N in (4, 7):
                E.set_history(0, P)
                g[(P, N)] = E.draft_at(0, seq[P], P, N)
        E.set_history(0, P0 - 1)  # one sampled draw: the target step at P0 − 1 again (the same bytes), then
        E.step([seq[P0 - 1]], [P0 - 1], [0], [0, 1], False)  # spec_begin samples the token at P0 and drafts from it
        tok = E.spec_begin(0, 7, 1.0, SEED, 0, top_p=0.95)
        got.append((g, (tok, E.spec_drafts(0), E.spec_conf(0))))
    if len(got) == 2:
        (g0, s0), (g1, s1) = got
        same = sum(g0[k] == g1[k] for k in g0)
        say(f"[draftms] drafts + confidences bitwise, current vs patch: greedy {same}/{len(g0)} blocks · sampled "
            f"{'identical' if s0 == s1 else 'DIFFER'} → {'YES' if same == len(g0) and s0 == s1 else 'NO'}")
        RES["draftms"]["bitwise"] = same == len(g0) and s0 == s1
    E.set_history(0, P0)
    os.environ["E55_DSP_PROF"] = "1"
    for N in depths:
        for _ in range(2):
            E.draft_at(0, seq[P0], P0, N)
    os.environ["E55_DSP_PROF"] = "0"
    if a.dms_sync:
        E.sync_mode(2)
        E.prof()
        E.draft_at(0, seq[P0], P0, 7)
        lines = E.prof()
        E.sync_mode(0)
        say("[draftms] depth 7, model.cc stages synchronized: " + " · ".join(
            f"{w}:{ln} {ms:.3f}" for w, ln, ms, n in sorted(lines, key=lambda x: x[1])))
    for p, v in dflt.items():
        E.patch(p, v)
    E.reset(0)


def tp2():
    global E, chat
    import ext
    from chat import Chat
    t0 = time.time()
    E = ext.load("engine", ext.engine_sources(), log=f"/tmp/build{a.rank}.log", ldflags=ext.ENGINE_LDFLAGS)
    E.init(a.model, a.table, a.rank, a.head_ip, 4, 8192, MAXR, 64, a.port)
    f0 = torch.cuda.mem_get_info()[0] / 2**30
    E.dspark_init()
    f1 = torch.cuda.mem_get_info()[0] / 2**30
    print(f"[mem] rank{a.rank} · 4 slots × 8192 · rows {MAXR} · logit rows 64: engine {f0:.1f} GiB free · + draft "
          f"{f1:.1f} GiB free", flush=True)
    say(f"[dspark] engine + draft up · {time.time() - t0:.0f} s")
    chat = Chat(a.model)
    for what in a.do.split(","):
        t = time.time()
        globals()[what]()
        say(f"[dspark] {what} · {time.time() - t:.0f} s")
    if R0:
        os.makedirs(a.out, exist_ok=True)
        json.dump(RES, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=str)
    E.close()


if __name__ == "__main__":
    ref_phase() if a.phase == "ref" else tp2()
