#!/usr/bin/env python3
"""dspark_accept.py — how much DSpark speculative decoding can give, measured before it is built: greedy acceptance of
the checkpoint's DSpark draft (tools/ref/dspark_ref.py, written from vLLM-ERNIE) against the engine's own greedy
continuations, per content type, plus the target's expert spread over verify-sized row blocks.

Phase 1, the engine (TP=2, both ranks; tools/tp2.sh appends --rank / --head-ip; only the scripts are synced):
  E55_GPU_LOCK=1 E55_SYNC_FILES="engine/tests/dspark_accept.py tools/ref/dspark_ref.py" \
    tools/tp2.sh dspark python3 /work/src/engine/tests/dspark_accept.py gen
  per prompt (chat template, thinking on): greedy decode with the engine — thinking up to --think-cap tokens (then
  </ernie:think> is forced), then the answer (--answer + 10 tokens or the end); then ONE step over prompt +
  continuation with taps on the rows the drafter needs: the residual after layers 39–42 (stream mean → the drafter's
  aux features) and every layer's routed expert ids. Rank 1 (node B) saves them to --data; rank 0 prints the text.
Phase 2, the drafter alone on B (no engine):
  tools/agent.sh dspark --gpu python3 engine/tests/dspark_accept.py draft
  for every anchor position of each segment (the thinking, the answer) the draft block of N tokens at once
  (independent of the others, so batched), then the greedy verify walk: from the segment start, accept the drafts'
  longest prefix equal to the engine's tokens, plus the bonus token, move on. Prints per content type: mean accepted
  tokens per verify pass, per-position acceptance, distinct experts per layer for 1 / 4 / 5 / 8 consecutive rows,
  the drafter's own expert spread, and tok/s estimates (labelled) from the measured step profile and the bandwidth
  floor. Results → <data>/results.json."""
import argparse
import hashlib
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

ap = argparse.ArgumentParser()
ap.add_argument("phase", choices=("gen", "draft"))
ap.add_argument("--model", default="/models/ERNIE-5.5-Flash-Preview")
ap.add_argument("--table", default="/work/cache/table.tsv")
ap.add_argument("--data", default="/work/dev/dspark/data")
ap.add_argument("--rank", type=int, default=0)
ap.add_argument("--head-ip", default="")
ap.add_argument("--port", type=int, default=29611)
ap.add_argument("--think-cap", type=int, default=300)
ap.add_argument("--answer", type=int, default=256)
ap.add_argument("--seg", type=int, default=256, help="anchor positions per segment")
ap.add_argument("--names", default="")
ap.add_argument("--ns", default="7,4,3", help="draft block sizes (num_speculative_tokens) to run")
ap.add_argument("--rope-check", action="store_true", help="also N=7 with SGLang's draft rope (YaRN, theta 160k)")
ap.add_argument("--aux-shift", type=int, default=0, help="-1: aux from layers 39–41 (a check of the id semantics)")
ap.add_argument("--limit", type=int, default=0, help="draft phase: only the first N anchors per segment (smoke)")
a = ap.parse_args()
R0 = a.rank == 0
MAXN = 8  # verify tokens needed past an anchor: up to 7 drafts + 1


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
PROMPTS = {
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


# ---------------------------------------------------------------------------------------------------- phase 1
def gen():
    import ext
    from chat import Chat, normalize_tools
    why = ext.stale("engine", ext.engine_sources())
    if why:
        say(f"[dspark] engine module would be rebuilt ({why}) — stopping (other agents are editing sources)")
        sys.exit(3)
    t0 = time.time()
    E = ext.load("engine", ext.engine_sources(), log=f"/tmp/build{a.rank}.log", ldflags=ext.ENGINE_LDFLAGS)
    E.init(a.model, a.table, a.rank, a.head_ip, 1, 4096, 4096, 1, a.port)
    say(f"[dspark] engine up · {time.time() - t0:.0f} s")
    chat = Chat(a.model)
    os.makedirs(a.data, exist_ok=True)
    names = a.names.split(",") if a.names else list(PROMPTS)
    for name in names:
        kw = PROMPTS[name]
        ids = chat.encode_chat(chat.render(kw["messages"], normalize_tools(kw.get("tools")), thinking=True,
                                           level="high"))
        T = len(ids)
        E.reset(0)
        seq = list(ids)
        t = time.time()
        lg = E.step(ids, list(range(T)), [0], [0, T], False)
        te, forced = -1, False
        while True:
            nxt = int(lg[0].argmax())
            if te < 0 and len(seq) - T >= a.think_cap and nxt != chat.think_end:
                nxt, forced = chat.think_end, True
            seq.append(nxt)
            if nxt == chat.think_end and te < 0:
                te = len(seq) - 1
            if nxt in chat.eos or (te >= 0 and len(seq) - 1 - te >= a.answer + MAXN + 2) or len(seq) >= 4000:
                break
            lg = E.step([nxt], [len(seq) - 1], [0], [0, 1], False)
        dt = time.time() - t
        M = len(seq)
        # segments: anchors [s0, s1); every token the walk verifies (up to anchor + MAXN) lies below L
        segs = []
        if te > T:
            L = te if forced else te + 1  # a forced </ernie:think> is not the model's
            segs.append(("think", T, min(T + a.seg, L - MAXN), L))
        if te >= 0:
            segs.append(("answer", te, min(te + a.seg, M - MAXN), M))
        segs = [s for s in segs if s[2] > s[1]]
        rows = sorted({p for _, s0, s1, _L in segs for p in range(max(0, s0 - 128), min(M, s1 + MAXN))})
        # one step over the whole sequence, taps on those rows
        E.reset(0)
        E.set_taps(rows)
        n1 = rows[-1] + 1
        E.fill(seq[:n1], list(range(n1)), [0], [0, n1])
        n = len(rows)
        res = E.taps(n, [39, 40, 41, 42])
        aux = res[0].mean(2)  # [4, n, H] bf16: vLLM's aux_recon.mean(dim=1) on the bf16 residual
        del res
        route = []
        for L0 in range(0, 43, 11):
            r = E.taps(n, list(range(L0, min(43, L0 + 11))))
            route.append(r[3].to(torch.int16).cpu())
            del r
        route = torch.cat(route)  # [43, n, 10]
        E.set_taps([])
        torch.cuda.empty_cache()
        h = hashlib.sha1(json.dumps(seq).encode()).hexdigest()[:12]
        print(f"[dspark] rank{a.rank} {name}: prompt {T} · thinking {(te if te >= 0 else M) - T}"
              f"{' (forced end)' if forced else ''} · answer {M - te - 1 if te >= 0 else 0} · seq sha1 {h}", flush=True)
        say(f"  decode {M - T} tokens in {dt:.1f} s · segments " +
            " · ".join(f"{k} anchors [{s0},{s1})" for k, s0, s1, _ in segs) + f" · {n} tapped rows")
        if R0:
            think = chat.decode(seq[T:te if te >= 0 else M], False)
            ans = chat.decode(seq[te + 1:], False) if te >= 0 else ""
            say("  thinking: " + think[:300].replace("\n", " ⏎ ") + (" …" if len(think) > 300 else ""))
            say("  answer:   " + ans[:600].replace("\n", " ⏎ ") + (" …" if len(ans) > 600 else ""))
        if a.rank == 1:  # node B, where phase 2 runs
            torch.save({"name": name, "seq": seq, "T": T, "te": te, "forced": forced, "segs": segs, "rows": rows,
                        "aux": aux.cpu(), "aux_layers": [39, 40, 41, 42], "route": route},
                       os.path.join(a.data, f"{name}.pt"))
        del aux
    E.close()
    say(f"[dspark] gen done · {time.time() - t0:.0f} s")


# ---------------------------------------------------------------------------------------------------- phase 2
def distinct(ids):
    """ids [..., k] int → number of distinct values along the last dim."""
    s = ids.sort(-1).values
    return (s[..., 1:] != s[..., :-1]).sum(-1) + 1


def walk(dr, seq, s0, s1, k):
    """Greedy verify walk with the first k drafts of each block: (passes, accepted tokens incl. the bonus)."""
    P, passes, acc = s0, 0, 0
    while P < s1:
        d = dr[P]
        m = 0
        while m < k and d[m] == seq[P + 1 + m]:
            m += 1
        passes += 1
        acc += m + 1
        P += m + 1
    return passes, acc


def ckpt_bytes(ck):
    """tensor name → bytes, from the shard headers."""
    out = {}
    for f in sorted(set(ck.map.values())):
        _, h, _ = ck._h(f)
        for k, m in h.items():
            if k != "__metadata__":
                out[k] = m["data_offsets"][1] - m["data_offsets"][0]
    return out


def draft():
    import dspark_ref as dsr
    import ernie_ref as ref
    t0 = time.time()
    ck = ref.Ckpt(a.model)
    D = dsr.Drafter(ck)
    print(f"[dspark] drafter loaded · {time.time() - t0:.0f} s", flush=True)
    names = a.names.split(",") if a.names else list(PROMPTS)
    ns = [int(x) for x in a.ns.split(",")]
    cfgs = [(n, "vllm") for n in ns] + ([(7, "sglang")] if a.rope_check else [])
    R = {}  # (content type, cfg) → stats
    wins = (1, 4, 5, 6, 8)

    def acc_add(key, **kv):
        d = R.setdefault(key, {})
        for k, v in kv.items():
            d[k] = d.get(k, 0) + v

    texp = {}  # content type → {w: [sum, count]}
    for name in names:
        f = os.path.join(a.data, f"{name}.pt")
        if not os.path.exists(f):
            print(f"[dspark] {name}: no data", flush=True)
            continue
        S = torch.load(f, weights_only=False)
        seq, rows, segs = S["seq"], S["rows"], S["segs"]
        n = len(rows)
        pos2row = torch.full((len(seq) + 1,), -1, dtype=torch.long)
        pos2row[torch.tensor(rows)] = torch.arange(n)
        sel = [S["aux_layers"].index(L + a.aux_shift) for L in dsr.TARGET_LAYERS]
        aux = S["aux"][sel].to(ref.DEV)                         # [3, n, H]
        aux = aux.permute(1, 0, 2).reshape(n, 3 * dsr.H)        # cat(aux40, aux41, aux42) per row
        with torch.no_grad():
            main_x = D.features(aux)
        rpos = torch.tensor(rows, device=ref.DEV)
        route = S["route"].long()                               # [43, n, 10]
        segs = [(k, s0, s1 if not a.limit else min(s1, s0 + a.limit), L) for k, s0, s1, L in segs]
        # target expert spread over w consecutive rows starting at each anchor (layers 1–42)
        for kind, s0, s1, L in segs:
            ct = "thinking" if kind == "think" else name
            for w in wins:
                idx = pos2row[torch.arange(s0, s1)[:, None] + torch.arange(w)[None, :]]  # [A, w]
                g = route[1:, idx.clamp(min=0)].reshape(42, s1 - s0, w * 10)
                d = distinct(g).float()
                e = texp.setdefault(ct, {}).setdefault(w, [0.0, 0])
                e[0] += d.sum().item()
                e[1] += d.numel()
        anchors = [p for _, s0, s1, _ in segs for p in range(s0, s1)]
        ap_ = torch.tensor(anchors, device=ref.DEV)
        atok = torch.tensor([seq[p] for p in anchors], device=ref.DEV)
        cidx = pos2row.to(ref.DEV)[(ap_[:, None] - 128 + torch.arange(128, device=ref.DEV)[None, :]).clamp(min=0)]
        cok = (ap_[:, None] - 128 + torch.arange(128, device=ref.DEV)[None, :] >= 0) & (cidx >= 0)
        cidx = cidx.clamp(min=0)
        rope_now = None
        for N, rope in cfgs:
            t = time.time()
            if rope != rope_now:
                D.set_rope(rope)
                with torch.no_grad():
                    ckv = D.context_kv(main_x, rpos)
                rope_now = rope
            ctx = [c[cidx] for c in ckv]                        # per stage [B, 128, 512]
            dr, routes = D.draft(atok, ap_, ctx, cok, N)
            dr = dr.cpu().tolist()
            # the drafter's own spread: distinct experts per stage over its N block rows
            dd = distinct(routes.reshape(dsr.STAGES, len(anchors), N * 10)).float().mean().item()
            del ctx, routes
            drp = dict(zip(anchors, dr))
            for kind, s0, s1, L in segs:
                ct = "thinking" if kind == "think" else name
                ks = sorted({N} | ({3, 4, 5} if N == 7 and rope == "vllm" else set()))
                for k in ks:
                    p_, c_ = walk(drp, seq, s0, s1, k)
                    acc_add((ct, N, k, rope), passes=p_, acc=c_)
                # per position: P(draft i right | drafts before right), over every anchor
                pos_ok = [0] * N
                pos_n = [0] * N
                for p in range(s0, s1):
                    for i in range(N):
                        pos_n[i] += 1
                        if drp[p][i] != seq[p + 1 + i]:
                            break
                        pos_ok[i] += 1
                acc_add((ct, N, N, rope), **{f"ok{i}": pos_ok[i] for i in range(N)},
                        **{f"n{i}": pos_n[i] for i in range(N)}, ddraft=dd * (s1 - s0), anchors=s1 - s0)
            print(f"[dspark] {name}: N={N} rope={rope} · {len(anchors)} blocks drafted · {time.time() - t:.0f} s",
                  flush=True)
        del main_x, aux, ckv
        torch.cuda.empty_cache()
    report(R, texp, ck)


def report(R, texp, ck):
    import dspark_ref as dsr
    by = ckpt_bytes(ck)
    is_exp = lambda k: ".experts." in k  # noqa: E731
    stage_ne = sum(v for k, v in by.items() if k.startswith("model.mtp.0.") and not is_exp(k))
    exp_one = sum(v for k, v in by.items() if k.startswith("model.mtp.0.block_sparse_moe.experts.0."))
    fproj = by["model.mtp.feature_projection.weight"] + by["model.mtp.feature_projection.weight_scale_inv"]
    head, mk2 = by["lm_head.weight"], by["model.mtp.markov_head.markov_w2.weight"]
    tgt_ne = sum(v for k, v in by.items() if k.startswith("model.layers.") and not is_exp(k)) + head
    frac = 4.56e9 / tgt_ne  # the target's measured non-expert bytes per rank per pass over its full non-expert bytes
    # bandwidths (per rank): floor 241 GB/s; today and at the 27 ms target, the target's own bytes per step over its
    # step time (non-expert 4.56 GB + 10 experts × 42 layers × 3.25 MB)
    step_bytes = 4.56e9 + 10 * 42 * 3.25e6
    bw = {"today": step_bytes / 36.0e-3, "t27": step_bytes / 27.0e-3, "floor": 241e9}

    def t_verify(rows_d, mode):  # rows_d: distinct experts per layer of the verify block
        if mode == "today":
            return (36.0 - 6.9) + 6.9 * rows_d / 10
        if mode == "t27":
            return 27.0 * (36.0 - 6.9) / 36.0 + 27.0 * 6.9 / 36.0 * rows_d / 10
        return 18.9 + 0.57 * rows_d

    def drafter_bytes(N, dd):
        """per rank per pass: the 3 stages (TP-split like the target) with dd distinct experts each (half of each
        expert per rank), feature_projection + the context KV projection (replicated in vLLM), the lm_head
        (vocab-split), N sequential Markov w2 reads (replicated, disable_tp)"""
        return (dsr.STAGES * (stage_ne * frac + dd * exp_one / 2) + fproj + head / 2 + N * mk2)

    out = {"drafter": {"stage_nonexpert_MB": stage_ne / 1e6, "expert_MB": exp_one / 1e6, "feature_proj_MB": fproj / 1e6,
                       "lm_head_MB": head / 1e6, "markov_w2_MB": mk2 / 1e6, "tp_fraction": frac}, "rows": []}
    print(f"\n[dspark] drafter weights: per stage non-expert {stage_ne / 1e6:.0f} MB (×{frac:.2f} per rank, the "
          f"target's split) · expert {exp_one / 1e6:.2f} MB · feature_projection {fproj / 1e6:.0f} MB · lm_head "
          f"{head / 1e6:.0f} MB (½ per rank) · Markov w2 {mk2 / 1e6:.0f} MB per sequential step", flush=True)
    print("[dspark] target experts: distinct per layer (mean over layers 1–42) for w consecutive rows", flush=True)
    for ct, d in texp.items():
        print(f"  {ct:10s} " + " · ".join(f"w{w} {s / c:.1f}" for w, (s, c) in sorted(d.items())), flush=True)
    cts = ["ko_prose", "en_prose", "py_code", "tool_call", "thinking"]
    print("\n[dspark] measured: mean accepted tokens per verify pass (incl. bonus) · ESTIMATED: pass time (verify + "
          "draft) and tok/s at today's kernels (36 ms step) / the 27 ms target / the bandwidth floor; need = accepted "
          "tokens a pass needs for 100 tok/s", flush=True)
    for ct in cts:
        if ct not in texp:
            continue
        print(f"  {ct}", flush=True)
        for key in sorted(k for k in R if k[0] == ct):
            _, N, k, rope = key
            s = R[key]
            if "passes" not in s:
                continue
            acc = s["acc"] / s["passes"]
            stat = R.get((ct, N, N, rope), {})
            dd = stat.get("ddraft", 0) / max(1, stat.get("anchors", 1))
            w = k + 1
            dexp = texp[ct].get(w) or texp[ct][min(texp[ct], key=lambda x: abs(x - w))]
            dv = dexp[0] / dexp[1]
            db = drafter_bytes(N, dd)
            line = {"type": ct, "N": N, "k": k, "rope": rope, "accepted": acc, "passes": s["passes"],
                    "verify_rows": w, "target_distinct": dv, "draft_distinct": dd, "drafter_GB_rank": db / 1e9}
            parts = []
            for mode in ("today", "t27", "floor"):
                tp = t_verify(dv, mode) + 1e3 * db / bw[mode]
                line[f"pass_ms_{mode}"], line[f"tok_s_{mode}"], line[f"need_{mode}"] = tp, 1e3 * acc / tp, tp / 10
                parts.append(f"{mode} {tp:.1f} ms → {1e3 * acc / tp:.0f} tok/s (need {tp / 10:.1f})")
            out["rows"].append(line)
            pos = ""
            if k == N and "ok0" in stat:
                pos = " · per position " + " ".join(f"{stat[f'ok{i}'] / max(1, stat[f'n{i}']):.2f}" for i in range(N))
            print(f"    N={N}{'' if k == N else f'→verify {k}'}{'' if rope == 'vllm' else ' (sglang rope)'}: "
                  f"accepted {acc:.2f} ({s['passes']} passes) · rows {w}: target distinct {dv:.1f}, drafter distinct "
                  f"{dd:.1f}, drafter {db / 1e9:.2f} GB/rank · " + " | ".join(parts) + pos, flush=True)
    json.dump(out, open(os.path.join(a.data, "results.json"), "w"), indent=1)
    print(f"[dspark] → {os.path.join(a.data, 'results.json')}", flush=True)


if __name__ == "__main__":
    if a.phase == "gen":
        gen()
    else:
        draft()
