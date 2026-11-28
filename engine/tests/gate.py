#!/usr/bin/env python3
"""gate.py — the correctness gate: the engine (TP=2, both ranks) against the reference forward (tools/ref/ernie_ref.py
dumps), layer by layer. Run through tools/tp2.sh (it appends --rank and --head-ip):
  tools/tp2.sh gate python3 /work/src/engine/tests/gate.py --ref /work/results/ref/v3 [--names a,b] [--pf both]
Step kinds as the server's (core/step_kind.h): every prompt step is a prefill step (prefill = True: where the
prefill-rule kernels of engine/tests/pf_rule.py run), every decode step a decode step. Per prompt:
  A. the prompt as prefill: one step (≤ 4096 tokens) or chunks of --chunk (8192, the server's): residual after every
     layer, attention / FFN outputs and MoE expert sets at the dump's positions (ernie_ref.sel_positions); logits there
     (top-1 agreement, KL, max |Δ|); next-token NLL, engine and reference side by side (a bug makes the engine
     systematically worse; rounding noise does not), top-1 agreement where the reference is confident (p ≥ 0.5)
  B. the same prompt through the caches: prefill in chunks of 1000, the rest less 32 as one chunk, the last 32
     positions one decode step each — logits of the last 32 against the reference and against A (bitwise while every
     kernel keeps the decode bits, i.e. with the prefill-rule patches off)
--pf: the prefill-rule patches (pf_rule.py), switched in this one process (E.patch, both ranks alike):
  both (default)  A and B with them off, then on. "off" is the noise floor — the engine's bitwise path against the
                  reference, two lossless implementations; "on" passes when, at A's positions,
                    KL(ref ‖ on)  ≤ 1.5 · KL(ref ‖ off) + 0.005
                    KL(off ‖ on)  ≤ 2 · KL(ref ‖ off) + 0.005     (no further apart than each is from the reference)
                    top-1(on)     ≥ top-1(off) − max(0.02, 2 / positions)
                    NLL(on)       ≤ NLL(off) + max(0.05, 0.01 · NLL(ref))
                  over all positions and, for a long prompt, over those ≥ 114,688 too (rows where P210 replaces the
                  base's long-context P150 / P172). Thresholds from HY4's P510 + P520 gate (11-27), which passed at
                  KL off → on 0.029 → 0.023 · 0.447 → 0.410 · 0.340 → 0.450 (1.32×) and NLL +0.05 at most.
                  Summary → <ref>/gate-pf<tag>.txt
  off | on        that state only · as-is: E55_PATCHES / the defaults, one pass
Engine dumps for tools/ref/noise_floor.py: <ref>/<name>.engine-<off|on|asis><tag>.safetensors (ids, pos_sel,
logits_sel, resid0 / resid21 / resid42, route) unless --no-save.
The long prompt ("long": tools/ref/prompts.py --long --add, dumped by ernie_ref.py --qchunk 64 --rows 16384
--key-slice): add it to --names; ctx, max rows and logit rows follow the prompts.
--gen N: after the prompts, up to N greedy tokens for each prompt, printed as text.
--isolate L,L…: each listed layer on its own (prompt --iso-name, every row tapped): the reference's layer
(tools/ref/ernie_ref.py) run on the engine's residual before it, against the engine's own outputs — attention, FFN,
expert sets and residual errors that do not carry earlier layers' differences.
--gen / --isolate run after the prompts, with the prefill-rule patches back at their starting state.
The rank that finds the dumps compares and writes <ref>/gate-<name><tag>.txt (the other needs <ref>/prompts.json only);
rank 0 prints too."""
import argparse
import json
import os
import sys
import time

import torch
from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools", "ref"))
import ernie_ref as ref  # noqa: E402
import ext  # noqa: E402
import pf_rule  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True)
ap.add_argument("--names", default="ko_short,en_tools,doc_long")
ap.add_argument("--model", default="/models/ERNIE-5.5-Flash-Preview")
ap.add_argument("--table", default="/work/cache/table.tsv")
ap.add_argument("--rank", type=int, required=True)
ap.add_argument("--head-ip", required=True)
ap.add_argument("--port", type=int, default=29611)
ap.add_argument("--pf", choices=["both", "off", "on", "as-is"], default="both")
ap.add_argument("--chunk", type=int, default=8192, help="A: prefill rows a step for prompts over 4096 tokens")
ap.add_argument("--tag", default="", help="suffix of the summary and dump files")
ap.add_argument("--no-save", action="store_true", help="no engine dumps")
ap.add_argument("--gen", type=int, default=0)
ap.add_argument("--isolate", default="")
ap.add_argument("--iso-name", default="en_tools")
a = ap.parse_args()
R0 = a.rank == 0
PF = pf_rule.names()
ONE = 4096  # A reads a prompt up to this long in one step
# the noise-floor judgement (--pf both; module docstring)
KL_X, KL_PAIR, KL_ABS = 1.5, 2.0, 0.005
TOP1_DROP = 0.02
NLL_UP, NLL_REL = 0.05, 0.01

t0 = time.time()
prompts = json.load(open(os.path.join(a.ref, "prompts.json")))
names = [n for n in a.names.split(",") if n in prompts]
have = {n: os.path.exists(os.path.join(a.ref, f"{n}.safetensors")) for n in names}
CMP = any(have.values())  # this rank compares


def say(*x):
    if CMP or R0:
        print(*x, flush=True)


for n in a.names.split(","):
    if n not in prompts:
        say(f"[gate] {n}: not in prompts.json")


def steps_of(T):
    """A's prefill steps [c0, c1) for a T-token prompt"""
    n = T if T <= ONE else a.chunk
    return [(c0, min(T, c0 + n)) for c0 in range(0, T, n)]


def rows_in(sel, c0, c1):
    return [p - c0 for p in sel if c0 <= p < c1]


Tmax = max([len(prompts[n]) for n in names] + ([len(prompts[a.iso_name])] if a.isolate else []) + [1])
ctx = 8192 if Tmax + a.gen + 64 <= 8192 else -(-(Tmax + a.gen + 64) // 1024) * 1024
max_rows = ONE if Tmax <= ONE else max(ONE, a.chunk)
LR = max([512] + [len(rows_in(ref.sel_positions(len(prompts[n])), c0, c1)) for n in names
                  for c0, c1 in steps_of(len(prompts[n]))])
LR = -(-LR // 64) * 64
E = ext.load("engine", ext.engine_sources(), log=f"/tmp/build{a.rank}.log", ldflags=ext.ENGINE_LDFLAGS)
say(f"[gate] rank {a.rank} · engine built · {time.time() - t0:.0f} s · compares: {CMP}")
E.init(a.model, a.table, a.rank, a.head_ip, 2, ctx, max_rows, LR, a.port)
start = {n: E.patch_on(n) for n in PF}
PASSES = {"both": [("off", False), ("on", True)], "off": [("off", False)], "on": [("on", True)],
          "as-is": [("asis", None)]}[a.pf]
say(f"[gate] ctx {ctx} · max rows {max_rows} · logit rows {LR} · prefill-rule patches {start} · passes "
    f"{[p for p, _ in PASSES]}")
V = 201216


def set_pf(on):
    if on is not None:
        for n in PF:
            E.patch(n, on)


def rel(x, y):
    x, y = x.float(), y.float()
    return ((x - y).norm() / y.norm().clamp_min(1e-30)).item()


def logit_stats(lg, ref_lg):
    """top-1 agreement, KL(ref ‖ lg) mean / max, max |Δ|"""
    lp, lq = torch.log_softmax(ref_lg.float(), -1), torch.log_softmax(lg.float(), -1)
    kl = (lp.exp() * (lp - lq)).sum(-1)
    top = (lg.argmax(-1) == ref_lg.argmax(-1)).float()
    return top.mean().item(), kl.mean().item(), kl.max().item(), (lg - ref_lg).abs().max().item()


def nll_stats(lg, ref_lg, nxt):
    """mean NLL of the next tokens (engine, reference) and top-1 agreement where the reference's top p ≥ 0.5"""
    lp, lq = torch.log_softmax(ref_lg.float(), -1), torch.log_softmax(lg.float(), -1)
    t = torch.tensor(nxt, device=lg.device)[:, None]
    conf = lp.max(-1).values.exp() >= 0.5
    agree = (lg.argmax(-1) == ref_lg.argmax(-1))[conf].float()
    return (-lq.gather(1, t).mean().item(), -lp.gather(1, t).mean().item(), agree.mean().item() if conf.any() else
            float("nan"), int(conf.sum()))


def read_A(ids, sel):
    """the prompt as prefill steps, taps and logits at sel → (logits [P, V], [resid, attn, ffn, route], s)"""
    T = len(ids)
    E.reset(0)
    lgs, tp = [], []
    t = time.time()
    for c0, c1 in steps_of(T):
        rows = rows_in(sel, c0, c1)
        E.set_taps(rows)
        lg = E.step(ids[c0:c1], list(range(c0, c1)), [0], [0, c1 - c0], False, rows, prefill=True)
        if rows:
            lgs.append(lg)
            tp.append(E.taps(len(rows))[:4])
    torch.cuda.synchronize()
    dt = time.time() - t
    E.set_taps([])
    taps = [tp[0][i] if len(tp) == 1 else torch.cat([x[i] for x in tp], 1) for i in range(4)]
    return (lgs[0] if len(lgs) == 1 else torch.cat(lgs)), taps, dt


def read_B(ids):
    """prefill in chunks of 1000 (the rest less 32 as one), then the last 32 positions as decode steps"""
    T = len(ids)
    E.reset(0)
    tail = min(32, T - 1)
    cut = T - tail
    pos = 0
    t = time.time()
    while pos < cut:
        n = min(1000, cut - pos)
        E.fill(ids[pos:pos + n], list(range(pos, pos + n)), [0], [0, n])  # (prefill kind)
        pos += n
    outs = [E.step([ids[p]], [p], [0], [0, 1], False)[0] for p in range(cut, T)]  # (decode kind)
    torch.cuda.synchronize()
    return torch.stack(outs), cut, time.time() - t


def one_pass(name, ids, sel, D, lab, on):
    T = len(ids)
    out = {"lines": []}
    L = out["lines"]
    lg, (r, at, ff, rt), dt = read_A(ids, sel)
    out["lg"] = lg
    L.append(f"[{lab}] A: {T} tokens as {len(steps_of(T))} prefill step(s) · {dt:.2f} s")
    say(L[-1])
    if D is not None:
        lay = []
        for l in range(43):
            same = 1.0
            if l > 0:
                ee, er = rt[l].long().sort(-1).values, D["route"][l][sel].long().sort(-1).values
                same = (ee == er).all(-1).float().mean().item()
            lay.append((rel(r[l], D["resid"][l]), rel(at[l], D["attn"][l]), rel(ff[l], D["ffn"][l]), same))
        out["layers"] = lay
        top1, kl, klmax, dmax = logit_stats(lg, D["logits_sel"])
        L.append(f"[{lab}] A logits at {len(sel)} positions: top-1 agree {top1:.3f} · KL mean {kl:.2e} max "
                 f"{klmax:.2e} · max |Δ| {dmax:.3f}")
        k = [i for i, p in enumerate(sel) if p + 1 < T]
        if k:
            ne, nr, ca, nc = nll_stats(lg[k], D["logits_sel"][k], [ids[sel[i] + 1] for i in k])
            L.append(f"[{lab}] A prompt next-token NLL at {len(k)} positions: engine {ne:.3f} · reference {nr:.3f} · "
                     f"top-1 agree where the reference is sure {ca:.3f} ({nc} positions)")
        if not a.no_save:
            f = os.path.join(a.ref, f"{name}.engine-{lab}{a.tag}.safetensors")
            save_file({"ids": torch.tensor(ids, dtype=torch.int32), "pos_sel": torch.tensor(sel, dtype=torch.int32),
                       "logits_sel": lg.float().cpu().contiguous(), "resid0": r[0].cpu().contiguous(),
                       "resid21": r[21].cpu().contiguous(), "resid42": r[42].cpu().contiguous(),
                       "route": rt.to(torch.int16).cpu().contiguous()}, f)
            L.append(f"[{lab}] engine dump → {f}")
    del r, at, ff, rt
    lgB, cut, dtb = read_B(ids)
    idx = [sel.index(p) for p in range(cut, T)]
    dA = (lgB - lg[idx]).abs().max().item()
    bits = torch.equal(lgB.view(torch.int32), lg[idx].view(torch.int32))
    note = ("expected: every kernel the decode bits" if on is False else
            "not expected: A's rows took the prefill-rule kernels, B's decode rows the decode ones" if on else "")
    klA = logit_stats(lgB, lg[idx])[1]
    line = (f"[{lab}] B cached path (chunks of 1000, then {T - cut} decode steps, {dtb:.1f} s): against A max |Δ| "
            f"{dA:.3g} · KL {klA:.2e} · bitwise {bits} ({note})")
    if D is not None:
        top1, kl, klmax, dmax = logit_stats(lgB, D["logits_sel"][idx])
        line += f" · against the reference top-1 {top1:.3f} · KL mean {kl:.2e} max {klmax:.2e} · max |Δ| {dmax:.3f}"
    L.append(line)
    say(L[-1])
    return out


def judge(what, sub, off, on, R, ids, sel):
    """on against the floor of off at the positions sel[i], i in sub → (ok, line)"""
    T = len(ids)
    t_off, kl_off = logit_stats(off[sub], R[sub])[:2]
    t_on, kl_on = logit_stats(on[sub], R[sub])[:2]
    kl_pair = logit_stats(on[sub], off[sub])[1]  # KL(off ‖ on)
    chk = [("KL(ref‖on)", kl_on, KL_X * kl_off + KL_ABS), ("KL(off‖on)", kl_pair, KL_PAIR * kl_off + KL_ABS),
           ("top-1 drop", t_off - t_on, max(TOP1_DROP, 2 / len(sub)))]
    k = [i for i in sub if sel[i] + 1 < T]
    nl = ""
    if k:
        nxt = [ids[sel[i] + 1] for i in k]
        n_off, n_ref = nll_stats(off[k], R[k], nxt)[:2]
        n_on = nll_stats(on[k], R[k], nxt)[0]
        chk.append(("NLL rise", n_on - n_off, max(NLL_UP, NLL_REL * n_ref)))
        nl = f" · NLL off {n_off:.3f} on {n_on:.3f} ref {n_ref:.3f}"
    ok = all(v <= lim for _, v, lim in chk)
    return ok, (f"{what}: {'PASS' if ok else 'FAIL'} at {len(sub)} positions · floor KL(ref‖off) {kl_off:.3e} "
                f"top-1 {t_off:.3f} · on top-1 {t_on:.3f}{nl} · " +
                " · ".join(f"{nm} {v:.3e} ≤ {lim:.3e}" + ("" if v <= lim else " ✗") for nm, v, lim in chk))


summary, all_ok = [], True
for name in names:
    ids = prompts[name]
    T = len(ids)
    sel = ref.sel_positions(T)
    D = load_file(os.path.join(a.ref, f"{name}.safetensors"), device="cuda") if have[name] else None
    if D is not None:
        assert D["ids"].tolist() == ids, f"{name}: the dump's ids are not prompts.json's"
        assert D["pos_sel"].tolist() == sel, f"{name}: the dump's positions are not ernie_ref.sel_positions"
    say(f"[gate] {name}: {T} tokens, {len(sel)} positions")
    got = {}
    for lab, on in PASSES:
        set_pf(on)
        got[lab] = one_pass(name, ids, sel, D, lab, on)
    for n in PF:
        E.patch(n, start[n])
    if D is None:
        del D
        torch.cuda.empty_cache()
        continue
    lines = [f"{name}: {T} tokens · passes {', '.join(got)}"]
    for g in got.values():
        lines += g["lines"]
    labs = list(got)
    lines.append("layer  " + "  ".join(f"resid {x:<4} attn {x:<4}  ffn {x:<4}  experts {x:<4}" for x in labs))
    for l in range(43):
        lines.append(f"{l:5d}  " + "  ".join("{:.2e}   {:.2e}   {:.2e}   {:.3f}       ".format(*got[x]["layers"][l])
                                             for x in labs))
    if "off" in got and "on" in got:
        Rl = D["logits_sel"]
        groups = [("all", list(range(len(sel))))]
        far = [i for i, p in enumerate(sel) if p >= ref.LONG_FROM]
        if len(far) >= 64:
            groups.append((f"≥ {ref.LONG_FROM}", far))
        for what, sub in groups:
            ok, line = judge(f"{name} {what}", sub, got["off"]["lg"], got["on"]["lg"], Rl, ids, sel)
            all_ok &= ok
            lines.append("noise floor " + line)
            summary.append(line)
    txt = "\n".join(lines)
    say(txt)
    open(os.path.join(a.ref, f"gate-{name}{a.tag}.txt"), "w").write(txt + "\n")
    del D, got
    torch.cuda.empty_cache()
if summary and CMP:
    summary.append(f"prefill-rule patches {PF}: {'PASS' if all_ok else 'FAIL'} (KL(ref‖on) ≤ {KL_X}·floor + {KL_ABS}, "
                   f"KL(off‖on) ≤ {KL_PAIR}·floor + {KL_ABS}, top-1 drop ≤ max({TOP1_DROP}, 2/n), NLL rise ≤ "
                   f"max({NLL_UP}, {NLL_REL}·NLL(ref)))")
    txt = "\n".join(summary)
    say("[gate] " + txt.replace("\n", "\n[gate] "))
    open(os.path.join(a.ref, f"gate-pf{a.tag}.txt"), "w").write(txt + "\n")
if a.isolate:
    name = a.iso_name
    ids = prompts[name]
    T = len(ids)
    E.reset(0)
    E.set_taps(list(range(T)))
    E.step(ids, list(range(T)), [0], [0, T], False, prefill=True)
    iso = [int(x) for x in a.isolate.split(",")]
    need = sorted({x for L in iso for x in (L - 1, L) if x >= 0})
    r, at, ff, rt, rm, po, cb = E.taps(T, need)  # only the layers the checks read
    E.set_taps([])
    ix = {L: i for i, L in enumerate(need)}
    r, at, ff, rt, rm, po, cb = ({L: x[ix[L]] for L in need} for x in (r, at, ff, rt, rm, po, cb))
    if R0:
        ck = ref.Ckpt(a.model)
        pos = torch.arange(T, device="cuda")
        cs_hca = ref.cos_sin(ref.inv_freq_yarn(), pos)
        cs8 = ref.cos_sin_dsa(pos)
        say(f"[gate] isolated layers on {name} ({T} tokens): the reference layer on the engine's input")
        say("layer  attn       ffn        experts  resid")
        with torch.no_grad():
            for L in iso:
                w = ref.W(ck, f"model.layers.{L}.")
                if L == 0:
                    e = ck.get("model.embed_tokens.weight")
                    Rin = e[torch.tensor(ids, device="cuda")][:, None, :].expand(T, 4, 4096).contiguous()
                    del e
                else:
                    Rin = r[L - 1]
                # the pre reads the previous post's unrounded fp32 residual: rebuilt from the engine's mid-layer taps
                # (layer L-1's residual after attention, its FFN output and its FFN pre's post / comb); its bf16
                # rounding must give the tapped residual back
                xin = None
                if L > 0:
                    Rchk, xin = ref.mhc_post(ff[L - 1], rm[L - 1], po[L - 1], cb[L - 1].view(T, 4, 4))
                    recon = (Rchk != Rin).float().mean().item()
                post, comb, inp = ref.mhc_pre(Rin, w, "attn", w("input_layernorm.weight"), xin)
                if L in ref.DSA_LAYERS:
                    ar = ref.dsa(inp, w, pos, cs8, cs8)[0]
                else:
                    ar = ref.hca(inp, w, pos, cs_hca)[0]
                Rm, xm = ref.mhc_post(at[L], Rin, post, comb)
                post, comb, inp = ref.mhc_pre(Rm, w, "ffn", w("post_attention_layernorm.weight"), xm)
                if L == 0:
                    fr, same = ref.dense_ffn(inp, w, "mlp."), 1.0
                else:
                    fr, route = ref.moe(inp, w, ck, L)
                    same = (rt[L].long().sort(-1).values == route.long().sort(-1).values).all(-1).float().mean().item()
                Ro, _ = ref.mhc_post(ff[L], Rm, post, comb)
                bk = " · ".join(f"[{b0},{min(b1, T)}) {rel(at[L][b0:b1], ar[b0:b1]):.1e}"
                                for b0, b1 in ((0, 128), (128, 256), (256, 1024), (1024, 2176), (2176, 1 << 30)) if b0 < T)
                say(f"{L:5d}  {rel(at[L], ar):.2e}  {rel(ff[L], fr):.2e}  {same:.3f}    {rel(r[L], Ro):.2e}  attn by "
                    f"position {bk}" + (f" · rebuilt input ≠ tapped {recon:.1e}" if L > 0 else ""))
                del w
                torch.cuda.empty_cache()
    del r, at, ff, rt
if a.gen:
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(os.path.join(a.model, "tokenizer.json"))
    for name in names:
        ids = list(prompts[name])
        E.reset(0)
        lg = E.step(ids, list(range(len(ids))), [0], [0, len(ids)], False, prefill=True)
        out = []
        t = time.time()
        for i in range(a.gen):
            nxt = int(lg[0].argmax())
            out.append(nxt)
            if nxt in (2, 8):
                break
            lg = E.step([nxt], [len(ids) + i], [0], [0, 1], False)
        dt = time.time() - t
        say(f"[gate] greedy {name}: {len(out)} tokens · {len(out) / dt:.1f} tok/s (plain decode)")
        say(tk.decode(out, skip_special_tokens=False))
E.close()
say(f"[gate] done · {time.time() - t0:.0f} s")
