#!/usr/bin/env python3
"""gate.py — the correctness gate: the engine (TP=2, both ranks) against the reference forward (tools/ref/ernie_ref.py
dumps), layer by layer. Run through tools/tp2.sh (it appends --rank and --head-ip):
  tools/tp2.sh gate python3 /work/src/engine/tests/gate.py --ref /work/results/ref/v1 [--names a,b]
Per prompt:
  A. one prefill step over the whole prompt: residual after every layer, attention / FFN outputs and MoE expert sets
     at the dump's positions; logits there (top-1 agreement, KL, max |Δ|)
  B. the same prompt through the caches: prefill in chunks of 1000, the rest less 32 as one chunk, the last 32
     positions one decode step each — logits of the last 32 against the reference
  Next-token NLL of the prompt at the dump's positions, engine and reference side by side (a bug makes the engine
  systematically worse; rounding noise does not), and top-1 agreement where the reference is confident (p ≥ 0.5).
--gen N: after the prompts, up to N greedy tokens for each prompt, printed as text.
--isolate L,L…: each listed layer on its own (prompt --iso-name, every row tapped): the reference's layer
  (tools/ref/ernie_ref.py) run on the engine's residual before it, against the engine's own outputs — attention, FFN,
  expert sets and residual errors that do not carry earlier layers' differences.
Rank 0 prints; rank 1 makes the same calls. Summary → <ref>/gate-<name>.txt"""
import argparse
import json
import os
import sys
import time

import torch
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ext  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True)
ap.add_argument("--names", default="ko_short,en_tools,doc_long")
ap.add_argument("--model", default="/models/ERNIE-5.5-Flash-Preview")
ap.add_argument("--table", default="/work/cache/table.tsv")
ap.add_argument("--rank", type=int, required=True)
ap.add_argument("--head-ip", required=True)
ap.add_argument("--port", type=int, default=29611)
ap.add_argument("--gen", type=int, default=0)
ap.add_argument("--isolate", default="")
ap.add_argument("--iso-name", default="en_tools")
a = ap.parse_args()
R0 = a.rank == 0


def say(*x):
    if R0:
        print(*x, flush=True)


t0 = time.time()
E = ext.load("engine", ext.engine_sources(), log=f"/tmp/build{a.rank}.log", ldflags=ext.ENGINE_LDFLAGS)
say(f"[gate] engine built · {time.time() - t0:.0f} s")
E.init(a.model, a.table, a.rank, a.head_ip, 2, 8192, 4096, 512, a.port)
prompts = json.load(open(os.path.join(a.ref, "prompts.json")))
V = 201216


def rel(x, y):
    x, y = x.float(), y.float()
    return ((x - y).norm() / y.norm().clamp_min(1e-30)).item()


def logit_stats(lg, ref):
    lp, lq = torch.log_softmax(ref.float(), -1), torch.log_softmax(lg.float(), -1)
    kl = (lp.exp() * (lp - lq)).sum(-1)
    top = (lg.argmax(-1) == ref.argmax(-1)).float()
    return top.mean().item(), kl.mean().item(), kl.max().item(), (lg - ref).abs().max().item()


def nll_stats(lg, ref, nxt):
    """mean NLL of the next tokens (engine, reference) and top-1 agreement where the reference's top p ≥ 0.5"""
    lp, lq = torch.log_softmax(ref.float(), -1), torch.log_softmax(lg.float(), -1)
    t = torch.tensor(nxt, device=lg.device)[:, None]
    conf = lp.max(-1).values.exp() >= 0.5
    agree = (lg.argmax(-1) == ref.argmax(-1))[conf].float()
    return (-lq.gather(1, t).mean().item(), -lp.gather(1, t).mean().item(), agree.mean().item() if conf.any() else
            float("nan"), int(conf.sum()))


for name in a.names.split(","):
    f = os.path.join(a.ref, f"{name}.safetensors")
    if not os.path.exists(f):
        say(f"[gate] {name}: no reference dump")
        continue
    D = load_file(f, device="cuda")
    ids = prompts[name]
    T = len(ids)
    sel = D["pos_sel"].tolist()
    lines = []
    # A: one step
    E.reset(0)
    E.set_taps(sel)
    t = time.time()
    lg = E.step(ids, list(range(T)), [0], [0, T], False, sel)
    torch.cuda.synchronize()
    dt = time.time() - t
    r, at, ff, rt = E.taps(len(sel))[:4]
    if R0:
        lines.append(f"{name}: {T} tokens · prefill step {dt:.2f} s")
        lines.append("layer  resid      attn       ffn        experts")
        for L in range(43):
            same = 1.0
            if L > 0:
                ee, er = rt[L].long().sort(-1).values, D["route"][L][sel].long().sort(-1).values
                same = (ee == er).all(-1).float().mean().item()
            lines.append(f"{L:5d}  {rel(r[L], D['resid'][L]):.2e}  {rel(at[L], D['attn'][L]):.2e}  "
                         f"{rel(ff[L], D['ffn'][L]):.2e}  {same:.3f}")
        top1, kl, klmax, dmax = logit_stats(lg, D["logits_sel"])
        lines.append(f"logits at {len(sel)} positions: top-1 agree {top1:.3f} · KL mean {kl:.2e} max {klmax:.2e} · "
                     f"max |Δ| {dmax:.3f}")
        k = [i for i, p in enumerate(sel) if p + 1 < T]
        ne, nr, ca, nc = nll_stats(lg[k], D["logits_sel"][k], [ids[sel[i] + 1] for i in k])
        lines.append(f"prompt next-token NLL at {len(k)} positions: engine {ne:.3f} · reference {nr:.3f} · "
                     f"top-1 agree where the reference is sure {ca:.3f} ({nc} positions)")
    E.set_taps([])
    # B: through the caches
    E.reset(0)
    tail = 32
    cut = T - tail
    pos = 0
    t = time.time()
    while pos < cut:
        n = min(1000, cut - pos) if cut - pos > 1000 else cut - pos
        E.step(ids[pos:pos + n], list(range(pos, pos + n)), [0], [0, n], False)
        pos += n
    outs = []
    for p in range(cut, T):
        outs.append(E.step([ids[p]], [p], [0], [0, 1], False)[0])
    torch.cuda.synchronize()
    dt = time.time() - t
    if R0:
        lgB = torch.stack(outs)
        idx = [sel.index(p) for p in range(cut, T)]
        top1, kl, klmax, dmax = logit_stats(lgB, D["logits_sel"][idx])
        same = (lgB - lg[idx]).abs().max().item()
        lines.append(f"cached path (chunks of 1000, then 32 single steps, {dt:.1f} s): top-1 agree {top1:.3f} · "
                     f"KL mean {kl:.2e} max {klmax:.2e} · max |Δ| {dmax:.3f} · against the one step max |Δ| {same:.3g}")
        txt = "\n".join(lines)
        print(txt, flush=True)
        open(os.path.join(a.ref, f"gate-{name}.txt"), "w").write(txt + "\n")
if a.isolate:
    name = a.iso_name
    ids = prompts[name]
    T = len(ids)
    E.reset(0)
    E.set_taps(list(range(T)))
    E.step(ids, list(range(T)), [0], [0, T], False)
    iso = [int(x) for x in a.isolate.split(",")]
    need = sorted({x for L in iso for x in (L - 1, L) if x >= 0})
    r, at, ff, rt, rm, po, cb = E.taps(T, need)  # only the layers the checks read
    E.set_taps([])
    ix = {L: i for i, L in enumerate(need)}
    r, at, ff, rt, rm, po, cb = ({L: x[ix[L]] for L in need} for x in (r, at, ff, rt, rm, po, cb))
    if R0:
        sys.path.insert(0, os.path.join(HERE, "..", "..", "tools", "ref"))
        import ernie_ref as ref
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
    for name in a.names.split(","):
        ids = list(prompts[name])
        E.reset(0)
        lg = E.step(ids, list(range(len(ids))), [0], [0, len(ids)], False)
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
