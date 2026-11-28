#!/usr/bin/env python3
"""noise_floor.py — the distance between two implementations at the gate's positions (pos_sel / logits_sel): top-1
agreement, mean KL(A ‖ B), the next-token NLL of each, residual relative error at layers 0 / 21 / 42, expert-set
overlap at layers 1 / 21 / 42. Two lossless implementations show the floor every other comparison is read against.
  python3 tools/ref/noise_floor.py <dir A> <suffix A> <dir B> <suffix B> [names]
A file is <dir><name><suffix>: an ernie_ref.py dump (suffix .safetensors; .slice.safetensors for --key-slice --rows),
or an engine dump of engine/tests/gate.py (.engine-off.safetensors / .engine-on.safetensors: the prefill-rule patches
off / on). E.g. on B, /work/results/ref/v3:
  ref vs engine (off):  noise_floor.py v3/ .safetensors v3/ .engine-off.safetensors
  engine off vs on:     noise_floor.py v3/ .engine-off.safetensors v3/ .engine-on.safetensors
  ref dense vs sliced:  noise_floor.py v3/ .safetensors v3/ .slice.safetensors doc_long
Compared on the positions both saved (a long prompt's engine dump has the reference's positions)."""
import os
import sys

import torch
from safetensors.torch import load_file


def resid(D, L, keep):
    if "resid" in D:
        return D["resid"][L][keep]
    k = f"resid{L}"
    return D[k][keep] if k in D else None


def route(D, L, pos):
    """[P, 10] expert ids at positions pos (a reference dump holds every row, an engine dump only its pos_sel)"""
    if "route" not in D:
        return None
    r = D["route"][L]
    if r.shape[0] == D["pos_sel"].numel():
        ix = {p: i for i, p in enumerate(D["pos_sel"].tolist())}
        return r[[ix[p] for p in pos]]
    return r[pos]


def main():
    da, sa, db, sb = sys.argv[1:5]
    names = sys.argv[5].split(",") if len(sys.argv) > 5 else ["ko_short", "en_tools", "doc_long"]
    print(f"{'prompt':10} {'pos':>5} {'top-1':>6} {'KL':>9} {'NLL A':>7} {'NLL B':>7} {'rel L0':>9} {'rel L21':>9} "
          f"{'rel L42':>9} {'exp L1/L21/L42':>16}")
    for n in names:
        fa, fb = os.path.join(da, n + sa), os.path.join(db, n + sb)
        if not (os.path.exists(fa) and os.path.exists(fb)):
            print(f"{n:10} (missing {fa if not os.path.exists(fa) else fb})")
            continue
        A, B = load_file(fa), load_file(fb)
        pa, pb = A["pos_sel"].tolist(), B["pos_sel"].tolist()
        both = sorted(set(pa) & set(pb))
        ia, ib = {p: i for i, p in enumerate(pa)}, {p: i for i, p in enumerate(pb)}
        ka, kb = [ia[p] for p in both], [ib[p] for p in both]
        a, b = A["logits_sel"][ka].double(), B["logits_sel"][kb].double()
        top1 = (a.argmax(-1) == b.argmax(-1)).double().mean().item()
        la, lb = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
        kl = (la.exp() * (la - lb)).sum(-1).mean().item()
        nll = ["—", "—"]
        ids = A["ids"] if "ids" in A else B.get("ids")
        if ids is not None:
            k = [i for i, p in enumerate(both) if p + 1 < ids.numel()]
            t = ids[[both[i] + 1 for i in k]].long()[:, None]
            nll = [f"{-x[k].gather(1, t).mean().item():.3f}" for x in (la, lb)]
        rel = []
        for L in (0, 21, 42):
            x, y = resid(B, L, kb), resid(A, L, ka)
            rel.append(f"{((x.float() - y.float()).norm() / y.float().norm()).item():9.2e}" if x is not None and
                       y is not None else f"{'—':>9}")
        ov = "—"
        if "route" in A and "route" in B:
            s = []
            for L in (1, 21, 42):
                ra, rb = route(A, L, both).long(), route(B, L, both).long()
                s.append((ra[:, :, None] == rb[:, None, :]).any(-1).float().mean().item())
            ov = "/".join(f"{v:.3f}" for v in s)
        print(f"{n:10} {len(both):5d} {top1:6.3f} {kl:9.2e} {nll[0]:>7} {nll[1]:>7} {rel[0]} {rel[1]} {rel[2]} "
              f"{ov:>16}")


if __name__ == "__main__":
    main()
