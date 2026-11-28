#!/usr/bin/env python3
"""pf_rule.py — the prefill-rule patches (rule of 2026-11-28): prefill-only kernels with their own accumulation order
(the reference's rounding points kept, deterministic, batch- and chunk-size invariant within prefill, the MoE combine in
slot order), chosen by the step's kind (core/step_kind.h: Engine::step(…, prefill)), never by M. They are not bitwise
the base, so:
  patch_check.py  holds them at one state in both runs (--pf off | on) and refuses to compare runs whose states differ;
  gate.py         judges them against the reference: --pf both runs every prompt with them off (the noise floor: the
                  engine's bitwise path vs tools/ref/ernie_ref.py) and on, and passes on when it stays at that floor;
  pf_check.py     checks, at the model level, that with them on a row's bits do not depend on how the prompt is read
                  (chunk 8192 / 2048 / odd pieces / a checkpoint restore part-way, repeated runs) and that decode-kind
                  steps never take them.
A new prefill-rule patch: its name goes into PF_RULE, and its patches.def line says "prefill rule" — names() fails when
the two disagree. A prefill-rule patch must be switchable at run time (E.patch: gate.py and pf_check.py turn it off and
on in one process), i.e. chosen at dispatch, with no load-time layout of its own."""
import os
import re

PF_RULE = (
    "hca_attn_1p",  # P210: HCA prefill attention, one online softmax over a row's keys
)
DEFS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "model", "patches.def")


def names(defs=DEFS):
    """PF_RULE, checked against patches.def: every name listed there, every line saying "prefill rule" listed here"""
    if not os.path.exists(defs):
        return list(PF_RULE)
    rows = re.findall(r'^E55_PATCH\(\s*(\d+),\s*(\w+),\s*\w+,\s*"(.*)"\)\s*$', open(defs, encoding="utf-8").read(), re.M)
    known = {n for _, n, _ in rows}
    gone = [n for n in PF_RULE if n not in known]
    if gone:
        raise SystemExit(f"[pf_rule] in pf_rule.PF_RULE but not in patches.def: {gone}")
    said = [f"P{k} {n}" for k, n, d in rows if "prefill rule" in d and n not in PF_RULE]
    if said:
        raise SystemExit(f"[pf_rule] patches.def says 'prefill rule' but pf_rule.PF_RULE lacks: {said}")
    return list(PF_RULE)


def env(on, pf, base=None):
    """E55_PATCHES with the prefill-rule patches pf forced on / off after what it held (the last token wins)"""
    toks = [t for t in (os.environ.get("E55_PATCHES", "") if base is None else base).split(",") if t]
    return ",".join(toks + [("+" if on else "-") + n for n in pf])


if __name__ == "__main__":
    print(" ".join(names()))
