#!/usr/bin/env python3
"""show.py — decode a reference dump's per-position top predictions next to the actual next token.
  python3 tools/ref/show.py <model_dir> <dump.safetensors> [first last]"""
import os
import sys

import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "server"))
from ernie_proto.tokens import SegmentTokenizer  # noqa: E402

tok = SegmentTokenizer(sys.argv[1])
d = load_file(sys.argv[2])
ids, top, val = d["ids"].tolist(), d["top_ids"], d["top_vals"]
a, b = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, len(ids))
for i in range(a, min(b, len(ids))):
    nxt = tok.decode([ids[i + 1]]) if i + 1 < len(ids) else "—"
    preds = " | ".join(f"{tok.decode([t])!r}:{v:.1f}" for t, v in zip(top[i, :4].tolist(), val[i, :4].tolist()))
    raw = ""
    if "raw_ids" in d:
        raw = " || raw " + " | ".join(f"{tok.decode([t])!r}:{v:.1f}" for t, v in zip(d["raw_ids"][i, :3].tolist(), d["raw_vals"][i, :3].tolist()))
    print(f"{i:4d} {tok.decode([ids[i]])!r:>16} → next {nxt!r:>14} · {preds}{raw}")
