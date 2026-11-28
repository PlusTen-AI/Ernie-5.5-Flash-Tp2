#!/usr/bin/env python3
"""prompts.py — the correctness gate's prompts as token ids (chat template of the checkpoint, thinking on).
  python3 tools/ref/prompts.py <model_dir> <out.json> [--long [N]] [--add]
Three conversations: a short Korean question (~100 tokens), an English tool-use turn (~1K) and a long document
question (~3K: past the DSA indexer's 2,048 + 128 window, so its top-k really selects; 20+ HCA compressed entries).
--long [N]: also "long", a question about this repository's own sources and docs (README, docs, engine/src, server,
tools — sorted, each under a "=== path ===" line) cut to at most N tokens (default 139,264 = 17 × 8192): its rows past
114,688 read ≥ 5 HCA key chunks (the base's long-context prefill kernels P150 / P172; the prefill-rule P210 takes
every prefill row). Written once and kept (the sources change): --add puts only "long" into an existing <out.json>,
its other prompts (and so their dumps) untouched."""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "server"))
from ernie_proto.encoding_ernie import render_chat  # noqa: E402
from ernie_proto.tokens import SegmentTokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("out")
ap.add_argument("--long", type=int, nargs="?", const=139264, default=0)
ap.add_argument("--add", action="store_true", help="only the long prompt, into the existing out.json")
a = ap.parse_args()
tok = SegmentTokenizer(a.model)


def encode(messages, tools=None):
    r = render_chat(messages, tools, add_generation_prompt=True)
    return tok.encode_segments(r.segments)


res = {}
if a.add:
    res = json.load(open(a.out))
else:
    readme = open(os.path.join(a.model, "README.md"), encoding="utf-8").read()
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string",
                       "enum": ["c", "f"]}}, "required": ["city"]}}}]
    convs = {
        "ko_short": dict(messages=[{"role": "user", "content": "서울에서 부산까지 KTX로 얼마나 걸려? 짧게 답해줘."}]),
        "en_tools": dict(messages=[
            {"role": "system", "content": "You are a helpful assistant. Use tools when they help."},
            {"role": "user", "content": "What's the weather in Paris right now, in celsius? " + readme[1500:4500]},
        ], tools=tools),
        "doc_long": dict(messages=[{"role": "user", "content": readme[:12000] +
                                    "\n\nSummarize the section on DSpark speculative decoding in three sentences."}]),
    }
    for name, kw in convs.items():
        res[name] = encode(kw["messages"], kw.get("tools"))
if a.long:
    pats = ["README.md", "docs/*.md", "engine/src/**/*.h", "engine/src/**/*.cc", "engine/src/**/*.cu",
            "engine/src/**/*.cuh", "engine/src/**/*.def", "server/*.py", "tools/*.py", "tools/ref/*.py"]
    files = []
    for p in pats:
        files += [f for f in sorted(glob.glob(os.path.join(ROOT, p), recursive=True)) if f not in files]
    text = "".join(f"=== {os.path.relpath(f, ROOT)} ===\n{open(f, encoding='utf-8', errors='replace').read()}\n"
                   for f in files)
    head = "Here are the sources and documents of an inference engine.\n\n"
    tail = ("\n\nWhich kernels does the prefill of one HCA layer launch, in order, and which patch chooses each? "
            "Answer briefly.")

    def conv(n):
        return encode([{"role": "user", "content": head + text[:n] + tail}])

    full = len(conv(len(text)))
    if full < a.long:
        sys.exit(f"[prompts] the sources give only {full} tokens < {a.long}")
    lo, hi = 0, len(text)  # the longest cut with ≤ N tokens
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if len(conv(mid)) <= a.long else (lo, mid)
    res["long"] = conv(lo)
for name, ids in res.items():
    print(f"{name}: {len(ids)} tokens · tail {tok.decode(ids[-12:])!r}")
json.dump(res, open(a.out, "w"))
