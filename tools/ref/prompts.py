#!/usr/bin/env python3
"""prompts.py — the correctness gate's prompts as token ids (chat template of the checkpoint, thinking on).
  python3 tools/ref/prompts.py <model_dir> <out.json>
Three conversations: a short Korean question (~100 tokens), an English tool-use turn (~1K) and a long document
question (~3K: past the DSA indexer's 2,048 + 128 window, so its top-k really selects; 20+ HCA compressed entries)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "server"))
from ernie_proto.encoding_ernie import render_chat  # noqa: E402
from ernie_proto.tokens import SegmentTokenizer  # noqa: E402

model, out = sys.argv[1], sys.argv[2]
tok = SegmentTokenizer(model)
readme = open(os.path.join(model, "README.md"), encoding="utf-8").read()
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
res = {}
for name, kw in convs.items():
    r = render_chat(kw["messages"], kw.get("tools"), add_generation_prompt=True)
    ids = tok.encode_segments(r.segments)
    res[name] = ids
    print(f"{name}: {len(ids)} tokens · tail {tok.decode(ids[-12:])!r}")
json.dump(res, open(out, "w"))
