"""Segments → token ids, as the checkpoint's tokenization_ernie.py (ErnieChatTokenizer.encode_segments, Apache-2.0,
Baidu) does it, without transformers: renderer control segments compile their literal added tokens to ids; all other
text — adjacent ordinary runs joined across segment boundaries — is BPE-encoded by tokenizer.json with its added tokens
removed, so client text can never become a control token. The ids-only path (serving): no offsets or masks."""
from __future__ import annotations

import json
import os
import re

from tokenizers import Tokenizer


class SegmentTokenizer:
    def __init__(self, model_dir: str):
        with open(os.path.join(model_dir, "tokenizer.json"), encoding="utf-8") as f:
            config = json.load(f)
        added = config.get("added_tokens", [])
        if any(t.get(k) for t in added for k in ("single_word", "lstrip", "rstrip", "normalized")):
            raise ValueError("segment encoding requires literal added tokens")
        self.special = {t["content"]: t["id"] for t in added}
        self.reserved = frozenset(self.special.values())
        self.pattern = re.compile("|".join(re.escape(s) for s in sorted(self.special, key=len, reverse=True)))
        config["added_tokens"] = []
        config["padding"] = None
        config["truncation"] = None
        self.ordinary = Tokenizer.from_str(json.dumps(config))
        self.full = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))  # decoding (added tokens kept)

    def encode_segments(self, segments) -> list[int]:
        runs: list = []  # str = ordinary text run, int = control token id
        buffer: list[str] = []

        def flush():
            if buffer:
                runs.append("".join(buffer))
                buffer.clear()

        for seg in segments:
            text = seg.text
            if seg.allow_special:
                cursor = 0
                for m in self.pattern.finditer(text):
                    if m.start() > cursor:
                        buffer.append(text[cursor:m.start()])
                    flush()
                    runs.append(self.special[m.group()])
                    cursor = m.end()
                text = text[cursor:]
            if text:
                buffer.append(text)
        flush()
        texts = [r for r in runs if isinstance(r, str)]
        encoded = iter(self.ordinary.encode_batch(texts, add_special_tokens=False) if texts else ())
        ids: list[int] = []
        for r in runs:
            if isinstance(r, int):
                ids.append(r)
                continue
            run_ids = next(encoded).ids
            if not self.reserved.isdisjoint(run_ids):
                raise ValueError("an ordinary BPE path produced an added-token id")
            ids.extend(run_ids)
        return ids

    def encode_text(self, text: str) -> list[int]:
        """Plain text (no control tokens), as the reference's encode_text default."""
        return self.ordinary.encode(text, add_special_tokens=False).ids

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return self.full.decode(list(ids), skip_special_tokens=skip_special_tokens)
