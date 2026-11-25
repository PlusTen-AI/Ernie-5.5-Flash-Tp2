#!/usr/bin/env python3
"""tensor_table.py — loader tensor table (TSV) for the ERNIE-5.5-Flash-Preview checkpoint. Stdlib only.
  python3 tools/tensor_table.py <checkpoint dir> <out.tsv>

Reads model.safetensors.index.json + every shard header and writes one row per tensor, sorted by (file, offset):
  name file offset nbytes dtype shape layer block part tp
  offset  absolute byte offset of the data in `file` (8 + header_len + data_offsets[0])
  shape   dims joined by 'x', scalar '-'
  layer   decoder layer, -1 = model-global, mtp = the bundled DSpark draft (model.mtp.*)
  block   embed lm_head head hca dsa dsa_indexer hc dense router moe_proj shared expert mtp (a draft stage's tensors
          take mtp_<the target layer's block>, part "<stage>.<part>"; the draft's own tensors block mtp)
  tp      TP=2 placement (engine/src/core/weights.cc):
            rep  both ranks hold the whole tensor
            col  dim 0 halved: rank r takes rows [r·N/2, (r+1)·N/2)
            row  dim 1 halved (input side; packed FP4 bytes and scale columns halve with it)
            skip not loaded (none in this checkpoint)
TP=2 plan (docs/design.md): attention heads, output groups and FFN intermediates split, everything that feeds both
halves (q / kv down projections, compressor, indexer, routers, latent projections, mHC) replicated; o_proj and down
projections row-split (partial sums, all-reduced); lm_head split by vocabulary rows.
"""
import json
import os
import re
import struct
import sys
from collections import defaultdict

COLS = "name file offset nbytes dtype shape layer block part tp".split()


def classify(name):
    """→ (layer, block, part, tp)"""
    if name == "model.embed_tokens.weight":
        return "-1", "embed", "weight", "rep"
    if name == "lm_head.weight":
        return "-1", "lm_head", "weight", "col"
    m = re.match(r"model\.(lm_head|mhc_contract|norm)\.(.+)", name)
    if m:
        return "-1", "head", f"{m.group(1)}.{m.group(2)}", "rep"
    if name.startswith("model.mtp."):  # the bundled DSpark draft: its 3 stages split like target layers
        rest = name[len("model.mtp."):]
        m = re.match(r"(\d+)\.(.+)", rest)
        if m:
            _, block, part, tp = classify(f"model.layers.{m.group(1)}.{m.group(2)}")
            return "mtp", "mtp_" + block, f"{m.group(1)}.{part}", tp
        if rest.startswith("markov_head.markov_w2."):  # [vocab, rank]: vocabulary rows split as lm_head
            return "mtp", "mtp", rest, "col"
        return "mtp", "mtp", rest, "rep"  # feature_projection / _norm, norm, mhc_contract, markov_w1
    m = re.match(r"model\.layers\.(\d+)\.(.+)", name)
    if not m:
        raise SystemExit(f"unclassified tensor {name}")
    L, p = m.group(1), m.group(2)
    if "hyper_connection" in p or p.endswith("layernorm.weight") and p.startswith(("input_", "post_attention_")):
        return L, "hc", p, "rep"
    if p.startswith("mlp."):  # layer 0 dense FFN
        q = p[4:]
        return L, "dense", q, "row" if q.startswith("down_proj.") else "col"
    if p.startswith("block_sparse_moe."):
        q = p[len("block_sparse_moe."):]
        if q.startswith("experts."):
            return L, "expert", q, "row" if ".w2." in q else "col"
        if q.startswith("shared_experts."):
            return L, "shared", q, "row" if q.startswith("shared_experts.w2.") else "col"
        if q.startswith(("gate.", "e_score_correction_bias")):
            return L, "router", q, "rep"
        return L, "moe_proj", q, "rep"  # fc1 / fc2 latent projections, latent_norm
    if p.startswith("self_attn."):
        q = p[len("self_attn."):]
        dsa = int(L) in (7, 14, 21, 28, 35, 42)
        if q.startswith("core_attention.indexer."):
            return L, "dsa_indexer", q, "rep"
        block = "dsa" if dsa else "hca"
        if q.startswith(("linear_q_up_proj", "q_b_proj", "kv_b_proj", "gate_proj", "linear_o_group_proj")):
            return L, block, q, "col"
        if q in ("core_attention.attn_sink", "core_attention.softmax_offset"):
            return L, block, q, "col"
        if q.startswith("o_proj."):
            return L, block, q, "row"
        return L, block, q, "rep"
    raise SystemExit(f"unclassified tensor {name}")


def main():
    ckpt, out = sys.argv[1], sys.argv[2]
    wm = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    rows = []
    for f in sorted(set(wm.values())):
        with open(os.path.join(ckpt, f), "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            a, b = v["data_offsets"]
            layer, block, part, tp = classify(k)
            rows.append(dict(name=k, file=f, offset=8 + n + a, nbytes=b - a, dtype=v["dtype"],
                             shape="x".join(map(str, v["shape"])) or "-", layer=layer, block=block, part=part, tp=tp))
    if len(rows) != len(wm):
        raise SystemExit(f"{len(rows)} tensors in the shards, {len(wm)} in the index")
    # every split must divide evenly, on the scale tiles too
    for r in rows:
        if r["tp"] in ("col", "row"):
            d = [int(x) for x in r["shape"].split("x")]
            i = 0 if r["tp"] == "col" else 1
            if d[i] % 2:
                raise SystemExit(f"{r['name']}: {r['tp']} split of odd dim {d}")
    rows.sort(key=lambda r: (r["file"], r["offset"]))
    with open(out, "w") as fo:
        fo.write("\t".join(COLS) + "\n")
        for r in rows:
            fo.write("\t".join(str(r[c]) for c in COLS) + "\n")
    per = defaultdict(int)
    for r in rows:
        if r["tp"] == "skip":
            continue
        per[r["block"]] += r["nbytes"] if r["tp"] == "rep" else r["nbytes"] // 2
    tot = sum(per.values())
    print(f"[table] {len(rows)} tensors → {out} · per rank {tot / 2**30:.1f} GiB: " +
          " · ".join(f"{k} {v / 2**30:.2f}" for k, v in sorted(per.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
