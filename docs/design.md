# Engine design — ERNIE 5.5 Flash on two GB10 (TP = 2)

The model's math is [model.md](model.md). This page is how the engine computes it.

## Precision: the checkpoint's own, natively

| tensors | stored | how they run |
|---|---|---|
| attention / dense / shared / latent projections | FP8 e4m3, 128×128 power-of-two scales | `kernels/gemm/fp8.cu`: activation → e4m3 with 1×128 UE8M0 scales (the reference's rule), `mma.sync kind::mxf8f6f4.block_scale` e4m3 × e4m3, the 128-block scale as UE8M0 per 32 (exact) |
| routed experts | MXFP4 (e2m1 + UE8M0 per 32) | `kernels/moe/moe.cu`: W4A8 — activations MXFP8 (e4m3, UE8M0 per 32), `kind::mxf8f6f4` e2m1 × e4m3 |
| DSA indexer | MXFP4 query and keys | `kernels/attn/dsa.cu`: `kind::mxf8f6f4` e2m1 × e2m1 |
| KV caches | HCA: e4m3 (UE8M0 per 64) + bf16 rope · DSA: e4m3 (pow2 per 128) + bf16 rope | attention dequantizes rows to bf16 (exact) and runs bf16 tensor cores, fp32 softmax |
| embedding, lm_head, routers, compressor, indexer wk / weights_proj | bf16 | bf16 tensor cores (`kernels/linear`, from Plus) |
| mHC mixers | fp32 | `kernels/mhc`: fp32 FMAs on CUDA cores in one fixed order (64 chunks), post fused with the next pre |

Every kernel is deterministic and batch-invariant: a row's result is the same chain of operations whatever M is or
whatever else is in the step (fixed K splits, fixed attention key chunks of 256, no float atomics). Decode steps and
speculative rounds therefore give the same bits for the same token, and so do chunked prefill and cache reuse.

**Step kind and the prefill rule.** Every step carries its kind, set by its caller (`Engine::step(…, prefill)`,
`core/step_kind.h`): a prompt chunk is prefill; a decode step, a verify step and a draft pass are decode. Kernels choose
by the kind, never by M alone: a 1-row prompt chunk after a checkpoint restore still takes the prefill kernels, and a
128-row verify of 16 sequences takes the decode kernels. Since 0.6.0 a prefill kernel may accumulate in its own order
instead of the decode kernels' (P210: HCA prefill attention as one online softmax over a row's keys in 16-key tiles,
no 256-key chunks or merge). It keeps the reference's rounding points, determinism, batch and chunk-size invariance
within prefill and the MoE combine's slot order, and is judged by the reference gate at the noise floor (`gate.py
--pf`: the rule's patches off and on against the reference) instead of bitwise against decode. Decode, verify and draft
rows keep their bits (patch_check), so speculative rounds stay plain decoding.

## TP = 2

Heads, output groups and FFN intermediates are split; whatever feeds both halves is replicated
([tools/tensor_table.py](../tools/tensor_table.py) is the placement):

- **HCA**: q / kv down projections, compressor, KV caches replicated (one shared KV row, MQA); heads 32 + 32, wo_a
  groups 4 + 4, o_proj row-split. Collectives: VHA's rank-16 term (fp32 all-reduce of Σ U·o), o_proj partial sums.
- **DSA**: q_a / kv_a, the indexer (all 64 heads: its scores sum over heads) and both caches replicated; MLA heads
  32 + 32 (W_UK / W_UV per head from kv_b_proj), o_proj row-split. Same two collectives.
- **MoE**: routers and fc1 / fc2 latent projections replicated; every routed expert's intermediate split 1024 + 1024
  (balanced: each rank reads half of each selected expert); shared expert split. One bf16 all-reduce of
  [latent partial 2048 ‖ shared partial 4096] (both rounded to bf16 first, as vLLM's two bf16 all-reduces), then
  latent_norm → fc2 on both ranks, bf16 + bf16.
- **Dense FFN** (layer 0): intermediate split, down row-split.
- **lm_head** split by vocabulary rows; logits all-gathered. mHC runs replicated on identical inputs.
- **Prefill** (8192-row chunks as lanes): the residual stream sequence-parallel (P094: per-row work on each rank's own
  rows, all-gather in / reduce-scatter out), attention output to the row owners by one o all-to-all (P100), whose peer
  half lands straight in the owner's NIC-registered receive buffer (P176).

Per rank: 73.7 GiB of weights. Caches per token: HCA 37 × 584 B / 128 (+ window and compressor state per slot),
DSA 6 × 724 B — about 4.5 KB per token per rank, so a 1M-token slot is ~4.5 GB.

## Phases

1. **Runs and answers correctly** — every kernel in its native format, NCCL collectives, plain scheduling. One gate:
   the engine against the reference forward ([tools/ref/ernie_ref.py](../tools/ref/ernie_ref.py), written from
   vLLM-ERNIE's code) layer by layer, prefill and the cached decode path ([engine/tests/gate.py](../engine/tests/gate.py)).
   Then the server (Plus's layer, ERNIE's protocol from the checkpoint).
2. **Speed** — measured first, then: DSpark drafts, RDMA all-reduce on both rails, overlap, PDL, larger GEMM / MoE
   tiles, FP4 indexer at the mxf4nvf4 rate, the indexer's exact threshold pre-filter, HCA compression on a side stream.

## DSpark speculative decoding (`engine/src/model/dspark.cc`)

- **Draft**: placed like the target (`tools/tensor_table.py`: stages split as target layers, `markov_w2` by vocabulary
  rows like `lm_head`, the rest replicated), every GEMM in the native kernels. Context rows: every target step writes,
  for each sequence's last ≤ 128 rows, the draft's context KV (feature_projection → each stage's kv) into a per-slot
  ring of 256 rows by position — plain steps and speculative rounds mix freely, and rows of rejected positions are
  rewritten before any draft reads them. Greedy drafts: each rank's vocabulary-half argmax, exchanged by one small
  all-reduce per Markov step; sampled drafts: full rows gathered and drawn by `sample_rows` (kept as the q rows).
- **Verify**: one target step over `[last, d_1 … d_k]` of every sequence (all logits), the accept on both ranks
  (`spec_accept`, P016 for T > 0; greedy: the target's argmax). Every kernel is batch-invariant, so a kept row's logits
  are bitwise plain decoding's. Rollback: the HCA window ring (by position % 128) rows of the rejected rows are saved
  before the step and put back; when the block closes a compressor group and the closing row is rejected, the state
  rows the new group overwrote come back and the kept rows' state is written from the step's stashed projections.
  Positional caches (DSA latent / indexer keys, compressed entries) need nothing.
- **Verify length**: fixed (`depth`), gated by the draft's confidence head (P121 conf_gate: verify draft j while
  Π_{i≤j} σ(conf_i) ≥ gate), or (the default, `[decode] verify = "cost"`, P143 verify_joint) chosen for the whole round:
  the total rows whose expected tokens (the confidence products) per ms of the round's measured cost by sequences and
  rows is best; never past the round's cap (P122 cap_verify).
- **Serving** (`server/scheduler.py`, `[decode]` in serve.toml): the decode step is one round over every decoding
  sequence with its own sampling settings; a prompt's last chunk is a plain step whose logits row gives the first token
  (`spec_begin`, then its drafts). Each sequence's cap — max_tokens, the thinking budget's and the loop guard's first
  possible forced token, a draft that is an end token, the one-call stop token or (tool_choice forcing) `</think>` —
  puts anything the server changes or ends on the round's last token (not yet run), so a slot's history is exactly
  plain decoding's and prefix reuse works as before.
- **Copy drafts** (P168, `[decode] copy`, `server/copy_draft.py` on rank 0): while one request decodes alone and its
  reply's tail repeats ≥ `copy_match` tokens of its history (an index of 4-token runs, the longest backward match), the
  round verifies the tokens that followed there — 7, then doubling while taken whole, up to `copy_rows` − 1 — instead
  of DSpark's drafts. Every row is drawn exactly as plain decoding draws that position (greedy argmax, or the request's
  sampling with the position's own counter) and a copied token is kept while the draw equals it, so the reply is plain
  decoding's; wide steps are bitwise one-row steps (batch invariance; checked to 64 rows, 140K context). Rounds shared
  by several requests never copy.
- **Prefix checkpoints** (P146, `[server] checkpoints`): at every prompt's end and every `checkpoint_every` prefill
  tokens a slot copies what cannot step back — each HCA layer's 128-row window ring, the compressor's open group, the
  draft's context rows — into KV-pool chunks (2.9–21 MiB each); a prompt that leaves the history part-way restores
  the longest checkpoint at or below the common prefix and reads only the rest (bitwise a full prefill's). `decode.dspark = false` serves plain decoding.

