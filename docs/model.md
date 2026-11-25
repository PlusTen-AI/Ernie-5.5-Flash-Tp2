# ERNIE 5.5 Flash — what one forward pass computes

The math this engine implements, written from the reference serving code: vLLM-ERNIE (`baidu/vLLM-ERNIE`
@ 4a971d3, Apache-2.0) on vLLM 0.32.0 (its DeepSeek-V4 attention, compressor and mHC code). Checkpoint
`baidu/ERNIE-5.5-Flash-Preview` @ de3ffd3e: blockwise FP8 (128×128, fp32 scales) everywhere, routed experts MXFP4.
Names in `code` are checkpoint tensor names under `model.layers.N.` unless said otherwise.

## Shapes

| | |
|---|---|
| hidden H | 4096, residual carried as 4 streams (mHC, `hc_mult` 4) |
| layers | 43: layer 0 dense FFN, 1–42 MoE; attention: 37 HCA + 6 DSA (layers 7, 14, 21, 28, 35, 42: `compress_ratios[i] = -2`) |
| vocab | 201,216 · bos 1 · eos 2 and 8 (`generation_config`) · pad 0 |
| HCA | 64 heads × 512 (448 nope + 64 rope), one shared KV (MQA), q_lora 1024, 8 output groups × o_lora 1024, window 128, compress 128 |
| DSA | MLA: 64 heads, q_lora 1024, kv_lora 512, qk 192 nope + 64 rope, v 256; indexer 64 heads × 128, top 2048 |
| MoE | 512 routed (top 10) in a 2048 latent, expert inter 2048, MXFP4 W4A8; 1 shared expert (inter 2048) on H |
| dense FFN | inter 14336 (layer 0) |
| draft | DSpark, 3 layers under `model.mtp.*` (block 7), below |

Every RMSNorm: `x · rsqrt(mean(x²) + 1e-6) · w` in fp32, out bf16 (except where noted).

## Linear layers (FP8 block)

Weight `W` e4m3 [N, K] with `weight_scale_inv` fp32 [N/128, K/128]; the activation is quantized per token in 1×128
groups: `s = 2^ceil(log2(amax / 448))` (UE8M0, `scale_fmt: ue8m0`), `q = sat_e4m3(x / s)`;
`y = Σ_kblocks (q·s) · (W·S)` accumulated in fp32, out bf16. Unquantized weights (bf16 in the checkpoint):
compressor `linear_wkv` / `linear_wgate` (run fp32: `mm(x, W.T, out_dtype=fp32)`), MoE router `gate.weight(_1)`
(fp32), indexer `wk`, `weights_proj`, embedding, `lm_head`, VHA `U`/`V`, norms.

## mHC (every sublayer: attention and FFN)

Per layer two mixers (`self_attention_hyper_connection`, `mlp_hyper_connection`): `fn` = `mapping_proj.weight`ᵀ
[24, 16384] fp32, `base` = `bias` [24], `scale` = (`alpha_pre`, `alpha_post`, `alpha_res`).
Residual `R` [T, 4, H] bf16. vLLM-ERNIE fuses each post with the next pre; on its decode path (≤ 32 tokens) the
mixer reads the post's fp32 sum before the bf16 rounding (on its prefill path a TF32 GEMM over the bf16 R — not
batch-invariant). The engine and the reference use the decode form for every T. Pre:

```
x      = r.flatten (fp32, 16384)                        # r: the preceding post's UNROUNDED fp32 sum (below)
mixes  = (x @ fnᵀ) · 1/(sqrt(Σx²/16384) + 1e-6)          # ERNIE's RMS factor (not rsqrt(mean+eps)); fp32 fn
pre    = sigmoid(mixes[0:4]·scale0 + base[0:4]) + 1e-6
post   = 2 · sigmoid(mixes[4:8]·scale1 + base[4:8])
comb   = softmax_row(mixes[8:24].view(4,4)·scale2 + base[8:24]) + 1e-6
comb  /= colsum(comb) + 1e-6;  19× { comb /= rowsum + 1e-6; comb /= colsum + 1e-6 }
input  = RMSNorm_w(bf16(Σ_j pre_j · R_j))                  # the stored bf16 R; w = input_layernorm / post_attention_layernorm
```

Post: `r_j = fma(c3j, R_3, fma(c2j, R_2, fma(c1j, R_1, fma(c0j, R_0, post_j · out))))` (fp32, c = comb[i][j]),
`R'_j = bf16(r_j)`. Layer 0: `R = [e, e, e, e]` (embedding broadcast); its first pre sums `fn` over the 4 streams
first (`fn.view(24, 4, 4096).sum(1)`) and applies it to e with the factor `1/(sqrt(Σe²/4096) + 1e-6)`. After layer 42: head contraction (`model.mhc_contract`, `fn` [4, 16384] = stored ᵀ, `scale` [1],
`base` [4]): `pre = sigmoid((x@fnᵀ)·rsqrt(mean(x²)+1e-6)·scale + base) + 1e-6` (here the usual rsqrt form),
`h = bf16(Σ_j pre_j R_j)` → `model.norm` → `lm_head` (bf16 weight, fp32 logits) → multimax.

Multimax (`model.lm_head.multimax_ranges` r, `multimax_ts` t, bf16 → fp32), elementwise on fp32 logits, this order:
`y = x + t0·relu(r0−x) + t1·relu(x−r1) + t2·relu(r2−x)² + t3·relu(x−r3)²`.

## HCA attention (37 layers)

```
qr  = RMSNorm_q(linear_q_down_proj(x))                      # [1024]   q_layernorm
kv  = RMSNorm_kv(linear_kv_proj(x))                         # [512]    kv_layernorm
q   = linear_q_up_proj(qr).view(64, 512)
q_h = q_h · rsqrt(mean(q_h²) + 1e-6)                        # per head, no weight
rope(q[..., 448:], pos); rope(kv[448:], pos)                # interleaved pairs, see RoPE
gate = sigmoid(gate_proj(qr))                               # [8 × 1024]
```

KV cache rows (SWA window and compressed entries share the format): 448 nope dims e4m3 with one UE8M0 scale per 64
dims (`e = ceil(log2(max(amax,1e-4)/448))`, values first rounded to bf16), 64 rope dims bf16. 584 bytes.

Compressor (`core_attention.compressor`, ratio 128, no overlap): per token `kv_t = x@Wkvᵀ`, `s_t = x@Wgateᵀ +
ape[pos % 128]` (fp32, [512] each). When `(pos+1) % 128 == 0` the group j = pos // 128 closes:
`c = Σ_t softmax_t(s)·kv_t` (per dim, over the group's 128 tokens) → `RMSNorm_norm(c)` → rope at position 128·j →
quantized into compressed entry j.

Attention for a query at position p: keys = compressed entries 0 … ⌊(p+1)/128⌋−1 plus the window p−127 … p (raw
tokens). K = V = the dequantized row (bf16 values, 512). `score = q·k / sqrt(512)`, softmax with the per-head sink
(`core_attention.attn_sink` fp32): `p_i = exp(s_i − m) / (Σ exp(s − m) + exp(sink − m))`. `o_h = Σ p_i v_i` [512].

Output: VHA `m = I + V Uᵀ` (`vha_postmix_V`, `vha_postmix_U` [64, 16]) mixes heads:
`o'_h = o_h + Σ_r V[h,r] · z_r`, `z = Σ_g U[g,:] o_g` (fp32; vLLM's TP form). Inverse rope on `o'[..., 448:]`
at p. Per group g (8 heads = 4096 dims): FP8 quant (1×128, UE8M0) and `z_g = o'_g @ wo_aᵀ_g` (`linear_o_group_proj`
[8 × 1024, 4096], FP8 block) → `bf16`; `out = o_proj(z ⊙ gate)` (`o_proj` [4096, 8192]).

## DSA attention (layers 7, 14, 21, 28, 35, 42)

```
q_c  = RMSNorm(q_a_proj(x))                       # [1024]   q_a_layernorm
q    = q_b_proj(q_c).view(64, 256)                # 192 nope | 64 rope
kvl  = kv_a_proj_with_mqa(x)                      # [512 kv_c | 64 k_pe]
kv_c = RMSNorm(kvl[:512])                         # kv_a_layernorm
rope(q[..., 192:], pos); rope(k_pe, pos)          # interleaved, theta 8e6, no YaRN
gate = sigmoid(gate_proj(q_c))                    # [64 × 256]
```

Cache row: kv_c as e4m3 with one power-of-two fp32 scale per 128 (`2^ceil(log2(max(amax/448, 1e-4)))`), k_pe bf16;
656 bytes. Attention over the selected positions (below), absorbed MLA: per head `kv_b_proj` [64·(192+256), 512]
gives `W_UK` (rows 0–191) and `W_UV` (rows 192–447): `q_lat = q_nope @ W_UK` [512],
`score = (q_lat·kv_c + q_pe·k_pe) / sqrt(256)`, softmax with the sink (`core_attention.softmax_offset`, bf16),
`o_h = (Σ p_i kv_c_i) @ W_UVᵀ` [256]. VHA as above on [64, 256], then `out = o_proj((o' ⊙ gate))`
(`o_proj` [4096, 16384]).

Indexer (`core_attention.indexer`), per token:
```
q   = wq_b(q_c).view(64, 128)                     # FP8 block linear
k   = LayerNorm(wk(x))                            # [128], with bias, eps 1e-6
w   = weights_proj(x) · 128^-0.5 · 64^-0.5        # [64] fp32
rope(q[..., :64], pos); rope(k[:64], pos)         # NeoX (half split) on the LEADING 64 dims, theta 8e6
q, k = Hadamard128(q), Hadamard128(k)             # orthonormal
q, k → MXFP4 (e2m1, UE8M0 per 32)                 # k cached: 64 B values + 4 B scales
#   scale = 2^(floor(log2(amax rounded: bits + 0x200000, exponent kept)) − 2), elements RNE to e2m1, |x| ≤ 6,
#   two per byte, even index in the low nibble (vLLM downcast_to_mxfp)
logit(t) = Σ_h relu(q_h · k_t) · w_h              # dequantized values, fp32 sums
```
Candidates are positions 0 … p−128 (older than the window); the top 2048 by logit (all if fewer), then the window
p−127 … p is appended: ≤ 2176 keys. (FlashIndexer head routing is approximate — not used.)

## RoPE

Interleaved pairs (2i, 2i+1) unless said; `cos/sin` from fp32 `inv_freq`, angles `pos · inv_freq` in fp32. HCA keeps
fp32 cos/sin; DSA and its indexer use vLLM's `get_rope` cache in the model dtype — the fp32 cos/sin rounded to bf16 —
with fp32 rotation math and one bf16 rounding of the output.
- HCA (all of a layer: q, window kv, compressed kv, inverse on o): last 64 of 512, theta 160,000, YaRN
  (factor 16, original 65,536, β 32/1, `mscale` 1): `low = floor(d(32))`, `high = ceil(d(1))`,
  `d(r) = 64·ln(65536/(2πr)) / (2 ln θ)`, clamp to [0, 63]; `ramp_i = clamp((i−low)/(high−low), 0, 1)`;
  `inv_freq_i = θ^(−2i/64) · (ramp_i / 16 + 1 − ramp_i)`.
- DSA MLA: last 64 of 256 (q) / the 64 k_pe dims, theta 8,000,000, plain.
- DSA indexer: NeoX on the first 64 of 128, theta 8,000,000, plain.

## FFN

SiTU-GLU (dense FFN, shared expert — vLLM's `situ_and_mul`): `g' = 2·tanh(g·0.25)·(1 + tanh(g·0.5))`,
`u' = 25·tanh(u·fl(1/25))`, `h = bf16(g'·u')` (fp32, in this order). Routed experts compute SiTU inside the MXFP4 MoE
kernel (`4·tanh(g/4)·sigmoid(g)·25·tanh(u/25)` in the engine; the reference backend's form is not in our sources).

Layer 0: `down_proj(SiTU(gate_proj(x), up_proj(x)))`, FP8 block linears.

MoE (`block_sparse_moe`):
```
l0, l1 = x @ gate.weightᵀ, x @ gate.weight_1ᵀ     # fp32 [512] each
s      = sigmoid(l0) + sigmoid(l1)
ids    = top10(s + e_score_correction_bias)          # slots in torch.topk(sorted=False) order: above the 10th value
                                                   # by id, then ties by id (vLLM-ERNIE's routing kernel)
w      = s[ids] / max(Σ s[ids], 1e-20) · routed_scaling_factor_param[ids]   # Σ: tree over 16 zero-padded slots
lat    = fc1_latent_proj(x)                        # [2048]  FP8 block
r      = Σ_k w_k · E_ids[k](lat)                   # expert: w2(SiTU(w1 lat, w3 lat)), MXFP4 W4A8
y      = fc2_latent_proj(RMSNorm_latent_norm(r)) + shared(x)     # shared: SiTU-GLU 4096 → 2048 → 4096, FP8 block
```
At TP = 2 (vLLM's form, the engine's too): each rank's partial r (its half of every expert's intermediate) and its
partial shared output are rounded to bf16 and all-reduced in bf16; `y = bf16(bf16 fc2 out + bf16 shared)`.
W4A8 expert GEMMs: weights e2m1 (int8 storage, two per byte, low nibble first) with UE8M0 per 32 along K
(`weight_scale_inv` F8_E8M0); the activation is MXFP8 (e4m3, UE8M0 per 32, `2^ceil(log2(amax/448))`); fp32 sums.

## DSpark draft (`model.mtp.*`)

The checkpoint's speculative drafter (vLLM-ERNIE `Ernie5_5DSparkForCausalLM`; plain PyTorch form
[tools/ref/dspark_ref.py](../tools/ref/dspark_ref.py); the engine's: `engine/src/model/dspark.cc`).

```
aux_L   = bf16(Σ_j R_j / 4)  of the residual after target layers L = 40, 41, 42   (vLLM aux layers 41, 42, 43)
main_x  = RMSNorm_feature_norm(feature_projection([aux40 ‖ aux41 ‖ aux42]))     # FP8 block [4096, 12288]
ctx_i   = KV row(rope(RMSNorm_kv_i(linear_kv_proj_i(main_x)), pos))           # per stage i, HCA row format
```
A draft at anchor P (the last emitted token t, not yet through the target), N rows (block size; the checkpoint's 7):
ids `[t, 1024 × (N−1)]` at positions P … P+N−1 → the target's embedding, 4 equal streams → 3 stages, each an ERNIE MoE
layer (mHC as the target — stage 0's first pre on the repeated embedding itself, the unfused form; HCA-style attention
without compressor: q / kv / gate / VHA / wo_a / o_proj as HCA; latent MoE with MXFP4 experts) whose attention keys
are the context rows P−128 … P−1 plus all N block rows (non-causal), sink softmax, scale 512^-0.5. RoPE: plain at
θ = 8e6 (`build_deepseek_v4_rope(compress_ratio = 1)`: YaRN formula with factor 1), fp32 cos / sin, interleaved,
on the last 64 dims (q, kv rows, inverse on o). Head: `mhc_contract` → `norm` → the target's `lm_head` (fp32).
Left to right, i = 0 … N−1: `logits_i = multimax(base_i + markov_w2 · markov_w1[:, prev])` (prev = t, then the
previous draft; w1 [256, V], w2 [V, 256], bf16 weights, fp32 sums), draft_i = argmax (greedy) or a draw.
Confidence (adaptive verify length): `conf_i = proj · [h_i, markov_w1[:, prev_i]]` (h_i the head's input before its
norm), the logit of P(draft i accepted | drafts before it were).

## Sampling defaults

`generation_config`: temperature 1.0, top_p 0.95, eos [2, 8]. Thinking is always on (README); the chat template and
output parser come from the checkpoint's `encoding_ernie.py` / `decoding_ernie.py`.
