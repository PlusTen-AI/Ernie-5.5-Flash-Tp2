#!/usr/bin/env python3
"""dspark_ref.py — the DSpark draft of ERNIE 5.5 Flash in plain PyTorch: the reference for the engine's DSpark.

Written from the reference serving code (Apache-2.0): vLLM-ERNIE @ 4a971d3 src/vllm_ernie/models/ernie5_5/nvidia/
dspark.py (the draft model) on vLLM (spec_decode/dspark/speculator.py: anchor-first block + sequential Markov
sampling; spec_decode/dflash/speculator.py: the block's ids / positions; attention/backends/mla/sparse_swa.py: the
non-causal SWA indices of the block; models/deepseek_v4/nvidia/dspark.py: the context-KV insert;
model_executor/models/qwen3_dspark.py: the Markov head; spec_decode/eagle/eagle3_utils.py: the aux layer ids;
vllm_ernie patches.py: multimax after the Markov bias). Numerics as tools/ref/ernie_ref.py (whose helpers it uses):
fp32 math with the reference's quantization steps (FP8 block linears, MXFP4 W4A8 experts, FP8 KV rows).

One drafting pass at anchor position P (vLLM's sample_from_anchor layout, N = num_speculative_tokens, default
dspark_block_size 7):
  context   every committed position c < P has main_x[c] = RMSNorm_feature_norm(feature_projection(
            cat(aux40[c], aux41[c], aux42[c]))), aux_L = mean over the 4 mHC streams of the bf16 residual after
            target layer L (vLLM captures idx+1 ∈ {41, 42, 43}); per draft stage i its context KV row
            = FP8 row(rope(kv_layernorm_i(linear_kv_proj_i(main_x[c])), c))
  query     N tokens at positions P … P+N−1: ids [t_P (the target's last token, the "anchor"), 1024 × (N−1)]
            (dspark_noise_token_id); embedding (the target's) → 4 equal mHC streams → 3 stages, each an ERNIE MoE
            decoder layer (mHC pre/post, HCA-style attention without compressor, latent MoE with MXFP4 experts)
  attention every block query sees the same keys: context rows max(P−128, 0) … P−1 plus all N block rows
            (non-causal inside the block); SWA rope: plain RoPE, theta = rope_theta 8e6 (vLLM-ERNIE; SGLang uses
            the HCA layers' YaRN at 160k — rope="sglang" selects that, for comparison only)
  head      mHC head contraction (model.mtp.mhc_contract) → model.mtp.norm → the target's lm_head (fp32 logits)
  sampling  left to right, i = 0 … N−1: logits_i + markov_w2 · markov_w1[prev] (prev = t_P, then the previous draft),
            then the target's multimax, then argmax (greedy) → draft token for position P+i+1
The confidence head (adaptive verification only) is not part of greedy drafting and is not implemented."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ernie_ref as ref  # noqa: E402

F32, BF = torch.float32, torch.bfloat16
DEV = ref.DEV
H, HC, V = ref.H, ref.HC, ref.V
NOISE = 1024          # dspark_noise_token_id (the block's mask token)
BLOCK = 7             # dspark_block_size (vLLM's default num_speculative_tokens for the bundled draft)
TARGET_LAYERS = (40, 41, 42)  # dspark_target_layer_ids: residual AFTER these target layers
WINDOW = 128          # sliding_window
STAGES = 3
_ALIAS = 1000         # ernie_ref.moe names experts "model.layers.<L>."; stage i is served as layer _ALIAS + i


class MtpCkpt:
    """The checkpoint seen by ernie_ref's layer helpers: "model.layers.<1000+i>." is "model.mtp.<i>.", and every
    tensor read is kept on the GPU (the 3 × 512 MXFP4 experts are ~9.8 GB raw), so repeated passes read the disk once."""

    def __init__(self, ck):
        self.ck, self.cache = ck, {}

    def get(self, name, dtype=None):
        for i in range(STAGES):
            p = f"model.layers.{_ALIAS + i}."
            if name.startswith(p):
                name = f"model.mtp.{i}." + name[len(p):]
        if name not in self.cache:
            self.cache[name] = self.ck.get(name)
        t = self.cache[name]
        return t if dtype is None else t.to(dtype)


def multimax(lg, r, t):
    """The target lm_head's multimax (fp32, vLLM-ERNIE's op order)."""
    return (lg + t[0] * torch.relu(r[0] - lg) + t[1] * torch.relu(lg - r[1])
            + t[2] * torch.relu(r[2] - lg).square() + t[3] * torch.relu(lg - r[3]).square())


class Drafter:
    def __init__(self, ck, rope="vllm"):
        self.ck = MtpCkpt(ck)
        self.w = [ref.W(self.ck, f"model.mtp.{i}.") for i in range(STAGES)]
        self.top = ref.W(self.ck, "model.mtp.")
        self.embed = ck.get("model.embed_tokens.weight")                   # bf16 [V, H] (the target's)
        self.head = ck.get("lm_head.weight").float()                       # fp32 [V, H] (head_dtype float32)
        self.mm_r = ck.get("model.lm_head.multimax_ranges", F32)
        self.mm_t = ck.get("model.lm_head.multimax_ts", F32)
        self.mk1 = ck.get("model.mtp.markov_head.markov_w1.weight").t().contiguous()  # stored [r, V] → [V, r] bf16
        # bf16 [V, r] held as fp32: bf16 inputs with fp32 accumulation, as vLLM's mm(out_dtype=fp32) head
        self.mk2f = ck.get("model.mtp.markov_head.markov_w2.weight").float()
        self.set_rope(rope)

    def set_rope(self, rope):
        """vllm: vLLM-ERNIE's build_deepseek_v4_rope(compress_ratio=1) — deepseek_yarn with factor 1 at rope_theta
        (plain RoPE), fp32 cos/sin as the HCA layers; sglang: compress_rope_theta with the HCA layers' YaRN."""
        self.rope = rope
        self.inv = (ref.inv_freq_yarn(theta=8000000.0, factor=1.0, orig=1048576) if rope == "vllm"
                    else ref.inv_freq_yarn())

    # ---- context ------------------------------------------------------------------------------------------------
    def features(self, aux):
        """aux bf16 [n, 3·H] (cat of the stream-mean residuals after layers 40, 41, 42) → main_x bf16 [n, H]."""
        return ref.rms(ref.lin8(aux, self.top.lin("feature_projection")), self.top("feature_norm.weight"))

    def context_kv(self, main_x, pos):
        """Per stage, the context KV cache rows (fp32 holding the dequantized FP8 / bf16 values) [n, 512]."""
        cs = ref.cos_sin(self.inv, pos)
        out = []
        for w in self.w:
            kv = ref.rms(ref.lin8(main_x, w.lin("self_attn.linear_kv_proj")), w("self_attn.kv_layernorm.weight"))
            kvr = kv.float()
            out.append(ref.kv_row_v4(torch.cat([kvr[:, :448], ref.rope_gptj(kvr[:, 448:], cs)], -1)))
        return out

    # ---- block forward ------------------------------------------------------------------------------------------
    def attn(self, x, w, cs, ctx_keys, ctx_ok, N):
        """x bf16 [B·N, H]; ctx_keys fp32 [B, 128, 512] / ctx_ok bool [B, 128]: the window before each block."""
        T = x.shape[0]
        B = T // N
        qr = ref.rms(ref.lin8(x, w.lin("self_attn.linear_q_down_proj")), w("self_attn.q_layernorm.weight"))
        kv = ref.rms(ref.lin8(x, w.lin("self_attn.linear_kv_proj")), w("self_attn.kv_layernorm.weight"))
        q = ref.lin8(qr, w.lin("self_attn.linear_q_up_proj")).view(T, 64, 512).float()
        q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + ref.EPS)
        q = ref.bf(torch.cat([q[..., :448], ref.rope_gptj(q[..., 448:], cs)], -1))
        kvr = kv.float()
        own = ref.kv_row_v4(torch.cat([kvr[:, :448], ref.rope_gptj(kvr[:, 448:], cs)], -1))  # the block's own rows
        keys = torch.cat([ctx_keys, own.view(B, N, 512)], 1)                                  # [B, 128 + N, 512]
        ok = torch.cat([ctx_ok, torch.ones(B, N, dtype=torch.bool, device=DEV)], 1)
        s = torch.einsum("bnhd,bkd->bnhk", q.view(B, N, 64, 512).float(), keys) * (512 ** -0.5)
        s = s.masked_fill(~ok[:, None, None, :], float("-inf"))
        pr = ref.softmax_sink(s, w("self_attn.core_attention.attn_sink", F32))
        o = ref.bf(torch.einsum("bnhk,bkd->bnhd", pr, keys)).reshape(T, 64, 512)
        o2 = ref.vha(o, w).float()
        o2 = torch.cat([o2[..., :448], ref.rope_gptj(o2[..., 448:], cs, inverse=True)], -1)
        oq = ref.act_q8(o2.reshape(T, 64 * 512))
        wa = w.lin("self_attn.linear_o_group_proj").view(8, 1024, 4096)
        z = ref.bf(torch.einsum("tgk,grk->tgr", oq.view(T, 8, 4096), wa)).reshape(T, 8192)
        gate = ref.bf(torch.sigmoid(ref.lin8(qr, w.lin("self_attn.gate_proj")).float()))
        return ref.lin8(ref.bf(z.float() * gate.float()), w.lin("self_attn.o_proj"))

    def forward(self, tok, pos, ctx, ctx_ok, N):
        """tok / pos [B·N] (blocks of N rows); ctx: per stage [B, 128, 512]; → head hidden bf16 [B·N, H], the
        stages' expert ids [STAGES, B·N, 10]."""
        T = tok.numel()
        cs = ref.cos_sin(self.inv, pos)
        R = self.embed[tok][:, None, :].expand(T, HC, H).contiguous()
        xr = R.float()  # stage 0's pre reads the repeated embedding itself (vLLM: mhc_pre on the 3-D input)
        routes = []
        for i, w in enumerate(self.w):
            post, comb, inp = ref.mhc_pre(R, w, "attn", w("input_layernorm.weight"), xr)
            a = self.attn(inp, w, cs, ctx[i], ctx_ok, N)
            R, xr = ref.mhc_post(a, R, post, comb)
            post, comb, inp = ref.mhc_pre(R, w, "ffn", w("post_attention_layernorm.weight"), xr)
            f, route = ref.moe(inp, w, self.ck, _ALIAS + i)
            R, xr = ref.mhc_post(f, R, post, comb)
            routes.append(route)
        x = R.reshape(T, HC * H).float()
        fn = self.top("mhc_contract.hc_head_fn", F32)  # stored [16384, 4]
        mixes = (x @ fn) * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + ref.EPS)
        pre = torch.sigmoid(mixes * self.top("mhc_contract.hc_head_scale", F32)[0] +
                            self.top("mhc_contract.hc_head_base", F32)) + ref.EPS
        h = ref.bf((pre[..., None] * R.float()).sum(1))
        return ref.rms(h, self.top("norm.weight")), torch.stack(routes)

    @torch.no_grad()
    def draft(self, anchor_tok, anchor_pos, ctx_rows, ctx_pos_ok, N=BLOCK):
        """B passes at once (each independent): anchor_tok / anchor_pos [B]; ctx_rows: per stage [B, 128, 512] (the
        context KV rows at positions P−128 … P−1), ctx_pos_ok [B, 128]. Returns drafts [B, N] (tokens for positions
        P+1 … P+N) and the stages' expert ids [STAGES, B, N, 10]."""
        B = anchor_tok.numel()
        tok = torch.full((B, N), NOISE, dtype=torch.long, device=DEV)
        tok[:, 0] = anchor_tok
        pos = anchor_pos[:, None] + torch.arange(N, device=DEV)[None, :]
        h, routes = self.forward(tok.reshape(-1), pos.reshape(-1), ctx_rows, ctx_pos_ok, N)
        h = h.view(B, N, H)
        prev = anchor_tok.clone()
        out = torch.empty(B, N, dtype=torch.long, device=DEV)
        for i in range(N):
            base = h[:, i].float() @ self.head.t()                                  # fp32 [B, V]
            bias = self.mk1[prev].float() @ self.mk2f.t()                          # Markov bias from prev
            lg = multimax(base + bias, self.mm_r, self.mm_t)
            prev = lg.argmax(-1)
            out[:, i] = prev
        return out, routes.view(STAGES, B, N, 10)
