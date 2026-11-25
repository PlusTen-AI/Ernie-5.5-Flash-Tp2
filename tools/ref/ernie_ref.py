#!/usr/bin/env python3
"""ernie_ref.py — reference forward of ERNIE 5.5 Flash in plain PyTorch: the answer key of the correctness gate.

Written from the reference serving code (vLLM-ERNIE @ 4a971d3 on vLLM 0.32.0; the math is docs/model.md), not from
this engine. One sequence from position 0 at a time, every layer in turn (its weights read from the checkpoint, then
dropped), fp32 math with the reference's quantization steps kept: FP8 block linears (activations 1×128 UE8M0), MXFP4
W4A8 experts (activations MXFP8), FP8 KV rows (HCA: UE8M0 per 64 + bf16 rope; DSA: pow2 per 128 + bf16 rope), the
MXFP4 indexer. Attention is computed densely with masks (the same keys the sparse kernels gather).

  python3 tools/ref/ernie_ref.py --model <ckpt> --prompts <prompts.json> --out <dir> [--names a,b] [--layers N]

Writes <out>/<name>.safetensors per prompt:
  ids [T] · pos_sel [P] (positions with saved states) · resid [L, P, 4, H] bf16 (residual after each layer) ·
  attn [L, P, H] / ffn [L, P, H] bf16 (sublayer outputs) · logits_sel [P, V] fp32 · top_ids / top_vals [T, 8] ·
  route_ids [L, T, 10] (MoE layers; -1 for layer 0) · dsa_topk [6, P, 2048] (indexer picks, -1 padded)
and prints the prompt's next-token accuracy and perplexity (a sanity check of the reference itself)."""
import argparse
import json
import math
import os
import time

import torch
from safetensors.torch import save_file

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
F32, BF = torch.float32, torch.bfloat16
DEV = "cuda"
H, HC, V = 4096, 4, 201216
EPS = 1e-6
DSA_LAYERS = (7, 14, 21, 28, 35, 42)


# ---------------------------------------------------------------------------------------------------------- checkpoint
class Ckpt:
    """Tensors read with pread into fresh memory and their file pages dropped right after (no mmap): on GB10 the GPU
    takes system memory, and cached shard pages starve it (the run was OOM-killed with mmapped shards)."""
    DT = {"BF16": torch.bfloat16, "F32": torch.float32, "F8_E4M3": torch.float8_e4m3fn, "I8": torch.int8,
          "F8_E8M0": torch.uint8, "U8": torch.uint8, "I32": torch.int32, "I64": torch.int64}

    def __init__(self, d):
        self.d = d
        self.map = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.hdr = {}

    def _h(self, f):
        if f not in self.hdr:
            fd = os.open(os.path.join(self.d, f), os.O_RDONLY)
            n = int.from_bytes(os.pread(fd, 8, 0), "little")
            self.hdr[f] = (fd, json.loads(os.pread(fd, n, 8)), 8 + n)
        return self.hdr[f]

    def get(self, name, dtype=None):
        fd, h, base = self._h(self.map[name])
        m = h[name]
        a, b = m["data_offsets"]
        buf = bytearray(b - a)
        mv, done = memoryview(buf), 0
        while done < b - a:
            k = os.preadv(fd, [mv[done:]], base + a + done)
            if k <= 0:
                raise IOError(name)
            done += k
        os.posix_fadvise(fd, base + a, b - a, os.POSIX_FADV_DONTNEED)
        t = torch.frombuffer(buf, dtype=self.DT[m["dtype"]]).reshape(m["shape"]).to(DEV)
        if m["dtype"] == "F8_E8M0":
            t = t.view(torch.uint8)
        return t if dtype is None else t.to(dtype)


# ---------------------------------------------------------------------------------------------------------- numerics
def bf(x):
    return x.to(BF)


def rms(x, w, eps=EPS):
    x = x.float()
    return bf(x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w.float())


def e4m3(x):  # x already scaled into ±448
    return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()


def act_q8(x, g=128):
    """Activation → FP8 e4m3 per g along the last dim, scale 2^ceil(log2(max(amax,1e-10)/448)); returns q·s (fp32)."""
    shp = x.shape
    x = x.float().reshape(*shp[:-1], shp[-1] // g, g)
    amax = x.abs().amax(-1, keepdim=True).clamp_min(1e-10)
    s = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    return (e4m3(x / s) * s).reshape(shp)


def dq_block(w8, s):
    n, k = w8.shape
    sc = s.float().repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)[:, :k]
    return w8.float() * sc


class W:
    """One layer's weights, read on first use."""

    def __init__(self, ck, prefix):
        self.ck, self.p, self.c = ck, prefix, {}

    def __call__(self, name, dtype=None):
        key = (name, dtype)
        if key not in self.c:
            self.c[key] = self.ck.get(self.p + name, dtype)
        return self.c[key]

    def lin(self, name):  # FP8 block weight, dequantized fp32 [N, K]
        key = (name, "dq")
        if key not in self.c:
            wn, sn = (name, name + "_scale_inv") if name.endswith("linear_o_group_proj") else \
                (name + ".weight", name + ".weight_scale_inv")
            self.c[key] = dq_block(self.ck.get(self.p + wn), self.ck.get(self.p + sn))
        return self.c[key]


def lin8(x, wd):
    return bf(act_q8(x) @ wd.t())


# rope ------------------------------------------------------------------------------------------------------------------
def inv_freq_plain(theta, d=64):
    return 1.0 / theta ** (torch.arange(0, d, 2, dtype=F32) / d)


def inv_freq_yarn(theta=160000.0, d=64, factor=16.0, orig=65536, b_fast=32, b_slow=1):
    pos_freqs = theta ** (torch.arange(0, d, 2, dtype=F32) / d)
    extra, inter = 1.0 / pos_freqs, 1.0 / (factor * pos_freqs)

    def corr(r):
        return d * math.log(orig / (r * 2 * math.pi)) / (2 * math.log(theta))

    low, high = max(math.floor(corr(b_fast)), 0), min(math.ceil(corr(b_slow)), d - 1)
    if low == high:
        high += 0.001
    mask = 1 - ((torch.arange(d // 2, dtype=F32) - low) / (high - low)).clamp(0, 1)
    return inter * (1 - mask) + extra * mask


def cos_sin(inv, pos):
    f = pos.to(F32)[:, None] * inv.to(DEV)[None, :]
    return f.cos(), f.sin()


def cos_sin_dsa(pos):
    """DSA and indexer: vLLM's get_rope cache in the model dtype, built on the GPU — fp32 inv_freq, angles
    fl(pos·inv_freq), fp32 cos / sin, then rounded to bf16 (HCA keeps fp32)."""
    inv = 1.0 / (8000000.0 ** (torch.arange(0, 64, 2, dtype=F32, device=DEV) / 64))
    return tuple(bf(c).float() for c in cos_sin(inv, pos))


def rope_gptj(x, cs, inverse=False):
    """Interleaved pairs over the last dim of x (fp32, [..., 64]); cs = (cos, sin) [T, 32] for x's leading T."""
    cos, sin = cs
    shape = [x.shape[0]] + [1] * (x.dim() - 2) + [cos.shape[-1]]
    cos, sin = cos.view(shape), sin.view(shape)
    if inverse:
        sin = -sin
    e, o = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = e * cos - o * sin
    out[..., 1::2] = o * cos + e * sin
    return out


def rope_neox(x, cs):
    cos, sin = cs
    shape = [x.shape[0]] + [1] * (x.dim() - 2) + [cos.shape[-1]]
    cos, sin = cos.view(shape), sin.view(shape)
    h = x.shape[-1] // 2
    a, b = x[..., :h], x[..., h:]
    return torch.cat([a * cos - b * sin, b * cos + a * sin], -1)


# KV rows ---------------------------------------------------------------------------------------------------------------
def kv_row_v4(x):
    """HCA cache row (fp32 in, [N, 512]): bf16, nope 448 → e4m3 UE8M0 per 64 (amax ≥ 1e-4), rope 64 stays bf16."""
    x = bf(x).float()
    nope = x[:, :448].reshape(-1, 7, 64)
    e = torch.ceil(torch.log2(nope.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448.0))
    s = torch.exp2(e)
    return torch.cat([(e4m3(nope / s) * s).reshape(-1, 448), x[:, 448:]], -1)


def kv_row_mla(kv_c):
    """DSA cache kv_c (bf16 [N, 512]): e4m3, one pow2 fp32 scale per 128 = 2^ceil(log2(max(amax/448, 1e-4)))."""
    x = kv_c.float().reshape(-1, 4, 128)
    s = torch.exp2(torch.ceil(torch.log2((x.abs().amax(-1, keepdim=True) / 448.0).clamp_min(1e-4))))
    return (e4m3(x / s) * s).reshape(-1, 512)


E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=F32)


def mxfp4_q(x):
    """vLLM downcast_to_mxfp + dequant (fp32 in [..., D], D % 32 == 0)."""
    shp = x.shape
    x = x.float().reshape(*shp[:-1], shp[-1] // 32, 32)
    amax = x.abs().amax(-1, keepdim=True)
    bits = amax.view(torch.int32)
    rounded = ((bits + 0x200000) & 0x7F800000).view(F32)
    rounded = torch.where(rounded == 0, torch.full_like(rounded, 2.0 ** -126), rounded)
    e = (torch.floor(torch.log2(rounded)) - 2).clamp(-127, 127)
    s = torch.exp2(e)
    v = (x / s)
    # RNE onto the e2m1 grid (|v| ≤ 6)
    grid = E2M1.to(x.device)
    a = v.abs().clamp(max=6.0)
    idx = torch.bucketize(a, (grid[1:] + grid[:-1]) / 2, right=False)  # ties → lower index; fix ties to even below
    mids = (grid[1:] + grid[:-1]) / 2
    tie = (idx < 7) & (a == mids[idx.clamp(max=6)])
    # at a tie between grid[i] and grid[i+1], bucketize(right=False) gives i; choose the even code
    idx = torch.where(tie & (idx % 2 == 1), idx + 1, idx)
    q = grid[idx] * torch.sign(v)
    return (q * s).reshape(shp)


def dq_mxfp4(w, s):
    """Expert weight int8 [N, K/2] (low nibble = even k) + e8m0 [N, K/32] → fp32 [N, K]."""
    b = w.view(torch.uint8)
    lo, hi = (b & 0xF).long(), (b >> 4).long()
    lut = torch.cat([E2M1, -E2M1]).to(w.device)
    vals = torch.stack([lut[lo], lut[hi]], -1).reshape(w.shape[0], -1)
    sc = torch.exp2(s.view(torch.uint8).float() - 127.0).repeat_interleave(32, 1)
    return vals * sc


def situ(g, u):
    """Dense FFN and shared expert: vLLM's situ_and_mul (csrc/activation_kernels.cu situ_activation, beta 4,
    linear_beta 25), the same operations in the same order (fp32 on the GPU)."""
    g, u = g.float(), u.float()
    return (0.5 * 4.0) * torch.tanh(g * 0.25) * (1.0 + torch.tanh(g * 0.5)) * (25.0 * torch.tanh(u * (1.0 / 25.0)))


def situ_routed(g, u):
    """Routed experts: the reference's MXFP4 MoE backend computes SiTU inside its kernel (not in the sources we have);
    the engine's W4A8 kernel uses this form."""
    g, u = g.float(), u.float()
    return 4.0 * torch.tanh(g / 4.0) * torch.sigmoid(g) * (25.0 * torch.tanh(u / 25.0))


HAD = None


def hadamard128(x):
    global HAD
    if HAD is None:
        h = torch.ones(1, 1)
        while h.shape[0] < 128:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        HAD = (h / 128 ** 0.5).to(DEV)
    return bf(x.float() @ HAD)


# ---------------------------------------------------------------------------------------------------------- blocks
def mhc_pre(R, w, kind, normw, x=None):
    """R bf16 [T, 4, H] (the stored residual) → post [T, 4], comb [T, 4, 4] (fp32), layer input bf16 [T, H].
    As vLLM-ERNIE's fused post + pre (its decode path, used here for every T): the mixer and its RMS factor read x, the
    preceding post's UNROUNDED fp32 output [T, 4, H]; the collapse reads the bf16 R. x = None is layer 0's first pre
    (R = [e, e, e, e]): the mixer summed over the 4 streams first, then applied to e over H."""
    pre_name = "self_attention_hyper_connection" if kind == "attn" else "mlp_hyper_connection"
    fn = w(f"{pre_name}.mapping_proj.weight", F32).t()  # [24, 16384]
    base = w(f"{pre_name}.bias", F32)
    sc = [w(f"{pre_name}.alpha_{a}", F32)[0] for a in ("pre", "post", "res")]
    T = R.shape[0]
    if x is None:
        e = R[:, 0].float()
        fnb = fn.contiguous().view(24, HC, H).sum(1)
        mixes = (e @ fnb.t()) * (1.0 / (torch.sqrt(e.pow(2).sum(-1, keepdim=True) / H) + EPS))
    else:
        x = x.reshape(T, HC * H).float()
        mixes = (x @ fn.t()) * (1.0 / (torch.sqrt(x.pow(2).sum(-1, keepdim=True) / (HC * H)) + EPS))
    pre = torch.sigmoid(mixes[:, :4] * sc[0] + base[:4]) + EPS
    post = torch.sigmoid(mixes[:, 4:8] * sc[1] + base[4:8]) * 2.0
    comb = torch.softmax(mixes[:, 8:].view(T, 4, 4) * sc[2] + base[8:].view(1, 4, 4), -1) + EPS
    comb = comb / (comb.sum(-2, keepdim=True) + EPS)
    for _ in range(19):
        comb = comb / (comb.sum(-1, keepdim=True) + EPS)
        comb = comb / (comb.sum(-2, keepdim=True) + EPS)
    ol = (pre[..., None] * R.float()).sum(1)
    sumsq = ol.pow(2).sum(-1, keepdim=True)
    inp = bf(bf(ol).float() * torch.rsqrt(sumsq / H + EPS) * normw.float())
    return post, comb, inp


def mhc_post(out, R, post, comb):
    """R'_j = Σ_i comb[i][j]·R_i + post_j·out as vLLM-ERNIE's fused kernel: post·out first, then the four terms in
    order as FMAs (fp64 stands in for fma's single rounding). Returns (R' bf16, the unrounded fp32 sum for the next
    pre)."""
    r = post[..., None] * out.float()[:, None, :]
    for i in range(HC):
        r = (r.double() + comb[:, i, :, None].double() * R[:, i].double()[:, None, :]).float()
    return bf(r), r


def vha(o, w):
    U, Vv = w("self_attn.vha_postmix_U", F32), w("self_attn.vha_postmix_V", F32)  # [64, 16]
    z = torch.einsum("thd,hr->trd", o.float(), U)
    return bf(o.float() + torch.einsum("trd,hr->thd", z, Vv))


def softmax_sink(s, sink):
    """s [..., heads, K] fp32 with -inf for masked keys, sink [heads] → probabilities (sink takes its share)."""
    m = torch.maximum(s.amax(-1), sink)
    e = torch.exp(s - m[..., None])
    return e / (e.sum(-1) + torch.exp(sink - m))[..., None]


def hca(x, w, pos, cs, qchunk=256):
    T = x.shape[0]
    qr = rms(lin8(x, w.lin("self_attn.linear_q_down_proj")), w("self_attn.q_layernorm.weight"))
    kv = rms(lin8(x, w.lin("self_attn.linear_kv_proj")), w("self_attn.kv_layernorm.weight"))
    q = lin8(qr, w.lin("self_attn.linear_q_up_proj")).view(T, 64, 512).float()
    q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + EPS)
    q = bf(torch.cat([q[..., :448], rope_gptj(q[..., 448:], cs)], -1))
    kvr = kv.float()
    kvr = torch.cat([kvr[:, :448], rope_gptj(kvr[:, 448:], cs)], -1)
    win = kv_row_v4(kvr)  # [T, 512] fp32 (bf16-exact values)
    # compressor
    cp = "self_attn.core_attention.compressor."
    ckv = x.float() @ w(cp + "linear_wkv.weight", F32).t()
    csc = x.float() @ w(cp + "linear_wgate.weight", F32).t() + w(cp + "ape", F32)[pos % 128]
    G = T // 128
    comp = torch.zeros(0, 512, device=DEV)
    if G:
        sm = torch.softmax(csc[:G * 128].view(G, 128, 512), 1)
        c = (sm * ckv[:G * 128].view(G, 128, 512)).sum(1)
        cn = c * torch.rsqrt(c.pow(2).mean(-1, keepdim=True) + EPS) * w(cp + "norm.weight", F32)
        gpos = torch.arange(G, device=DEV) * 128
        ccs = (cs[0][gpos], cs[1][gpos])
        comp = kv_row_v4(torch.cat([cn[:, :448], rope_gptj(cn[:, 448:], ccs)], -1))
    keys = torch.cat([win, comp], 0)  # [T + G, 512]
    sink = w("self_attn.core_attention.attn_sink", F32)
    o = torch.empty(T, 64, 512, device=DEV, dtype=BF)
    kidx = torch.arange(T + G, device=DEV)
    for a in range(0, T, qchunk):
        b = min(T, a + qchunk)
        p = pos[a:b, None]
        is_win = kidx[None, :] < T
        ok_win = is_win & (kidx[None, :] <= p) & (kidx[None, :] >= p - 127)
        ok_cmp = (~is_win) & ((kidx[None, :] - T) < (p + 1) // 128)
        mask = ok_win | ok_cmp  # [tc, K]
        s = torch.einsum("thd,kd->thk", q[a:b].float(), keys) * (512 ** -0.5)
        s = s.masked_fill(~mask[:, None, :], float("-inf"))
        pr = softmax_sink(s, sink)
        o[a:b] = bf(torch.einsum("thk,kd->thd", pr, keys))
    o2 = vha(o, w).float()
    o2 = torch.cat([o2[..., :448], rope_gptj(o2[..., 448:], cs, inverse=True)], -1)
    oq = act_q8(o2.reshape(T, 64 * 512))
    wa = w.lin("self_attn.linear_o_group_proj").view(8, 1024, 4096)
    z = bf(torch.einsum("tgk,grk->tgr", oq.view(T, 8, 4096), wa)).reshape(T, 8192)
    gate = bf(torch.sigmoid(lin8(qr, w.lin("self_attn.gate_proj")).float()))
    return lin8(bf(z.float() * gate.float()), w.lin("self_attn.o_proj")), None


def dsa(x, w, pos, cs8, csn, qchunk=256):
    T = x.shape[0]
    qc = rms(lin8(x, w.lin("self_attn.q_a_proj")), w("self_attn.q_a_layernorm.weight"))
    kvl = lin8(x, w.lin("self_attn.kv_a_proj_with_mqa"))
    kvc = rms(kvl[:, :512], w("self_attn.kv_a_layernorm.weight"))
    kpe = bf(rope_gptj(kvl[:, 512:].float(), cs8))
    gate = bf(torch.sigmoid(lin8(qc, w.lin("self_attn.gate_proj")).float()))
    q = lin8(qc, w.lin("self_attn.q_b_proj")).view(T, 64, 256)
    qpe = bf(rope_gptj(q[..., 192:].float(), cs8))
    # indexer
    ip = "self_attn.core_attention.indexer."
    qi = lin8(qc, w.lin(ip + "wq_b")).view(T, 64, 128).float()
    kw = bf(x.float() @ torch.cat([w(ip + "wk.weight", F32), w(ip + "weights_proj.weight", F32)], 0).t())
    k = bf(torch.nn.functional.layer_norm(kw[:, :128].float(), (128,), w(ip + "k_norm.weight", F32),
                                          w(ip + "k_norm.bias", F32), 1e-6))
    wts = kw[:, 128:].float() * (128 ** -0.5 * 64 ** -0.5)
    qi = bf(torch.cat([rope_neox(qi[..., :64], csn), qi[..., 64:]], -1))
    kf = k.float()
    k = bf(torch.cat([rope_neox(kf[:, :64], csn), kf[:, 64:]], -1))
    qi4, k4 = mxfp4_q(hadamard128(qi).float()), mxfp4_q(hadamard128(k).float())
    # MLA
    kvb = w.lin("self_attn.kv_b_proj").view(64, 448, 512)
    wuk, wuv = bf(kvb[:, :192]).float(), bf(kvb[:, 192:]).float()
    qlat = bf(torch.einsum("thn,hnc->thc", q[..., :192].float(), wuk))
    kc = kv_row_mla(kvc)  # [T, 512] fp32
    sink = w("self_attn.core_attention.softmax_offset", F32)
    o = torch.empty(T, 64, 256, device=DEV, dtype=BF)
    topk_all = torch.full((T, 2048), -1, dtype=torch.int32, device=DEV)
    tk = torch.arange(T, device=DEV)
    for a in range(0, T, qchunk):
        b = min(T, a + qchunk)
        p = pos[a:b, None]
        lg = (torch.relu(torch.einsum("thd,kd->thk", qi4[a:b], k4)) * wts[a:b, :, None]).sum(1)  # [tc, T]
        cand = tk[None, :] <= p - 128
        lg = lg.masked_fill(~cand, float("-inf"))
        n = cand.sum(-1)
        kk = min(2048, T)
        top = lg.topk(kk, -1).indices
        valid = torch.arange(kk, device=DEV)[None, :] < n[:, None].clamp(max=2048)
        # invalid picks go to a spare column T (dropped): pointing them at 0 let their False overwrite position 0
        sel = torch.zeros(b - a, T + 1, dtype=torch.bool, device=DEV)
        sel.scatter_(1, torch.where(valid, top, torch.full_like(top, T)), True)
        sel = sel[:, :T] | ((tk[None, :] <= p) & (tk[None, :] >= p - 127))
        tsel = torch.where(valid, top, torch.full_like(top, -1)).sort(-1, descending=True).values[:, :2048]
        topk_all[a:b, :tsel.shape[1]] = tsel.int()
        s = (torch.einsum("thc,kc->thk", qlat[a:b].float(), kc) +
             torch.einsum("thr,kr->thk", qpe[a:b].float(), kpe.float())) * (256 ** -0.5)
        s = s.masked_fill(~sel[:, None, :], float("-inf"))
        pr = softmax_sink(s, sink)
        olat = bf(torch.einsum("thk,kc->thc", pr, kc))
        o[a:b] = bf(torch.einsum("thc,hvc->thv", olat.float(), wuv))
    o2 = vha(o, w)
    return lin8(bf(o2.reshape(T, 64 * 256).float() * gate.float()), w.lin("self_attn.o_proj")), topk_all


def dense_ffn(x, w, pre):
    g, u = lin8(x, w.lin(pre + "gate_proj")), lin8(x, w.lin(pre + "up_proj"))
    return lin8(bf(situ(g, u)), w.lin(pre + "down_proj"))


def moe(x, w, ck, layer):
    T = x.shape[0]
    mp = "block_sparse_moe."
    lg = x.float() @ torch.cat([w(mp + "gate.weight", F32), w(mp + "gate.weight_1", F32)], 0).t()
    # vLLM-ERNIE's routing kernel: top 10 of sigmoid sums + bias, slots in torch.topk(sorted=False) order (above the
    # 10th value by id, then ties by id), weights renormalized by a tree sum over 16 zero-padded slots
    s = torch.sigmoid(lg[:, :512]) + torch.sigmoid(lg[:, 512:])
    v, i = (s + w(mp + "e_score_correction_bias", F32)).sort(dim=-1, descending=True, stable=True)
    ids = i[:, :10]
    ids = ids.gather(1, torch.where(v[:, :10] > v[:, 9:10], ids, ids + 512).argsort(-1))
    wt = s.gather(1, ids)
    t = torch.nn.functional.pad(wt, (0, 6))
    t = t[:, :8] + t[:, 8:]
    t = t[:, :4] + t[:, 4:]
    t = t[:, :2] + t[:, 2:]
    wt = wt / (t[:, :1] + t[:, 1:]).clamp_min(1e-20) * w(mp + "gate.routed_scaling_factor_param", F32)[ids]
    lat = lin8(x, w.lin(mp + "fc1_latent_proj"))
    latq = act_q8(lat, 32)
    # vLLM at TP = 2 (fused_moe runner, shared experts with reduce_results=False): each rank holds half of every
    # expert's intermediate (rank r: [r·1024, (r+1)·1024)) and half of the shared expert's; each rank's routed latent
    # and shared output leave in bf16 and are all-reduced in bf16; latent_norm → fc2, then bf16 + bf16.
    acc = [torch.zeros(T, 2048, device=DEV) for _ in range(2)]
    base = f"model.layers.{layer}.{mp}experts."
    for e in ids.unique().tolist():
        rows, slot = torch.where(ids == e)
        ep = f"{base}{e}."
        w1 = dq_mxfp4(ck.get(ep + "w1.weight"), ck.get(ep + "w1.weight_scale_inv"))
        w3 = dq_mxfp4(ck.get(ep + "w3.weight"), ck.get(ep + "w3.weight_scale_inv"))
        w2 = dq_mxfp4(ck.get(ep + "w2.weight"), ck.get(ep + "w2.weight_scale_inv"))
        a = latq[rows]
        hq = act_q8(situ_routed(a @ w1.t(), a @ w3.t()), 32)
        for r in range(2):
            k = slice(r * 1024, (r + 1) * 1024)
            acc[r].index_add_(0, rows, wt[rows, slot, None] * (hq[:, k] @ w2[:, k].t()))
    lat = bf(bf(acc[0]).float() + bf(acc[1]).float())
    yr = lin8(rms(lat, w(mp + "latent_norm.weight")), w.lin(mp + "fc2_latent_proj"))
    g = lin8(x, w.lin(mp + "shared_experts.w1"))
    u = lin8(x, w.lin(mp + "shared_experts.w3"))
    h, w2s = bf(situ(g, u)), w.lin(mp + "shared_experts.w2")
    sh = bf(sum(lin8(h[:, r * 1024:(r + 1) * 1024], w2s[:, r * 1024:(r + 1) * 1024]).float() for r in range(2)))
    return bf(yr.float() + sh.float()), ids


# ---------------------------------------------------------------------------------------------------------- driver
def run(ck, ids, n_layers, sel_every=16, sel_tail=32):
    T = len(ids)
    pos = torch.arange(T, device=DEV)
    sel = sorted(set(range(0, T, sel_every)) | set(range(max(0, T - sel_tail), T)))
    psel = torch.tensor(sel, device=DEV)
    cs_hca = cos_sin(inv_freq_yarn(), pos)
    cs8 = cos_sin_dsa(pos)
    csn = cs8  # NeoX uses the same frequencies, half-split pairing
    emb = ck.get("model.embed_tokens.weight")
    x = emb[torch.tensor(ids, device=DEV)]
    del emb
    R = x[:, None, :].expand(T, HC, H).contiguous()
    xr = None  # the last post's unrounded fp32 output (None before layer 0)
    out = {"resid": [], "attn": [], "ffn": [], "route": [], "dsa": []}
    for L in range(n_layers):
        t0 = time.time()
        w = W(ck, f"model.layers.{L}.")
        post, comb, inp = mhc_pre(R, w, "attn", w("input_layernorm.weight"), xr)
        if L in DSA_LAYERS:
            a, tk = dsa(inp, w, pos, cs8, csn)
            out["dsa"].append(tk[psel])
        else:
            a, _ = hca(inp, w, pos, cs_hca)
        R, xr = mhc_post(a, R, post, comb)
        post, comb, inp = mhc_pre(R, w, "ffn", w("post_attention_layernorm.weight"), xr)
        if L == 0:
            f = dense_ffn(inp, w, "mlp.")
            route = torch.full((T, 10), -1, dtype=torch.int64, device=DEV)
        else:
            f, route = moe(inp, w, ck, L)
        R, xr = mhc_post(f, R, post, comb)
        out["resid"].append(R[psel].cpu())
        out["attn"].append(a[psel].cpu())
        out["ffn"].append(f[psel].cpu())
        out["route"].append(route.to(torch.int16).cpu())
        del w
        torch.cuda.empty_cache()
        print(f"  layer {L:2d} {'DSA' if L in DSA_LAYERS else 'HCA'} {time.time() - t0:5.1f}s "
              f"|R| {R.float().pow(2).mean().sqrt().item():.4f}", flush=True)
    res = {"ids": torch.tensor(ids, dtype=torch.int32), "pos_sel": psel.int().cpu()}
    for k in ("resid", "attn", "ffn", "route"):
        res[k] = torch.stack(out[k])
    if out["dsa"]:
        res["dsa_topk"] = torch.stack(out["dsa"]).cpu()
    if n_layers == 43:
        x = R.reshape(T, HC * H).float()
        fn = ck.get("model.mhc_contract.hc_head_fn", F32).t()
        mixes = (x @ fn.t()) * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
        pre = torch.sigmoid(mixes * ck.get("model.mhc_contract.hc_head_scale", F32)[0] +
                            ck.get("model.mhc_contract.hc_head_base", F32)) + EPS
        h = rms(bf((pre[..., None] * R.float()).sum(1)), ck.get("model.norm.weight"))
        r_ = ck.get("model.lm_head.multimax_ranges", F32)
        t_ = ck.get("model.lm_head.multimax_ts", F32)
        print(f"  multimax ranges {r_.tolist()} ts {t_.tolist()}", flush=True)
        head = ck.get("lm_head.weight").float()
        top_ids, top_vals, nll, hit = [], [], 0.0, 0
        raw_ids, raw_vals = [], []
        logits_sel = []
        for a in range(0, T, 256):
            b = min(T, a + 256)
            lg = h[a:b].float() @ head.t()
            rv, ri = lg.topk(8, -1)
            raw_ids.append(ri.int().cpu())
            raw_vals.append(rv.cpu())
            lg = (lg + t_[0] * torch.relu(r_[0] - lg) + t_[1] * torch.relu(lg - r_[1])
                  + t_[2] * torch.relu(r_[2] - lg).square() + t_[3] * torch.relu(lg - r_[3]).square())
            tv, ti = lg.topk(8, -1)
            top_ids.append(ti.int().cpu())
            top_vals.append(tv.cpu())
            nxt = torch.tensor(ids[a + 1:b + 1], device=DEV)
            m = nxt.numel()
            if m:
                lp = torch.log_softmax(lg[:m], -1)
                nll += -lp.gather(1, nxt[:, None]).sum().item()
                hit += (ti[:m, 0] == nxt).sum().item()
            keep = [i - a for i in sel if a <= i < b]
            if keep:
                logits_sel.append(lg[keep].cpu())
        res["top_ids"], res["top_vals"] = torch.cat(top_ids), torch.cat(top_vals)
        res["logits_sel"] = torch.cat(logits_sel)
        res["raw_ids"], res["raw_vals"] = torch.cat(raw_ids), torch.cat(raw_vals)
        print(f"  hidden after norm: RMS {h.float().pow(2).mean().sqrt().item():.3f} · raw top logit mean "
              f"{res['raw_vals'][:, 0].mean().item():.2f} max {res['raw_vals'][:, 0].max().item():.2f}", flush=True)
        print(f"  next-token accuracy {hit / (T - 1):.3f} · perplexity {math.exp(nll / (T - 1)):.2f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--names", default="")
    ap.add_argument("--layers", type=int, default=43)
    a = ap.parse_args()
    ck = Ckpt(a.model)
    prompts = json.load(open(a.prompts))
    os.makedirs(a.out, exist_ok=True)
    for name, ids in prompts.items():
        if a.names and name not in a.names.split(","):
            continue
        t0 = time.time()
        print(f"[ref] {name}: {len(ids)} tokens", flush=True)
        with torch.no_grad():
            res = run(ck, ids, a.layers)
        save_file({k: v.contiguous() for k, v in res.items()}, os.path.join(a.out, f"{name}.safetensors"))
        print(f"[ref] {name} done · {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
