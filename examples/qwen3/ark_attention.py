# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""ARK GQA attention for Qwen3: QKV projections, QK-norm, RoPE, GQA expand,
scaled dot-product with causal mask, output projection.

Implementation note — no ark.transpose
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``ark.transpose`` combined with other ops in a single eval graph triggers
``cudaErrorMisalignedAddress`` on A100.  Individual ops (rmsnorm, rope,
matmul) work in isolation.  As a workaround, all reshape/transpose logic
uses torch; ARK handles matmul, composed RMSNorm, and RoPE via separate
small eval() calls.
"""

import math

import torch
import ark

from .qwen3_config import Qwen3Config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def precompute_ark_rope_freqs(head_dim, max_seq_len, theta=1e6):
    """Precompute interleaved [cos, sin] RoPE frequencies for ``ark.rope``.

    Returns:
        fp16 tensor of shape ``(1, 1, max_seq_len, head_dim)`` on CPU.
    """
    freqs = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    cos_vals = torch.cos(angles)
    sin_vals = torch.sin(angles)
    interleaved = torch.stack([cos_vals, sin_vals], dim=-1)
    interleaved = interleaved.reshape(max_seq_len, head_dim)
    return interleaved.unsqueeze(0).unsqueeze(0).half()


def ark_rmsnorm(x, weight, eps):
    """Composed RMSNorm using ARK primitives (fp32 accumulation).

    Args:
        x: 3-D or 4-D ARK-compatible tensor ``(..., dim)``.
        weight: 1-D scale parameter ``(dim,)``.
        eps: epsilon for numerical stability.

    Returns:
        ARK tensor (fp16) with the same shape as *x*.
    """
    x_f32 = ark.cast(x, ark.fp32)
    x2 = ark.mul(x_f32, x_f32)
    mean = ark.reduce_mean(x2, axis=-1)
    mean_eps = ark.add(mean, eps)
    rrms = ark.rsqrt(mean_eps)
    x_normed = ark.mul(x_f32, rrms)

    dim = (
        weight.shape[-1]
        if isinstance(weight, torch.Tensor)
        else weight.shape()[-1]
    )
    w_f32 = ark.cast(weight, ark.fp32)
    w_f32 = ark.reshape(w_f32, [1, 1, 1, dim])
    x_scaled = ark.mul(x_normed, w_f32)
    return ark.cast(x_scaled, ark.fp16)


# ---------------------------------------------------------------------------
# Full GQA attention — staged eval, torch reshape/transpose
# ---------------------------------------------------------------------------


def ark_gqa_attention(
    x,
    q_w,
    k_w,
    v_w,
    o_w,
    qk_q_w,
    qk_k_w,
    rope_freqs,
    mask,
    cfg,
):
    """ARK GQA attention — staged evaluation.

    All weight/input arguments are **torch tensors on CUDA**.

    Returns an ARK tensor; call ``.eval()`` to get a torch.Tensor.
    """
    batch, seq = x.shape[0], x.shape[1]
    n_q = cfg.n_q_heads
    n_kv = cfg.n_kv_heads
    hd = cfg.head_dim
    n_rep = n_q // n_kv

    # ---- Stage 1: QKV projections (ARK matmul) ----
    ark.init()
    q = ark.matmul(x, q_w, transpose_other=True).eval()
    ark.init()
    k = ark.matmul(x, k_w, transpose_other=True).eval()
    ark.init()
    v = ark.matmul(x, v_w, transpose_other=True).eval()

    # ---- Reshape + transpose in torch ----
    q = q.reshape(batch, seq, n_q, hd).transpose(1, 2).contiguous()
    k = k.reshape(batch, seq, n_kv, hd).transpose(1, 2).contiguous()
    v = v.reshape(batch, seq, n_kv, hd).transpose(1, 2).contiguous()

    # ---- Stage 2: QK-norm (ARK composed RMSNorm) ----
    ark.init()
    q = ark_rmsnorm(q, qk_q_w, cfg.rms_norm_eps).eval()
    ark.init()
    k = ark_rmsnorm(k, qk_k_w, cfg.rms_norm_eps).eval()

    # ---- Stage 3: RoPE (ARK) ----
    ark.init()
    q = ark.rope(q, rope_freqs).eval()
    ark.init()
    k = ark.rope(k, rope_freqs).eval()

    # ---- GQA expand (torch) ----
    if n_rep > 1:
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)

    # ---- Stage 4: Attention scores (ARK matmul + scale) ----
    q3 = q.reshape(batch * n_q, seq, hd).contiguous()
    k3 = k.reshape(batch * n_q, seq, hd).contiguous()
    v3 = v.reshape(batch * n_q, seq, hd).contiguous()

    ark.init()
    scores = ark.matmul(q3, k3, transpose_other=True)
    scores = ark.mul(scores, 1.0 / math.sqrt(hd))
    scores = scores.eval()

    # ---- Mask + softmax (torch) ----
    if mask is not None:
        scores = scores.reshape(batch, n_q, seq, seq) + mask
        scores = scores.reshape(batch * n_q, seq, seq)
    attn_w = torch.softmax(scores.float(), dim=-1).half()

    # ---- Stage 5: Weighted sum (ARK matmul) ----
    ark.init()
    out = ark.matmul(attn_w, v3).eval()

    # ---- Output reshape (torch) ----
    out = out.reshape(batch, n_q, seq, hd)
    out = out.transpose(1, 2).contiguous()
    out = out.reshape(batch, seq, n_q * hd)

    # ---- Stage 6: Output projection (ARK matmul) ----
    ark.init()
    result = ark.matmul(out, o_w, transpose_other=True).eval()

    # Wrap as trivial ARK graph so callers can use .eval()
    ark.init()
    return ark.copy(result)
