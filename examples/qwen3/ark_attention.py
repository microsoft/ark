# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Qwen3 GQA attention: torch-only pipeline.

All ops (QKV projection, QK-norm, RoPE, attention, output projection) use
torch.  ARK ops (``ark_rmsnorm``, ``precompute_ark_rope_freqs``) are kept
dormant for re-enablement after the upstream composed-graph fix lands (Q6).
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


# NOTE: Intentionally duplicates qwen3_ref.precompute_rope_freqs / apply_rope.
# Kept as local copies so ARK ops can be swapped back in without coupling
# to the reference module.
def precompute_torch_rope_freqs(head_dim, max_seq_len, theta=1e6):
    """Precompute complex RoPE frequencies for ``torch_rope``.

    Returns:
        complex64 tensor of shape ``(max_seq_len, head_dim // 2)`` on CPU.
    """
    freqs = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(angles), angles)


def torch_rope(x, freqs):
    """Apply rotary position embeddings using pure torch ops.

    Replaces ``ark.rope`` which crashes with the upstream composed-graph
    planner bug at 4-D shapes like ``(1, 4, 128, 32)``.

    Args:
        x: ``(batch, n_heads, seq, head_dim)`` real fp16 tensor on CUDA.
        freqs: ``(max_seq_len, head_dim // 2)`` complex64 tensor.

    Returns:
        Rotated tensor, same shape and dtype as *x*.
    """
    batch, n_heads, seq, hd = x.shape
    x_complex = torch.view_as_complex(
        x.float().reshape(batch, n_heads, seq, hd // 2, 2)
    )
    freqs = freqs[:seq].unsqueeze(0).unsqueeze(0)
    x_rotated = torch.view_as_real(x_complex * freqs)
    return x_rotated.reshape(batch, n_heads, seq, hd).to(x.dtype)


# NOTE: Dormant — torch_rmsnorm is used in production because even the
# 2-D flatten workaround still hits planner bugs at certain shapes.
# Kept for re-enablement when the upstream ARK fix lands.
def ark_rmsnorm(x, weight, eps):
    """Composed RMSNorm using ARK primitives (fp32 accumulation).

    Flattens the input to 2-D in **torch** (a zero-copy view) so the ARK
    graph contains no reshape ops.  This avoids a planner/kernel bug that
    causes ``cudaErrorMisalignedAddress`` when ``ark.reshape`` appears in
    the graph for certain shapes.

    Handles ``ark.init()`` / ``.eval()`` internally.

    Args:
        x: ``torch.Tensor`` on CUDA, any shape; the last dimension is
           the normalization dimension.
        weight: 1-D ``torch.Tensor`` scale parameter ``(dim,)``.
        eps: epsilon for numerical stability.

    Returns:
        ``torch.Tensor`` (fp16) with the same shape as *x*.
    """
    orig_shape = list(x.shape)
    dim = orig_shape[-1]
    n = 1
    for s in orig_shape[:-1]:
        n *= s

    # Flatten in torch — no ARK reshape op in the graph.
    x_2d = x.reshape(n, dim)

    ark.init()
    x_f32 = ark.cast(x_2d, ark.fp32)
    x2 = ark.mul(x_f32, x_f32)
    mean = ark.reduce_mean(x2, axis=-1)
    mean_eps = ark.add(mean, eps)
    rrms = ark.rsqrt(mean_eps)
    x_normed = ark.mul(x_f32, rrms)

    w_f32 = ark.cast(weight, ark.fp32)
    w_f32 = ark.reshape(w_f32, [1, dim])
    x_scaled = ark.mul(x_normed, w_f32)
    result_2d = ark.cast(x_scaled, ark.fp16).eval()

    # Unflatten in torch.
    return result_2d.reshape(orig_shape)


def torch_rmsnorm(x, weight, eps):
    """RMSNorm via pure torch ops (fp32 accumulation).

    Replaces the ARK composed graph which crashes with
    ``cudaErrorMisalignedAddress`` at shapes >= ``(1,4,128,32)``.

    Args:
        x: ``torch.Tensor`` on CUDA, any shape; the last dimension is
           the normalization dimension.
        weight: 1-D ``torch.Tensor`` scale parameter ``(dim,)``.
        eps: epsilon for numerical stability.

    Returns:
        ``torch.Tensor`` (fp16) with the same shape as *x*.
    """
    x_f32 = x.float()
    rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    x_normed = x_f32 / rms
    return (x_normed * weight.float()).half()


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
    cfg: Qwen3Config,
):
    """ARK GQA attention — staged evaluation.

    All weight/input arguments are **torch tensors on CUDA**.

    Returns the result wrapped in a trivial ``ark.copy`` graph for
    ``.eval()`` API consistency; all computation is eager (torch + ARK ops).
    """
    batch, seq = x.shape[0], x.shape[1]
    n_q = cfg.n_q_heads
    n_kv = cfg.n_kv_heads
    hd = cfg.head_dim
    n_rep = n_q // n_kv

    # ---- Stage 1: QKV projections (torch matmul) ----
    q = torch.matmul(x, q_w.t())
    k = torch.matmul(x, k_w.t())
    v = torch.matmul(x, v_w.t())

    # ---- Reshape + transpose in torch ----
    q = q.reshape(batch, seq, n_q, hd).transpose(1, 2).contiguous()
    k = k.reshape(batch, seq, n_kv, hd).transpose(1, 2).contiguous()
    v = v.reshape(batch, seq, n_kv, hd).transpose(1, 2).contiguous()

    # ---- Stage 2: QK-norm (torch RMSNorm — ARK composed graph crashes) ----
    q = torch_rmsnorm(q, qk_q_w, cfg.rms_norm_eps)
    k = torch_rmsnorm(k, qk_k_w, cfg.rms_norm_eps)

    # ---- Stage 3: RoPE (torch — ARK composed graph crashes at 4D) ----
    q = torch_rope(q, rope_freqs)
    k = torch_rope(k, rope_freqs)

    # ---- GQA expand (torch) ----
    if n_rep > 1:
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)

    # ---- Stage 4: Attention scores (torch matmul + scale) ----
    scale = 1.0 / math.sqrt(hd)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # ---- Mask + softmax (torch) ----
    if mask is not None:
        scores = scores + mask
    attn_w = torch.softmax(scores.float(), dim=-1).half()

    # ---- Stage 5: Weighted sum (torch matmul) ----
    out = torch.matmul(attn_w, v)

    # ---- Output reshape (torch) ----
    out = out.transpose(1, 2).contiguous()
    out = out.reshape(batch, seq, n_q * hd)

    # ---- Stage 6: Output projection (torch matmul) ----
    result = torch.matmul(out, o_w.t())

    # Wrap as trivial ARK graph so callers can use .eval()
    ark.init()
    return ark.copy(result)
