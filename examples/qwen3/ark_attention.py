# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""ARK GQA attention for Qwen3: QKV projections, QK-norm, RoPE, GQA expand,
scaled dot-product with causal mask, output projection.

Implementation note — staged eval
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Compiling the entire attention pipeline into a single ARK kernel triggers a
``cudaErrorMisalignedAddress`` at runtime (observed on A100 CI with large
op-count graphs mixing matmul, transpose, and element-wise ops).  The tested
individual ops (rmsnorm, rope, matmul) work in isolation.  As a workaround
the pipeline is split into stages, each evaluated separately.  This adds
launch overhead but guarantees correctness; a single-kernel version can be
revisited after the root cause is identified in the ARK runtime.
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
    angles = torch.outer(t, freqs)  # (seq, head_dim // 2)
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
# Staged helpers — each builds a small graph and eval()s it
# ---------------------------------------------------------------------------


def _eval_qkv_proj(x, q_w, k_w, v_w):
    """QKV linear projections.  Returns three torch tensors."""
    ark.init()
    q_out = ark.matmul(x, q_w, transpose_other=True).eval()
    ark.init()
    k_out = ark.matmul(x, k_w, transpose_other=True).eval()
    ark.init()
    v_out = ark.matmul(x, v_w, transpose_other=True).eval()
    return q_out, k_out, v_out


def _eval_qknorm_rope(q, k, qk_q_w, qk_k_w, rope_freqs, cfg):
    """QK-norm + RoPE on Q and K.  Returns two torch tensors in (B,H,S,D)."""
    B, S = q.shape[0], q.shape[1]
    hd = cfg.head_dim

    # Reshape (B,S,H,D) for both Q and K, apply QK-norm
    ark.init()
    q4 = ark.reshape(q, [B, S, cfg.n_q_heads, hd])
    q4 = ark.transpose(q4, [0, 2, 1, 3])
    q4 = ark_rmsnorm(q4, qk_q_w, cfg.rms_norm_eps)
    q4 = ark.rope(q4, rope_freqs)
    q_out = q4.eval()

    ark.init()
    k4 = ark.reshape(k, [B, S, cfg.n_kv_heads, hd])
    k4 = ark.transpose(k4, [0, 2, 1, 3])
    k4 = ark_rmsnorm(k4, qk_k_w, cfg.rms_norm_eps)
    k4 = ark.rope(k4, rope_freqs)
    k_out = k4.eval()

    return q_out, k_out


def _eval_attention(q, k, v, mask, cfg):
    """Scaled dot-product attention.  All inputs/outputs are torch tensors.

    q: (B, n_q, S, D), k: (B, n_kv, S, D), v: (B, n_kv, S, D)
    Returns: (B, n_q, S, D)
    """
    B = q.shape[0]
    S = q.shape[2]
    n_q = cfg.n_q_heads
    n_kv = cfg.n_kv_heads
    hd = cfg.head_dim
    n_rep = n_q // n_kv

    # GQA expand (torch — simple and avoids ARK copy issues)
    if n_rep > 1:
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)

    # Flatten to 3-D for matmul
    q3 = q.reshape(B * n_q, S, hd)
    k3 = k.reshape(B * n_q, S, hd)
    v3 = v.reshape(B * n_q, S, hd)

    # Scores
    ark.init()
    scores = ark.matmul(q3, k3, transpose_other=True)
    scores = ark.mul(scores, 1.0 / math.sqrt(hd))
    scores_out = scores.eval()  # (B*n_q, S, S)

    # Add mask (torch — broadcast is trivial)
    if mask is not None:
        scores_4d = scores_out.reshape(B, n_q, S, S)
        scores_4d = scores_4d + mask
        scores_out = scores_4d.reshape(B * n_q, S, S)

    # Softmax (torch — avoids fp16 precision issues in composed ARK softmax)
    attn_w = torch.softmax(scores_out.float(), dim=-1).half()

    # Weighted sum
    ark.init()
    out = ark.matmul(attn_w, v3)
    return out.eval().reshape(B, n_q, S, hd)


def _eval_output_proj(attn_out, o_w, cfg):
    """Output projection.  attn_out: (B, n_q, S, D) → (B, S, hidden)."""
    B = attn_out.shape[0]
    S = attn_out.shape[2]

    # Transpose (B,H,S,D) → (B,S,H,D) → (B,S,H*D) in torch
    out = attn_out.transpose(1, 2).contiguous()
    out = out.reshape(B, S, cfg.n_q_heads * cfg.head_dim)

    ark.init()
    result = ark.matmul(out, o_w, transpose_other=True)
    return result.eval()


# ---------------------------------------------------------------------------
# Public API
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

    Returns:
        torch.Tensor of shape ``(B, S, hidden_dim)``.
    """
    # Stage 1: QKV projections
    q, k, v = _eval_qkv_proj(x, q_w, k_w, v_w)

    # Stage 2: QK-norm + RoPE (returns 4-D (B,H,S,D))
    q, k = _eval_qknorm_rope(q, k, qk_q_w, qk_k_w, rope_freqs, cfg)

    # Stage 3: V also needs transpose (no norm/rope)
    B, S = x.shape[0], x.shape[1]
    v = (
        v.reshape(B, S, cfg.n_kv_heads, cfg.head_dim)
        .transpose(1, 2)
        .contiguous()
    )

    # Stage 4: Attention (scores + softmax + weighted sum)
    attn_out = _eval_attention(q, k, v, mask, cfg)

    # Stage 5: Output projection
    result = _eval_output_proj(attn_out, o_w, cfg)

    # Wrap as a trivial ARK graph so callers can use .eval()
    ark.init()
    return ark.copy(result)
