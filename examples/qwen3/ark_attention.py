# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""ARK GQA attention for Qwen3: QKV projections, QK-norm, RoPE, GQA expand,
scaled dot-product with causal mask, output projection.

All functions build ARK computation graphs.  Call ``.eval()`` on the
returned tensor to compile, execute, and retrieve a ``torch.Tensor``.

Implementation note — 3-D matmul only
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ARK's default 4-D batched-GEMM batch-stride calculation (``ops_matmul.cpp``)
does not match the ``(B, H, S, D)`` attention layout; the llama example works
around this with explicit ``BatchStride*`` PlannerContext config.  To keep this
module self-contained (no PlannerContext), all attention matmuls use 3-D
tensors ``(B*H, S, D)`` so the batch stride is computed correctly by default.
QK-norm and RoPE still operate on 4-D ``(B, H, S, D)`` tensors (element-wise
/ reduce ops are unaffected by the batch-stride issue).
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
    # Interleave: [cos0, sin0, cos1, sin1, ...]
    interleaved = torch.stack([cos_vals, sin_vals], dim=-1)  # (S, D/2, 2)
    interleaved = interleaved.reshape(max_seq_len, head_dim)
    return interleaved.unsqueeze(0).unsqueeze(0).half()  # (1,1,S,D)


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

    # Determine the head dim from the weight tensor.
    dim = (
        weight.shape[-1]
        if isinstance(weight, torch.Tensor)
        else weight.shape()[-1]
    )
    w_f32 = ark.cast(weight, ark.fp32)
    w_f32 = ark.reshape(w_f32, [1, 1, 1, dim])
    x_scaled = ark.mul(x_normed, w_f32)
    return ark.cast(x_scaled, ark.fp16)


def _gqa_expand(x, total_kv, n_rep, seq, head_dim):
    """Broadcast-copy KV heads in 3-D layout.

    ``(B*n_kv, S, D)`` → ``(B*n_kv*n_rep, S, D)``

    Each KV head is replicated *n_rep* times using ``ark.copy`` broadcast.
    """
    sd = seq * head_dim
    x_src = ark.reshape(x, [total_kv, 1, sd])
    x_dst = ark.tensor([total_kv, n_rep, sd], ark.fp16)
    x_copied = ark.copy(x_src, x_dst)
    return ark.reshape(x_copied, [total_kv * n_rep, seq, head_dim])


# ---------------------------------------------------------------------------
# Full GQA attention graph
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
    """Build ARK GQA attention graph and return the (unevaluated) output tensor.

    All weight/input arguments are **torch tensors on CUDA** (auto-converted
    by ``ark._ensure_ark``).

    Args:
        x:          ``(B, S, hidden_dim)`` input activations, fp16.
        q_w:        ``(n_q*D, hidden_dim)`` query projection weight.
        k_w:        ``(n_kv*D, hidden_dim)`` key projection weight.
        v_w:        ``(n_kv*D, hidden_dim)`` value projection weight.
        o_w:        ``(hidden_dim, n_q*D)`` output projection weight.
        qk_q_w:     ``(D,)`` QK-norm query scale.
        qk_k_w:     ``(D,)`` QK-norm key scale.
        rope_freqs: ``(1, 1, S, D)`` interleaved cos/sin, fp16.
        mask:       ``(1, 1, S, S)`` additive causal mask or ``None``.
        cfg:        :class:`Qwen3Config`.

    Returns:
        ARK tensor of shape ``(B, S, hidden_dim)``.  Call ``.eval()``
        to execute.
    """
    batch, seq = x.shape[0], x.shape[1]
    n_q = cfg.n_q_heads
    n_kv = cfg.n_kv_heads
    hd = cfg.head_dim
    n_rep = n_q // n_kv

    # --- QKV projections (x @ W^T) — 3-D matmul ---
    q = ark.matmul(x, q_w, transpose_other=True)
    k = ark.matmul(x, k_w, transpose_other=True)
    v = ark.matmul(x, v_w, transpose_other=True)

    # --- Reshape (B, S, H, D) and transpose to (B, H, S, D) ---
    # Transpose is a physical copy in ARK, producing a contiguous tensor.
    q = ark.transpose(ark.reshape(q, [batch, seq, n_q, hd]), [0, 2, 1, 3])
    k = ark.transpose(ark.reshape(k, [batch, seq, n_kv, hd]), [0, 2, 1, 3])
    v = ark.transpose(ark.reshape(v, [batch, seq, n_kv, hd]), [0, 2, 1, 3])

    # --- QK-norm (per-head RMSNorm along head_dim, 4-D) ---
    q = ark_rmsnorm(q, qk_q_w, cfg.rms_norm_eps)
    k = ark_rmsnorm(k, qk_k_w, cfg.rms_norm_eps)

    # --- RoPE (4-D) ---
    q = ark.rope(q, rope_freqs)
    k = ark.rope(k, rope_freqs)

    # --- Flatten to 3-D (B*H, S, D) for matmul ---
    q = ark.reshape(q, [batch * n_q, seq, hd])
    k = ark.reshape(k, [batch * n_kv, seq, hd])
    v = ark.reshape(v, [batch * n_kv, seq, hd])

    # --- GQA head expansion (3-D) ---
    if n_rep > 1:
        k = _gqa_expand(k, batch * n_kv, n_rep, seq, hd)
        v = _gqa_expand(v, batch * n_kv, n_rep, seq, hd)

    # --- Scaled dot-product attention (3-D matmul) ---
    scale = 1.0 / math.sqrt(hd)
    scores = ark.matmul(q, k, transpose_other=True)  # (B*n_q, S, S)
    scores = ark.mul(scores, scale)

    if mask is not None:
        # mask is (1, 1, S, S); reshape to (1, S, S) for 3-D broadcast.
        scores = ark.reshape(scores, [batch, n_q, seq, seq])
        scores = ark.add(scores, mask)
        scores = ark.reshape(scores, [batch * n_q, seq, seq])

    # Softmax in fp32 for numerical parity with torch reference.
    scores = ark.cast(scores, ark.fp32)
    attn_w = ark.softmax(scores)
    attn_w = ark.cast(attn_w, ark.fp16)

    # Weighted sum over values.
    out = ark.matmul(attn_w, v)  # (B*n_q, S, D)

    # --- Reshape back: (B*H, S, D) → (B, S, H*D) ---
    out = ark.reshape(out, [batch, n_q, seq, hd])
    out = ark.transpose(out, [0, 2, 1, 3])  # (B, S, H, D)
    out = ark.reshape(out, [batch, seq, n_q * hd])

    # --- Output projection (3-D matmul) ---
    out = ark.matmul(out, o_w, transpose_other=True)
    return out
