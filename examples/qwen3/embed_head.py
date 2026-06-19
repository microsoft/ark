# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Qwen3 token embedding, final RMSNorm, and lm_head helpers.

The ARK path composes existing ops only: ``embedding``, elementwise math,
``reduce_mean``, ``rsqrt``, ``cast``, and ``matmul``.  The torch helpers are
references for local equivalence tests and do not launch or drive SGLang.
"""

import ark

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - tests skip when torch is absent.
    torch = None
    F = None

QWEN3_HIDDEN_SIZE = 4096
QWEN3_VOCAB_SIZE = 151936
QWEN3_RMS_EPS = 1e-6


def _ark_dtype_from_tensor(tensor, default=None):
    """Return the ARK dtype for an ARK or torch tensor."""
    if default is not None:
        return default
    if isinstance(tensor, ark.Tensor):
        return tensor.dtype()
    if torch is not None and isinstance(tensor, torch.Tensor):
        return ark.DataType.from_torch(tensor.dtype)
    raise TypeError("out_dtype is required for non-tensor inputs")


def qwen3_token_embedding(tokens, embed_weight):
    """Apply the Qwen3 token embedding table with ARK's embedding op."""
    return ark.embedding(tokens, embed_weight)


def qwen3_final_rmsnorm(
    hidden,
    norm_weight,
    eps=QWEN3_RMS_EPS,
    out_dtype=None,
):
    """Apply final RMSNorm with fp32 reduction and dtype-restored output."""
    dst_dtype = _ark_dtype_from_tensor(hidden, out_dtype)
    hidden_fp32 = ark.cast(hidden, ark.fp32)
    hidden_shape = hidden_fp32.shape()
    weight_fp32 = ark.cast(norm_weight, ark.fp32)
    if weight_fp32.shape() == [hidden_shape[-1]] and len(hidden_shape) > 1:
        weight_fp32 = ark.reshape(
            weight_fp32, [1] * (len(hidden_shape) - 1) + [hidden_shape[-1]]
        )
    hidden_sq = ark.mul(hidden_fp32, hidden_fp32)
    mean_sq = ark.reduce_mean(hidden_sq, axis=-1)
    rms_inv = ark.rsqrt(ark.add(mean_sq, eps))
    normalized = ark.mul(ark.mul(hidden_fp32, rms_inv), weight_fp32)
    if dst_dtype == ark.fp32:
        return normalized
    return ark.cast(normalized, dst_dtype)


def qwen3_lm_head(hidden, lm_head_weight):
    """Project normalized hidden states with the Qwen3 lm_head weight."""
    return ark.matmul(hidden, lm_head_weight, transpose_other=True)


def qwen3_embed_head(
    tokens,
    embed_weight,
    norm_weight,
    lm_head_weight,
    eps=QWEN3_RMS_EPS,
):
    """Run token embedding -> final RMSNorm -> lm_head with ARK ops."""
    hidden = qwen3_token_embedding(tokens, embed_weight)
    hidden = qwen3_final_rmsnorm(hidden, norm_weight, eps=eps)
    return qwen3_lm_head(hidden, lm_head_weight)


def torch_qwen3_final_rmsnorm(hidden, norm_weight, eps=QWEN3_RMS_EPS):
    """Torch reference for Qwen3 final RMSNorm."""
    hidden_fp32 = hidden.float()
    variance = (hidden_fp32 * hidden_fp32).mean(dim=-1, keepdim=True)
    hidden_fp32 = hidden_fp32 * torch.rsqrt(variance + eps)
    hidden_fp32 = hidden_fp32 * norm_weight.float()
    return hidden_fp32.to(hidden.dtype)


def torch_qwen3_embed_head(
    tokens,
    embed_weight,
    norm_weight,
    lm_head_weight,
    eps=QWEN3_RMS_EPS,
):
    """Torch reference for token embedding -> final RMSNorm -> lm_head."""
    hidden = F.embedding(tokens.long(), embed_weight)
    hidden = torch_qwen3_final_rmsnorm(hidden, norm_weight, eps=eps)
    return hidden @ lm_head_weight.t()
