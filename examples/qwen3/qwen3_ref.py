# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Pure-torch Qwen3-8B reference model (random weights, fixed seed, fp16).

Implements: RMSNorm, RoPE, QK-norm, GQA attention, SwiGLU MLP,
TransformerBlock, and Qwen3Model. No ARK dependency.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .qwen3_config import Qwen3Config

_DEFAULT_SEED = 42


def _get_dtype(cfg: Qwen3Config) -> torch.dtype:
    dt = getattr(torch, cfg.dtype, None)
    if not isinstance(dt, torch.dtype):
        raise ValueError(f"Invalid dtype string in config: {cfg.dtype!r}")
    return dt


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_normed = x.float() / rms
        return (x_normed * self.weight.float()).to(x.dtype)


def precompute_rope_freqs(
    head_dim: int, max_seq_len: int, theta: float = 1e6
) -> torch.Tensor:
    """Precompute complex RoPE frequencies of shape (max_seq_len, head_dim//2)."""
    freqs = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)  # (seq, head_dim//2)
    return torch.polar(torch.ones_like(angles), angles)  # complex64


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings.

    Args:
        x: (batch, n_heads, seq, head_dim)  — real tensor.
        freqs: (seq, head_dim//2) — complex tensor.
    """
    # Reshape to pairs and view as complex
    batch, n_heads, seq, hd = x.shape
    x_complex = torch.view_as_complex(
        x.float().reshape(batch, n_heads, seq, hd // 2, 2)
    )
    freqs = freqs[:seq].unsqueeze(0).unsqueeze(0)  # (1, 1, seq, hd//2)
    x_rotated = torch.view_as_real(x_complex * freqs)
    return x_rotated.reshape(batch, n_heads, seq, hd).to(x.dtype)


class QKNorm(nn.Module):
    """Per-head RMS normalization applied to Q and K projections."""

    def __init__(self, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.q_norm = RMSNorm(head_dim, eps=eps)
        self.k_norm = RMSNorm(head_dim, eps=eps)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize Q and K. Inputs: (batch, n_heads, seq, head_dim)."""
        orig_shape = q.shape
        q = self.q_norm(q.reshape(-1, orig_shape[-1])).reshape(orig_shape)
        k = self.k_norm(k.reshape(-1, orig_shape[-1])).reshape(k.shape)
        return q, k


class GQAAttention(nn.Module):
    """Grouped-query attention with QK-norm and RoPE."""

    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.n_q_heads = cfg.n_q_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = self.n_q_heads // self.n_kv_heads

        dtype = _get_dtype(cfg)
        self.q_proj = nn.Linear(
            cfg.hidden_dim, cfg.n_q_heads * cfg.head_dim, bias=False
        ).to(dtype)
        self.k_proj = nn.Linear(
            cfg.hidden_dim, cfg.n_kv_heads * cfg.head_dim, bias=False
        ).to(dtype)
        self.v_proj = nn.Linear(
            cfg.hidden_dim, cfg.n_kv_heads * cfg.head_dim, bias=False
        ).to(dtype)
        self.o_proj = nn.Linear(
            cfg.n_q_heads * cfg.head_dim, cfg.hidden_dim, bias=False
        ).to(dtype)
        self.qk_norm = QKNorm(cfg.head_dim, eps=cfg.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq, _ = x.shape

        q = self.q_proj(x).reshape(batch, seq, self.n_q_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, seq, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, seq, self.n_kv_heads, self.head_dim)

        # (batch, heads, seq, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # QK-norm before RoPE
        q, k = self.qk_norm(q, k)

        # RoPE
        q = apply_rope(q, rope_freqs)
        k = apply_rope(k, rope_freqs)

        # Expand KV heads for GQA
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            attn_weights = attn_weights + mask

        attn_weights = F.softmax(attn_weights.float(), dim=-1).to(x.dtype)
        out = torch.matmul(attn_weights, v)

        out = out.transpose(1, 2).reshape(batch, seq, -1)
        return self.o_proj(out)


class SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        dtype = _get_dtype(cfg)
        self.gate_proj = nn.Linear(
            cfg.hidden_dim, cfg.intermediate_dim, bias=False
        ).to(dtype)
        self.up_proj = nn.Linear(
            cfg.hidden_dim, cfg.intermediate_dim, bias=False
        ).to(dtype)
        self.down_proj = nn.Linear(
            cfg.intermediate_dim, cfg.hidden_dim, bias=False
        ).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Single Qwen3 transformer block: pre-norm attention + pre-norm MLP."""

    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_dim, eps=cfg.rms_norm_eps)
        self.attn = GQAAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.hidden_dim, eps=cfg.rms_norm_eps)
        self.mlp = SwiGLUMLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), rope_freqs, mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Qwen3Model(nn.Module):
    """Qwen3 causal language model (random weights, no real checkpoint).

    Uses a fixed seed for reproducible random initialization.
    """

    def __init__(self, cfg: Qwen3Config, seed: int = _DEFAULT_SEED):
        super().__init__()
        self.cfg = cfg
        dtype = _get_dtype(cfg)

        torch.manual_seed(seed)

        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim).to(dtype)
        self.layers = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.final_norm = RMSNorm(cfg.hidden_dim, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False).to(
            dtype
        )

        # Precompute RoPE frequencies (cpu, moved to device in forward)
        self.register_buffer(
            "rope_freqs",
            precompute_rope_freqs(
                cfg.head_dim, cfg.max_seq_len, cfg.rope_theta
            ),
            persistent=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: (batch, seq) integer token IDs.

        Returns:
            Logits tensor of shape (batch, seq, vocab_size).
        """
        batch, seq = input_ids.shape
        x = self.embed(input_ids)

        # Causal mask: upper-triangular -inf
        mask = torch.full(
            (seq, seq), float("-inf"), device=x.device, dtype=x.dtype
        )
        mask = torch.triu(mask, diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)

        rope_freqs = self.rope_freqs.to(x.device)

        for layer in self.layers:
            x = layer(x, rope_freqs, mask)

        x = self.final_norm(x)
        return self.lm_head(x)
