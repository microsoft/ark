# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Qwen3 SwiGLU MLP: torch matmul + torch silu·gate fallback.

Computes: down_proj(SiLU(gate_proj(x)) * up_proj(x))

All matmul uses torch.matmul (full-ARK matmul deferred to Q10).
SiLU·gate fusion uses torch ops (F.silu(gate) * up) because the upstream
ARK composed-graph planner bug crashes at intermediate_dim=12288 (same
bug class as the 4-D shape crash documented in Q4).

The ARK path (``ark_silu_gate``) is retained dormant for re-enablement
after the upstream fix lands.
"""

import torch
import torch.nn.functional as F

import ark

from .qwen3_config import Qwen3Config

# ---------------------------------------------------------------------------
# SiLU·gate implementations
# ---------------------------------------------------------------------------


def ark_silu_gate(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SiLU(gate) * up using ARK primitives.

    NOTE: Dormant — crashes with the upstream ARK composed-graph planner
    bug at shapes like (2048, 12288). Kept for re-enablement after
    the upstream fix lands.

    Args:
        gate: (N, intermediate_dim) fp16 tensor on CUDA.
        up: (N, intermediate_dim) fp16 tensor on CUDA.

    Returns:
        (N, intermediate_dim) fp16 tensor.
    """
    ark.init()
    # SiLU(x) = x * sigmoid(x)
    sig = ark.sigmoid(gate)
    silu = ark.mul(gate, sig)
    result = ark.mul(silu, up)
    return result.eval()


def torch_silu_gate(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SiLU(gate) * up using pure torch ops.

    Replaces ``ark_silu_gate`` which crashes with the upstream
    composed-graph planner bug at intermediate_dim=12288.

    Args:
        gate: (N, intermediate_dim) fp16 tensor on CUDA.
        up: (N, intermediate_dim) fp16 tensor on CUDA.

    Returns:
        (N, intermediate_dim) fp16 tensor.
    """
    return F.silu(gate) * up


# ---------------------------------------------------------------------------
# Full SwiGLU MLP
# ---------------------------------------------------------------------------


def ark_swiglu_mlp(
    x: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
    cfg: Qwen3Config,
) -> torch.Tensor:
    """SwiGLU MLP: down_proj(SiLU(gate_proj(x)) * up_proj(x)).

    All weight/input arguments are torch tensors on CUDA.
    Matmul uses torch.matmul; silu·gate uses torch fallback.

    Args:
        x: (B, S, hidden_dim) fp16 input tensor.
        gate_w: (intermediate_dim, hidden_dim) gate projection weight.
        up_w: (intermediate_dim, hidden_dim) up projection weight.
        down_w: (hidden_dim, intermediate_dim) down projection weight.
        cfg: Qwen3Config instance.

    Returns:
        (B, S, hidden_dim) fp16 output tensor wrapped in ark.copy
        for .eval() API consistency.
    """
    orig_shape = x.shape  # (B, S, hidden_dim)
    batch_seq = orig_shape[0] * orig_shape[1]

    # Flatten to 2D for matmul
    x_2d = x.reshape(batch_seq, cfg.hidden_dim)

    # Gate and up projections (torch matmul)
    gate = torch.matmul(x_2d, gate_w.t())  # (B*S, intermediate_dim)
    up = torch.matmul(x_2d, up_w.t())  # (B*S, intermediate_dim)

    # SiLU·gate fusion (torch fallback — ARK crashes at intermediate_dim=12288)
    hidden = torch_silu_gate(gate, up)  # (B*S, intermediate_dim)

    # Down projection (torch matmul)
    out_2d = torch.matmul(hidden, down_w.t())  # (B*S, hidden_dim)

    # Reshape back to (B, S, hidden_dim)
    result = out_2d.reshape(orig_shape)

    # Wrap as trivial ARK graph so callers can use .eval()
    ark.init()
    return ark.copy(result)
