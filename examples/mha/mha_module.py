# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Multi-Head Attention as an ``ark.Module`` subclass.

Demonstrates:
  - Subclassing ``ark.Module`` with ``ark.parameter()``.
  - Using the functional API (``ark.matmul``, ``ark.reshape``,
    ``ark.transpose``, ``ark.softmax``, ``ark.mul``).
  - Numerical validation against a manual PyTorch reference implementation.

Run: ``python examples/mha/mha_module.py``
"""

import math
import numpy as np
import torch
import torch.nn as nn
import ark

# ---------- hyperparameters ----------
BATCH = 1
SEQ = 64
D_MODEL = 128
N_HEADS = 4
D_K = D_MODEL // N_HEADS  # 32


# ---------- ARK Module ----------
class MultiHeadAttention(ark.Module):
    """Scaled dot-product multi-head attention (no bias, no mask)."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_k)

        # Projection weights: [d_model, d_model]
        self.wq = ark.parameter([d_model, d_model], ark.fp16)
        self.wk = ark.parameter([d_model, d_model], ark.fp16)
        self.wv = ark.parameter([d_model, d_model], ark.fp16)
        self.wo = ark.parameter([d_model, d_model], ark.fp16)

    def forward(self, x):
        # x: [BATCH, SEQ, D_MODEL]
        B = BATCH
        S = SEQ
        H = self.n_heads
        dk = self.d_k

        # Linear projections
        q = ark.matmul(x, self.wq)  # [B, S, D]
        k = ark.matmul(x, self.wk)
        v = ark.matmul(x, self.wv)

        # Reshape to [B, S, H, dk] then transpose to [B, H, S, dk]
        q = ark.transpose(ark.reshape(q, [B, S, H, dk]), [0, 2, 1, 3])
        k = ark.transpose(ark.reshape(k, [B, S, H, dk]), [0, 2, 1, 3])
        v = ark.transpose(ark.reshape(v, [B, S, H, dk]), [0, 2, 1, 3])

        # Scaled dot-product attention
        # scores: [B, H, S, S]
        scores = ark.matmul(q, k, transpose_other=True)
        scores = ark.mul(scores, self.scale)
        attn = ark.softmax(scores)  # along last axis
        # context: [B, H, S, dk]
        context = ark.matmul(attn, v)

        # Transpose back and reshape: [B, S, D]
        context = ark.reshape(
            ark.transpose(context, [0, 2, 1, 3]),
            [B, S, H * dk],
        )

        # Output projection
        out = ark.matmul(context, self.wo)
        return out


# ---------- PyTorch reference ----------
def pytorch_mha(x_np, wq, wk, wv, wo):
    """Manual MHA in PyTorch matching the ARK implementation above."""
    x = torch.from_numpy(x_np).cuda()
    B, S, D = x.shape
    H = N_HEADS
    dk = D_K

    q = x @ torch.from_numpy(wq).cuda()
    k = x @ torch.from_numpy(wk).cuda()
    v = x @ torch.from_numpy(wv).cuda()

    q = q.view(B, S, H, dk).permute(0, 2, 1, 3)
    k = k.view(B, S, H, dk).permute(0, 2, 1, 3)
    v = v.view(B, S, H, dk).permute(0, 2, 1, 3)

    scores = q @ k.transpose(-2, -1) / math.sqrt(dk)
    attn = torch.softmax(scores, dim=-1)
    context = attn @ v

    context = context.permute(0, 2, 1, 3).contiguous().view(B, S, D)
    out = context @ torch.from_numpy(wo).cuda()
    return out.cpu().numpy()


# ---------- main ----------
def main():
    ark.init()

    # Build graph
    x_ark = ark.tensor([BATCH, SEQ, D_MODEL], ark.fp16)
    model = MultiHeadAttention(D_MODEL, N_HEADS)
    y_ark = model(x_ark)

    # Launch runtime
    runtime = ark.Runtime()
    runtime.launch()

    # Random inputs and weights (small magnitude for fp16 stability)
    rng = np.random.RandomState(42)
    x_np = (rng.randn(BATCH, SEQ, D_MODEL) * 0.02).astype(np.float16)
    wq_np = (rng.randn(D_MODEL, D_MODEL) * 0.02).astype(np.float16)
    wk_np = (rng.randn(D_MODEL, D_MODEL) * 0.02).astype(np.float16)
    wv_np = (rng.randn(D_MODEL, D_MODEL) * 0.02).astype(np.float16)
    wo_np = (rng.randn(D_MODEL, D_MODEL) * 0.02).astype(np.float16)

    x_ark.from_numpy(x_np)
    model.load_state_dict({"wq": wq_np, "wk": wk_np, "wv": wv_np, "wo": wo_np})

    # Run ARK
    runtime.run()
    y_host = y_ark.to_numpy()

    # Run PyTorch reference
    y_ref = pytorch_mha(x_np, wq_np, wk_np, wv_np, wo_np)

    # Validate
    np.testing.assert_allclose(y_host, y_ref, atol=1e-2, rtol=1e-2)
    max_err = np.max(np.abs(y_host - y_ref))
    print(f"MHA module test passed  (max error: {max_err:.6f})")


if __name__ == "__main__":
    main()
