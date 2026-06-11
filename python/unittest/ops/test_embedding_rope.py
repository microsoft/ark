# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Numerical tests for embedding and rope ops."""

import pytest

from common import ark

torch = pytest.importorskip("torch")
import torch.nn.functional as F

DEVICE = "cuda:0"


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16]
)
def test_embedding(dtype):
    vocab_size, embed_dim = 100, 64
    indices = torch.randint(0, vocab_size, (4, 8), device=DEVICE).to(
        torch.int32
    )
    weight = torch.randn(vocab_size, embed_dim, dtype=dtype, device=DEVICE)
    result = ark.embedding(indices, weight).eval()
    expected = F.embedding(indices, weight)
    assert torch.allclose(
        result, expected, atol=0, rtol=0
    ), f"max_diff={(result - expected).abs().max()}"


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16]
)
def test_rope(dtype):
    """Test rotary positional embedding against PyTorch complex-multiply reference.
    ARK's rope computes element-wise complex multiplication on consecutive pairs:
      c[2k]   = a[2k]*b[2k]   - a[2k+1]*b[2k+1]
      c[2k+1] = a[2k]*b[2k+1] + a[2k+1]*b[2k]
    """
    shape = (1, 1, 8, 64)
    x = torch.randn(shape, dtype=dtype, device=DEVICE)
    other = torch.randn(shape, dtype=dtype, device=DEVICE)
    result = ark.rope(x, other).eval()
    # PyTorch reference: complex multiply on paired elements
    a = x.reshape(*shape[:-1], -1, 2)
    b = other.reshape(*shape[:-1], -1, 2)
    expected = torch.stack(
        [
            a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1],
            a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0],
        ],
        dim=-1,
    ).reshape(shape)
    atol = 1e-5 if dtype == torch.float32 else 5e-2
    assert torch.allclose(
        result, expected, atol=atol, rtol=1e-3
    ), f"max_diff={(result - expected).abs().max()}"
