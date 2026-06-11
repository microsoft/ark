# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Numerical tests for composite ops: softmax, layernorm."""

import pytest

from common import ark

torch = pytest.importorskip("torch")
import torch.nn.functional as F

DEVICE = "cuda:0"


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("shape", [(4, 8, 256), (8, 16), (3, 13, 127), (4, 1)])
def test_softmax(dtype, shape):
    a = torch.randn(shape, dtype=dtype, device=DEVICE)
    result = ark.softmax(a).eval()
    expected = F.softmax(a, dim=-1)
    atol = {torch.float32: 1e-5, torch.float16: 1e-3, torch.bfloat16: 5e-2}[
        dtype
    ]
    assert torch.allclose(
        result, expected, atol=atol, rtol=1e-3
    ), f"max_diff={(result - expected).abs().max()}"


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("shape", [(4, 8, 256), (8, 16), (3, 13, 127)])
def test_layernorm(dtype, shape):
    a = torch.randn(shape, dtype=dtype, device=DEVICE)
    result = ark.layernorm(a, eps=1e-6).eval()
    mean = a.mean(dim=-1, keepdim=True)
    var = ((a - mean) ** 2).mean(dim=-1, keepdim=True)
    expected = (a - mean) / torch.sqrt(var + 1e-6)
    atol = {torch.float32: 1e-4, torch.float16: 1e-2, torch.bfloat16: 5e-2}[
        dtype
    ]
    assert torch.allclose(
        result, expected, atol=atol, rtol=1e-4
    ), f"max_diff={(result - expected).abs().max()}"
