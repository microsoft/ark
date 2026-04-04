# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Numerical tests for composite ops: softmax, layernorm."""

import pytest
import torch
import torch.nn.functional as F
from conftest import ark, DEVICE


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_softmax(dtype):
    shape = (4, 8, 256)
    a = torch.randn(shape, dtype=dtype, device=DEVICE)
    result = ark.softmax(a).eval()
    expected = F.softmax(a, dim=-1)
    atol = 1e-5 if dtype == torch.float32 else 1e-3
    assert torch.allclose(result, expected, atol=atol, rtol=1e-3), (
        f"max_diff={(result - expected).abs().max()}"
    )


def test_layernorm():
    shape = (4, 8, 256)
    a = torch.randn(shape, dtype=torch.float32, device=DEVICE)
    result = ark.layernorm(a, eps=1e-6).eval()
    mean = a.mean(dim=-1, keepdim=True)
    var = ((a - mean) ** 2).mean(dim=-1, keepdim=True)
    expected = (a - mean) / torch.sqrt(var + 1e-6)
    assert torch.allclose(result, expected, atol=1e-4, rtol=1e-4), (
        f"max_diff={(result - expected).abs().max()}"
    )
