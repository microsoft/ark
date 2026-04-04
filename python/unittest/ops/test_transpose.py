# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Numerical tests for transpose op."""

import pytest
import torch
from conftest import ark, DEVICE


@pytest.mark.parametrize(
    "perm, shape",
    [
        ([0, 1, 3, 2], [2, 3, 64, 128]),
        ([0, 2, 3, 1], [2, 3, 64, 128]),
        ([0, 2, 1, 3], [2, 3, 64, 128]),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_transpose(perm, shape, dtype):
    a = torch.randn(shape, dtype=dtype, device=DEVICE)
    result = ark.transpose(a, perm).eval()
    expected = a.permute(perm).contiguous()
    assert torch.allclose(result, expected, atol=0, rtol=0), (
        f"max_diff={(result - expected).abs().max()}"
    )
