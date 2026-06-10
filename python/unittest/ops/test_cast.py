# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Numerical tests for cast op."""

import pytest

from common import ark

torch = pytest.importorskip("torch")

DEVICE = "cuda:0"

# Note: byte↔{fp32,fp16,int32} casts not tested — ark.cast does not expose byte conversions in the Python API.


@pytest.mark.parametrize(
    "src_dtype, dst_dtype, ark_dst",
    [
        (torch.float16, torch.float32, ark.fp32),
        (torch.float32, torch.float16, ark.fp16),
        (torch.bfloat16, torch.float32, ark.fp32),
        (torch.float32, torch.bfloat16, ark.bf16),
    ],
)
def test_cast_float(src_dtype, dst_dtype, ark_dst):
    a = torch.randn(4, 2, 1024, dtype=torch.float32, device=DEVICE).to(
        src_dtype
    )
    result = ark.cast(a, ark_dst).eval()
    expected = a.to(dst_dtype)
    assert result.dtype == dst_dtype
    assert torch.allclose(result, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "src_dtype, dst_dtype, ark_dst",
    [
        (torch.float32, torch.int32, ark.int32),
        (torch.int32, torch.float32, ark.fp32),
        (torch.float16, torch.int32, ark.int32),
        (torch.int32, torch.float16, ark.fp16),
    ],
)
def test_cast_int(src_dtype, dst_dtype, ark_dst):
    a = (
        torch.arange(4 * 2 * 1024, device=DEVICE)
        .reshape(4, 2, 1024)
        % 1000
    ).to(src_dtype)
    result = ark.cast(a, ark_dst).eval()
    expected = a.to(dst_dtype)
    assert result.dtype == dst_dtype
    assert torch.allclose(result, expected, atol=0, rtol=0)
