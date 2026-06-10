# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Numerical tests for reduce ops: reduce_sum, reduce_max, reduce_mean."""

import pytest

from common import ark

torch = pytest.importorskip("torch")

DEVICE = "cuda:0"


@pytest.mark.parametrize("axis", [0, 1, 2, 3])
def test_reduce_sum_fp32(axis):
    shape = [7, 2, 4, 1024]
    a = torch.randn(shape, dtype=torch.float32, device=DEVICE) * 0.1
    result = ark.reduce_sum(a, axis=axis).eval()
    expected = torch.sum(a, dim=axis, keepdim=True)
    atol = shape[axis] * 1e-5
    assert torch.allclose(
        result, expected, atol=atol, rtol=1e-4
    ), f"axis={axis}, max_diff={(result - expected).abs().max()}"


@pytest.mark.parametrize("axis", [0, 3])
def test_reduce_sum_fp16(axis):
    shape = [7, 2, 4, 1024]
    a = torch.randn(shape, dtype=torch.float16, device=DEVICE) * 0.1
    result = ark.reduce_sum(a, axis=axis).eval()
    expected = torch.sum(a, dim=axis, keepdim=True)
    atol = shape[axis] * 2e-2
    assert torch.allclose(
        result, expected, atol=atol, rtol=1e-2
    ), f"axis={axis}, max_diff={(result - expected).abs().max()}"


@pytest.mark.parametrize("axis", [0, 3])
def test_reduce_sum_bf16(axis):
    shape = [7, 2, 4, 1024]
    a = torch.randn(shape, dtype=torch.bfloat16, device=DEVICE) * 0.1
    result = ark.reduce_sum(a, axis=axis).eval()
    expected = torch.sum(a, dim=axis, keepdim=True)
    atol = shape[axis] * 2e-2
    assert torch.allclose(
        result, expected, atol=atol, rtol=1e-2
    ), f"axis={axis}, max_diff={(result - expected).abs().max()}"


def test_reduce_sum_no_keepdims():
    shape = [7, 2, 4, 1024]
    a = torch.randn(shape, dtype=torch.float16, device=DEVICE) * 0.1
    result = ark.reduce_sum(a, axis=3, keepdims=False).eval()
    expected = torch.sum(a, dim=3, keepdim=False)
    atol = shape[3] * 2e-2
    assert torch.allclose(
        result, expected, atol=atol, rtol=1e-2
    ), f"max_diff={(result - expected).abs().max()}"


def test_reduce_max_fp32():
    a = torch.randn(1, 1, 2, 8192, dtype=torch.float32, device=DEVICE)
    result = ark.reduce_max(a, axis=-1).eval()
    expected = torch.max(a, dim=-1, keepdim=True).values
    assert torch.allclose(result, expected, atol=0, rtol=0)


def test_reduce_mean_fp32():
    a = torch.randn(1, 1, 2, 8192, dtype=torch.float32, device=DEVICE) * 0.1
    result = ark.reduce_mean(a, axis=-1).eval()
    expected = torch.mean(a, dim=-1, keepdim=True)
    assert torch.allclose(result, expected, atol=1e-4, rtol=1e-4)
