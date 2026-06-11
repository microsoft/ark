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
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_reduce_sum_half(axis, dtype):
    shape = [7, 2, 4, 1024]
    a = torch.randn(shape, dtype=dtype, device=DEVICE) * 0.1
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_reduce_max(dtype):
    a = torch.randn(1, 1, 2, 8192, dtype=dtype, device=DEVICE)
    result = ark.reduce_max(a, axis=-1).eval()
    expected = torch.max(a, dim=-1, keepdim=True).values
    assert torch.allclose(result, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_reduce_mean(dtype):
    a = torch.randn(1, 1, 2, 8192, dtype=dtype, device=DEVICE) * 0.1
    result = ark.reduce_mean(a, axis=-1).eval()
    expected = torch.mean(a, dim=-1, keepdim=True)
    atol = 1e-4 if dtype == torch.float32 else 1e-2
    rtol = 1e-4 if dtype == torch.float32 else 1e-2
    assert torch.allclose(result, expected, atol=atol, rtol=rtol)


def test_reduce_sum_fused_tile():
    """WarpWise reduce_sum with a multi-row Tile (rows > 1, reduce-axis dim = 1)."""
    shape = [1, 1, 4, 1024]
    a = torch.randn(shape, dtype=torch.float32, device=DEVICE) * 0.1
    with ark.PlannerContext(
        config={
            "NumWarps": 1,
            "SramBytes": 256,
            "ImplType": "WarpWise",
            "Tile": [1, 1, 2, 1],
        }
    ):
        result = ark.reduce_sum(a, axis=3).eval()
    expected = torch.sum(a, dim=3, keepdim=True)
    atol = shape[3] * 1e-5
    assert torch.allclose(
        result, expected, atol=atol, rtol=1e-4
    ), f"max_diff={(result - expected).abs().max()}"


def test_reduce_sum_fused_tile_elementwise():
    """ElementWise reduce_sum on axis 2 with a non-default Tile."""
    shape = [1, 2, 8, 512]
    a = torch.randn(shape, dtype=torch.float32, device=DEVICE) * 0.1
    with ark.PlannerContext(
        config={
            "NumWarps": 1,
            "SramBytes": 0,
            "ImplType": "ElementWise",
            "Tile": [1, 1, 1, 64],
        }
    ):
        result = ark.reduce_sum(a, axis=2).eval()
    expected = torch.sum(a, dim=2, keepdim=True)
    atol = shape[2] * 1e-5
    assert torch.allclose(
        result, expected, atol=atol, rtol=1e-4
    ), f"max_diff={(result - expected).abs().max()}"


def test_reduce_tile_axis_validation():
    """Tile dimension on the reduce axis != 1 must raise RuntimeError."""
    shape = [1, 1, 4, 1024]
    a = torch.randn(shape, dtype=torch.float32, device=DEVICE) * 0.1
    with pytest.raises(RuntimeError):
        with ark.PlannerContext(
            config={
                "NumWarps": 1,
                "SramBytes": 256,
                "ImplType": "WarpWise",
                "Tile": [1, 1, 1, 4],
            }
        ):
            ark.reduce_sum(a, axis=3).eval()
