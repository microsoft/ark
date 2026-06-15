# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Regression tests for composed-graph shapes that previously caused
cudaErrorMisalignedAddress or wrong results.

Root cause: Cast and RoPE kernels hardcode NelemPerThread=2, but the
default tile selection could choose tile_y=1 when H>W, causing
vectorized loads/stores at misaligned addresses.
"""

import pytest

from common import ark

torch = pytest.importorskip("torch")
import torch.nn.functional as F

DEVICE = "cuda:0"


# ---------------------------------------------------------------------------
# Cast at 4-D production shapes (was: misaligned address when H > W)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        (1, 4, 128, 32),
        (1, 4, 16, 32),
        (64, 32),
        (2048, 128),
        (1, 4, 128, 1),
        (1, 4, 128, 2),
    ],
)
@pytest.mark.parametrize(
    "src_dtype, dst_dtype, ark_dst",
    [
        (torch.float16, torch.float32, ark.fp32),
        (torch.float32, torch.float16, ark.fp16),
        (torch.bfloat16, torch.float32, ark.fp32),
    ],
)
def test_cast_shapes(shape, src_dtype, dst_dtype, ark_dst):
    """Cast must produce correct output at shapes where H > W."""
    x = torch.randn(shape, dtype=src_dtype, device=DEVICE)
    result = ark.cast(x, ark_dst).eval()
    expected = x.to(dst_dtype)
    assert (
        result.dtype == dst_dtype
    ), f"expected {dst_dtype}, got {result.dtype}"
    assert torch.allclose(
        result, expected, atol=0, rtol=0
    ), f"cast {src_dtype}->{dst_dtype} shape={shape} max_diff={(result - expected).abs().max()}"


# ---------------------------------------------------------------------------
# RoPE at 4-D production shapes (was: wrong output when H > W)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        (1, 4, 128, 32),
        (1, 4, 16, 32),
        (1, 1, 8, 64),
        (1, 32, 128, 128),
        (1, 4, 128, 2),
    ],
)
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32, torch.bfloat16]
)
def test_rope_shapes(shape, dtype):
    """RoPE must match the complex-multiply reference at production shapes."""
    x = torch.randn(shape, dtype=dtype, device=DEVICE)
    other = torch.randn(shape, dtype=dtype, device=DEVICE)
    result = ark.rope(x, other).eval()
    # Reference: complex multiply on consecutive pairs
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
    ), f"rope shape={shape} dtype={dtype} max_diff={(result - expected).abs().max()}"


# ---------------------------------------------------------------------------
# Composed layernorm at 4-D shapes (exercises cast + reduce + broadcast)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        (1, 4, 128, 32),
        (64, 32),
        (4, 8, 256),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_layernorm_shapes(shape, dtype):
    """Composed layernorm must be correct at shapes that trigger H>W tiles."""
    a = torch.randn(shape, dtype=dtype, device=DEVICE)
    result = ark.layernorm(a, eps=1e-6).eval()
    mean = a.mean(dim=-1, keepdim=True)
    var = ((a - mean) ** 2).mean(dim=-1, keepdim=True)
    expected = (a - mean) / torch.sqrt(var + 1e-6)
    atol = 1e-4 if dtype == torch.float32 else 1e-2
    assert torch.allclose(
        result, expected, atol=atol, rtol=1e-3
    ), f"layernorm shape={shape} dtype={dtype} max_diff={(result - expected).abs().max()}"


# ---------------------------------------------------------------------------
# Composed RMSNorm at 4-D shapes (was: cudaErrorMisalignedAddress)
# ---------------------------------------------------------------------------


def _torch_rmsnorm(x, eps=1e-6):
    """Pure-torch RMSNorm reference (fp32 accumulation)."""
    x_fp32 = x.float()
    rms = torch.sqrt((x_fp32 * x_fp32).mean(dim=-1, keepdim=True) + eps)
    return (x_fp32 / rms).to(x.dtype)


def _ark_rmsnorm(x_tensor, out_dtype=None, eps=1e-6):
    """Composed ARK RMSNorm: cast->square->reduce_mean->add->rsqrt->mul->cast."""
    x_fp32 = ark.cast(x_tensor, ark.fp32)
    sq = ark.mul(x_fp32, x_fp32)
    mean_sq = ark.reduce_mean(sq, axis=-1)
    rms_inv = ark.rsqrt(ark.add(mean_sq, eps))
    out_fp32 = ark.mul(x_fp32, rms_inv)
    return ark.cast(out_fp32, out_dtype if out_dtype is not None else ark.fp16)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 4, 128, 32),
        (64, 32),
        (1, 4, 16, 32),
    ],
)
def test_rmsnorm_composed(shape):
    """Composed RMSNorm must not crash and must match torch reference."""
    x = torch.randn(shape, dtype=torch.float16, device=DEVICE)
    result = _ark_rmsnorm(x).eval()
    expected = _torch_rmsnorm(x)
    assert torch.allclose(
        result, expected, atol=5e-3, rtol=1e-3
    ), f"rmsnorm shape={shape} max_diff={(result - expected).abs().max()}"


# ---------------------------------------------------------------------------
# Composed softmax at shapes with H > W
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        (1, 4, 128, 32),
        (64, 32),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_softmax_h_gt_w(shape, dtype):
    """Softmax at H>W shapes (does not use cast, but exercises reduce+broadcast tiles)."""
    a = torch.randn(shape, dtype=dtype, device=DEVICE)
    result = ark.softmax(a).eval()
    expected = F.softmax(a, dim=-1)
    atol = 1e-5 if dtype == torch.float32 else 1e-3
    assert torch.allclose(
        result, expected, atol=atol, rtol=1e-3
    ), f"softmax shape={shape} dtype={dtype} max_diff={(result - expected).abs().max()}"
