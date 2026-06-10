# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Numerical tests for arithmetic ops: add, sub, mul, div (tensor and scalar)."""

import pytest

from common import ark

torch = pytest.importorskip("torch")

DEVICE = "cuda:0"


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16]
)
def test_add(dtype):
    a = torch.randn(8192, dtype=dtype, device=DEVICE)
    b = torch.randn(8192, dtype=dtype, device=DEVICE)
    assert torch.allclose(ark.add(a, b).eval(), a + b, atol=0, rtol=0)


def test_add_broadcast():
    a = torch.randn(4, 1024, dtype=torch.float16, device=DEVICE)
    b = torch.randn(1, 1024, dtype=torch.float16, device=DEVICE)
    assert torch.allclose(ark.add(a, b).eval(), a + b, atol=0, rtol=0)


def test_add_broadcast_3d():
    a = torch.randn(3, 1, 1024, dtype=torch.float16, device=DEVICE)
    b = torch.randn(1, 4, 1, dtype=torch.float16, device=DEVICE)
    assert torch.allclose(ark.add(a, b).eval(), a + b, atol=0, rtol=0)


def test_sub():
    dtype = torch.float32
    a = torch.randn(8192, dtype=dtype, device=DEVICE)
    b = torch.randn(8192, dtype=dtype, device=DEVICE)
    assert torch.allclose(ark.sub(a, b).eval(), a - b, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_mul(dtype):
    a = torch.randn(8192, dtype=dtype, device=DEVICE)
    b = torch.randn(8192, dtype=dtype, device=DEVICE)
    assert torch.allclose(ark.mul(a, b).eval(), a * b, atol=0, rtol=0)


def test_mul_broadcast():
    a = torch.randn(4, 1024, dtype=torch.float16, device=DEVICE)
    b = torch.randn(1, 1024, dtype=torch.float16, device=DEVICE)
    assert torch.allclose(ark.mul(a, b).eval(), a * b, atol=0, rtol=0)


def test_mul_broadcast_3d():
    a = torch.randn(3, 1, 1024, dtype=torch.float16, device=DEVICE)
    b = torch.randn(1, 4, 1, dtype=torch.float16, device=DEVICE)
    assert torch.allclose(ark.mul(a, b).eval(), a * b, atol=0, rtol=0)


def test_div_fp32():
    a = torch.randn(8192, dtype=torch.float32, device=DEVICE)
    b = torch.randn(8192, dtype=torch.float32, device=DEVICE).abs() + 0.01
    assert torch.allclose(ark.div(a, b).eval(), a / b, atol=0, rtol=0)


# Scalar operations

FACTOR = 0.75


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("shape", [(4, 2, 1), (4, 2, 1024)])
def test_scalar_mul(dtype, shape):
    a = torch.randn(shape, dtype=dtype, device=DEVICE)
    assert torch.allclose(ark.mul(a, FACTOR).eval(), a * FACTOR, atol=0, rtol=0)


@pytest.mark.parametrize("shape", [(4, 2, 1), (4, 2, 1024)])
def test_scalar_add(shape):
    a = torch.randn(shape, dtype=torch.float16, device=DEVICE)
    assert torch.allclose(ark.add(a, FACTOR).eval(), a + FACTOR, atol=0, rtol=0)


@pytest.mark.parametrize("shape", [(4, 2, 1), (4, 2, 1024)])
def test_scalar_sub(shape):
    a = torch.randn(shape, dtype=torch.float16, device=DEVICE)
    assert torch.allclose(ark.sub(a, FACTOR).eval(), a - FACTOR, atol=0, rtol=0)


@pytest.mark.parametrize("shape", [(4, 2, 1), (4, 2, 1024)])
def test_scalar_div(shape):
    a = torch.randn(shape, dtype=torch.float16, device=DEVICE)
    assert torch.allclose(
        ark.div(a, FACTOR).eval(), a / FACTOR, atol=1e-3, rtol=1e-3
    )


# Constant & scalar copy
# Scalar copy only; tensor copy tested separately.


def test_constant_fp16():
    out = ark.constant(7, (4, 2, 50), ark.fp16).eval()
    assert (out == 7).all()


def test_constant_fp32():
    out = ark.constant(7, (1,), ark.fp32).eval()
    assert out.item() == 7.0


def test_copy_scalar_fp16():
    t = torch.zeros(4, 2, 50, dtype=torch.float16, device=DEVICE)
    out = ark.copy(7.0, ark.Tensor.from_torch(t)).eval()
    assert (out == 7).all()


def test_copy_scalar_fp32():
    out = ark.copy(7.0).eval()
    assert out.item() == 7.0
