# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Numerical tests for unary math ops: exp, gelu, relu, sigmoid, sqrt, rsqrt."""

import pytest
import torch
import torch.nn.functional as F
from conftest import ark, DEVICE


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_exp(dtype):
    a = torch.randn(4, 2, 1024, dtype=dtype, device=DEVICE)
    atol = 1e-5 if dtype == torch.float32 else 1e-2
    assert torch.allclose(ark.exp(a).eval(), torch.exp(a), atol=atol, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gelu(dtype):
    a = torch.randn(4, 2, 1024, dtype=dtype, device=DEVICE)
    atol = 1e-5 if dtype == torch.float32 else 1e-2
    assert torch.allclose(ark.gelu(a).eval(), F.gelu(a, approximate="tanh"), atol=atol, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_relu(dtype):
    a = torch.randn(4, 2, 1024, dtype=dtype, device=DEVICE)
    assert torch.allclose(ark.relu(a).eval(), F.relu(a), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sigmoid(dtype):
    a = torch.randn(4, 2, 1024, dtype=dtype, device=DEVICE)
    atol = 1e-5 if dtype == torch.float32 else 1e-2
    assert torch.allclose(ark.sigmoid(a).eval(), torch.sigmoid(a), atol=atol, rtol=0)


def test_sqrt_fp32():
    a = torch.rand(4, 2, 1024, dtype=torch.float32, device=DEVICE) + 0.01
    assert torch.allclose(ark.sqrt(a).eval(), torch.sqrt(a), atol=1e-6, rtol=0)


def test_rsqrt_fp32():
    a = torch.rand(4, 2, 1024, dtype=torch.float32, device=DEVICE) + 0.01
    assert torch.allclose(ark.rsqrt(a).eval(), torch.rsqrt(a), atol=1e-4, rtol=0)
