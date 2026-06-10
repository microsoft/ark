# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Numerical tests for matmul: NN, NT, TN, TT, batched."""

import pytest

from common import ark

torch = pytest.importorskip("torch")

DEVICE = "cuda:0"


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16]
)
def test_matmul_nn(dtype):
    M, N, K = 256, 256, 512
    a = torch.randn(M, K, dtype=dtype, device=DEVICE)
    b = torch.randn(K, N, dtype=dtype, device=DEVICE)
    result = ark.matmul(a, b).eval()
    expected = a @ b
    atol = 1e-3 if dtype == torch.float32 else 3e-1
    assert torch.allclose(
        result, expected, atol=atol, rtol=1e-2
    ), f"max_diff={(result - expected).abs().max()}"


def test_matmul_nt():
    M, N, K = 256, 256, 512
    a = torch.randn(M, K, dtype=torch.float16, device=DEVICE)
    b = torch.randn(N, K, dtype=torch.float16, device=DEVICE)
    result = ark.matmul(a, b, transpose_other=True).eval()
    expected = a @ b.t()
    assert torch.allclose(
        result, expected, atol=3e-1, rtol=1e-2
    ), f"max_diff={(result - expected).abs().max()}"


def test_matmul_tn():
    M, N, K = 256, 256, 512
    a = torch.randn(K, M, dtype=torch.float16, device=DEVICE)
    b = torch.randn(K, N, dtype=torch.float16, device=DEVICE)
    result = ark.matmul(a, b, transpose_input=True).eval()
    expected = a.t() @ b
    assert torch.allclose(
        result, expected, atol=3e-1, rtol=1e-2
    ), f"max_diff={(result - expected).abs().max()}"


def test_matmul_tt():
    M, N, K = 256, 256, 512
    a = torch.randn(K, M, dtype=torch.float16, device=DEVICE)
    b = torch.randn(N, K, dtype=torch.float16, device=DEVICE)
    result = ark.matmul(a, b, transpose_input=True, transpose_other=True).eval()
    expected = a.t() @ b.t()
    assert torch.allclose(
        result, expected, atol=3e-1, rtol=1e-2
    ), f"max_diff={(result - expected).abs().max()}"


def test_matmul_batched():
    B, M, N, K = 4, 256, 256, 512
    a = torch.randn(B, M, K, dtype=torch.float16, device=DEVICE)
    b = torch.randn(B, K, N, dtype=torch.float16, device=DEVICE)
    result = ark.matmul(a, b).eval()
    expected = a @ b
    assert torch.allclose(
        result, expected, atol=3e-1, rtol=1e-2
    ), f"max_diff={(result - expected).abs().max()}"
