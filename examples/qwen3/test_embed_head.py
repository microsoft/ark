# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Equivalence tests for Qwen3 embedding, final RMSNorm, and lm_head.

The tests compare the ARK composed graph with a torch reference for fp16 and
bf16.  They cover decode-shaped input and a CI-safe prefill-shaped input.
"""

import os
import sys

import pytest

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    pytest.skip("torch is not installed", allow_module_level=True)

try:
    import ark
except ImportError:
    pytest.skip("ark is not installed", allow_module_level=True)

try:
    from .embed_head import (
        qwen3_embed_head,
        qwen3_final_rmsnorm,
        qwen3_lm_head,
        qwen3_token_embedding,
        torch_qwen3_embed_head,
        torch_qwen3_final_rmsnorm,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from embed_head import (  # noqa: E402
        qwen3_embed_head,
        qwen3_final_rmsnorm,
        qwen3_lm_head,
        qwen3_token_embedding,
        torch_qwen3_embed_head,
        torch_qwen3_final_rmsnorm,
    )

DEVICE = "cuda:0"


@pytest.fixture(autouse=True)
def _reset_ark():
    """Reset the singleton graph before each parametrized test."""
    ark.init()


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")


def _assert_close(result, expected, atol, rtol):
    max_diff = (result - expected).abs().max().item()
    assert torch.allclose(result, expected, atol=atol, rtol=rtol), (
        f"max_diff={max_diff} result_dtype={result.dtype} "
        f"expected_dtype={expected.dtype}"
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_token_embedding_matches_torch(dtype):
    """ARK embedding must match torch index semantics for int32 tokens."""
    _require_cuda()
    tokens = torch.tensor(
        [[0, 3, 5, 7], [9, 1, 11, 2]], dtype=torch.int32, device=DEVICE
    )
    weight = torch.randn(16, 64, dtype=dtype, device=DEVICE)

    result = qwen3_token_embedding(tokens, weight).eval()
    expected = F.embedding(tokens.long(), weight)

    assert result.shape == expected.shape
    _assert_close(result, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 1, 128), (1, 128, 256)])
def test_final_rmsnorm_matches_torch(dtype, shape):
    """ARK final RMSNorm must use fp32 reduction and restore input dtype."""
    _require_cuda()
    torch.manual_seed(100 + shape[-1])
    hidden = torch.randn(shape, dtype=dtype, device=DEVICE) * 0.2
    norm_weight = torch.randn(shape[-1], dtype=dtype, device=DEVICE) * 0.1

    result = qwen3_final_rmsnorm(hidden, norm_weight).eval()
    expected = torch_qwen3_final_rmsnorm(hidden, norm_weight)

    assert result.dtype == dtype
    assert result.shape == expected.shape
    atol = 5e-3 if dtype == torch.float16 else 2e-2
    _assert_close(result, expected, atol=atol, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_lm_head_matches_torch(dtype):
    """ARK lm_head must project hidden states with weight transposition."""
    _require_cuda()
    torch.manual_seed(200)
    hidden = torch.randn(1, 8, 128, dtype=dtype, device=DEVICE) * 0.1
    lm_head_weight = torch.randn(320, 128, dtype=dtype, device=DEVICE) * 0.02

    result = qwen3_lm_head(hidden, lm_head_weight).eval()
    expected = hidden @ lm_head_weight.t()

    assert result.shape == expected.shape
    _assert_close(result, expected, atol=1e-1, rtol=5e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "tokens_shape,hidden_size,vocab_size",
    [
        ((1, 1), 256, 512),  # decode-shaped
        ((1, 128), 256, 512),  # CI-safe prefill-shaped
    ],
)
def test_composed_embed_head_matches_torch(
    dtype, tokens_shape, hidden_size, vocab_size
):
    """ARK composed embedding -> final RMSNorm -> lm_head matches torch."""
    _require_cuda()
    torch.manual_seed(300 + tokens_shape[1])
    tokens = torch.randint(
        0, vocab_size, tokens_shape, dtype=torch.int32, device=DEVICE
    )
    embed_weight = torch.randn(
        vocab_size, hidden_size, dtype=dtype, device=DEVICE
    ) * 0.2
    norm_weight = torch.randn(hidden_size, dtype=dtype, device=DEVICE) * 0.1
    lm_head_weight = torch.randn(
        vocab_size, hidden_size, dtype=dtype, device=DEVICE
    ) * 0.02

    result = qwen3_embed_head(
        tokens, embed_weight, norm_weight, lm_head_weight
    ).eval()
    expected = torch_qwen3_embed_head(
        tokens, embed_weight, norm_weight, lm_head_weight
    )

    assert result.shape == (*tokens_shape, vocab_size)
    _assert_close(result, expected, atol=2e-1, rtol=8e-2)


def test_composed_qwen_hidden_decode_shape_bf16():
    """Decode path covers Qwen3 hidden width with a CI-safe vocab shard."""
    _require_cuda()
    dtype = torch.bfloat16
    tokens_shape = (1, 1)
    hidden_size = 4096
    vocab_size = 1024
    torch.manual_seed(4096)
    tokens = torch.randint(
        0, vocab_size, tokens_shape, dtype=torch.int32, device=DEVICE
    )
    embed_weight = torch.randn(
        vocab_size, hidden_size, dtype=dtype, device=DEVICE
    ) * 0.2
    norm_weight = torch.randn(hidden_size, dtype=dtype, device=DEVICE) * 0.1
    lm_head_weight = torch.randn(
        vocab_size, hidden_size, dtype=dtype, device=DEVICE
    ) * 0.02

    result = qwen3_embed_head(
        tokens, embed_weight, norm_weight, lm_head_weight
    ).eval()
    expected = torch_qwen3_embed_head(
        tokens, embed_weight, norm_weight, lm_head_weight
    )

    assert result.shape == (1, 1, vocab_size)
    _assert_close(result, expected, atol=5e-1, rtol=1e-1)
