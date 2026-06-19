# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Equivalence tests for Qwen3 embedding, final RMSNorm, and lm_head.

Each ARK case runs in its own subprocess.  This keeps the tests focused on
component correctness and avoids reusing ARK's singleton executor after a
previous CUDA graph has run.
"""

import json
import os
import subprocess
import sys

import pytest

try:
    import torch
except ImportError:
    pytest.skip("torch is not installed", allow_module_level=True)

try:
    from ._env import _subprocess_env
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _env import _subprocess_env


_WORKER_SCRIPT = r"""
import json
import sys

qwen3_dir = sys.argv[1]
case = json.loads(sys.argv[2])
sys.path.insert(0, qwen3_dir)

import ark
import torch
import torch.nn.functional as F

from embed_head import (
    qwen3_embed_head,
    qwen3_final_rmsnorm,
    qwen3_lm_head,
    qwen3_token_embedding,
    torch_qwen3_embed_head,
    torch_qwen3_final_rmsnorm,
)

DEVICE = "cuda:0"


def _dtype(name):
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise AssertionError(f"unsupported dtype: {name}")


def _assert_close(result, expected, atol, rtol):
    max_diff = (result - expected).abs().max().item()
    assert torch.allclose(result, expected, atol=atol, rtol=rtol), (
        f"max_diff={max_diff} result_dtype={result.dtype} "
        f"expected_dtype={expected.dtype}"
    )


if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")

ark.init()
kind = case["kind"]
dtype = _dtype(case["dtype"])

if kind == "embedding":
    tokens = torch.tensor(
        [[0, 3, 5, 7], [9, 1, 11, 2]], dtype=torch.int32, device=DEVICE
    )
    weight = torch.randn(16, 64, dtype=dtype, device=DEVICE)
    result = qwen3_token_embedding(tokens, weight).eval()
    expected = F.embedding(tokens.long(), weight)
    assert result.shape == expected.shape
    _assert_close(result, expected, atol=0, rtol=0)
elif kind == "rmsnorm":
    shape = tuple(case["shape"])
    torch.manual_seed(100 + shape[-1])
    hidden = torch.randn(shape, dtype=dtype, device=DEVICE) * 0.2
    norm_weight = torch.randn(shape[-1], dtype=dtype, device=DEVICE) * 0.1
    result = qwen3_final_rmsnorm(hidden, norm_weight).eval()
    expected = torch_qwen3_final_rmsnorm(hidden, norm_weight)
    assert result.dtype == dtype
    assert result.shape == expected.shape
    atol = 5e-3 if dtype == torch.float16 else 2e-2
    _assert_close(result, expected, atol=atol, rtol=2e-2)
elif kind == "lm_head":
    torch.manual_seed(200)
    hidden = torch.randn(1, 8, 128, dtype=dtype, device=DEVICE) * 0.1
    lm_head_weight = torch.randn(320, 128, dtype=dtype, device=DEVICE) * 0.02
    result = qwen3_lm_head(hidden, lm_head_weight).eval()
    expected = hidden @ lm_head_weight.t()
    assert result.shape == expected.shape
    _assert_close(result, expected, atol=1e-1, rtol=5e-2)
elif kind == "composed":
    tokens_shape = tuple(case["tokens_shape"])
    hidden_size = case["hidden_size"]
    vocab_size = case["vocab_size"]
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
elif kind == "qwen_hidden_decode":
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
else:
    raise AssertionError(f"unsupported case: {kind}")
"""

CASES = [{"kind": "embedding", "dtype": dtype} for dtype in ("fp16", "bf16")]
CASES += [
    {"kind": "rmsnorm", "dtype": dtype, "shape": shape}
    for dtype in ("fp16", "bf16")
    for shape in ([1, 1, 128], [1, 128, 256])
]
CASES += [{"kind": "lm_head", "dtype": dtype} for dtype in ("fp16", "bf16")]
CASES += [
    {
        "kind": "composed",
        "dtype": dtype,
        "tokens_shape": tokens_shape,
        "hidden_size": 256,
        "vocab_size": 512,
    }
    for dtype in ("fp16", "bf16")
    for tokens_shape in ([1, 1], [1, 128])
]
CASES += [{"kind": "qwen_hidden_decode", "dtype": "bf16"}]


def _case_id(case):
    fields = [case["kind"], case["dtype"]]
    if "shape" in case:
        fields.append("x".join(str(v) for v in case["shape"]))
    if "tokens_shape" in case:
        fields.append("tokens" + "x".join(str(v) for v in case["tokens_shape"]))
    return "-".join(fields)


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")


def _tail(text, limit=1200):
    return text.strip()[-limit:]


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_embed_head_case(case):
    """Run one ARK equivalence case in a fresh Python process."""
    _require_cuda()
    qwen3_dir = os.path.dirname(os.path.abspath(__file__))
    proc = subprocess.run(
        [sys.executable, "-c", _WORKER_SCRIPT, qwen3_dir, json.dumps(case)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd="/",
        env=_subprocess_env(1),
        timeout=180,
        check=False,
    )
    assert proc.returncode == 0, (
        f"case={case} exit={proc.returncode}\n"
        f"stdout={_tail(proc.stdout)}\n"
        f"stderr={_tail(proc.stderr)}"
    )
