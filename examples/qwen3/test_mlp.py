# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Equivalence tests: ARK SwiGLU MLP vs torch reference.

All GPU tests are gated with ``skipif(not cuda)``.
"""

import subprocess
import sys
import os

import pytest
import torch

_CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not _CUDA, reason="CUDA not available")

_SEED = 42


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _small_cfg():
    """Return a small Qwen3Config suitable for unit tests."""
    from .qwen3_config import Qwen3Config

    return Qwen3Config(
        n_layers=1,
        hidden_dim=128,
        n_q_heads=4,
        n_kv_heads=2,
        head_dim=32,
        intermediate_dim=256,
        rms_norm_eps=1e-6,
        rope_theta=1e6,
        max_seq_len=256,
    )


def _build_ref_mlp(cfg):
    """Instantiate a torch SwiGLUMLP with fixed seed on CUDA."""
    from .qwen3_ref import SwiGLUMLP

    torch.manual_seed(_SEED)
    return SwiGLUMLP(cfg).cuda().half().eval()


def _run_mlp_equivalence(cfg, B, S, seed_offset=0):
    """Run ARK vs torch MLP equivalence for given cfg, B, S."""
    from .ark_mlp import ark_swiglu_mlp
    from .equiv import assert_close

    mlp = _build_ref_mlp(cfg)

    torch.manual_seed(_SEED + seed_offset)
    x = torch.randn(B, S, cfg.hidden_dim, device="cuda", dtype=torch.float16)

    with torch.no_grad():
        ref = mlp(x)

    with torch.no_grad():
        ark_out = ark_swiglu_mlp(
            x,
            mlp.gate_proj.weight.detach(),
            mlp.up_proj.weight.detach(),
            mlp.down_proj.weight.detach(),
            cfg,
        ).eval()

    assert ark_out.shape == ref.shape
    assert_close(ark_out, ref, atol=5e-3, rtol=5e-3, msg=f"MLP B={B} S={S}")


# ---------------------------------------------------------------------------
# Intermediate check: SiLU·gate
# ---------------------------------------------------------------------------


@requires_cuda
def test_silu_gate():
    """torch_silu_gate matches F.silu(gate) * up."""
    import torch.nn.functional as F

    from .ark_mlp import torch_silu_gate
    from .equiv import assert_close

    torch.manual_seed(_SEED)
    gate = torch.randn(64, 256, device="cuda", dtype=torch.float16)
    up = torch.randn(64, 256, device="cuda", dtype=torch.float16)

    ref = F.silu(gate) * up
    out = torch_silu_gate(gate, up)

    assert_close(out, ref, atol=1e-6, rtol=1e-6, msg="SiLU·gate mismatch")


# ---------------------------------------------------------------------------
# Full MLP equivalence tests
# ---------------------------------------------------------------------------


@requires_cuda
def test_mlp_small():
    """ARK MLP matches SwiGLUMLP at B=1, S=16 (small shape)."""
    _run_mlp_equivalence(_small_cfg(), B=1, S=16, seed_offset=10)


@requires_cuda
def test_mlp_prefill():
    """ARK MLP matches SwiGLUMLP at B=1, S=128 (prefill shape)."""
    _run_mlp_equivalence(_small_cfg(), B=1, S=128, seed_offset=11)


@requires_cuda
def test_mlp_decode():
    """ARK MLP matches SwiGLUMLP at B=1, S=1 (decode step)."""
    _run_mlp_equivalence(_small_cfg(), B=1, S=1, seed_offset=12)


@requires_cuda
def test_mlp_batch():
    """ARK MLP matches SwiGLUMLP at B=2, S=16 (multi-batch)."""
    _run_mlp_equivalence(_small_cfg(), B=2, S=16, seed_offset=13)


# ---------------------------------------------------------------------------
# Output shape and dtype
# ---------------------------------------------------------------------------


@requires_cuda
def test_mlp_output_shape():
    """ARK MLP output has correct shape and dtype."""
    from .ark_mlp import ark_swiglu_mlp

    cfg = _small_cfg()
    mlp = _build_ref_mlp(cfg)

    B, S = 2, 32
    torch.manual_seed(_SEED + 20)
    x = torch.randn(B, S, cfg.hidden_dim, device="cuda", dtype=torch.float16)

    with torch.no_grad():
        out = ark_swiglu_mlp(
            x,
            mlp.gate_proj.weight.detach(),
            mlp.up_proj.weight.detach(),
            mlp.down_proj.weight.detach(),
            cfg,
        ).eval()

    assert out.shape == (B, S, cfg.hidden_dim)
    assert out.dtype == torch.float16


# ---------------------------------------------------------------------------
# xfail: ARK silu·gate at (2048, 12288) — upstream ARK planner bug
# ---------------------------------------------------------------------------


@requires_cuda
@pytest.mark.xfail(
    reason="ARK planner bug: composed graph crashes at (2048, 12288)",
    strict=False,
)
def test_ark_silu_gate_large_xfail():
    """Document that ark_silu_gate crashes at Qwen3-8B intermediate_dim.

    Same class of upstream ARK composed-graph planner bug as Q4's
    4-D shape crash. Runs in a subprocess to avoid poisoning the
    CUDA context.
    """
    script = (
        "import torch, ark\n"
        "torch.manual_seed(42)\n"
        "gate = torch.randn(2048, 12288, device='cuda', dtype=torch.float16)\n"
        "up = torch.randn(2048, 12288, device='cuda', dtype=torch.float16)\n"
        "ark.init()\n"
        "sig = ark.sigmoid(gate)\n"
        "silu = ark.mul(gate, sig)\n"
        "result = ark.mul(silu, up)\n"
        "out = result.eval()\n"
        "assert out.shape == (2048, 12288)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        env=os.environ.copy(),
    )
    assert (
        result.returncode == 0
    ), f"Subprocess exited {result.returncode}: {result.stderr[-500:]}"
