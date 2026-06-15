# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for ARK all-reduce wrapper at Qwen3 TP shapes.

Two tiers:
  - **CPU-only (always run in CI):** validation logic — alignment checks,
    dtype guards, contiguity guards, flatten/reshape round-trip.
  - **Multi-GPU (skip if ``torch.cuda.device_count() < 2``):** functional
    correctness via ``multiprocessing`` — each rank fills its tensor with
    ``rank + 1``, runs all-reduce, asserts output == sum(1..world_size).

The CI runner has 1 GPU, so multi-GPU tests skip cleanly.
"""

import os
import subprocess
import sys

import pytest
import torch

from .ark_allreduce import validate_allreduce_input

_CUDA = torch.cuda.is_available()
_NUM_GPUS = torch.cuda.device_count() if _CUDA else 0

requires_multi_gpu = pytest.mark.skipif(
    _NUM_GPUS < 2,
    reason=f"Need >= 2 GPUs, have {_NUM_GPUS}",
)


# -----------------------------------------------------------------------
# Tier 1: CPU-only validation tests (always run)
# -----------------------------------------------------------------------


class TestValidation:
    """Tests for ``validate_allreduce_input`` — no GPU required."""

    def test_rejects_fp32(self):
        """float32 dtype raises ValueError."""
        x = torch.randn(4096, dtype=torch.float32)
        with pytest.raises(ValueError, match="float16"):
            validate_allreduce_input(x, world_size=2)

    def test_rejects_bf16(self):
        """bfloat16 dtype raises ValueError."""
        x = torch.randn(4096, dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="float16"):
            validate_allreduce_input(x, world_size=2)

    def test_rejects_non_contiguous(self):
        """Non-contiguous tensor raises ValueError."""
        x = torch.randn(8, 4096, dtype=torch.float16)[:, ::2]
        assert not x.is_contiguous()
        with pytest.raises(ValueError, match="contiguous"):
            validate_allreduce_input(x, world_size=2)

    def test_rejects_bad_alignment_tp2(self):
        """Element count not divisible by 4*2=8 raises ValueError."""
        # 7 elements — not divisible by 8
        x = torch.randn(7, dtype=torch.float16)
        with pytest.raises(ValueError, match="divisible"):
            validate_allreduce_input(x, world_size=2)

    def test_rejects_bad_alignment_tp8(self):
        """Element count not divisible by 4*8=32 raises ValueError."""
        # 24 elements — divisible by 8 but not by 32
        x = torch.randn(24, dtype=torch.float16)
        with pytest.raises(ValueError, match="divisible"):
            validate_allreduce_input(x, world_size=8)

    def test_accepts_prefill_shape_tp8(self):
        """Prefill shape (2048, 4096) with TP=8 passes validation."""
        x = torch.randn(2048, 4096, dtype=torch.float16)
        validate_allreduce_input(x, world_size=8)  # no exception

    def test_accepts_decode_shape_tp8(self):
        """Decode shape (1, 4096) with TP=8 passes validation."""
        x = torch.randn(1, 4096, dtype=torch.float16)
        validate_allreduce_input(x, world_size=8)  # no exception

    def test_accepts_1d_tensor(self):
        """1-D tensor with aligned count passes validation."""
        x = torch.randn(4096, dtype=torch.float16)
        validate_allreduce_input(x, world_size=2)  # no exception

    def test_accepts_tp2(self):
        """Element count divisible by 4*2=8 passes validation."""
        x = torch.randn(32, dtype=torch.float16)
        validate_allreduce_input(x, world_size=2)  # no exception

    def test_rejects_world_size_zero(self):
        """world_size=0 raises ValueError (avoids ZeroDivisionError)."""
        x = torch.randn(4096, dtype=torch.float16)
        with pytest.raises(ValueError, match="world_size"):
            validate_allreduce_input(x, world_size=0)

    def test_rejects_world_size_negative(self):
        """Negative world_size raises ValueError."""
        x = torch.randn(4096, dtype=torch.float16)
        with pytest.raises(ValueError, match="world_size"):
            validate_allreduce_input(x, world_size=-1)


class TestFlattenReshapeLogic:
    """Verify flatten/reshape round-trip logic used by ark_allreduce (CPU tensors, no ARK dependency)."""

    def test_2d_roundtrip(self):
        """Flatten to 1-D and reshape back preserves data and shape."""
        shape = (2048, 4096)
        x = torch.randn(shape, dtype=torch.float16)
        x_flat = x.reshape(-1)
        assert x_flat.shape == (2048 * 4096,)
        x_back = x_flat.reshape(shape)
        assert x_back.shape == shape
        assert torch.equal(x, x_back)

    def test_1d_roundtrip(self):
        """1-D tensor reshape(-1) is a no-op."""
        x = torch.randn(4096, dtype=torch.float16)
        x_flat = x.reshape(-1)
        assert torch.equal(x, x_flat)

    def test_decode_shape(self):
        """Decode shape (1, 4096) flattens to (4096,) and back."""
        shape = (1, 4096)
        x = torch.randn(shape, dtype=torch.float16)
        x_flat = x.reshape(-1)
        assert x_flat.shape == (4096,)
        x_back = x_flat.reshape(shape)
        assert torch.equal(x, x_back)


# -----------------------------------------------------------------------
# Tier 2: Multi-GPU functional tests (skip on 1-GPU CI)
# -----------------------------------------------------------------------

_ALLREDUCE_WORKER_SCRIPT = '''
"""Worker script for multi-GPU all-reduce test.

Launched as a subprocess to avoid CUDA context pollution in the test process.
Each rank fills its tensor with (rank + 1), runs all-reduce, and checks
that the result equals sum(1..world_size).
"""
import sys
import torch
import ark

rank = int(sys.argv[1])
world_size = int(sys.argv[2])
n_elements = int(sys.argv[3])

ark.set_rank(rank)
ark.set_world_size(world_size)

# Fill with rank + 1
x = torch.full((n_elements,), rank + 1, dtype=torch.float16, device=f"cuda:{rank}")

ark.init()
result = ark.all_reduce_packet(x, rank, world_size)

with ark.Runtime() as rt:
    rt.launch(device_id=rank)
    rt.run()
    out = result.to_torch()

# Expected: sum of (1 + 2 + ... + world_size)
expected = world_size * (world_size + 1) / 2
if not torch.allclose(out, torch.full_like(out, expected), atol=1e-2, rtol=1e-2):
    bad = (out - expected).abs().max().item()
    print(f"FAIL rank={rank}: max_diff={bad}", file=sys.stderr)
    sys.exit(1)
print(f"PASS rank={rank}")
'''


def _run_allreduce_subprocess(
    world_size: int, n_elements: int, timeout: int = 120
):
    """Spawn *world_size* workers, each running the all-reduce script."""
    procs = []
    for rank in range(world_size):
        p = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _ALLREDUCE_WORKER_SCRIPT,
                str(rank),
                str(world_size),
                str(n_elements),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": ",".join(
                    str(i) for i in range(world_size)
                ),
            },
        )
        procs.append(p)

    failures = []
    for rank, p in enumerate(procs):
        stdout, stderr = p.communicate(timeout=timeout)
        if p.returncode != 0:
            failures.append(
                f"rank {rank} exit={p.returncode}: "
                f"{stderr.decode().strip()[-500:]}"
            )

    if failures:
        raise AssertionError(
            f"All-reduce failed for {len(failures)}/{world_size} ranks:\n"
            + "\n".join(failures)
        )


# TODO: test ark_allreduce() wrapper end-to-end once subprocess import path is resolved


@requires_multi_gpu
def test_allreduce_decode_tp2():
    """All-reduce at decode shape (4096 elems) with TP=2."""
    _run_allreduce_subprocess(world_size=2, n_elements=4096)


@requires_multi_gpu
def test_allreduce_prefill_tp2():
    """All-reduce at prefill shape (8,388,608 elems) with TP=2."""
    _run_allreduce_subprocess(world_size=2, n_elements=2048 * 4096)


@requires_multi_gpu
@pytest.mark.skipif(
    _NUM_GPUS < 8,
    reason=f"Need >= 8 GPUs, have {_NUM_GPUS}",
)
def test_allreduce_prefill_tp8():
    """All-reduce at prefill shape (8,388,608 elems) with TP=8."""
    _run_allreduce_subprocess(world_size=8, n_elements=2048 * 4096)


@requires_multi_gpu
@pytest.mark.skipif(
    _NUM_GPUS < 8,
    reason=f"Need >= 8 GPUs, have {_NUM_GPUS}",
)
def test_allreduce_decode_tp8():
    """All-reduce at decode shape (4096 elems) with TP=8."""
    _run_allreduce_subprocess(world_size=8, n_elements=4096)
