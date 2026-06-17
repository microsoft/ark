# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Equivalence tests for ARK fused-packet all-reduce at Qwen3 TP shapes.

Verifies that ``ark.all_reduce_packet`` produces the same result as a
torch all-reduce (sum) across ranks. Tests are d2h-safe: each worker
copies the result to CPU (``result.to_torch().cpu()``) AFTER stopping
the ARK runtime, then asserts on the host with ``torch.allclose``.

No torch GPU kernel is issued while the ARK runtime is launched.

Requires ≥2 GPUs; skips gracefully on single-GPU machines.
"""

import json
import os
import subprocess
import sys

import pytest
import torch

try:
    from ._env import _subprocess_env
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _env import _subprocess_env


def _gpu_count() -> int:
    """Return available CUDA device count (0 if CUDA unavailable)."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


# Worker script executed in each subprocess rank.
# Uses a deterministic seed per rank so the expected sum is reproducible.
_WORKER_SCRIPT = '''
"""Worker: run ARK all-reduce and verify result on CPU."""
import json
import os
import sys

import torch
import ark
from ark.executor import Executor

rank = int(sys.argv[1])
world_size = int(sys.argv[2])
n_elements = int(sys.argv[3])

ark.init()
ark.set_rank(rank)
ark.set_world_size(world_size)

# --- Input: deterministic per-rank values (BEFORE ARK launch) ---
torch.manual_seed(42 + rank)
x = torch.randn(n_elements, dtype=torch.float16, device=f"cuda:{rank}")

# Build ARK graph (no GPU kernel launched yet).
result = ark.all_reduce_packet(x, rank, world_size)

with ark.Runtime() as rt:
    rt.launch(device_id=rank)
    # Single iteration — correctness, not throughput.
    rt.run(iter=1)
    rt.stop()

# --- D2H transfer AFTER runtime stopped (safe: no ARK loop kernel live) ---
result_cpu = result.to_torch().cpu()

# --- Expected: sum of all ranks' inputs ---
# Regenerate all ranks' inputs on CPU and sum them.
expected = torch.zeros(n_elements, dtype=torch.float16)
for r in range(world_size):
    torch.manual_seed(42 + r)
    expected += torch.randn(n_elements, dtype=torch.float16)

# FP16 all-reduce may accumulate rounding; use relaxed tolerance.
close = torch.allclose(result_cpu, expected, rtol=1e-2, atol=1e-2)

# Report result as JSON on stdout (only rank 0 for simplicity).
if rank == 0:
    max_diff = (result_cpu - expected).abs().max().item()
    print(json.dumps({
        "rank": rank,
        "world_size": world_size,
        "n_elements": n_elements,
        "pass": close,
        "max_diff": max_diff,
    }))
    sys.stdout.flush()

# Workaround: mscclpp's UnixSocketServer destructor races during normal
# Python shutdown (static destruction order is undefined across TUs),
# causing SIGABRT.  Executor.reset() forces orderly mscclpp teardown,
# then os._exit() skips Python's atexit / gc finalizers entirely.
Executor.reset()
os._exit(0 if close else 1)
'''


def _run_allreduce_test(world_size: int, n_elements: int, timeout: int = 120):
    """Spawn *world_size* workers and assert all-reduce correctness."""
    procs = []
    for rank in range(world_size):
        procs.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    _WORKER_SCRIPT,
                    str(rank),
                    str(world_size),
                    str(n_elements),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd="/",
                env=_subprocess_env(world_size),
            )
        )

    errors = []
    result_json = None
    try:
        for rank, p in enumerate(procs):
            try:
                out, err = p.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
                errors.append(f"rank {rank}: timed out after {timeout}s")
                continue
            if p.returncode != 0:
                errors.append(
                    f"rank {rank}: exit={p.returncode} "
                    f"stderr={err.decode().strip()[-300:]}"
                )
            if rank == 0 and out.strip():
                result_json = json.loads(out.decode().strip())
    finally:
        for p in procs:
            p.kill()
            p.wait()

    assert not errors, "\n".join(errors)
    assert result_json is not None, "rank 0 produced no output"
    assert result_json[
        "pass"
    ], f"allclose failed: max_diff={result_json['max_diff']}"


# ---------- Decode shape (1, 4096) = 4096 elements ----------


@pytest.mark.skipif(_gpu_count() < 2, reason="need ≥2 GPUs")
def test_allreduce_decode_tp2():
    """Decode (1,4096) all-reduce at TP=2."""
    _run_allreduce_test(world_size=2, n_elements=4096)


@pytest.mark.skipif(_gpu_count() < 8, reason="need ≥8 GPUs")
def test_allreduce_decode_tp8():
    """Decode (1,4096) all-reduce at TP=8."""
    _run_allreduce_test(world_size=8, n_elements=4096)


# ---------- Prefill shape (2048, 4096) = 8388608 elements ----------


@pytest.mark.skipif(_gpu_count() < 2, reason="need ≥2 GPUs")
def test_allreduce_prefill_tp2():
    """Prefill (2048,4096) all-reduce at TP=2."""
    _run_allreduce_test(world_size=2, n_elements=2048 * 4096)


@pytest.mark.skipif(_gpu_count() < 8, reason="need ≥8 GPUs")
def test_allreduce_prefill_tp8():
    """Prefill (2048,4096) all-reduce at TP=8."""
    _run_allreduce_test(world_size=8, n_elements=2048 * 4096)
