# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Equivalence tests for a minimal Qwen3 decode row-parallel TP slice.

Each rank owns one shard of the input hidden dimension, computes its local
partial output with ARK matmul, then reduces the partial outputs with
``ark.all_reduce_packet``. The CPU reference uses the same row-parallel
sharding math:

    y = sum_r x[:, r*K:(r+1)*K] @ w[r*K:(r+1)*K, :]

The workers copy ARK outputs to CPU after the ARK runtime stops and only then
run torch comparisons. No torch GPU work is issued while the ARK runtime is
launched.
"""

import os
import subprocess
import sys

import pytest

try:
    import torch
except ImportError:
    pytest.skip("torch is not installed", allow_module_level=True)

try:
    from ._env import _load_worker_result, _subprocess_env
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _env import _load_worker_result, _subprocess_env


HIDDEN_SIZE = 4096


def _gpu_count() -> int:
    """Return available CUDA device count (0 if CUDA unavailable)."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


# This intentionally mirrors bench_tp.py's ARK graph/lifecycle while adding
# only post-rt.stop() CPU reference checks so the benchmark stays latency-only.
_WORKER_SCRIPT = r'''
"""Worker: run one ARK row-parallel TP decode slice."""
import json
import os
import sys

import torch
import ark
from ark.executor import Executor

rank = int(sys.argv[1])
world_size = int(sys.argv[2])
hidden_size = int(sys.argv[3])

local_hidden = hidden_size // world_size
if hidden_size % world_size != 0:
    raise RuntimeError("hidden_size must be divisible by world_size")

def make_shard(shard_rank):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(20260619 + shard_rank)
    x = 0.05 * torch.randn((1, local_hidden), generator=gen)
    w = 0.05 * torch.randn((local_hidden, hidden_size), generator=gen)
    return x.to(torch.float16), w.to(torch.float16)

ark.init()
ark.set_rank(rank)
ark.set_world_size(world_size)
ark.Model.set_device_id(rank)
torch.cuda.set_device(rank)

x_cpu, w_cpu = make_shard(rank)
x = x_cpu.to(device=f"cuda:{rank}")
w = w_cpu.to(device=f"cuda:{rank}")
torch.cuda.synchronize(rank)

partial = ark.matmul(x, w)
reduce_op = ark.all_reduce_packet
result = reduce_op(partial, rank, world_size)

with ark.Runtime() as rt:
    rt.launch(device_id=rank)
    if world_size > 1:
        rt.barrier()
    rt.run(iter=1)
    if world_size > 1:
        rt.barrier()
    rt.stop()

result_cpu = result.to_torch().cpu()

expected = torch.zeros((1, hidden_size), dtype=torch.float32)
for shard_rank in range(world_size):
    x_ref, w_ref = make_shard(shard_rank)
    expected += x_ref.float().matmul(w_ref.float())
expected = expected.to(torch.float16)

close = torch.allclose(result_cpu, expected, rtol=5e-2, atol=5e-2)
max_diff = (result_cpu - expected).abs().max().item()

print(json.dumps({
    "rank": rank,
    "world_size": world_size,
    "hidden_size": hidden_size,
    "local_hidden": local_hidden,
    "result_shape": list(result_cpu.shape),
    "result_dtype": str(result_cpu.dtype),
    "route": reduce_op.__name__,
    "pass": close,
    "max_diff": max_diff,
}))
sys.stdout.flush()

Executor.reset()
os._exit(0 if close else 1)
'''


def _tail(data, limit=500):
    """Return a short decoded tail for subprocess diagnostics."""
    return data.decode(errors="replace").strip()[-limit:]


def _run_tp_test(world_size: int, timeout: int = 180):
    """Spawn *world_size* workers and assert TP slice correctness."""
    env = _subprocess_env(world_size)
    errors = []
    results = []
    procs = []
    try:
        for rank in range(world_size):
            try:
                procs.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-c",
                            _WORKER_SCRIPT,
                            str(rank),
                            str(world_size),
                            str(HIDDEN_SIZE),
                        ],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        cwd="/",
                        env=env,
                    )
                )
            except OSError as exc:
                errors.append(f"rank {rank}: launch failed: {exc}")
                break

        if not errors:
            for rank, proc in enumerate(procs):
                try:
                    out, err = proc.communicate(timeout=timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                    errors.append(f"rank {rank}: timed out after {timeout}s")
                    continue
                result = _load_worker_result(out)
                if proc.returncode != 0:
                    errors.append(
                        f"rank {rank}: exit={proc.returncode} "
                        f"stderr={_tail(err, 300)}"
                    )
                if result is None:
                    errors.append(
                        f"rank {rank}: stdout contained no JSON result "
                        f"stdout_tail={_tail(out)} stderr_tail={_tail(err)}"
                    )
                else:
                    results.append(result)
    finally:
        for proc in procs:
            proc.kill()
            proc.wait()

    assert not errors, "\n".join(errors)
    assert len(results) == world_size
    for result in results:
        assert result["route"] == "all_reduce_packet"
        assert result["result_shape"] == [1, HIDDEN_SIZE]
        assert result["result_dtype"] == "torch.float16"
        assert result["pass"], (
            f"rank {result['rank']} allclose failed: "
            f"max_diff={result['max_diff']}"
        )


@pytest.mark.skipif(_gpu_count() < 2, reason="need ≥2 GPUs")
def test_tp_decode_row_parallel_tp2():
    """Qwen3 decode row-parallel TP slice at TP=2."""
    _run_tp_test(world_size=2)


@pytest.mark.skipif(_gpu_count() < 8, reason="need ≥8 GPUs")
def test_tp_decode_row_parallel_tp8():
    """Qwen3 decode row-parallel TP slice at TP=8."""
    _run_tp_test(world_size=8)
