# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Qwen3 fixed-layout KV-cache slot equivalence tests.

The worker stops the ARK runtime before any torch D2H copy or comparison. No
Torch GPU work is issued while the ARK loop kernel is launched.
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


_WORKER_SCRIPT = r'''
"""Worker: validate ARK KV-cache slot update/read behavior."""
import json
import os
import sys

import torch
import ark
from ark.executor import Executor

mode = sys.argv[1]

ark.init()
torch.cuda.set_device(0)

if mode == "equiv":
    max_seq = 4
    iters = 3
    slot_shape = (2, 3)
    cache = torch.zeros((max_seq,) + slot_shape, dtype=torch.float16, device="cuda:0")
    token_cpu = torch.arange(1, 7, dtype=torch.float16).reshape(slot_shape)
    token = token_cpu.to(device="cuda:0")
    position = torch.zeros(1, dtype=torch.int32, device="cuda:0")
    torch.cuda.synchronize(0)

    slot = ark.kv_cache_slot(cache, token, position)

    with ark.Runtime() as rt:
        rt.launch(device_id=0, loop_mode=True)
        rt.run(iter=iters)
        rt.stop()

    cache_cpu = cache.cpu()
    slot_cpu = slot.to_torch().cpu()
    position_cpu = position.cpu()

    expected_cache = torch.zeros((max_seq,) + slot_shape, dtype=torch.float16)
    expected_cache[:iters] = token_cpu
    ok = (
        torch.equal(cache_cpu, expected_cache)
        and torch.equal(slot_cpu, token_cpu)
        and int(position_cpu.item()) == iters
    )
    print(json.dumps({
        "mode": mode,
        "pass": ok,
        "position": int(position_cpu.item()),
    }))
    sys.stdout.flush()
    Executor.reset()
    os._exit(0 if ok else 1)

if mode == "last":
    max_seq = 4
    slot_shape = (2, 3)
    cache = torch.zeros((max_seq,) + slot_shape, dtype=torch.float16, device="cuda:0")
    token_cpu = torch.arange(1, 7, dtype=torch.float16).reshape(slot_shape)
    token = token_cpu.to(device="cuda:0")
    position = torch.full((1,), max_seq - 1, dtype=torch.int32, device="cuda:0")
    torch.cuda.synchronize(0)

    slot = ark.kv_cache_slot(cache, token, position)

    with ark.Runtime() as rt:
        rt.launch(device_id=0, loop_mode=True)
        rt.run(iter=1)
        rt.stop()

    cache_cpu = cache.cpu()
    slot_cpu = slot.to_torch().cpu()
    position_cpu = position.cpu()

    expected_cache = torch.zeros((max_seq,) + slot_shape, dtype=torch.float16)
    expected_cache[max_seq - 1] = token_cpu
    ok = (
        torch.equal(cache_cpu, expected_cache)
        and torch.equal(slot_cpu, token_cpu)
        and int(position_cpu.item()) == max_seq
    )
    print(json.dumps({
        "mode": mode,
        "pass": ok,
        "position": int(position_cpu.item()),
    }))
    sys.stdout.flush()
    Executor.reset()
    os._exit(0 if ok else 1)

if mode == "graph_read":
    max_seq = 4
    iters = 3
    slot_shape = (2, 3)
    cache = torch.zeros((max_seq,) + slot_shape, dtype=torch.float16, device="cuda:0")
    token_cpu = torch.arange(1, 7, dtype=torch.float16).reshape(slot_shape)
    token = token_cpu.to(device="cuda:0")
    position = torch.zeros(1, dtype=torch.int32, device="cuda:0")
    torch.cuda.synchronize(0)

    cache_ark = ark.Tensor.from_torch(cache)
    token_ark = ark.Tensor.from_torch(token)
    position_ark = ark.Tensor.from_torch(position)
    ark.kv_cache_slot(cache_ark, token_ark, position_ark)
    later_cache_copy = ark.copy(cache_ark)

    with ark.Runtime() as rt:
        rt.launch(device_id=0, loop_mode=True)
        rt.run(iter=iters)
        rt.stop()

    copy_cpu = later_cache_copy.to_torch().cpu()
    position_cpu = position.cpu()

    expected = torch.zeros((max_seq,) + slot_shape, dtype=torch.float16)
    expected[:iters] = token_cpu
    ok = torch.equal(copy_cpu, expected) and int(position_cpu.item()) == iters
    print(json.dumps({
        "mode": mode,
        "pass": ok,
        "position": int(position_cpu.item()),
    }))
    sys.stdout.flush()
    Executor.reset()
    os._exit(0 if ok else 1)

if mode == "two_slots":
    max_seq = 4
    slot_shape = (2, 3)
    cache = torch.zeros((max_seq,) + slot_shape, dtype=torch.float16, device="cuda:0")
    token0_cpu = torch.arange(1, 7, dtype=torch.float16).reshape(slot_shape)
    token1_cpu = token0_cpu + 10
    token0 = token0_cpu.to(device="cuda:0")
    token1 = token1_cpu.to(device="cuda:0")
    position = torch.zeros(1, dtype=torch.int32, device="cuda:0")
    torch.cuda.synchronize(0)

    slot0 = ark.kv_cache_slot(cache, token0, position)
    slot1 = ark.kv_cache_slot(cache, token1, position)
    read0 = ark.copy(slot0)
    read1 = ark.copy(slot1)

    with ark.Runtime() as rt:
        rt.launch(device_id=0, loop_mode=True)
        rt.run(iter=1)
        rt.stop()

    cache_cpu = cache.cpu()
    read0_cpu = read0.to_torch().cpu()
    read1_cpu = read1.to_torch().cpu()
    position_cpu = position.cpu()

    expected_cache = torch.zeros((max_seq,) + slot_shape, dtype=torch.float16)
    expected_cache[0] = token0_cpu
    expected_cache[1] = token1_cpu
    ok = (
        torch.equal(cache_cpu, expected_cache)
        and torch.equal(read0_cpu, token0_cpu)
        and torch.equal(read1_cpu, token1_cpu)
        and int(position_cpu.item()) == 2
    )
    print(json.dumps({
        "mode": mode,
        "pass": ok,
        "position": int(position_cpu.item()),
    }))
    sys.stdout.flush()
    Executor.reset()
    os._exit(0 if ok else 1)

if mode == "explicit_output":
    slot_shape = (2, 3)
    cache = torch.zeros((2,) + slot_shape, dtype=torch.float16, device="cuda:0")
    token_cpu = torch.arange(1, 7, dtype=torch.float16).reshape(slot_shape)
    token = token_cpu.to(device="cuda:0")
    position = torch.zeros(1, dtype=torch.int32, device="cuda:0")
    output = torch.zeros(slot_shape, dtype=torch.float16, device="cuda:0")
    torch.cuda.synchronize(0)

    slot = ark.kv_cache_slot(cache, token, position, output=output)

    with ark.Runtime() as rt:
        rt.launch(device_id=0, loop_mode=True)
        rt.run(iter=1)
        rt.stop()

    output_cpu = output.cpu()
    slot_cpu = slot.to_torch().cpu()
    position_cpu = position.cpu()
    ok = (
        torch.equal(output_cpu, token_cpu)
        and torch.equal(slot_cpu, token_cpu)
        and int(position_cpu.item()) == 1
    )
    print(json.dumps({
        "mode": mode,
        "pass": ok,
        "position": int(position_cpu.item()),
    }))
    sys.stdout.flush()
    Executor.reset()
    os._exit(0 if ok else 1)

if mode == "oob":
    cache = torch.zeros((2, 2, 3), dtype=torch.float16, device="cuda:0")
    token = torch.ones((2, 3), dtype=torch.float16, device="cuda:0")
    position = torch.full((1,), 2, dtype=torch.int32, device="cuda:0")
    torch.cuda.synchronize(0)
    slot = ark.kv_cache_slot(cache, token, position)
    try:
        with ark.Runtime() as rt:
            rt.launch(device_id=0, loop_mode=True)
            rt.run(iter=1)
            rt.stop()
    except Exception as exc:  # GpuError from the device-side bounds assert.
        print(json.dumps({"mode": mode, "pass": True, "error": str(exc)}))
        sys.stdout.flush()
        os._exit(0)
    print(json.dumps({"mode": mode, "pass": False, "error": "no error"}))
    sys.stdout.flush()
    os._exit(1)

raise SystemExit(f"unknown mode: {mode}")
'''


def _gpu_count() -> int:
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def _tail(data, limit=600):
    return data.decode(errors="replace").strip()[-limit:]


def _run_worker(mode: str, timeout: int = 120):
    proc = subprocess.Popen(
        [sys.executable, "-c", _WORKER_SCRIPT, mode],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd="/",
        env=_subprocess_env(1),
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        pytest.fail(
            f"worker {mode} timed out after {timeout}s\n"
            f"stdout={_tail(out)}\nstderr={_tail(err)}"
        )
    result = _load_worker_result(out)
    if proc.returncode != 0:
        err_lower = err.lower()
        if mode == "oob" and (
            b"assert" in err_lower
            or b"trap" in err_lower
            or b"illegal" in err_lower
        ):
            return {"mode": mode, "pass": True, "error": _tail(err)}
        pytest.fail(
            f"worker {mode} failed rc={proc.returncode}\n"
            f"stdout={_tail(out)}\nstderr={_tail(err)}"
        )
    if result is None:
        pytest.fail(f"worker {mode} produced no JSON\nstderr={_tail(err)}")
    return result


@pytest.mark.skipif(_gpu_count() < 1, reason="CUDA GPU is required")
def test_kv_cache_slot_updates_multiple_positions_and_reads_slot():
    """One launched runtime updates positions 0, 1, and 2."""
    result = _run_worker("equiv")

    assert result["pass"] is True
    assert result["position"] == 3


@pytest.mark.skipif(_gpu_count() < 1, reason="CUDA GPU is required")
def test_kv_cache_slot_updates_last_valid_position():
    """The last valid cache slot is writable and advances position."""
    result = _run_worker("last")

    assert result["pass"] is True
    assert result["position"] == 4


@pytest.mark.skipif(_gpu_count() < 1, reason="CUDA GPU is required")
def test_later_graph_op_reads_updated_external_cache():
    """A later ARK op consumes multi-position cache updates."""
    result = _run_worker("graph_read")

    assert result["pass"] is True
    assert result["position"] == 3


@pytest.mark.skipif(_gpu_count() < 1, reason="CUDA GPU is required")
def test_two_kv_cache_slots_share_position_in_one_graph_launch():
    """Later graph ops read two runtime-selected cache slots."""
    result = _run_worker("two_slots")

    assert result["pass"] is True
    assert result["position"] == 2


@pytest.mark.skipif(_gpu_count() < 1, reason="CUDA GPU is required")
def test_kv_cache_slot_writes_explicit_output_tensor():
    """A caller-provided output tensor receives the selected slot."""
    result = _run_worker("explicit_output")

    assert result["pass"] is True
    assert result["position"] == 1


@pytest.mark.skipif(_gpu_count() < 1, reason="CUDA GPU is required")
def test_kv_cache_slot_out_of_bounds_position_fails():
    """A runtime position outside [0, max_seq) fails instead of aliasing."""
    result = _run_worker("oob")

    assert result["pass"] is True
