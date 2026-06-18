# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Benchmark end-to-end ``ark.all_reduce`` latency on torch input.

Measures single-iteration latency for Qwen3 TP decode (1, 4096) and prefill
(2048, 4096) shapes, including registered-memory staging when needed. Each
rank runs as its own process and the parent reports max-rank latency.

    python -m examples.qwen3.bench_allreduce --world-size 8

TIMING METHOD (critical): ARK runs a PERSISTENT loop kernel that owns all SMs
between ``rt.launch()`` and ``rt.stop()`` — by design. Torch synchronization is
safe before ``rt.launch()``, but any torch GPU op issued while the runtime is
live (``torch.cuda.synchronize``, ``torch.cuda.Event``, ``torch.allclose``, ...)
can never be scheduled and deadlocks. So we time with plain host wall-clock
around a single ``rt.run(iter=1)`` (which host-blocks on ARK's own completion
flags, not ``cudaDeviceSynchronize``) and align ranks with ``rt.barrier()``.
NO torch device sync while launched, NO CUDA events.
"""

import argparse
import os
import pathlib
import subprocess
import sys

try:
    from ._env import _load_worker_result, _subprocess_env
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _env import _load_worker_result, _subprocess_env

_WORKER_SCRIPT = '''
"""Worker: time torch-input ARK all-reduce without torch ops while launched."""
import json
import os
import sys
import time

import torch
import ark
from ark.executor import Executor

rank = int(sys.argv[1])
world_size = int(sys.argv[2])
n_elements = int(sys.argv[3])
label = sys.argv[4]

ark.init()
ark.set_rank(rank)
ark.set_world_size(world_size)

# Input is created and synchronized BEFORE launch, while no ARK loop kernel is
# live (safe). The benchmark includes any staging done by ark.all_reduce.
x = torch.randn(n_elements, dtype=torch.float16, device=f"cuda:{rank}")
torch.cuda.synchronize(rank)
result = ark.all_reduce(x, rank, world_size)

with ark.Runtime() as rt:
    rt.launch(device_id=rank)

    if world_size > 1:
        rt.barrier()

    t0 = time.perf_counter()
    rt.run(iter=1)
    host_s = time.perf_counter() - t0
    if world_size > 1:
        rt.barrier()

    rt.stop()

latency_us = host_s * 1e6

print(json.dumps({
    "rank": rank,
    "label": label,
    "world_size": world_size,
    "n_elements": n_elements,
    "latency_us": round(latency_us, 3),
}))
sys.stdout.flush()

# Workaround: mscclpp's UnixSocketServer destructor races during normal
# Python shutdown (static destruction order is undefined across TUs),
# causing SIGABRT.  Executor.reset() forces orderly mscclpp teardown,
# then os._exit() skips Python's atexit / gc finalizers entirely.
Executor.reset()
os._exit(0)
'''

# SGLang PROFILE.md Q7 nccl / comm target: 214.69 ms over 657 calls
# on 8xA100 TP=8, batch=1 decode-dominated Qwen3-8B.
_DECODE_TARGET_MS = 214.69 / 657.0
# SGLang PROFILE.md Q7P prefill all-reduce component target.
_PREFILL_TARGET_MS = 0.188
_TARGETS_MS = {"decode": _DECODE_TARGET_MS, "prefill": _PREFILL_TARGET_MS}
_GATE_NAMES = {"decode": "allreduce", "prefill": "allreduce_prefill"}

SHAPES = {
    "decode": ("decode  (1, 4096)", 4096),
    "prefill": ("prefill (2048, 4096)", 2048 * 4096),
}


def _current_git_sha():
    """Return the source checkout SHA used for this benchmark."""
    root = pathlib.Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_bench(world_size, timeout, shape):
    results = []
    any_failed = False
    shapes = SHAPES.values() if shape == "all" else [SHAPES[shape]]
    for label, n_elements in shapes:
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
                        label,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd="/",
                    env=_subprocess_env(world_size),
                )
            )
        shape_failed = False
        rank_results = []
        try:
            for rank, p in enumerate(procs):
                try:
                    out, err = p.communicate(timeout=timeout)
                except subprocess.TimeoutExpired:
                    shape_failed = True
                    print(
                        f"ERROR rank={rank} {label}: timed out after "
                        f"{timeout}s",
                        file=sys.stderr,
                    )
                    break
                if p.returncode != 0:
                    shape_failed = True
                    print(
                        f"ERROR rank={rank} {label}: "
                        f"{err.decode().strip()[-500:]}",
                        file=sys.stderr,
                    )
                result = _load_worker_result(out)
                if result is None:
                    shape_failed = True
                    print(
                        f"ERROR rank={rank} {label}: no result",
                        file=sys.stderr,
                    )
                else:
                    rank_results.append(result)
            if not shape_failed and len(rank_results) == world_size:
                rank_results.sort(key=lambda d: d["rank"])
                max_result = max(rank_results, key=lambda d: d["latency_us"])
                results.append(
                    {
                        "label": max_result["label"],
                        "world_size": world_size,
                        "n_elements": max_result["n_elements"],
                        "max_rank": max_result["rank"],
                        "latency_us": max_result["latency_us"],
                        "rank_latencies_us": [
                            d["latency_us"] for d in rank_results
                        ],
                    }
                )
            else:
                any_failed = True
                if not shape_failed:
                    print(
                        f"ERROR {label}: expected {world_size} rank results, "
                        f"got {len(rank_results)}",
                        file=sys.stderr,
                    )
        finally:
            for p in procs:
                p.kill()
                p.wait()

    print(f"\n{'=' * 72}")
    print(
        f"ARK all_reduce torch-input latency  |  TP={world_size}  "
        f"(single iteration, max rank, includes staging)"
    )
    print(f"{'=' * 72}")
    print(f"{'Shape':<24}{'Elements':>12}{'Max rank':>10}{'ARK us':>10}")
    print(f"{'-' * 72}")
    for d in results:
        print(
            f"{d['label']:<24}{d['n_elements']:>12,}"
            f"{d['max_rank']:>10}{d['latency_us']:>10.2f}"
        )
    print(f"{'=' * 72}\n")
    return results, any_failed


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Benchmark end-to-end ark.all_reduce latency on torch input "
            "at Qwen3 TP shapes, including registered-memory staging "
            "when needed"
        )
    )
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument(
        "--shape",
        choices=("decode", "prefill", "all"),
        default="all",
        help="Qwen3 shape to benchmark; the Q7P perf gate uses prefill",
    )
    args = ap.parse_args()

    # Repeated-iteration timing is intentionally unsupported until packet flag
    # rotation/reset exists.
    print(f"BENCH_SHA sha={_current_git_sha()}")
    results, any_failed = run_bench(args.world_size, args.timeout, args.shape)

    gate_shape = "prefill" if args.shape == "all" else args.shape
    gate_result = [
        r for r in results if r["n_elements"] == SHAPES[gate_shape][1]
    ]
    if gate_result:
        ark_ms = gate_result[0]["latency_us"] / 1000.0
    else:
        ark_ms = 999999.0
    sglang_ms = _TARGETS_MS[gate_shape]
    ratio = ark_ms / sglang_ms
    print(
        f"PERF_GATE name={_GATE_NAMES[gate_shape]}"
        f" ark_ms={ark_ms:.4f}"
        f" sglang_ms={sglang_ms:.4f}"
        f" ratio={ratio:.4f}"
    )
    if any_failed:
        print("ERROR: one or more benchmark workers failed", file=sys.stderr)
        raise SystemExit(1)
    if not gate_result:
        print(
            f"ERROR: {gate_shape} benchmark produced no result",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
