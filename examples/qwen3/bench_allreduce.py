# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Benchmark Qwen3 ARK all-reduce route latency on torch input.

Measures single-iteration latency for Qwen3 TP decode (1, 4096) and prefill
(2048, 4096) shapes. Each rank runs as its own process and the parent reports
max-rank latency.

    python -m examples.qwen3.bench_allreduce --world-size 8 --shape prefill

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
shape_key = sys.argv[4]
route_arg = sys.argv[5]
head_sha = sys.argv[6]

ark.init()
ark.set_rank(rank)
ark.set_world_size(world_size)

# Input is created and synchronized BEFORE launch, while no ARK loop kernel is
# live (safe). The benchmark includes any staging done by the selected route.
x = torch.randn(n_elements, dtype=torch.float16, device=f"cuda:{rank}")
torch.cuda.synchronize(rank)

if route_arg == "auto":
    result = ark.all_reduce(x, rank, world_size)
    route_used = "packet" if shape_key == "decode" else "prefill"
elif route_arg == "packet":
    result = ark.all_reduce_packet(x, rank, world_size)
    route_used = "packet"
elif route_arg == "prefill":
    result = ark.all_reduce_prefill(x, rank, world_size)
    route_used = "prefill"
else:
    raise ValueError(f"unknown route: {route_arg}")

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
    "shape": shape_key,
    "route": route_used,
    "head_sha": head_sha,
    "world_size": world_size,
    "n_elements": n_elements,
    "latency_us": round(latency_us, 3),
}))
sys.stdout.flush()

# Workaround: mscclpp's UnixSocketServer destructor races during normal
# Python shutdown (static destruction order is undefined across TUs),
# causing SIGABRT. Executor.reset() forces orderly mscclpp teardown,
# then os._exit() skips Python's atexit / gc finalizers entirely.
Executor.reset()
os._exit(0)
'''

# SGLang PROFILE.md Q7 nccl / comm target: 214.69 ms over 657 calls
# on 8xA100 TP=8, batch=1 decode-dominated Qwen3-8B.
_DECODE_TARGET_MS = 214.69 / 657.0
# Q7P strict prefill target from PROFILE.md evidence used by the perf gate.
_PREFILL_TARGET_MS = 0.188

SHAPES = {
    "decode": {
        "label": "decode  (1, 4096)",
        "n_elements": 4096,
        "target_ms": _DECODE_TARGET_MS,
        "gate_name": "allreduce_decode",
    },
    "prefill": {
        "label": "prefill (2048, 4096)",
        "n_elements": 2048 * 4096,
        "target_ms": _PREFILL_TARGET_MS,
        "gate_name": "allreduce_prefill",
    },
}


def _repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _head_sha():
    try:
        return subprocess.check_output(
            ["git", "-C", _repo_root(), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_bench(world_size, timeout, shape, route, head_sha):
    results = []
    any_failed = False
    shape_keys = list(SHAPES) if shape == "all" else [shape]
    for shape_key in shape_keys:
        spec = SHAPES[shape_key]
        label = spec["label"]
        n_elements = spec["n_elements"]
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
                        shape_key,
                        route,
                        head_sha,
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
                        "shape": shape_key,
                        "label": label,
                        "route": max_result["route"],
                        "head_sha": head_sha,
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

    print(f"\n{'=' * 96}")
    print(
        f"ARK Qwen3 all-reduce latency  |  TP={world_size}  "
        f"route={route}  head_sha={head_sha}"
    )
    print(f"{'=' * 96}")
    print(
        f"{'Shape':<24}{'Route':>10}{'Elements':>12}"
        f"{'Max rank':>10}{'ARK us':>10}"
    )
    print(f"{'-' * 96}")
    for d in results:
        print(
            f"{d['label']:<24}{d['route']:>10}{d['n_elements']:>12,}"
            f"{d['max_rank']:>10}{d['latency_us']:>10.2f}"
        )
        ark_ms = d["latency_us"] / 1000.0
        print(
            f"BENCH_RESULT name=allreduce_{d['shape']}"
            f" head_sha={d['head_sha']} route={d['route']}"
            f" world_size={d['world_size']} n_elements={d['n_elements']}"
            f" max_rank={d['max_rank']} ark_ms={ark_ms:.4f}"
        )
    print(f"{'=' * 96}\n")
    return results, any_failed


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Benchmark Qwen3 ARK all-reduce latency on torch input at decode "
            "and prefill TP shapes"
        )
    )
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument(
        "--shape",
        choices=("decode", "prefill", "all"),
        default="all",
        help="Qwen3 shape to benchmark; default keeps the decode perf gate",
    )
    ap.add_argument(
        "--route",
        choices=("auto", "packet", "prefill"),
        default="auto",
        help="Route to time. auto uses ark.all_reduce dispatch.",
    )
    args = ap.parse_args()

    # Repeated-iteration timing is intentionally unsupported until packet flag
    # rotation/reset exists.
    head_sha = _head_sha()
    if head_sha == "unknown":
        print("ERROR: head_sha=unknown", file=sys.stderr)
        raise SystemExit(1)

    results, any_failed = run_bench(
        args.world_size, args.timeout, args.shape, args.route, head_sha
    )

    gate_shape = "decode" if args.shape in ("all", "decode") else "prefill"
    gate_results = [r for r in results if r["shape"] == gate_shape]
    if gate_results:
        ark_ms = gate_results[0]["latency_us"] / 1000.0
    else:
        ark_ms = 999999.0
    sglang_ms = SHAPES[gate_shape]["target_ms"]
    ratio = ark_ms / sglang_ms
    print(
        f"PERF_GATE name={SHAPES[gate_shape]['gate_name']}"
        f" ark_ms={ark_ms:.4f}"
        f" sglang_ms={sglang_ms:.4f}"
        f" ratio={ratio:.4f}"
    )
    if any_failed:
        print("ERROR: one or more benchmark workers failed", file=sys.stderr)
        raise SystemExit(1)
    if not gate_results:
        print(f"ERROR: {gate_shape} benchmark produced no result", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
