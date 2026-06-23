# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Benchmark end-to-end ``ark.all_reduce_packet`` latency.

Measures single-iteration latency for Qwen3 TP decode (1, 4096) and prefill
(2048, 4096) shapes. The default run reports both input modes:

- ``external``: torch CUDA placeholder input; includes registered-memory
  staging.
- ``internal``: ARK-owned input tensor; excludes external staging.

Each rank runs as its own process and the parent reports max-rank latency.

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
import subprocess
import sys

try:
    from ._env import _load_worker_result, _subprocess_env
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _env import _load_worker_result, _subprocess_env

_WORKER_SCRIPT = '''
"""Worker: time ARK all-reduce without torch ops while launched."""
import json
import os
import sys
import time

import numpy as np
import torch
import ark
from ark.executor import Executor

rank = int(sys.argv[1])
world_size = int(sys.argv[2])
n_elements = int(sys.argv[3])
shape_name = sys.argv[4]
label = sys.argv[5]
input_mode = sys.argv[6]

ark.init()
ark.set_rank(rank)
ark.set_world_size(world_size)

if input_mode == "external":
    # Torch input is created and synchronized BEFORE launch, while no ARK loop
    # kernel is live (safe). The measured graph includes all_reduce_packet's
    # staging copy into ARK-managed registered memory.
    x = torch.randn(n_elements, dtype=torch.float16, device=f"cuda:{rank}")
    torch.cuda.synchronize(rank)
elif input_mode == "internal":
    # ARK owns this tensor's storage. It is allocated by rt.launch(), then
    # initialized from host memory before timing. This excludes external staging
    # and avoids torch GPU work while the persistent runtime owns the device.
    x = ark.tensor([n_elements], ark.fp16)
    x_host = np.full((n_elements,), rank + 1, dtype=np.float16)
else:
    raise ValueError(f"unknown input_mode: {input_mode}")

result = ark.all_reduce_packet(x, rank, world_size)

with ark.Runtime() as rt:
    rt.launch(device_id=rank)

    if input_mode == "internal":
        x.from_numpy(x_host)

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
    "shape": shape_name,
    "label": label,
    "input_mode": input_mode,
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

SHAPES = {
    "decode": ("decode  (1, 4096)", 4096),
    "prefill": ("prefill (2048, 4096)", 2048 * 4096),
}

INPUT_MODES = ("external", "internal")


def run_bench(world_size, timeout, shape, input_mode):
    results = []
    any_failed = False
    shapes = SHAPES.items() if shape == "all" else [(shape, SHAPES[shape])]
    input_modes = INPUT_MODES if input_mode == "all" else (input_mode,)
    for shape_name, (label, n_elements) in shapes:
        for mode in input_modes:
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
                            shape_name,
                            label,
                            mode,
                        ],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        cwd="/",
                        env=_subprocess_env(world_size),
                    )
                )
            shape_failed = False
            rank_results = []
            result_label = f"{label} mode={mode}"
            try:
                for rank, p in enumerate(procs):
                    try:
                        out, err = p.communicate(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        shape_failed = True
                        print(
                            f"ERROR rank={rank} {result_label}: timed out "
                            f"after {timeout}s",
                            file=sys.stderr,
                        )
                        break
                    if p.returncode != 0:
                        shape_failed = True
                        print(
                            f"ERROR rank={rank} {result_label}: "
                            f"{err.decode().strip()[-500:]}",
                            file=sys.stderr,
                        )
                    result = _load_worker_result(out)
                    if result is None:
                        shape_failed = True
                        print(
                            f"ERROR rank={rank} {result_label}: no result",
                            file=sys.stderr,
                        )
                    else:
                        rank_results.append(result)
                if not shape_failed and len(rank_results) == world_size:
                    rank_results.sort(key=lambda d: d["rank"])
                    max_result = max(
                        rank_results, key=lambda d: d["latency_us"]
                    )
                    results.append(
                        {
                            "shape": max_result["shape"],
                            "label": max_result["label"],
                            "input_mode": max_result["input_mode"],
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
                            f"ERROR {result_label}: expected {world_size} "
                            f"rank results, got {len(rank_results)}",
                            file=sys.stderr,
                        )
            finally:
                for p in procs:
                    p.kill()
                    p.wait()

    print(f"\n{'=' * 88}")
    print(
        f"ARK all_reduce_packet latency  |  TP={world_size}  "
        f"(single iteration, max rank)"
    )
    print(f"{'=' * 88}")
    print(
        f"{'Shape':<24}{'Mode':<12}{'Elements':>12}"
        f"{'Max rank':>10}{'ARK us':>10}"
    )
    print(f"{'-' * 88}")
    for d in results:
        print(
            f"{d['label']:<24}{d['input_mode']:<12}{d['n_elements']:>12,}"
            f"{d['max_rank']:>10}{d['latency_us']:>10.2f}"
        )
    print(f"{'=' * 88}")
    # Supplemental per-shape/mode machine-readable rows. The root perf gate
    # validates these rows for TP=2/8 external/internal decode coverage.
    for d in results:
        ark_ms = d["latency_us"] / 1000.0
        print(
            f"RESULT name=allreduce shape={d['shape']} "
            f"tp={d['world_size']} mode={d['input_mode']} "
            f"ark_ms={ark_ms:.4f} latency_us={d['latency_us']:.3f}"
        )
    print()
    return results, any_failed


def _perf_gate_ark_ms(results):
    """Return compatibility PERF_GATE latency from external decode only."""
    decode_external = [
        r
        for r in results
        if r["n_elements"] == SHAPES["decode"][1]
        and r["input_mode"] == "external"
    ]
    if decode_external:
        return decode_external[0]["latency_us"] / 1000.0
    return 999999.0


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Benchmark end-to-end ark.all_reduce_packet latency at Qwen3 TP "
            "shapes. External mode includes registered-memory staging; "
            "internal mode uses ARK-owned input storage."
        )
    )
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument(
        "--shape",
        choices=("decode", "prefill", "all"),
        default="all",
        help="Qwen3 shape to benchmark; the perf gate uses decode",
    )
    ap.add_argument(
        "--input-mode",
        choices=("external", "internal", "all"),
        default="all",
        help="Input storage mode to benchmark",
    )
    args = ap.parse_args()

    # Repeated-iteration timing is intentionally unsupported until packet flag
    # rotation/reset exists.
    results, any_failed = run_bench(
        args.world_size, args.timeout, args.shape, args.input_mode
    )

    ark_ms = _perf_gate_ark_ms(results)
    ratio = ark_ms / _DECODE_TARGET_MS
    print(
        f"PERF_GATE name=allreduce"
        f" ark_ms={ark_ms:.4f}"
        f" sglang_ms={_DECODE_TARGET_MS:.4f}"
        f" ratio={ratio:.4f}"
    )
    if any_failed:
        print("ERROR: one or more benchmark workers failed", file=sys.stderr)
        raise SystemExit(1)
    if ark_ms == 999999.0:
        print(
            "ERROR: external decode benchmark produced no result for PERF_GATE",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
