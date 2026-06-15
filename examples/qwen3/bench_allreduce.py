# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Microbenchmark: ARK fused-packet all-reduce at Qwen3 TP shapes.

Measures latency for both prefill (2048, 4096) and decode (1, 4096)
shapes at TP=2 and TP=8.  Run out-of-band on a multi-GPU node:

    # TP=2, 2 GPUs
    python -m examples.qwen3.bench_allreduce --world-size 2

    # TP=8, 8 GPUs (from repo root)
    python -m examples.qwen3.bench_allreduce --world-size 8

Each rank is launched as a separate process to avoid CUDA context issues.
Uses torch.cuda.Event for steady-state timing.
"""

import argparse
import os
import subprocess
import sys

_WORKER_SCRIPT = '''
"""Worker for all-reduce microbenchmark."""
import sys
import json

import torch
import ark

rank = int(sys.argv[1])
world_size = int(sys.argv[2])
n_elements = int(sys.argv[3])
label = sys.argv[4]

ark.set_rank(rank)
ark.set_world_size(world_size)

x = torch.randn(n_elements, dtype=torch.float16, device=f"cuda:{rank}")

ark.init()
result = ark.all_reduce_packet(x, rank, world_size)

with ark.Runtime() as rt:
    rt.launch(device_id=rank)

    # Warm up
    for _ in range(5):
        rt.run()

    # Measure
    torch.cuda.synchronize(rank)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    n_iters = 100
    start.record(torch.cuda.Stream(torch.device(f"cuda:{rank}")))
    for _ in range(n_iters):
        rt.run()
    end.record(torch.cuda.Stream(torch.device(f"cuda:{rank}")))
    torch.cuda.synchronize(rank)

    elapsed_ms = start.elapsed_time(end)
    mean_us = elapsed_ms * 1000.0 / n_iters

    if rank == 0:
        print(json.dumps({
            "label": label,
            "world_size": world_size,
            "n_elements": n_elements,
            "mean_us": round(mean_us, 2),
            "n_iters": n_iters,
        }))
'''

# Primary benchmark shape: decode (1, 4096) = 4096 elements.
# SGLang baseline target (decode, TP=2, A100 NVLink) — no PROFILE.md
# yet; value will be updated once profiling is done.
_SGLANG_DECODE_MS = 0.01  # placeholder until PROFILE.md exists

SHAPES = [
    ("decode  (1, 4096)", 4096),
    ("prefill (2048, 4096)", 2048 * 4096),
]


def run_bench(world_size: int):
    """Run all-reduce bench for all shapes at the given world_size.

    Returns a list of parsed JSON result dicts from rank-0 workers,
    or an empty list if all workers failed.
    """
    import json as _json

    results = []
    for label, n_elements in SHAPES:
        procs = []
        for rank in range(world_size):
            p = subprocess.Popen(
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
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": ",".join(
                        str(i) for i in range(world_size)
                    ),
                },
            )
            procs.append(p)

        for rank, p in enumerate(procs):
            stdout, stderr = p.communicate(timeout=300)
            if p.returncode != 0:
                print(
                    f"ERROR rank={rank}: {stderr.decode().strip()[-500:]}",
                    file=sys.stderr,
                )
            if rank == 0 and stdout.strip():
                results.append(_json.loads(stdout.decode().strip()))

    # Print summary table
    print(f"\n{'='*60}")
    print(f"ARK fused-packet all-reduce  |  TP={world_size}")
    print(f"{'='*60}")
    print(f"{'Shape':<30} {'Elements':>12} {'Latency (us)':>14}")
    print(f"{'-'*60}")
    for d in results:
        print(f"{d['label']:<30} {d['n_elements']:>12,} {d['mean_us']:>14.2f}")
    print(f"{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ARK fused-packet all-reduce at Qwen3 TP shapes"
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=2,
        help="Number of TP ranks (default: 2)",
    )
    args = parser.parse_args()
    results = run_bench(args.world_size)

    # Emit PERF_GATE line for the decode shape (primary gate metric).
    sglang_ms = _SGLANG_DECODE_MS
    decode_results = [r for r in results if r["n_elements"] == 4096]
    if decode_results:
        ark_ms = decode_results[0]["mean_us"] / 1000.0
    else:
        # Workers failed (codegen limitation: cannot offset external
        # buffer from all_reduce_packet, codegen.cpp:318).
        ark_ms = 999999.0
    ratio = ark_ms / sglang_ms if sglang_ms > 0 else 999999.0
    print(
        f"PERF_GATE name=allreduce"
        f" ark_ms={ark_ms:.4f}"
        f" sglang_ms={sglang_ms:.4f}"
        f" ratio={ratio:.4f}"
    )


if __name__ == "__main__":
    main()
