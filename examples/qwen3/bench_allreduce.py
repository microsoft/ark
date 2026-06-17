# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Microbenchmark: ARK fused-packet all-reduce at Qwen3 TP shapes.

Measures steady-state latency for decode (1, 4096) and prefill (2048, 4096)
at TP=2 / TP=8. Each rank runs as its own process.

    python -m examples.qwen3.bench_allreduce --world-size 8

TIMING METHOD (critical): ARK runs a PERSISTENT loop kernel that owns all SMs
between ``rt.launch()`` and ``rt.stop()`` — by design. Any torch GPU op issued
while the runtime is live (``torch.cuda.synchronize``, ``torch.cuda.Event``,
``torch.allclose``, ...) can never be scheduled and deadlocks. So we time with
plain host wall-clock around ``rt.run(iter=N)`` (which host-blocks on ARK's own
completion flags, not ``cudaDeviceSynchronize``) and align ranks with
``rt.barrier()``. NO torch device sync, NO CUDA events.
"""

import argparse
import json
import os
import subprocess
import sys

try:
    from ._env import _subprocess_env
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _env import _subprocess_env

_WORKER_SCRIPT = '''
"""Worker: time ARK all-reduce without any torch GPU op while launched."""
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
warmup = int(sys.argv[5])
n_iters = int(sys.argv[6])

ark.init()
ark.set_rank(rank)
ark.set_world_size(world_size)

# Input is created BEFORE launch, while no ARK loop kernel is live (safe).
x = torch.randn(n_elements, dtype=torch.float16, device=f"cuda:{rank}")
result = ark.all_reduce_packet(x, rank, world_size)

with ark.Runtime() as rt:
    rt.launch(device_id=rank)

    # Warm up (blocks on ARK completion flags, not cudaDeviceSynchronize).
    rt.run(iter=warmup)
    if world_size > 1:
        rt.barrier()

    # Steady-state: one batched run of n_iters, host wall-clock around it.
    t0 = time.perf_counter()
    rt.run(iter=n_iters)
    host_s = time.perf_counter() - t0
    if world_size > 1:
        rt.barrier()

    # Cross-check: ARK's own device-measured elapsed since launch (ms).
    dev_ms = rt.stop()

mean_us = host_s * 1e6 / n_iters

if rank == 0:
    print(json.dumps({
        "label": label,
        "world_size": world_size,
        "n_elements": n_elements,
        "mean_us": round(mean_us, 3),
        "n_iters": n_iters,
        "dev_ms_since_launch": round(dev_ms, 3),
    }))
    sys.stdout.flush()

# Executor.reset() forces full mscclpp teardown before os._exit skips Python
# cleanup.
Executor.reset()
os._exit(0)
'''

# mscclpp-NCCL ceiling (8xA100, fp16, measured nccl-tests all_reduce_perf):
#   decode  (1,4096)  8KB : ~11.7 us   (plain NCCL ~21-24 us)
#   prefill (2048,4096)16MB: ~188 us   (plain NCCL ~219-222 us)
# These are the real per-call targets ARK must beat (NOT the 5.96 ms/layer
# SGLang amortized figure, which is a whole decode trace / 36 layers).
_MSCCLPP_CEIL_US = {4096: 11.7, 2048 * 4096: 188.0}

SHAPES = [
    ("decode  (1, 4096)", 4096),
    ("prefill (2048, 4096)", 2048 * 4096),
]


def run_bench(world_size, warmup, n_iters, timeout):
    results = []
    for label, n_elements in SHAPES:
        procs = []
        for rank in range(world_size):
            procs.append(
                subprocess.Popen(
                    [
                        sys.executable, "-c", _WORKER_SCRIPT,
                        str(rank), str(world_size), str(n_elements), label,
                        str(warmup), str(n_iters),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd="/",
                    env=_subprocess_env(world_size),
                )
            )
        try:
            for rank, p in enumerate(procs):
                try:
                    out, err = p.communicate(timeout=timeout)
                except subprocess.TimeoutExpired:
                    print(f"ERROR rank={rank} {label}: timed out after "
                          f"{timeout}s", file=sys.stderr)
                    break
                if p.returncode != 0:
                    print(f"ERROR rank={rank} {label}: "
                          f"{err.decode().strip()[-500:]}", file=sys.stderr)
                if rank == 0 and out.strip():
                    results.append(json.loads(out.decode().strip()))
        finally:
            for p in procs:
                p.kill()
                p.wait()

    print(f"\n{'=' * 72}")
    print(f"ARK fused-packet all-reduce  |  TP={world_size}  "
          f"(warmup={warmup}, iters={n_iters})")
    print(f"{'=' * 72}")
    print(f"{'Shape':<24}{'Elements':>12}{'ARK us':>10}"
          f"{'mscclpp us':>12}{'ARK/ceil':>10}")
    print(f"{'-' * 72}")
    for d in results:
        ceil = _MSCCLPP_CEIL_US.get(d["n_elements"])
        ratio = f"{d['mean_us'] / ceil:.2f}x" if ceil else "-"
        ceil_s = f"{ceil:.1f}" if ceil else "-"
        print(f"{d['label']:<24}{d['n_elements']:>12,}{d['mean_us']:>10.2f}"
              f"{ceil_s:>12}{ratio:>10}")
    print(f"{'=' * 72}\n")
    return results


def main():
    ap = argparse.ArgumentParser(
        description="Benchmark ARK fused-packet all-reduce at Qwen3 TP shapes"
    )
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    results = run_bench(args.world_size, args.warmup, args.iters, args.timeout)

    # PERF_GATE on the decode shape vs the mscclpp ceiling (the real target).
    decode = [r for r in results if r["n_elements"] == 4096]
    if decode:
        ark_us = decode[0]["mean_us"]
    else:
        ark_us = 999999.0  # workers failed
    ceil_us = _MSCCLPP_CEIL_US[4096]
    ratio = ark_us / ceil_us if ceil_us > 0 else 999999.0
    print(f"PERF_GATE name=allreduce_decode"
          f" ark_us={ark_us:.3f}"
          f" mscclpp_ceil_us={ceil_us:.3f}"
          f" ratio={ratio:.3f}")


if __name__ == "__main__":
    main()
