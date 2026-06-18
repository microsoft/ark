# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Benchmark end-to-end ``ark.all_reduce_packet`` latency on torch input.

Measures single-iteration latency for Qwen3 TP decode (1, 4096) and prefill
(2048, 4096) shapes, including registered-memory staging when needed. Each
rank runs as its own process.

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
# live (safe). The benchmark includes any staging done by ark.all_reduce_packet.
x = torch.randn(n_elements, dtype=torch.float16, device=f"cuda:{rank}")
torch.cuda.synchronize(rank)
result = ark.all_reduce_packet(x, rank, world_size)

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

if rank == 0:
    print(json.dumps({
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

# mscclpp-NCCL ceiling from the local Q7 profile (8xA100, fp16, 8 KB).
_DECODE_TARGET_MS = 11.7 / 1000.0

SHAPES = [
    ("decode  (1, 4096)", 4096),
    ("prefill (2048, 4096)", 2048 * 4096),
]


def run_bench(world_size, timeout):
    results = []
    for label, n_elements in SHAPES:
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
        shape_result = None
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
                if rank == 0 and out.strip():
                    shape_result = json.loads(out.decode().strip())
            if not shape_failed and shape_result is not None:
                results.append(shape_result)
            elif not shape_failed:
                print(f"ERROR rank=0 {label}: no result", file=sys.stderr)
        finally:
            for p in procs:
                p.kill()
                p.wait()

    print(f"\n{'=' * 72}")
    print(
        f"ARK all_reduce_packet torch-input latency  |  TP={world_size}  "
        f"(single iteration, includes staging)"
    )
    print(f"{'=' * 72}")
    print(f"{'Shape':<24}{'Elements':>12}{'ARK us':>10}")
    print(f"{'-' * 72}")
    for d in results:
        print(
            f"{d['label']:<24}{d['n_elements']:>12,}"
            f"{d['latency_us']:>10.2f}"
        )
    print(f"{'=' * 72}\n")
    return results


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Benchmark end-to-end ark.all_reduce_packet latency on torch input "
            "at Qwen3 TP shapes, including registered-memory staging when needed"
        )
    )
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    # Repeated-iteration timing is intentionally unsupported until packet flag
    # rotation/reset exists.
    results = run_bench(args.world_size, args.timeout)

    decode = [r for r in results if r["n_elements"] == 4096]
    if decode:
        ark_ms = decode[0]["latency_us"] / 1000.0
    else:
        ark_ms = 999999.0
    ratio = ark_ms / _DECODE_TARGET_MS
    print(
        f"PERF_GATE name=allreduce_decode"
        f" ark_ms={ark_ms:.4f}"
        f" sglang_ms={_DECODE_TARGET_MS:.4f}"
        f" ratio={ratio:.4f}"
    )


if __name__ == "__main__":
    main()
