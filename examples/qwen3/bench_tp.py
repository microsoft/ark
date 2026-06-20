# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Benchmark a minimal Qwen3 decode row-parallel TP slice in ARK.

Each rank computes a local ``[1, 4096 / TP] x [4096 / TP, 4096]`` matmul
and reduces the partial ``[1, 4096]`` output with ``ark.all_reduce_packet``.
The parent reports max-rank latency and records the selected packet
all-reduce route.

The SGLang target is the PROFILE.md decode-dominated communication bucket:
214.69 ms over 657 calls = 0.3268 ms per call. This benchmark reports the
real ARK TP slice latency; the perf gate fails when the ratio exceeds 1.0.
"""

import argparse
import os
import re
import subprocess
import sys

try:
    from ._env import _load_worker_result, _subprocess_env
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _env import _load_worker_result, _subprocess_env


HIDDEN_SIZE = 4096
_TP_TARGET_MS = 214.69 / 657.0
_SENTINEL_MS = 999999.0
_PACKET_ROUTE = "all_reduce_packet"
_BASE_BRANCH = "qwen3-allreduce-bench"

_WORKER_SCRIPT = r'''
"""Worker: time one ARK row-parallel TP decode slice."""
import json
import os
import sys
import time

import torch
import ark
from ark.executor import Executor

rank = int(sys.argv[1])
world_size = int(sys.argv[2])
hidden_size = int(sys.argv[3])

local_hidden = hidden_size // world_size
if hidden_size % world_size != 0:
    raise RuntimeError("hidden_size must be divisible by world_size")

gen = torch.Generator(device="cpu")
gen.manual_seed(20260619 + rank)
x_cpu = (0.05 * torch.randn((1, local_hidden), generator=gen)).to(
    torch.float16
)
w_cpu = (0.05 * torch.randn((local_hidden, hidden_size), generator=gen)).to(
    torch.float16
)

ark.init()
ark.set_rank(rank)
ark.set_world_size(world_size)
ark.Model.set_device_id(rank)
torch.cuda.set_device(rank)

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
    t0 = time.perf_counter()
    rt.run(iter=1)
    host_s = time.perf_counter() - t0
    if world_size > 1:
        rt.barrier()
    rt.stop()

print(json.dumps({
    "rank": rank,
    "world_size": world_size,
    "hidden_size": hidden_size,
    "local_hidden": local_hidden,
    "route": reduce_op.__name__,
    "latency_ms": round(host_s * 1000.0, 6),
}))
sys.stdout.flush()

# Keep the result live through execution so the compiler cannot drop it.
_ = result
Executor.reset()
os._exit(0)
'''


def _tail(data, limit=500):
    """Return a short decoded tail for subprocess diagnostics."""
    return data.decode(errors="replace").strip()[-limit:]


def _is_sha(value):
    """Return True when *value* looks like a Git SHA."""
    return re.fullmatch(r"[0-9a-fA-F]{7,40}", value) is not None


def _repo_root():
    """Return the source repository root for this example file."""
    return os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    )


def _resolve_head_sha():
    """Return the current source SHA, or ``unknown`` when unavailable."""
    for name in (
        "ARK_HEAD_SHA",
        "GITHUB_SHA",
        "CI_COMMIT_SHA",
        "BUILD_SOURCEVERSION",
        "GIT_COMMIT",
    ):
        value = os.environ.get(name, "").strip()
        if _is_sha(value):
            return value
    try:
        value = subprocess.check_output(
            ["git", "-C", _repo_root(), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    if _is_sha(value):
        return value
    return "unknown"


def _resolve_base_sha():
    """Return the PR target branch SHA, or ``unknown`` if unavailable."""
    for name in ("ARK_BASE_SHA", "GITHUB_BASE_SHA", "BASE_SHA"):
        value = os.environ.get(name, "").strip()
        if _is_sha(value):
            return value
    for ref in (f"origin/{_BASE_BRANCH}", _BASE_BRANCH):
        try:
            value = subprocess.check_output(
                ["git", "-C", _repo_root(), "rev-parse", ref],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            continue
        if _is_sha(value):
            return value
    return "unknown"


def run_bench(world_size, timeout, hidden_size):
    """Return (ark_ms, route, failed) for one TP decode benchmark."""
    if world_size < 1:
        print(f"ERROR: invalid world_size={world_size}", file=sys.stderr)
        return _SENTINEL_MS, "unknown", True
    try:
        env = _subprocess_env(world_size)
    except Exception as exc:  # noqa: BLE001 - fail closed with PERF_GATE.
        print(f"ERROR: cannot build worker env: {exc}", file=sys.stderr)
        return _SENTINEL_MS, "unknown", True

    failed = False
    procs = []
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
                        str(hidden_size),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd="/",
                    env=env,
                )
            )
        except OSError as exc:
            failed = True
            print(f"ERROR rank={rank}: launch failed: {exc}", file=sys.stderr)
            break

    results = []
    try:
        for rank, proc in enumerate(procs):
            try:
                out, err = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                failed = True
                proc.kill()
                proc.wait()
                print(
                    f"ERROR rank={rank}: timed out after {timeout}s",
                    file=sys.stderr,
                )
                break
            result = _load_worker_result(out)
            if proc.returncode != 0:
                failed = True
                print(
                    f"ERROR rank={rank}: exit={proc.returncode} "
                    f"stderr={_tail(err, 300)}",
                    file=sys.stderr,
                )
            if result is None:
                failed = True
                print(
                    f"ERROR rank={rank}: no JSON result "
                    f"stdout_tail={_tail(out)} stderr_tail={_tail(err)}",
                    file=sys.stderr,
                )
            else:
                results.append(result)
    finally:
        for proc in procs:
            proc.kill()
            proc.wait()

    route = results[0].get("route", "unknown") if results else "unknown"
    if len(results) != world_size:
        failed = True
        route = "unknown"
        print(
            "ERROR: incomplete worker results; route unknown",
            file=sys.stderr,
        )
    if any(r.get("route") != _PACKET_ROUTE for r in results):
        failed = True
        route = "unknown"
        print(
            "ERROR: worker did not report all_reduce_packet route",
            file=sys.stderr,
        )
    if failed:
        return _SENTINEL_MS, route, True
    return max(r["latency_ms"] for r in results), route, False


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark ARK Qwen3 decode row-parallel TP matmul plus "
            "packet all-reduce"
        )
    )
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    args = parser.parse_args()

    ark_ms, route, failed = run_bench(
        world_size=args.world_size,
        timeout=args.timeout,
        hidden_size=args.hidden_size,
    )
    head_sha = _resolve_head_sha()
    base_sha = _resolve_base_sha()
    if head_sha == "unknown" or base_sha == "unknown" or route != _PACKET_ROUTE:
        failed = True
    ratio = ark_ms / _TP_TARGET_MS
    print(
        f"PERF_GATE name=tp"
        f" ark_ms={ark_ms:.4f}"
        f" sglang_ms={_TP_TARGET_MS:.4f}"
        f" ratio={ratio:.4f}"
        f" route={route}"
        f" head_sha={head_sha}"
        f" base_sha={base_sha}"
    )
    if failed or ratio >= 1.0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
