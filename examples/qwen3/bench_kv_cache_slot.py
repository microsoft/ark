# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Benchmark ARK fixed-layout KV-cache slot update/read.

The gate target is the Qwen3 TP=8 attention decode budget from PROFILE.md:
20.93 ms over 5*128 token-steps, or 0.0327 ms/token.
"""

import argparse
import os
import sys
import time

try:
    from ._env import _has_compiled_ark
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _env import _has_compiled_ark

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)


def _prepend_compiled_pythonpath():
    candidates = []
    ark_root = os.environ.get("ARK_ROOT")
    if ark_root:
        ark_root = os.path.abspath(ark_root)
        candidates.append(os.path.join(ark_root, "python"))
        candidates.append(os.path.join(ark_root, "build", "python"))
    candidates.append(os.path.join(_REPO_ROOT, "build", "python"))

    for candidate in candidates:
        candidate = os.path.normpath(candidate)
        if _has_compiled_ark(candidate):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            return


_prepend_compiled_pythonpath()

try:
    import torch
except Exception as exc:
    print(f"ERROR: failed to import torch: {exc}", file=sys.stderr)
    torch = None

try:
    import ark
except Exception as exc:
    print(f"ERROR: failed to import ark: {exc}", file=sys.stderr)
    ark = None


_SGLANG_MS = 20.93 / 640.0
_SENTINEL_MS = 999999.0


def _perf_gate_line(ark_ms):
    ratio = ark_ms / _SGLANG_MS
    print(
        f"PERF_GATE name=kv_cache_slot"
        f" ark_ms={ark_ms:.4f}"
        f" sglang_ms={_SGLANG_MS:.4f}"
        f" ratio={ratio:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ARK fixed-layout KV-cache slot update/read"
    )
    parser.add_argument("--iters", type=int, default=512)
    parser.add_argument("--max-seq", type=int, default=2048)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    args = parser.parse_args()
    emitted_gate_line = False

    try:
        available = (
            torch is not None and ark is not None and torch.cuda.is_available()
        )
    except Exception:
        available = False
    if not available:
        _perf_gate_line(_SENTINEL_MS)
        emitted_gate_line = True
        raise SystemExit(1)
    if args.iters < 1 or args.iters > args.max_seq:
        _perf_gate_line(_SENTINEL_MS)
        emitted_gate_line = True
        raise SystemExit("--iters must satisfy 1 <= --iters <= --max-seq")

    try:
        ark.init()
        torch.cuda.set_device(0)
        slot_shape = (args.kv_heads, args.head_dim)
        cache = torch.zeros(
            (args.max_seq,) + slot_shape, dtype=torch.bfloat16, device="cuda:0"
        )
        token = (
            torch.arange(
                args.kv_heads * args.head_dim,
                dtype=torch.float32,
                device="cuda:0",
            )
            .reshape(slot_shape)
            .to(torch.bfloat16)
        )
        position = torch.zeros(1, dtype=torch.int32, device="cuda:0")
        torch.cuda.synchronize(0)

        slot = ark.kv_cache_slot(cache, token, position)

        with ark.Runtime() as rt:
            rt.launch(device_id=0, loop_mode=True)
            start = time.perf_counter()
            rt.run(iter=args.iters)
            elapsed_s = time.perf_counter() - start
            rt.stop()

        # Proof after stopping ARK: the graph ran and advanced in-cache state.
        position_cpu = int(position.cpu().item())
        cache_cpu = cache[: args.iters].cpu()
        token_cpu = token.cpu()
        slot_cpu = slot.to_torch().cpu()
        expected_cache = token_cpu.expand(args.iters, *slot_shape)
        proof_ok = (
            position_cpu == args.iters
            and torch.equal(cache_cpu, expected_cache)
            and torch.equal(slot_cpu, token_cpu)
        )
        ark_ms = (
            elapsed_s * 1000.0 / float(args.iters) if proof_ok else _SENTINEL_MS
        )
        _perf_gate_line(ark_ms)
        emitted_gate_line = True
        if not proof_ok:
            raise SystemExit(1)

        # Avoid Python shutdown/destructor races after valid ARK evidence.
        sys.stdout.flush()
        from ark.executor import Executor

        Executor.reset()
        os._exit(0)
    except Exception:
        if not emitted_gate_line:
            _perf_gate_line(_SENTINEL_MS)
        raise


if __name__ == "__main__":
    main()
