# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Benchmark ARK fixed-layout KV-cache slot update/read.

This primitive has no stamped SGLang KV-slot-specific component target in
PROFILE.md, so this script reports ARK latency only.
"""

import argparse
import time

try:
    import torch
except ImportError:
    torch = None

try:
    import ark
except ImportError:
    ark = None


def _bench_line(ark_ms, proof):
    print(f"ARK_BENCH name=kv_cache_slot ark_ms={ark_ms:.4f} proof={proof}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ARK fixed-layout KV-cache slot update/read"
    )
    parser.add_argument("--iters", type=int, default=512)
    parser.add_argument("--max-seq", type=int, default=2048)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    args = parser.parse_args()

    if torch is None or ark is None or not torch.cuda.is_available():
        _bench_line(999999.0, "fail")
        raise SystemExit(1)
    if args.iters < 1 or args.iters > args.max_seq:
        _bench_line(999999.0, "fail")
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
            elapsed_s * 1000.0 / float(args.iters) if proof_ok else 999999.0
        )
        _bench_line(ark_ms, "pass" if proof_ok else "fail")
        if not proof_ok:
            raise SystemExit(1)
    except Exception:
        _bench_line(999999.0, "fail")
        raise


if __name__ == "__main__":
    main()
