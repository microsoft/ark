# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Benchmark ARK graph-reused KV-cache decode attention.

The SGLang target is the Q12A attention-cache budget from
``sglang-eval/PROFILE.md``: 20.93 ms over 640 decode token-steps.
"""

import argparse
import sys
import time

import torch

try:
    from .kv_cache_decode import (
        QWEN3_DECODE_CONFIG,
        KVCacheDecodeConfig,
        KVCacheDecodeGraph,
        make_prefix_mask,
    )
except ImportError:
    from kv_cache_decode import (  # type: ignore
        QWEN3_DECODE_CONFIG,
        KVCacheDecodeConfig,
        KVCacheDecodeGraph,
        make_prefix_mask,
    )


SGLANG_KV_CACHE_DECODE_MS = 20.93 / 640.0


def format_perf_gate(
    ark_ms: float, sglang_ms: float = SGLANG_KV_CACHE_DECODE_MS
):
    """Return the required machine-readable perf-gate line."""

    ratio = ark_ms / sglang_ms
    return (
        f"PERF_GATE name=kv_cache_decode"
        f" ark_ms={ark_ms:.4f}"
        f" sglang_ms={sglang_ms:.4f}"
        f" ratio={ratio:.4f}"
    )


def _make_tokens(config: KVCacheDecodeConfig, count: int, device: torch.device):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(2026)
    q_cpu = torch.randn(
        count,
        config.num_q_heads,
        config.head_dim,
        generator=gen,
        dtype=torch.float32,
    ).to(config.dtype)
    k_cpu = torch.randn(
        count,
        config.num_kv_heads,
        config.head_dim,
        generator=gen,
        dtype=torch.float32,
    ).to(config.dtype)
    v_cpu = torch.randn(
        count,
        config.num_kv_heads,
        config.head_dim,
        generator=gen,
        dtype=torch.float32,
    ).to(config.dtype)
    return q_cpu.to(device), k_cpu.to(device), v_cpu.to(device)


def measure_ark_ms(
    config: KVCacheDecodeConfig,
    positions: list[int],
    warmup: int,
    device: torch.device,
) -> float:
    """Measure average per-token latency over a real decode position sequence."""

    if warmup < 1:
        raise ValueError("warmup must be at least 1")
    if not positions:
        raise ValueError("positions must be non-empty")
    if min(positions) < 0 or max(positions) >= config.max_seq_len:
        raise ValueError("positions must be inside the configured cache length")

    timed_steps = len(positions)
    total = warmup + timed_steps
    q, k, v = _make_tokens(config, total, device)
    warm_k_cache = torch.zeros(
        config.num_kv_heads,
        config.max_seq_len,
        config.head_dim,
        dtype=config.dtype,
        device=device,
    )
    warm_v_cache = torch.zeros_like(warm_k_cache)
    timed_k_cache = torch.zeros_like(warm_k_cache)
    timed_v_cache = torch.zeros_like(warm_k_cache)
    warm_outputs = torch.empty(
        warmup,
        config.num_q_heads,
        config.head_dim,
        dtype=config.dtype,
        device=device,
    )
    timed_outputs = torch.empty(
        timed_steps,
        config.num_q_heads,
        config.head_dim,
        dtype=config.dtype,
        device=device,
    )
    warm_mask = make_prefix_mask(config, positions[0], device)
    timed_masks = [
        make_prefix_mask(config, position, device) for position in positions
    ]

    graph = KVCacheDecodeGraph(config)
    warm_bindings = [
        graph.bindings(
            q[i],
            k[i],
            v[i],
            warm_k_cache,
            warm_v_cache,
            warm_mask,
            warm_outputs[i],
            positions[0],
        )
        for i in range(warmup)
    ]
    timed_bindings = [
        graph.bindings(
            q[warmup + i],
            k[warmup + i],
            v[warmup + i],
            timed_k_cache,
            timed_v_cache,
            timed_masks[i],
            timed_outputs[i],
            position,
        )
        for i, position in enumerate(positions)
    ]

    torch.cuda.synchronize(device)
    graph.launch(warm_bindings[0], device_id=device.index or 0)
    try:
        graph.run()
        for binding in warm_bindings[1:]:
            graph.run(binding)
        start = time.perf_counter()
        for binding in timed_bindings:
            graph.run(binding)
        elapsed = time.perf_counter() - start
    finally:
        graph.stop()
    return elapsed * 1000.0 / timed_steps


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark ARK KV-cache decode attention for Q12A"
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=QWEN3_DECODE_CONFIG.max_seq_len
    )
    parser.add_argument(
        "--num-q-heads", type=int, default=QWEN3_DECODE_CONFIG.num_q_heads
    )
    parser.add_argument(
        "--num-kv-heads", type=int, default=QWEN3_DECODE_CONFIG.num_kv_heads
    )
    parser.add_argument(
        "--head-dim", type=int, default=QWEN3_DECODE_CONFIG.head_dim
    )
    parser.add_argument(
        "--start-position",
        type=int,
        default=0,
        help="first measured decode position",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=QWEN3_DECODE_CONFIG.max_seq_len,
        help="number of measured decode token-steps",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    config = KVCacheDecodeConfig(
        max_seq_len=args.max_seq_len,
        num_q_heads=args.num_q_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
    )
    device = torch.device(f"cuda:{args.device}")
    ark_ms = 999999.0
    try:
        positions = list(
            range(args.start_position, args.start_position + args.steps)
        )
        ark_ms = measure_ark_ms(
            config=config,
            positions=positions,
            warmup=args.warmup,
            device=device,
        )
    except Exception as exc:  # pragma: no cover - exercised only on GPU failure
        print(
            f"ERROR: ARK KV-cache decode benchmark failed: {exc}",
            file=sys.stderr,
        )
        print(format_perf_gate(ark_ms))
        raise SystemExit(1) from exc
    print(format_perf_gate(ark_ms))


if __name__ == "__main__":
    main()
