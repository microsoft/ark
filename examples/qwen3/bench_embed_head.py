# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""ARK-only latency benchmark for Qwen3 embedding -> final norm -> lm_head.

The benchmark uses torch only to allocate finite input tensors before ARK launch.
No torch GPU operation is issued while the ARK runtime is launched.
"""

import argparse
import math
import os
import sys
import time

try:
    import torch
except ImportError as exc:  # pragma: no cover - CI has torch.
    raise SystemExit(f"torch is required: {exc}")

import ark

try:
    from .embed_head import (
        QWEN3_HIDDEN_SIZE,
        QWEN3_VOCAB_SIZE,
        qwen3_embed_head,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from embed_head import (  # noqa: E402
        QWEN3_HIDDEN_SIZE,
        QWEN3_VOCAB_SIZE,
        qwen3_embed_head,
    )

# PROFILE.md records Q8-relevant remaining model work as "other" = 0.57 ms
# and notes "embed/lm_head <0.2%" for TP=8 batch=1 decode-dominated Qwen3-8B.
_SGLANG_TARGET_MS = 0.57
_FAILURE_MS = 999999.0

SHAPES = {
    "decode": (1, 1, QWEN3_HIDDEN_SIZE, QWEN3_VOCAB_SIZE),
    "ci-prefill": (1, 128, 512, 1024),
}


def _torch_dtype(name):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def _run_bench(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    batch, seq_len, hidden_size, vocab_size = SHAPES[args.shape]
    dtype = _torch_dtype(args.dtype)
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(args.device)

    ark.init()

    # Finite setup data.  This happens before ARK launch and is not timed.
    tokens = torch.zeros(
        (batch, seq_len), dtype=torch.int32, device=device
    )
    embed_weight = torch.zeros(
        (vocab_size, hidden_size), dtype=dtype, device=device
    )
    norm_weight = torch.ones(hidden_size, dtype=dtype, device=device)
    lm_head_weight = torch.zeros(
        (vocab_size, hidden_size), dtype=dtype, device=device
    )
    torch.cuda.synchronize(args.device)

    logits = qwen3_embed_head(
        tokens, embed_weight, norm_weight, lm_head_weight
    )

    with ark.Runtime() as rt:
        rt.launch(device_id=args.device)
        for _ in range(args.warmup):
            rt.run(iter=1)

        t0 = time.perf_counter()
        for _ in range(args.iters):
            rt.run(iter=1)
        elapsed_s = time.perf_counter() - t0

    # Keep the graph result live until after runtime stop.
    if logits.shape() != [batch, seq_len, vocab_size]:
        raise RuntimeError(f"unexpected output shape: {logits.shape()}")

    ark_ms = elapsed_s * 1000.0 / args.iters
    if not math.isfinite(ark_ms) or ark_ms <= 0:
        raise RuntimeError(f"invalid latency: {ark_ms}")
    return ark_ms


def _print_perf_gate(ark_ms):
    ratio = ark_ms / _SGLANG_TARGET_MS
    print(
        f"PERF_GATE name=embed_head"
        f" ark_ms={ark_ms:.4f}"
        f" sglang_ms={_SGLANG_TARGET_MS:.4f}"
        f" ratio={ratio:.4f}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ARK Qwen3 embed -> final RMSNorm -> lm_head"
    )
    parser.add_argument("--shape", choices=SHAPES.keys(), default="decode")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    if args.iters <= 0:
        print("ERROR: --iters must be positive", file=sys.stderr)
        _print_perf_gate(_FAILURE_MS)
        return 1
    if args.warmup < 0:
        print("ERROR: --warmup must be non-negative", file=sys.stderr)
        _print_perf_gate(_FAILURE_MS)
        return 1

    try:
        ark_ms = _run_bench(args)
    except Exception as exc:  # Print a fail-closed PERF_GATE line.
        print(f"ERROR: {exc}", file=sys.stderr)
        _print_perf_gate(_FAILURE_MS)
        return 1

    _print_perf_gate(ark_ms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
