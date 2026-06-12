#!/usr/bin/env python3
"""Measure SGLang Qwen3-8B prefill TTFT and decode per-token latency.

Uses the /generate endpoint with ignore_eos: true (the /v1/chat/completions
endpoint silently ignores ignore_eos in sglang v0.4.x–v0.5.x).

Metrics:
  - Prefill TTFT: prompt≈2048 tokens (default), max_new_tokens=1.
    Time from request send to first (only) token received.
  - Decode per-token latency: prompt≈2048 tokens (default), max_new_tokens=128.
    total_time / output_tokens (approximation; prefill << decode for 128 output tokens).

Reports median over N trials (default 5).
Captures GPU clocks via nvidia-smi at start of run.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from typing import Any

try:
    import requests
except ImportError:
    print(
        "Error: 'requests' package required. Install: pip install requests",
        file=sys.stderr,
    )
    sys.exit(1)


def build_prompt(num_tokens: int) -> str:
    """Build a prompt that is approximately `num_tokens` tokens long.

    Uses a repeating word pattern. Actual token count depends on the
    tokenizer, but "word " ≈ 1 token for most BPE tokenizers, so we
    slightly over-generate to ensure we hit the target length.
    """
    # Pad by 1.2x to account for BPE tokenizer variance (some words may produce <1 token).
    word = "hello "
    return word * int(num_tokens * 1.2)


def capture_gpu_clocks() -> str:
    """Capture current GPU clocks via nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,clocks.gr,clocks.mem,clocks.max.gr,clocks.max.mem",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return "nvidia-smi not available"


def send_request(
    base_url: str, prompt: str, max_new_tokens: int
) -> dict[str, Any]:
    """Send a /generate request and measure timing.

    Returns dict with keys: total_ms, output_tokens, output_text.
    """
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
        },
    }

    start = time.perf_counter()
    resp = requests.post(f"{base_url}/generate", json=payload, timeout=120)
    total = time.perf_counter() - start

    resp.raise_for_status()
    data = resp.json()

    # The /generate endpoint returns the generated text.
    output_text = data.get("text", "")
    # Rough token count from meta if available, else estimate from spaces.
    meta = data.get("meta_info", {})
    output_tokens = meta.get("completion_tokens", len(output_text.split()))
    if "completion_tokens" not in meta:
        print(
            "Warning: completion_tokens missing from meta_info, estimating from word count",
            file=sys.stderr,
        )

    return {
        "total_ms": total * 1000,
        "output_tokens": output_tokens,
        "output_text": output_text[:200],  # truncate for display
    }


def measure_ttft(base_url: str, prompt: str) -> float:
    """Measure prefill TTFT: prompt -> 1 token."""
    result = send_request(base_url, prompt, max_new_tokens=1)
    return result["total_ms"]


def measure_decode(
    base_url: str, prompt: str, max_new_tokens: int
) -> dict[str, Any]:
    """Measure decode per-token latency: prompt -> max_new_tokens tokens."""
    result = send_request(base_url, prompt, max_new_tokens=max_new_tokens)

    output_tokens = result["output_tokens"]
    total_ms = result["total_ms"]

    # Per-token latency: we don't have streaming TTFT here, so we estimate
    # per-token as total / output_tokens for the decode portion.
    # For a more precise split, use streaming — but for a baseline this
    # is sufficient since prefill << decode for 128 output tokens.
    if output_tokens == 0:
        print("Warning: server returned 0 output tokens", file=sys.stderr)
    per_token_ms = total_ms / max(output_tokens, 1)

    return {
        "total_ms": total_ms,
        "output_tokens": output_tokens,
        "per_token_ms": per_token_ms,
    }


def run_trials(
    base_url: str,
    prompt: str,
    num_trials: int,
    num_warmup: int,
    prompt_tokens: int,
) -> dict[str, Any]:
    """Run TTFT and decode measurements, return median results."""
    print(f"Running {num_warmup} warmup request(s) ...")
    for i in range(num_warmup):
        send_request(base_url, prompt, max_new_tokens=1)
        print(f"  warmup {i+1}/{num_warmup} done")

    # --- TTFT (prefill) ---
    print(
        f"\nMeasuring TTFT ({num_trials} trials, prompt≈{prompt_tokens} tokens, max_new_tokens=1) ..."
    )
    ttft_values = []
    for i in range(num_trials):
        ttft = measure_ttft(base_url, prompt)
        ttft_values.append(ttft)
        print(f"  trial {i+1}/{num_trials}: TTFT = {ttft:.2f} ms")

    # --- Decode per-token ---
    print(
        f"\nMeasuring decode latency ({num_trials} trials, prompt≈{prompt_tokens} tokens, max_new_tokens=128) ..."
    )
    decode_values = []
    for i in range(num_trials):
        result = measure_decode(base_url, prompt, max_new_tokens=128)
        decode_values.append(result["per_token_ms"])
        print(
            f"  trial {i+1}/{num_trials}: {result['per_token_ms']:.2f} ms/token "
            f"({result['output_tokens']} tokens in {result['total_ms']:.1f} ms)"
        )

    return {
        "ttft_median_ms": statistics.median(ttft_values),
        "ttft_all_ms": ttft_values,
        "decode_per_token_median_ms": statistics.median(decode_values),
        "decode_all_ms": decode_values,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure SGLang Qwen3-8B prefill TTFT and decode latency."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=30000,
        help="SGLang server port (default: 30000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="SGLang server host (default: localhost)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of measurement trials (default: 5)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup requests (default: 3)",
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=2048,
        help="Approximate prompt length in tokens (default: 2048)",
    )
    parser.add_argument(
        "--output",
        default="baseline_results.json",
        help="Path for JSON output file (default: baseline_results.json)",
    )
    args = parser.parse_args()

    if args.prompt_tokens < 1:
        parser.error("--prompt-tokens must be >= 1")
    if args.trials < 1:
        parser.error("--trials must be >= 1")

    base_url = f"http://{args.host}:{args.port}"

    # Check server health
    print(f"Checking server at {base_url} ...")
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        resp.raise_for_status()
    except (
        requests.ConnectionError,
        requests.Timeout,
        requests.HTTPError,
    ) as e:
        print(f"Error: cannot reach server at {base_url}: {e}", file=sys.stderr)
        sys.exit(1)
    print("Server is healthy.\n")

    # Capture GPU clocks
    print("GPU clocks:")
    gpu_clocks = capture_gpu_clocks()
    print(gpu_clocks)
    print()

    # Build prompt
    prompt = build_prompt(args.prompt_tokens)

    # Run measurements
    results = run_trials(
        base_url, prompt, args.trials, args.warmup, args.prompt_tokens
    )

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Prompt length:           ~{args.prompt_tokens} tokens")
    print(f"Trials:                  {args.trials}")
    print(f"Prefill TTFT (median):   {results['ttft_median_ms']:.2f} ms")
    print(
        f"  all:                   {[f'{v:.2f}' for v in results['ttft_all_ms']]}"
    )
    print(
        f"Decode per-token (median): {results['decode_per_token_median_ms']:.2f} ms/token"
    )
    print(
        f"  all:                   {[f'{v:.2f}' for v in results['decode_all_ms']]}"
    )
    print(f"GPU clocks:\n{gpu_clocks}")
    print("=" * 60)

    # Write JSON results
    output = {
        "prompt_tokens": args.prompt_tokens,
        "trials": args.trials,
        "ttft_median_ms": results["ttft_median_ms"],
        "ttft_all_ms": results["ttft_all_ms"],
        "decode_per_token_median_ms": results["decode_per_token_median_ms"],
        "decode_all_ms": results["decode_all_ms"],
        "gpu_clocks": gpu_clocks,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nJSON results written to {args.output}")


if __name__ == "__main__":
    main()
