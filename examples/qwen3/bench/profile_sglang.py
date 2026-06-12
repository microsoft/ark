#!/usr/bin/env python3
"""Profile SGLang Qwen3-8B with torch.profiler for prefill and decode phases.

Uses SGLang's ``/start_profile`` / ``/stop_profile`` server endpoints to
trigger server-side ``torch.profiler``, then sends HTTP requests to exercise
prefill (prompt=2048, max_new_tokens=1) and decode (prompt=2048,
max_new_tokens=128) phases.  Chrome-trace JSON files are saved on the server's
filesystem (default ``/tmp/sglang_profile/``).

Profiler configuration:
    schedule:       wait=2, warmup=1, active=3
    record_shapes:  True
    with_stack:     True
    output:         tensorboard_trace_handler

Usage (against a running SGLang server — see reproduce_profile.md)::

    python profile_sglang.py --port 30000 --output-dir /tmp/sglang_profile
    python profile_sglang.py --phase prefill --trials 5
    python profile_sglang.py --phase decode  --trials 5

Prerequisites:
    - SGLang server running (see reproduce_profile.md)
    - ``requests`` Python package
    - Pinned SGLang image: lmsysorg/sglang:v0.4.6.post1-cu124
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

try:
    import torch.profiler as _torch_profiler  # noqa: F401 — used by helpers
except ImportError:
    _torch_profiler = None  # type: ignore[assignment]

try:
    import requests as _requests
except ImportError:
    _requests = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Profiler configuration helpers (testable without GPU)
# ---------------------------------------------------------------------------

# Default schedule parameters
SCHEDULE_WAIT = 2
SCHEDULE_WARMUP = 1
SCHEDULE_ACTIVE = 3
DEFAULT_OUTPUT_DIR = "/tmp/sglang_profile"
DEFAULT_PORT = 30000
DEFAULT_PROMPT_LEN = 2048
DEFAULT_DECODE_TOKENS = 128
DEFAULT_TRIALS = 5

# Pinned SGLang image (matches Q1 pinned image)
SGLANG_IMAGE_TAG = "lmsysorg/sglang:v0.4.6.post1-cu124"


# Canonical schedule constructor — used server-side or by external tooling.
# Tested when torch is available to keep the specification in sync with the constants.
def build_profiler_schedule() -> Any:
    """Build a ``torch.profiler.schedule`` with the canonical parameters.

    Returns:
        A callable schedule suitable for ``torch.profiler.profile(schedule=...)``.

    Raises:
        RuntimeError: If ``torch`` is not installed.
    """
    if _torch_profiler is None:
        raise RuntimeError("torch is required to build the profiler schedule")
    return _torch_profiler.schedule(
        wait=SCHEDULE_WAIT,
        warmup=SCHEDULE_WARMUP,
        active=SCHEDULE_ACTIVE,
    )


def build_profiler_config(output_dir: str = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    """Return the ``torch.profiler.profile`` keyword arguments as a dict.

    This dict is the *specification*; it can be passed directly to
    ``torch.profiler.profile(**config)`` on the server side, or serialised
    for documentation.

    Returns:
        Dict with keys matching ``torch.profiler.profile`` kwargs.
    """
    config: dict[str, Any] = {
        "activities": ["cpu", "cuda"],
        "schedule": {
            "wait": SCHEDULE_WAIT,
            "warmup": SCHEDULE_WARMUP,
            "active": SCHEDULE_ACTIVE,
        },
        "record_shapes": True,
        "with_stack": True,
        "output_dir": output_dir,
    }
    return config


# ---------------------------------------------------------------------------
# Prompt construction (reuses Q1 pattern)
# ---------------------------------------------------------------------------


def build_prompt(num_tokens: int) -> str:
    """Build a prompt that is approximately *num_tokens* tokens long.

    Uses a repeating word pattern.  Actual count depends on the tokenizer;
    ``"hello "`` ≈ 1 token for most BPE tokenizers.  Over-generates by 1.2×
    to compensate for BPE variance.
    """
    return "hello " * int(num_tokens * 1.2)


# ---------------------------------------------------------------------------
# SGLang HTTP helpers
# ---------------------------------------------------------------------------


def _require_requests() -> None:
    if _requests is None:
        raise ImportError("'requests' package required. Install: pip install requests")


def start_server_profile(base_url: str) -> None:
    """POST /start_profile to begin server-side torch.profiler."""
    _require_requests()
    resp = _requests.post(f"{base_url}/start_profile")
    resp.raise_for_status()


def stop_server_profile(base_url: str) -> None:
    """POST /stop_profile to stop server-side torch.profiler and flush trace."""
    _require_requests()
    resp = _requests.post(f"{base_url}/stop_profile")
    resp.raise_for_status()


def send_generate(
    base_url: str,
    prompt: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Send a /generate request and return the JSON response."""
    _require_requests()
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
        },
    }
    start = time.perf_counter()
    resp = _requests.post(f"{base_url}/generate", json=payload, timeout=300)
    elapsed_ms = (time.perf_counter() - start) * 1000
    resp.raise_for_status()
    result = resp.json()
    result["elapsed_ms"] = elapsed_ms
    return result


# ---------------------------------------------------------------------------
# Profiling workflow
# ---------------------------------------------------------------------------


def run_phase(
    base_url: str,
    phase: str,
    prompt_len: int,
    max_new_tokens: int,
    trials: int,
) -> list[dict[str, Any]]:
    """Run *trials* requests for a single phase (prefill or decode).

    Starts/stops server-side profiling around the request batch.

    Returns:
        List of per-trial result dicts.
    """
    prompt = build_prompt(prompt_len)
    results: list[dict[str, Any]] = []

    print(f"[{phase}] Starting server-side profiler ...")
    start_server_profile(base_url)

    for i in range(trials):
        print(f"[{phase}] Trial {i + 1}/{trials} "
              f"(prompt≈{prompt_len}, max_new_tokens={max_new_tokens})")
        result = send_generate(base_url, prompt, max_new_tokens)
        results.append(result)
        print(f"  elapsed: {result['elapsed_ms']:.1f} ms")

    print(f"[{phase}] Stopping server-side profiler ...")
    stop_server_profile(base_url)
    return results


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Profile SGLang Qwen3-8B prefill/decode with torch.profiler",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"SGLang server port (default {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="SGLang server host (default localhost)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Server-side trace output directory (default {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--phase",
        choices=["prefill", "decode", "both"],
        default="both",
        help="Which phase to profile (default both)",
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=DEFAULT_PROMPT_LEN,
        help=f"Prompt length in tokens (default {DEFAULT_PROMPT_LEN})",
    )
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=DEFAULT_DECODE_TOKENS,
        help=f"Number of tokens to generate in decode phase (default {DEFAULT_DECODE_TOKENS})",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help=f"Number of trials per phase (default {DEFAULT_TRIALS})",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    base_url = f"http://{args.host}:{args.port}"

    print(f"SGLang server: {base_url}")
    print(f"Profiler config: {json.dumps(build_profiler_config(args.output_dir), indent=2)}")
    print()

    if args.phase in ("prefill", "both"):
        run_phase(
            base_url,
            phase="prefill",
            prompt_len=args.prompt_len,
            max_new_tokens=1,
            trials=args.trials,
        )
        print()

    if args.phase in ("decode", "both"):
        run_phase(
            base_url,
            phase="decode",
            prompt_len=args.prompt_len,
            max_new_tokens=args.decode_tokens,
            trials=args.trials,
        )
        print()

    print("Done. Traces saved on server at:", args.output_dir)
    print("Copy traces locally, then run: bash analyze_profile.sh <trace.json>")


if __name__ == "__main__":
    main()
