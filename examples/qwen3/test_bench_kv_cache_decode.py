# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for the KV-cache decode perf-gate line."""

import os
import re
import sys

try:
    from .bench_kv_cache_decode import (
        SGLANG_KV_CACHE_DECODE_MS,
        format_perf_gate,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from bench_kv_cache_decode import (  # type: ignore
        SGLANG_KV_CACHE_DECODE_MS,
        format_perf_gate,
    )


def test_kv_cache_decode_perf_target_matches_profile_budget():
    """Q12A target is PROFILE.md attention budget 20.93 ms / 640 steps."""

    assert SGLANG_KV_CACHE_DECODE_MS == 20.93 / 640.0
    assert f"{SGLANG_KV_CACHE_DECODE_MS:.4f}" == "0.0327"


def test_kv_cache_decode_perf_gate_line_format():
    """The benchmark emits one machine-readable PERF_GATE line."""

    line = format_perf_gate(ark_ms=2 * SGLANG_KV_CACHE_DECODE_MS)
    pattern = re.compile(
        r"^PERF_GATE name=kv_cache_decode "
        r"ark_ms=([0-9]+\.[0-9]{4}) "
        r"sglang_ms=([0-9]+\.[0-9]{4}) "
        r"ratio=([0-9]+\.[0-9]{4})$"
    )
    match = pattern.match(line)
    assert match is not None
    assert match.group(2) == "0.0327"
    assert match.group(3) == "2.0000"
