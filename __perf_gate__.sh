#!/usr/bin/env bash
set -u -o pipefail

: "${ARK_ROOT:=$PWD}"
export ARK_ROOT
if [ -n "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="$ARK_ROOT/python:$PYTHONPATH"
else
    export PYTHONPATH="$ARK_ROOT/python"
fi

# SGLang PROFILE.md Qwen3 TP=8 attention decode target: 20.93 ms / 640 token-steps.
target_ms="0.0327"
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

status=0
python3 ../examples/qwen3/bench_kv_cache_slot.py >"$tmpdir/out" 2>"$tmpdir/err" || status=$?

python3 - "$tmpdir/out" "$tmpdir/err" "$status" "$target_ms" <<'PY'
import os
import re
import sys

out_path = sys.argv[1]
err_path = sys.argv[2]
status = int(sys.argv[3])
target_ms = float(sys.argv[4])
pattern = re.compile(
    r"^PERF_GATE name=kv_cache_slot "
    r"ark_ms=([0-9]+(?:\.[0-9]+)?) "
    r"sglang_ms=([0-9]+(?:\.[0-9]+)?) "
    r"ratio=([0-9]+(?:\.[0-9]+)?)$"
)

fallback_ratio = 999999.0 / target_ms
fallback = (
    "PERF_GATE name=kv_cache_slot "
    f"ark_ms=999999.0000 sglang_ms={target_ms:.4f} "
    f"ratio={fallback_ratio:.4f}"
)
lines = [
    ln
    for ln in open(out_path, encoding="utf-8").read().splitlines()
    if ln.startswith("PERF_GATE ")
]
if status or os.path.getsize(err_path) != 0 or len(lines) != 1:
    print(fallback)
    raise SystemExit(1)

match = pattern.match(lines[0])
if match is None:
    print(fallback)
    raise SystemExit(1)

line = lines[0]
print(line)
ark_ms = float(match.group(1))
sglang_ms = float(match.group(2))
ratio = float(match.group(3))
expected_ratio = ark_ms / sglang_ms
if (
    abs(sglang_ms - target_ms) > 0.00005
    or abs(ratio - expected_ratio) > 0.002
    or ark_ms >= 999999.0
    or ratio > 1.0
):
    raise SystemExit(1)
PY
