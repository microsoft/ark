#!/usr/bin/env bash
set -u -o pipefail

: "${ARK_ROOT:=$PWD}"
export ARK_ROOT
export PYTHONPATH="${PYTHONPATH:-$ARK_ROOT/python}"

# SGLang PROFILE.md Qwen3 TP=8 attention decode target: 20.93 ms / 640 token-steps.
target_ms="0.0327"
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

status=0
python3 ../examples/qwen3/bench_kv_cache_slot.py >"$tmpdir/out" 2>"$tmpdir/err" || status=$?

line=$(python3 - "$tmpdir/out" <<'PY'
import re
import sys

text = open(sys.argv[1], encoding="utf-8").read().splitlines()
lines = [ln for ln in text if ln.startswith("PERF_GATE ")]
pattern = re.compile(
    r"^PERF_GATE name=kv_cache_slot "
    r"ark_ms=([0-9]+(?:\.[0-9]+)?) "
    r"sglang_ms=([0-9]+(?:\.[0-9]+)?) "
    r"ratio=([0-9]+(?:\.[0-9]+)?)$"
)
if len(lines) != 1 or pattern.match(lines[0]) is None:
    raise SystemExit(1)
print(lines[0])
PY
) || line="PERF_GATE name=kv_cache_slot ark_ms=999999.0000 sglang_ms=$target_ms ratio=30581009.1743"

printf '%s\n' "$line"
python3 - "$line" "$status" "$target_ms" <<'PY'
import re
import sys

line = sys.argv[1]
status = int(sys.argv[2])
target_ms = float(sys.argv[3])
match = re.match(
    r"^PERF_GATE name=kv_cache_slot "
    r"ark_ms=([0-9]+(?:\.[0-9]+)?) "
    r"sglang_ms=([0-9]+(?:\.[0-9]+)?) "
    r"ratio=([0-9]+(?:\.[0-9]+)?)$",
    line,
)
if match is None:
    raise SystemExit(1)
ark_ms = float(match.group(1))
sglang_ms = float(match.group(2))
ratio = float(match.group(3))
if (
    status
    or abs(sglang_ms - target_ms) > 0.00005
    or ark_ms >= 999999.0
    or ratio > 1.0
):
    raise SystemExit(1)
PY
