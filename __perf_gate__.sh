#!/usr/bin/env bash
set -u -o pipefail

: "${ARK_ROOT:=$PWD}"
export ARK_ROOT
if [ -n "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="$ARK_ROOT/python:$PYTHONPATH"
else
    export PYTHONPATH="$ARK_ROOT/python"
fi

allreduce_target_ms=$(python3 - <<'PY'
import importlib.util
import pathlib

path = pathlib.Path("../examples/qwen3/bench_allreduce.py")
spec = importlib.util.spec_from_file_location("bench_allreduce", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(f"{module._DECODE_TARGET_MS:.4f}")
PY
)

# SGLang PROFILE.md Qwen3 TP=8 attention decode target: 20.93 ms / 640 token-steps.
kv_cache_target_ms="0.0327"
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

gate_status=0

allreduce_status=0
python3 ../examples/qwen3/bench_allreduce.py --world-size 2 --shape decode \
    >"$tmpdir/allreduce_tp2.out" 2>"$tmpdir/allreduce_tp2.err" || allreduce_status=1
python3 ../examples/qwen3/bench_allreduce.py --world-size 8 --shape decode \
    >"$tmpdir/allreduce_tp8.out" 2>"$tmpdir/allreduce_tp8.err" || allreduce_status=1

python3 - \
    "$tmpdir/allreduce_tp2.out" \
    "$tmpdir/allreduce_tp8.out" \
    "$allreduce_status" \
    "$allreduce_target_ms" <<'PY' || gate_status=1
import re
import sys

out_paths = sys.argv[1:3]
status = int(sys.argv[3])
target_ms = float(sys.argv[4])
pattern = re.compile(
    r"^PERF_GATE name=allreduce "
    r"ark_ms=([0-9]+(?:\.[0-9]+)?) "
    r"sglang_ms=([0-9]+(?:\.[0-9]+)?) "
    r"ratio=([0-9]+(?:\.[0-9]+)?)$"
)

values = []
for out_path in out_paths:
    lines = [
        ln
        for ln in open(out_path, encoding="utf-8").read().splitlines()
        if ln.startswith("PERF_GATE name=allreduce ")
    ]
    if len(lines) != 1:
        continue
    match = pattern.match(lines[0])
    if match is None:
        continue
    values.append(float(match.group(1)))

ark_ms = max(values) if len(values) == 2 else 999999.0
ratio = ark_ms / target_ms
print(
    f"PERF_GATE name=allreduce ark_ms={ark_ms:.4f} "
    f"sglang_ms={target_ms:.4f} ratio={ratio:.4f}"
)
if status or len(values) != 2 or ark_ms >= target_ms:
    raise SystemExit(1)
PY

kv_cache_status=0
ARK_LOG_LEVEL=WARN python3 ../examples/qwen3/bench_kv_cache_slot.py \
    >"$tmpdir/kv_cache_slot.out" 2>"$tmpdir/kv_cache_slot.err" || kv_cache_status=$?

python3 - \
    "$tmpdir/kv_cache_slot.out" \
    "$tmpdir/kv_cache_slot.err" \
    "$kv_cache_status" \
    "$kv_cache_target_ms" <<'PY' || gate_status=1
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
    if ln.startswith("PERF_GATE name=kv_cache_slot ")
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

exit "$gate_status"
