#!/usr/bin/env bash
set -euo pipefail

: "${ARK_ROOT:=$PWD}"
export ARK_ROOT
export PYTHONPATH="${PYTHONPATH:-$ARK_ROOT/python}"

head_sha=$(git -C .. rev-parse HEAD)
target_ms=$(python3 - <<'PY'
import importlib.util
import pathlib

path = pathlib.Path("../examples/qwen3/bench_allreduce.py")
spec = importlib.util.spec_from_file_location("bench_allreduce", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(f"{module._PREFILL_TARGET_MS:.4f}")
PY
)

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
status=0
python3 ../examples/qwen3/bench_allreduce.py --world-size 2 --shape prefill >"$tmpdir/tp2.log" 2>"$tmpdir/tp2.err" || status=1
python3 ../examples/qwen3/bench_allreduce.py --world-size 8 --shape prefill >"$tmpdir/tp8.log" 2>"$tmpdir/tp8.err" || status=1

read -r ark_ms parse_status < <(python3 - "$tmpdir/tp2.log" "$tmpdir/tp8.log" "$status" "$head_sha" "$target_ms" <<'PY'
import re
import sys

log_paths = [(2, sys.argv[1]), (8, sys.argv[2])]
status = int(sys.argv[3])
head_sha = sys.argv[4]
target_ms = float(sys.argv[5])
errors = []
values = []

if status:
    errors.append("one or more benchmark workers failed")

for world_size, path in log_paths:
    text = open(path, encoding="utf-8").read()
    if f"BENCH_SHA sha={head_sha}" not in text:
        errors.append(f"TP={world_size}: missing or stale BENCH_SHA")
    matches = re.findall(
        r"PERF_GATE name=allreduce_prefill\s+"
        r"ark_ms=([0-9.]+)\s+sglang_ms=([0-9.]+)\s+ratio=([0-9.]+)",
        text,
    )
    if len(matches) != 1:
        errors.append(f"TP={world_size}: expected one allreduce_prefill line")
        continue
    ark_ms = float(matches[0][0])
    sglang_ms = float(matches[0][1])
    if abs(sglang_ms - target_ms) > 1e-4:
        errors.append(f"TP={world_size}: wrong target {sglang_ms:.4f}")
    if ark_ms >= 999999.0:
        errors.append(f"TP={world_size}: sentinel latency")
    if ark_ms > target_ms:
        errors.append(f"TP={world_size}: ark_ms {ark_ms:.4f} > {target_ms:.4f}")
    values.append(ark_ms)

for error in errors:
    print(f"ERROR: {error}", file=sys.stderr)
if errors or len(values) != 2:
    print("999999.0000 1")
else:
    print(f"{max(values):.4f} 0")
PY
)
ratio=$(python3 - "$ark_ms" "$target_ms" <<'PY'
import sys

print(f"{float(sys.argv[1]) / float(sys.argv[2]):.4f}")
PY
)
printf 'PERF_GATE name=allreduce_prefill ark_ms=%s sglang_ms=%s ratio=%s\n' "$ark_ms" "$target_ms" "$ratio"
python3 - "$ark_ms" "$target_ms" "$status" "$parse_status" <<'PY'
import sys

ark_ms = float(sys.argv[1])
target_ms = float(sys.argv[2])
status = int(sys.argv[3])
parse_status = int(sys.argv[4])
if status or parse_status or ark_ms > target_ms:
    raise SystemExit(1)
PY
