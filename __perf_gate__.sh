#!/usr/bin/env bash
set -uo pipefail

: "${ARK_ROOT:=$PWD}"
export ARK_ROOT
export PYTHONPATH="${PYTHONPATH:-$ARK_ROOT/python}"

target_ms=$(python3 - <<'PY'
import importlib.util
import pathlib

path = pathlib.Path("../examples/qwen3/bench_allreduce.py")
spec = importlib.util.spec_from_file_location("bench_allreduce", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(f"{module._DECODE_TARGET_MS:.4f}")
PY
)

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
status=0
python3 ../examples/qwen3/bench_allreduce.py --world-size 2 --shape decode >"$tmpdir/tp2.log" 2>"$tmpdir/tp2.err" || status=1
python3 ../examples/qwen3/bench_allreduce.py --world-size 8 --shape decode >"$tmpdir/tp8.log" 2>"$tmpdir/tp8.err" || status=1

ark_ms=$(python3 - "$tmpdir/tp2.log" "$tmpdir/tp8.log" "$status" <<'PY'
import re
import sys

values = []
valid_evidence = True
for name in sys.argv[1:3]:
    text = open(name, encoding="utf-8").read()
    perf = re.search(r"PERF_GATE name=allreduce_decode\s+ark_ms=([0-9.]+)", text)
    bench = re.search(
        r"BENCH_RESULT name=allreduce_decode\s+head_sha=(\S+)\s+"
        r"route=(\S+)\s+.*?\s+ark_ms=([0-9.]+)",
        text,
    )
    if perf:
        values.append(float(perf.group(1)))
    if not bench or bench.group(1) == "unknown" or bench.group(2) != "packet":
        valid_evidence = False
if int(sys.argv[3]) or len(values) != 2 or not valid_evidence:
    print("999999.0000")
else:
    print(f"{max(values):.4f}")
PY
)
ratio=$(python3 - "$ark_ms" "$target_ms" <<'PY'
import sys

print(f"{float(sys.argv[1]) / float(sys.argv[2]):.4f}")
PY
)
printf 'PERF_GATE name=allreduce ark_ms=%s sglang_ms=%s ratio=%s\n' "$ark_ms" "$target_ms" "$ratio"
python3 - "$ark_ms" "$target_ms" "$status" <<'PY'
import sys

ark_ms = float(sys.argv[1])
target_ms = float(sys.argv[2])
status = int(sys.argv[3])
if status or ark_ms >= target_ms:
    raise SystemExit(1)
PY
