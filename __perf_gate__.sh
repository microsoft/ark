#!/usr/bin/env bash
set -uo pipefail

: "${ARK_ROOT:=$PWD}"
export ARK_ROOT
export PYTHONPATH="${PYTHONPATH:-$ARK_ROOT/python}"

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

tp_target_ms=$(python3 - <<'PY'
import importlib.util
import pathlib

path = pathlib.Path("../examples/qwen3/bench_tp.py")
spec = importlib.util.spec_from_file_location("bench_tp", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(f"{module._TP_TARGET_MS:.4f}")
PY
)

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

status=0
python3 ../examples/qwen3/bench_allreduce.py --world-size 2 --shape decode >"$tmpdir/allreduce-tp2.log" 2>"$tmpdir/allreduce-tp2.err" || status=1
python3 ../examples/qwen3/bench_allreduce.py --world-size 8 --shape decode >"$tmpdir/allreduce-tp8.log" 2>"$tmpdir/allreduce-tp8.err" || status=1

allreduce_ark_ms=$(python3 - "$tmpdir/allreduce-tp2.log" "$tmpdir/allreduce-tp8.log" "$status" <<'PY'
import re
import sys

values = []
for name in sys.argv[1:3]:
    text = open(name, encoding="utf-8").read()
    match = re.search(r"PERF_GATE name=allreduce\s+ark_ms=([0-9.]+)", text)
    if match:
        values.append(float(match.group(1)))
if int(sys.argv[3]) or len(values) != 2:
    print("999999.0000")
else:
    print(f"{max(values):.4f}")
PY
)
allreduce_ratio=$(python3 - "$allreduce_ark_ms" "$allreduce_target_ms" <<'PY'
import sys

print(f"{float(sys.argv[1]) / float(sys.argv[2]):.4f}")
PY
)
if ! python3 - "$allreduce_ark_ms" "$allreduce_target_ms" "$status" <<'PY'
import sys

ark_ms = float(sys.argv[1])
target_ms = float(sys.argv[2])
status = int(sys.argv[3])
if status or ark_ms >= target_ms:
    raise SystemExit(1)
PY
then
    printf 'PERF_GATE name=allreduce ark_ms=%s sglang_ms=%s ratio=%s\n' "$allreduce_ark_ms" "$allreduce_target_ms" "$allreduce_ratio"
    exit 1
fi

tp_status=0
python3 ../examples/qwen3/bench_tp.py --world-size 8 >"$tmpdir/tp.log" 2>"$tmpdir/tp.err" || tp_status=1
read -r tp_ark_ms tp_ratio < <(python3 - "$tmpdir/tp.log" "$tp_status" "$tp_target_ms" <<'PY'
import re
import sys

text = open(sys.argv[1], encoding="utf-8").read()
status = int(sys.argv[2])
target = float(sys.argv[3])
match = re.search(r"PERF_GATE name=tp\s+ark_ms=([0-9.]+)\s+sglang_ms=([0-9.]+)\s+ratio=([0-9.]+)", text)
if status or match is None:
    ark_ms = 999999.0
    ratio = ark_ms / target
else:
    ark_ms = float(match.group(1))
    ratio = float(match.group(3))
print(f"{ark_ms:.4f} {ratio:.4f}")
PY
)
printf 'PERF_GATE name=tp ark_ms=%s sglang_ms=%s ratio=%s\n' "$tp_ark_ms" "$tp_target_ms" "$tp_ratio"
python3 - "$tp_ark_ms" "$tp_target_ms" "$tp_status" <<'PY'
import sys

ark_ms = float(sys.argv[1])
target_ms = float(sys.argv[2])
status = int(sys.argv[3])
if status or ark_ms >= target_ms:
    raise SystemExit(1)
PY
