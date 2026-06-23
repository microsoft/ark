#!/usr/bin/env bash
set -uo pipefail

: "${ARK_ROOT:=$PWD}"
export ARK_ROOT
export PYTHONPATH="${PYTHONPATH:-$ARK_ROOT/python}"

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
status=0
python3 ../examples/qwen3/bench_allreduce.py --world-size 2 --shape decode --input-mode all \
  >"$tmpdir/tp2.log" 2>"$tmpdir/tp2.err" || status=1
python3 ../examples/qwen3/bench_allreduce.py --world-size 8 --shape decode --input-mode all \
  >"$tmpdir/tp8.log" 2>"$tmpdir/tp8.err" || status=1

python3 - "$status" "$tmpdir/tp2.log" "$tmpdir/tp8.log" <<'PY'
import importlib.util
import pathlib
import re
import sys

status = int(sys.argv[1])
logs = sys.argv[2:]
path = pathlib.Path("../examples/qwen3/bench_allreduce.py")
spec = importlib.util.spec_from_file_location("bench_allreduce", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
target_ms = module._DECODE_TARGET_MS

pattern = re.compile(
    r"RESULT name=allreduce shape=decode tp=(\d+) "
    r"mode=(external|internal) ark_ms=([0-9.]+)"
)
values = {}
for log in logs:
    text = pathlib.Path(log).read_text(encoding="utf-8")
    for match in pattern.finditer(text):
        values[(int(match.group(1)), match.group(2))] = float(match.group(3))

expected = {
    (2, "external"),
    (2, "internal"),
    (8, "external"),
    (8, "internal"),
}
missing = expected - values.keys()
sentinel = [v for v in values.values() if v <= 0.0 or v >= 999999.0]
ark_ms = 999999.0 if status or missing or sentinel else max(values.values())
ratio = ark_ms / target_ms
print(
    f"PERF_GATE name=allreduce ark_ms={ark_ms:.4f} "
    f"sglang_ms={target_ms:.4f} ratio={ratio:.4f}"
)
if status or missing or sentinel or ark_ms > target_ms:
    raise SystemExit(1)
PY
