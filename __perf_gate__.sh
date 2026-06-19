#!/usr/bin/env bash
set -euo pipefail

: "${ARK_ROOT:=$PWD}"
export ARK_ROOT
export PYTHONPATH="${PYTHONPATH:-$ARK_ROOT/python}"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ -f "$script_dir/examples/qwen3/bench_allreduce.py" ]]; then
    repo_root=$script_dir
elif [[ -f "$script_dir/../examples/qwen3/bench_allreduce.py" ]]; then
    repo_root=$(cd -- "$script_dir/.." && pwd)
elif [[ -f "$PWD/../examples/qwen3/bench_allreduce.py" ]]; then
    repo_root=$(cd -- "$PWD/.." && pwd)
else
    echo "ERROR: cannot locate examples/qwen3/bench_allreduce.py" >&2
    exit 1
fi
bench="$repo_root/examples/qwen3/bench_allreduce.py"

# bench_allreduce._DECODE_TARGET_MS cites PROFILE.md: 214.69 ms / 657 calls.
target_ms=$(python3 - "$bench" <<'PY'
import importlib.util
import sys

path = sys.argv[1]
spec = importlib.util.spec_from_file_location("bench_allreduce", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(f"{module._DECODE_TARGET_MS:.4f}")
PY
)

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
status=0
python3 "$bench" --world-size 2 --shape decode >"$tmpdir/tp2.log" 2>"$tmpdir/tp2.err" || status=1
python3 "$bench" --world-size 8 --shape decode >"$tmpdir/tp8.log" 2>"$tmpdir/tp8.err" || status=1

ark_ms=$(python3 - "$tmpdir/tp2.log" "$tmpdir/tp8.log" "$status" <<'PY'
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
