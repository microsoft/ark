#!/usr/bin/env bash
set -uo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bench_py="$script_dir/examples/qwen3/bench_allreduce.py"
if [[ ! -f "$bench_py" ]]; then
  echo "ERROR: missing benchmark: $bench_py" >&2
  exit 1
fi

: "${ARK_ROOT:=$PWD}"
export ARK_ROOT
export PYTHONPATH="${PYTHONPATH:-$ARK_ROOT/python}"

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
status=0
python3 "$bench_py" --world-size 2 --shape decode --input-mode all \
  >"$tmpdir/tp2.log" 2>"$tmpdir/tp2.err" || status=1
python3 "$bench_py" --world-size 8 --shape decode --input-mode all \
  >"$tmpdir/tp8.log" 2>"$tmpdir/tp8.err" || status=1

python3 - "$status" "$bench_py" "$tmpdir/tp2.log" "$tmpdir/tp8.log" <<'PY'
import importlib.util
import pathlib
import sys

status = int(sys.argv[1])
path = pathlib.Path(sys.argv[2])
logs = sys.argv[3:]
spec = importlib.util.spec_from_file_location("bench_allreduce", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
target_ms = module._DECODE_TARGET_MS
texts = [pathlib.Path(log).read_text(encoding="utf-8") for log in logs]
ark_ms = (
    999999.0 if status else module._decode_gate_ark_ms_from_logs(texts)
)
ratio = ark_ms / target_ms
print(
    f"PERF_GATE name={module._DECODE_GATE_NAME} ark_ms={ark_ms:.4f} "
    f"sglang_ms={target_ms:.4f} ratio={ratio:.4f}"
)
if status or ark_ms >= 999999.0 or ark_ms >= target_ms:
    raise SystemExit(1)
PY
