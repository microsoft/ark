#!/usr/bin/env bash
set -euo pipefail

: "${ARK_ROOT:=$PWD}"
if ! compgen -G "$ARK_ROOT/python/ark/core*.so" >/dev/null; then
    if compgen -G "$PWD/build/python/ark/core*.so" >/dev/null; then
        ARK_ROOT="$PWD/build"
    fi
fi
export ARK_ROOT
export PYTHONPATH="$ARK_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"

bench=../examples/qwen3/bench_allreduce.py
if [[ ! -f "$bench" ]]; then
    bench=examples/qwen3/bench_allreduce.py
fi
if [[ ! -f "$bench" ]]; then
    printf 'PERF_GATE name=allreduce ark_ms=999999.0000 sglang_ms=0.3268 ratio=3059972.4602\n'
    exit 1
fi

target_ms=$(python3 - "$bench" <<'PY'
import importlib.util
import pathlib
import sys

# bench_allreduce.py records PROFILE.md Q7 comm target: 214.69 ms / 657 calls.
path = pathlib.Path(sys.argv[1])
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
