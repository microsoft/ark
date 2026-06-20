#!/usr/bin/env bash
set -uo pipefail

: "${ARK_ROOT:=$PWD}"
export ARK_ROOT
export PYTHONPATH="${PYTHONPATH:-$PWD/python}"

if [ -f ../examples/qwen3/bench_tp.py ]; then
    examples_dir=../examples/qwen3
elif [ -f examples/qwen3/bench_tp.py ]; then
    examples_dir=examples/qwen3
else
    echo 'PERF_GATE name=tp ark_ms=999999.0000 sglang_ms=0.3268 ratio=3060223.3127 route=unknown head_sha=unknown base_sha=unknown'
    exit 1
fi

# SGLang target is 214.69 ms / 657 calls = 0.3268 ms in PROFILE.md.
out_file=$(mktemp)
trap 'rm -f "$out_file"' EXIT
python3 "$examples_dir/bench_tp.py" --world-size 8 --timeout 600 >"$out_file" 2>&1
status=$?
perf_gate_count=$(grep -c '^PERF_GATE ' "$out_file" || true)
if [ "$perf_gate_count" -eq 1 ]; then
    cat "$out_file"
    exit "$status"
fi

grep -v '^PERF_GATE ' "$out_file" || true
echo 'PERF_GATE name=tp ark_ms=999999.0000 sglang_ms=0.3268 ratio=3060223.3127 route=unknown head_sha=unknown base_sha=unknown'
exit 1
