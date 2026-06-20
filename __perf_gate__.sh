#!/usr/bin/env bash
set -euo pipefail

: "${ARK_ROOT:=$PWD}"
export ARK_ROOT
export PYTHONPATH="$ARK_ROOT/python:${PYTHONPATH:-}"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=""
for candidate in "$script_dir" "$script_dir/.." "$PWD" "$PWD/.."; do
    if [ -f "$candidate/examples/qwen3/bench_tp.py" ]; then
        repo_root=$(cd "$candidate" && pwd)
        break
    fi
done

if [ -z "$repo_root" ]; then
    echo 'PERF_GATE name=tp ark_ms=999999.0000 sglang_ms=0.3268 ratio=3060223.3127 route=unknown head_sha=unknown base_sha=unknown'
    exit 1
fi

if [ -z "${ARK_HEAD_SHA:-}" ]; then
    head_sha=$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || true)
    if [[ "$head_sha" =~ ^[0-9a-fA-F]{7,40}$ ]]; then
        export ARK_HEAD_SHA="$head_sha"
    fi
fi
if [ -z "${ARK_BASE_SHA:-}" ]; then
    base_sha=$(git -C "$repo_root" rev-parse origin/qwen3-allreduce-bench 2>/dev/null || true)
    if [[ "$base_sha" =~ ^[0-9a-fA-F]{7,40}$ ]]; then
        export ARK_BASE_SHA="$base_sha"
    fi
fi

# SGLang target is 214.69 ms / 657 calls = 0.3268 ms in PROFILE.md.
out_file=$(mktemp)
trap 'rm -f "$out_file"' EXIT
set +e
python3 "$repo_root/examples/qwen3/bench_tp.py" --world-size 8 --timeout 600 >"$out_file" 2>&1
status=$?
set -e
perf_gate_count=$(grep -c '^PERF_GATE ' "$out_file" || true)
if [ "$perf_gate_count" -eq 1 ]; then
    cat "$out_file"
    exit "$status"
fi

grep -v '^PERF_GATE ' "$out_file" || true
echo 'PERF_GATE name=tp ark_ms=999999.0000 sglang_ms=0.3268 ratio=3060223.3127 route=unknown head_sha=unknown base_sha=unknown'
exit 1
