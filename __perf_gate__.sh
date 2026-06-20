#!/usr/bin/env bash
set -euo pipefail

_emit_sentinel() {
    echo 'PERF_GATE name=tp ark_ms=999999.0000 sglang_ms=0.3268 ratio=3060223.3127 route=unknown head_sha=unknown base_sha=unknown'
}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=""
for candidate in "$script_dir" "$script_dir/ark" "$script_dir/.." "$PWD" "$PWD/.." "$PWD/../ark"; do
    if [ -f "$candidate/examples/qwen3/bench_tp.py" ]; then
        repo_root=$(cd "$candidate" && pwd)
        break
    fi
done

if [ -z "$repo_root" ]; then
    _emit_sentinel
    exit 1
fi

if [ -z "${ARK_ROOT:-}" ]; then
    if [ -f "$PWD/python/ark/__init__.py" ]; then
        ARK_ROOT=$PWD
    else
        ARK_ROOT=$repo_root
    fi
fi
export ARK_ROOT
if [ -n "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="$ARK_ROOT/python:$PYTHONPATH"
else
    export PYTHONPATH="$ARK_ROOT/python"
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

_valid_perf_gate_line() {
    local line=$1
    local numeric='[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?'
    [[ "$line" =~ (^|[[:space:]])name=tp($|[[:space:]]) ]] || return 1
    [[ "$line" =~ (^|[[:space:]])route=all_reduce_packet($|[[:space:]]) ]] || return 1
    [[ "$line" =~ (^|[[:space:]])head_sha=[0-9a-fA-F]{7,40}($|[[:space:]]) ]] || return 1
    [[ "$line" =~ (^|[[:space:]])base_sha=[0-9a-fA-F]{7,40}($|[[:space:]]) ]] || return 1
    [[ "$line" =~ (^|[[:space:]])ark_ms=$numeric($|[[:space:]]) ]] || return 1
    [[ "$line" =~ (^|[[:space:]])sglang_ms=0\.3268($|[[:space:]]) ]] || return 1
    [[ "$line" =~ (^|[[:space:]])ratio=($numeric)($|[[:space:]]) ]] || return 1
    local ratio=${BASH_REMATCH[2]}
    awk -v ratio="$ratio" 'BEGIN { exit !(ratio < 1.0) }' || return 1
}

out_file=$(mktemp)
trap 'rm -f "$out_file"' EXIT
set +e
python3 "$repo_root/examples/qwen3/bench_tp.py" --world-size 8 --timeout 600 >"$out_file" 2>&1
status=$?
set -e

perf_gate_count=$(grep -c '^PERF_GATE ' "$out_file" || true)
if [ "$perf_gate_count" -eq 1 ]; then
    perf_gate_line=$(grep '^PERF_GATE ' "$out_file")
    if _valid_perf_gate_line "$perf_gate_line"; then
        echo "$perf_gate_line"
        exit "$status"
    fi
fi

_emit_sentinel
exit 1
