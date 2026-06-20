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

# bench_tp.py owns benchmark policy and emits the canonical line; this
# wrapper only accepts one well-formed passing packet-route line as a
# fail-closed CI guard, otherwise it emits the sentinel.
_valid_perf_gate_line() {
    local line=$1
    local numeric='([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?'
    [[ "$line" =~ (^|[[:space:]])name=tp($|[[:space:]]) ]] || return 1
    [[ "$line" =~ (^|[[:space:]])route=all_reduce_packet($|[[:space:]]) ]] || return 1
    [[ "$line" =~ (^|[[:space:]])head_sha=[0-9a-fA-F]{7,40}($|[[:space:]]) ]] || return 1
    [[ "$line" =~ (^|[[:space:]])base_sha=[0-9a-fA-F]{7,40}($|[[:space:]]) ]] || return 1
    [[ "$line" =~ (^|[[:space:]])ark_ms=($numeric)($|[[:space:]]) ]] || return 1
    PERF_GATE_ARK_MS=${BASH_REMATCH[2]}
    [[ "$line" =~ (^|[[:space:]])sglang_ms=0\.3268($|[[:space:]]) ]] || return 1
    [[ "$line" =~ (^|[[:space:]])ratio=($numeric)($|[[:space:]]) ]] || return 1
    PERF_GATE_RATIO=${BASH_REMATCH[2]}
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
        awk -v ratio="$PERF_GATE_RATIO" -v ark_ms="$PERF_GATE_ARK_MS" \
            'BEGIN { exit !(ratio < 1.0 && ark_ms < 0.3268) }' || {
            if awk -v ratio="$PERF_GATE_RATIO" \
                'BEGIN { exit !(ratio >= 1.0) }'; then
                echo "$perf_gate_line"
            else
                _emit_sentinel
            fi
            exit 1
        }
        echo "$perf_gate_line"
        exit "$status"
    fi
fi

_emit_sentinel
exit 1
