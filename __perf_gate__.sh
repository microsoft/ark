#!/usr/bin/env bash
set -uo pipefail

: "${ARK_ROOT:=$PWD}"
export ARK_ROOT
export PYTHONPATH="${PYTHONPATH:-$PWD/python}"

for py in \
    "$PWD/python" \
    "$PWD/build/python" \
    "$PWD/build-release/python" \
    "$PWD"/build/*/python \
    "$PWD"/build-release/*/python \
    "$PWD"/../build/*/python \
    "$PWD"/../build-release/*/python \
    /*/build/python \
    /*/build-release/python \
    /*/build/*/python \
    /*/build-release/*/python; do
    if ls "$py"/ark/core*.so >/dev/null 2>&1; then
        export PYTHONPATH="$py:$PYTHONPATH"
        export ARK_ROOT="$(dirname "$py")"
        break
    fi
done

if [ -f ../examples/qwen3/bench_tp.py ]; then
    examples_dir=../examples/qwen3
elif [ -f examples/qwen3/bench_tp.py ]; then
    examples_dir=examples/qwen3
else
    echo 'PERF_GATE name=tp ark_ms=999999.0000 sglang_ms=0.3268 ratio=3059972.4602 route=unknown head_sha=unknown base_sha=unknown'
    exit 1
fi

repo_root=$(cd "$examples_dir/../.." 2>/dev/null && pwd || true)
if [ -z "${ARK_HEAD_SHA:-}" ] && [ -n "$repo_root" ]; then
    head_sha=$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || true)
    if [[ "$head_sha" =~ ^[0-9a-fA-F]{7,40}$ ]]; then
        export ARK_HEAD_SHA="$head_sha"
    fi
fi
if [ -z "${ARK_BASE_SHA:-}" ] && [ -n "$repo_root" ]; then
    base_sha=$(git -C "$repo_root" rev-parse origin/qwen3-allreduce-bench 2>/dev/null || true)
    if [[ "$base_sha" =~ ^[0-9a-fA-F]{7,40}$ ]]; then
        export ARK_BASE_SHA="$base_sha"
    fi
fi

# SGLang target is 214.69 ms / 657 calls = 0.3268 ms in PROFILE.md.
python3 "$examples_dir/bench_tp.py" --world-size 8 --timeout 600
