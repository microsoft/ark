#!/usr/bin/env bash
set -uo pipefail

: "${ARK_ROOT:=$PWD}"
export ARK_ROOT
export PYTHONPATH="${PYTHONPATH:-$PWD/python}"

for py in "$PWD/python" "$PWD/build/python" "$PWD"/build/*/python "$PWD"/../build/*/python; do
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
    echo 'PERF_GATE name=tp ark_ms=999999.0000 sglang_ms=0.3268 ratio=3059972.4602 route=unknown head_sha=unknown'
    exit 1
fi

if [ -z "${ARK_HEAD_SHA:-}" ]; then
    repo_root=$(cd "$examples_dir/../.." 2>/dev/null && pwd || true)
    if [ -n "$repo_root" ]; then
        head_sha=$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || true)
        if [[ "$head_sha" =~ ^[0-9a-fA-F]{7,40}$ ]]; then
            export ARK_HEAD_SHA="$head_sha"
        fi
    fi
fi

tp_target_ms=$(python3 - "$examples_dir/bench_tp.py" <<'PY'
import importlib.util
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("bench_tp", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(f"{module._TP_TARGET_MS:.4f}")
PY
)

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

tp_status=0
python3 "$examples_dir/bench_tp.py" --world-size 8 --timeout 600 >"$tmpdir/tp.log" 2>"$tmpdir/tp.err" || tp_status=1
read -r tp_ark_ms tp_ratio tp_route tp_head_sha tp_parse_failed < <(python3 - "$tmpdir/tp.log" "$tp_target_ms" <<'PY'
import re
import sys

text = open(sys.argv[1], encoding="utf-8").read()
target = float(sys.argv[2])
match = re.search(
    r"PERF_GATE name=tp\s+ark_ms=([0-9.]+)"
    r"\s+sglang_ms=([0-9.]+)\s+ratio=([0-9.]+)"
    r"\s+route=([^\s]+)\s+head_sha=([^\s]+)",
    text,
)
if match is None:
    ark_ms = 999999.0
    ratio = ark_ms / target
    route = "unknown"
    head_sha = "unknown"
    parse_failed = 1
else:
    ark_ms = float(match.group(1))
    ratio = float(match.group(3))
    route = match.group(4)
    head_sha = match.group(5)
    parse_failed = 0
print(f"{ark_ms:.4f} {ratio:.4f} {route} {head_sha} {parse_failed}")
PY
)
printf 'PERF_GATE name=tp ark_ms=%s sglang_ms=%s ratio=%s route=%s head_sha=%s\n' "$tp_ark_ms" "$tp_target_ms" "$tp_ratio" "$tp_route" "$tp_head_sha"
python3 - "$tp_ark_ms" "$tp_ratio" "$tp_route" "$tp_head_sha" "$tp_status" "$tp_parse_failed" <<'PY'
import sys

ark_ms = float(sys.argv[1])
ratio = float(sys.argv[2])
route = sys.argv[3]
head_sha = sys.argv[4]
status = int(sys.argv[5])
parse_failed = int(sys.argv[6])
if (
    status
    or parse_failed
    or ark_ms >= 999999.0
    or ratio >= 1.0
    or route != "all_reduce_packet"
    or head_sha == "unknown"
):
    raise SystemExit(1)
PY
