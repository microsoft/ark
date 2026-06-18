#!/usr/bin/env bash
set -euo pipefail

: "${ARK_ROOT:=$PWD}"
export ARK_ROOT
export PYTHONPATH="${PYTHONPATH:-$ARK_ROOT/python}"

bench=""
for p in \
  "$PWD/../examples/qwen3/bench_allreduce.py" \
  "$PWD/examples/qwen3/bench_allreduce.py" \
  "$ARK_ROOT/../examples/qwen3/bench_allreduce.py" \
  "$ARK_ROOT/examples/qwen3/bench_allreduce.py"; do
  if [[ -f "$p" ]]; then
    bench=$(realpath "$p")
    break
  fi
done

repo_root=""
if [[ -n "$bench" ]]; then
  repo_root=$(realpath "$(dirname "$bench")/../..")
fi

has_compiled_ark() {
  compgen -G "$1/ark/core*.so" >/dev/null || \
    compgen -G "$1/ark/core*.pyd" >/dev/null
}

py_paths=()
add_py_path() {
  if [[ -n "$1" && -d "$1" ]]; then
    py_paths+=("$1")
  fi
}

build_root=""
for root in "$ARK_ROOT" "$PWD" "$PWD/build" "$repo_root/build"; do
  if [[ -n "$root" ]] && has_compiled_ark "$root/python"; then
    add_py_path "$root/python"
    if [[ -z "$build_root" ]]; then
      build_root=$(realpath "$root")
    fi
  fi
done
if [[ -n "$repo_root" ]]; then
  add_py_path "$repo_root"
fi
if [[ ${#py_paths[@]} -gt 0 ]]; then
  joined=$(IFS=:; echo "${py_paths[*]}")
  export PYTHONPATH="$joined${PYTHONPATH:+:$PYTHONPATH}"
fi
if [[ -n "$build_root" ]]; then
  export ARK_ROOT="$build_root"
fi

# PROFILE.md target cited by examples/qwen3/bench_allreduce.py:
# 214.69 ms over 657 decode-dominated Qwen3 comm calls.
target_ms=$(python3 - "$bench" <<'PY'
import ast
import pathlib
import sys

path = pathlib.Path(sys.argv[1]) if sys.argv[1] else None
if path and path.is_file():
    module = ast.parse(path.read_text())
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_DECODE_TARGET_MS":
                    value = eval(
                        compile(ast.Expression(node.value), str(path), "eval"),
                        {"__builtins__": {}},
                    )
                    print(f"{value:.10f}")
                    raise SystemExit(0)
print(f"{214.69 / 657.0:.10f}")
PY
)

log_dir=${Q7P2_PERF_LOG_DIR:-}
if [[ -z "$log_dir" ]]; then
  if [[ -n "$repo_root" ]]; then
    log_dir="$repo_root"
  else
    log_dir="$PWD"
  fi
fi
mkdir -p "$log_dir"
tp2_log="$log_dir/q7p2_allreduce_tp2.log"
tp8_log="$log_dir/q7p2_allreduce_tp8.log"

commit_sha="unknown"
if [[ -n "$repo_root" ]] && \
  git -C "$repo_root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  commit_sha=$(git -C "$repo_root" rev-parse HEAD)
fi

run_decode() {
  local world_size=$1
  local log_path=$2
  {
    printf 'Q7.2 SHA: %s\n' "$commit_sha"
    printf 'Command: python3 %s --world-size %s --shape decode\n' \
      "$bench" "$world_size"
    printf 'Timing: single iteration, max rank, decode no-copy gate\n'
  } >"$log_path"
  if [[ -z "$bench" ]]; then
    printf 'ERROR: examples/qwen3/bench_allreduce.py not found\n' >>"$log_path"
    return 1
  fi
  python3 "$bench" --world-size "$world_size" --shape decode \
    >>"$log_path" 2>&1
}

status=0
run_decode 2 "$tp2_log" || status=1
run_decode 8 "$tp8_log" || status=1

parse_out=$(python3 - "$tp2_log" "$tp8_log" <<'PY'
import pathlib
import re
import sys

values = []
missing = 0
sentinel = 0
for arg in sys.argv[1:]:
    text = pathlib.Path(arg).read_text(errors="replace")
    match = re.search(
        r"PERF_GATE name=allreduce\s+ark_ms=([0-9.]+)\s+sglang_ms=([0-9.]+)\s+ratio=([0-9.]+)",
        text,
    )
    if not match:
        values.append(999999.0)
        missing += 1
        continue
    ark_ms = float(match.group(1))
    values.append(ark_ms)
    if ark_ms >= 999999.0:
        sentinel += 1
print(f"{max(values):.4f} {missing} {sentinel}")
PY
)
read -r ark_ms missing sentinel <<<"$parse_out"
if [[ "$missing" != "0" || "$sentinel" != "0" ]]; then
  status=1
fi

ratio=$(python3 - "$ark_ms" "$target_ms" <<'PY'
import sys
ark_ms = float(sys.argv[1])
target_ms = float(sys.argv[2])
print(f"{ark_ms / target_ms:.4f}")
PY
)
printf 'PERF_GATE name=allreduce ark_ms=%s sglang_ms=%.4f ratio=%s\n' \
  "$ark_ms" "$target_ms" "$ratio"

# Q7 copy-staged decode baseline from qwen3-allreduce-bench logs:
# TP=2 0.0588 ms, TP=8 0.0637 ms. Allow 20% noise on the max-rank value.
python3 - "$ark_ms" "$target_ms" "$status" <<'PY'
import sys
ark_ms = float(sys.argv[1])
target_ms = float(sys.argv[2])
status = int(sys.argv[3])
q7_copy_max_ms = 0.0637
if status or ark_ms >= target_ms or ark_ms > q7_copy_max_ms * 1.20:
    raise SystemExit(1)
PY
