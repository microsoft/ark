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
if [[ -n "$build_root" ]]; then
  if [[ ${#py_paths[@]} -gt 0 ]]; then
    joined=$(IFS=:; echo "${py_paths[*]}")
    export PYTHONPATH="$joined${PYTHONPATH:+:$PYTHONPATH}"
  fi
  export ARK_ROOT="$build_root"
else
  # Source-only python/ark shadows the wheel installed by the perf harness.
  # Drop that path when no build-tree extension exists, but keep the repo root
  # so examples.qwen3 imports still work.
  filtered_py_paths=()
  source_py=""
  if [[ -n "$repo_root" && -d "$repo_root/python" ]]; then
    source_py=$(realpath "$repo_root/python")
  fi
  if [[ -n "$repo_root" ]]; then
    filtered_py_paths+=("$repo_root")
  fi
  if [[ -n "${PYTHONPATH:-}" ]]; then
    IFS=: read -r -a existing_py_paths <<<"$PYTHONPATH"
    for py_path in "${existing_py_paths[@]}"; do
      if [[ -z "$py_path" ]]; then
        continue
      fi
      if [[ -n "$source_py" && -e "$py_path" &&
            $(realpath "$py_path") == "$source_py" ]]; then
        continue
      fi
      filtered_py_paths+=("$py_path")
    done
  fi
  if [[ ${#filtered_py_paths[@]} -gt 0 ]]; then
    joined=$(IFS=:; echo "${filtered_py_paths[*]}")
    export PYTHONPATH="$joined"
  else
    unset PYTHONPATH
  fi
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

commit_sha=${Q7P2_COMMIT_SHA:-}
if [[ -z "$commit_sha" && -n "$repo_root" ]] && \
  git -C "$repo_root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  commit_sha=$(git -C "$repo_root" rev-parse HEAD)
fi
if [[ -z "$commit_sha" ]]; then
  commit_sha="unknown"
fi

# Tuned Q7.2 decode no-copy schedule from A100 TP sweep:
# 2 blocks per peer, 1 warp per block. Each trial is still one ARK runtime
# iteration and reports max-rank latency; the gate uses the median across
# independent trial processes to reject transient node jitter, not failed ranks.
trials=${Q7P2_PERF_TRIALS:-3}
blocks_per_peer=${Q7P2_BLOCKS_PER_PEER:-2}
num_warps=${Q7P2_NUM_WARPS:-1}

decode_config() {
  local world_size=$1
  python3 - "$world_size" "$blocks_per_peer" "$num_warps" <<'PY'
import json
import sys

world_size = int(sys.argv[1])
blocks_per_peer = int(sys.argv[2])
num_warps = int(sys.argv[3])
num_tasks = blocks_per_peer * (world_size - 1)
print(json.dumps({
    "PacketType": "mscclpp::LL16Packet",
    "SramBytes": 0,
    "Tile": [1, 1],
    "NumTasks": num_tasks,
    "NumProcs": num_tasks,
    "NumWarps": num_warps,
}, separators=(",", ":")))
PY
}

summarize_log() {
  local log_path=$1
  local run_status=$2
  python3 - "$log_path" "$target_ms" "$trials" "$run_status" <<'PY'
import pathlib
import re
import statistics
import sys

path = pathlib.Path(sys.argv[1])
target_ms = float(sys.argv[2])
trials = int(sys.argv[3])
run_status = int(sys.argv[4])
text = path.read_text(errors="replace")
values = [
    float(m.group(1))
    for m in re.finditer(
        r"PERF_GATE name=allreduce\s+"
        r"ark_ms=([0-9.]+)\s+sglang_ms=([0-9.]+)\s+"
        r"ratio=([0-9.]+)",
        text,
    )
]
if run_status or len(values) < trials or any(v >= 999999.0 for v in values):
    ark_ms = 999999.0
    status = 1
else:
    ark_ms = statistics.median(values[-trials:])
    status = 0
ratio = ark_ms / target_ms
print(
    "Summary: median of "
    f"{trials} single iteration, max rank decode no-copy trials"
)
print(
    f"PERF_GATE name=allreduce ark_ms={ark_ms:.4f} "
    f"sglang_ms={target_ms:.4f} ratio={ratio:.4f}"
)
raise SystemExit(status)
PY
}

run_decode() {
  local world_size=$1
  local log_path=$2
  local cfg
  cfg=$(decode_config "$world_size")
  {
    printf 'Q7.2 SHA: %s\n' "$commit_sha"
    printf 'Command: python3 %s --world-size %s --shape decode\n' \
      "$bench" "$world_size"
    printf 'Timing: single iteration, max rank, median of %s trials\n' \
      "$trials"
    printf 'Planner config: %s\n' "$cfg"
  } >"$log_path"
  if [[ -z "$bench" ]]; then
    printf 'ERROR: examples/qwen3/bench_allreduce.py not found\n' \
      >>"$log_path"
    summarize_log "$log_path" 1 >>"$log_path" || true
    return 1
  fi

  local run_status=0
  for trial in $(seq 1 "$trials"); do
    printf '\nTrial %s/%s\n' "$trial" "$trials" >>"$log_path"
    ARK_QWEN3_ALLREDUCE_CONFIG="$cfg" \
      python3 "$bench" --world-size "$world_size" --shape decode \
      >>"$log_path" 2>&1 || run_status=1
  done

  local summary_status=0
  summarize_log "$log_path" "$run_status" >>"$log_path" || summary_status=$?
  if [[ "$summary_status" != "0" ]]; then
    return 1
  fi
  return "$run_status"
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
    matches = re.findall(
        r"PERF_GATE name=allreduce\s+"
        r"ark_ms=([0-9.]+)\s+sglang_ms=([0-9.]+)\s+"
        r"ratio=([0-9.]+)",
        text,
    )
    if not matches:
        values.append(999999.0)
        missing += 1
        continue
    ark_ms = float(matches[-1][0])
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
