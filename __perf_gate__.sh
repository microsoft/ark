#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_root=""
for candidate in "$script_dir" "$PWD" .. ../ark; do
  if [[ -f "$candidate/examples/qwen3/bench_allreduce.py" ]]; then
    source_root=$(cd "$candidate" && pwd)
    break
  fi
done

emit_sentinel() {
  local target_ms="0.1880"
  if [[ -n "${source_root:-}" ]]; then
    target_ms=$(python3 - "$source_root/examples/qwen3/bench_allreduce.py" <<'PY'
import importlib.util
import sys

path = sys.argv[1]
spec = importlib.util.spec_from_file_location("bench_allreduce", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(f"{module._PREFILL_TARGET_MS:.4f}")
PY
)
  fi
  local ratio
  ratio=$(python3 - "$target_ms" <<'PY'
import sys

print(f"{999999.0 / float(sys.argv[1]):.4f}")
PY
)
  printf 'PERF_GATE name=allreduce_prefill ark_ms=999999.0000 sglang_ms=%s ratio=%s\n' "$target_ms" "$ratio"
}

if [[ -z "$source_root" ]]; then
  emit_sentinel
  exit 1
fi

has_compiled_ark() {
  compgen -G "$1/python/ark/core*.so" >/dev/null || \
    compgen -G "$1/python/ark/core*.pyd" >/dev/null
}

if [[ -z "${ARK_ROOT:-}" ]]; then
  if has_compiled_ark "$PWD"; then
    ARK_ROOT=$(cd "$PWD" && pwd)
  elif has_compiled_ark "$source_root/build"; then
    ARK_ROOT=$(cd "$source_root/build" && pwd)
  else
    ARK_ROOT=$(cd "$PWD" && pwd)
  fi
fi
export ARK_ROOT
export PYTHONPATH="$ARK_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"

bench_py="$source_root/examples/qwen3/bench_allreduce.py"

# Verify the benchmark code was loaded from this checkout, not another ARK tree.
head_sha=$(git -C "$source_root" rev-parse HEAD 2>/dev/null || echo unknown)
target_ms=$(python3 - "$bench_py" <<'PY'
import importlib.util
import sys

path = sys.argv[1]
spec = importlib.util.spec_from_file_location("bench_allreduce", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(f"{module._PREFILL_TARGET_MS:.4f}")
PY
)

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
status=0
python3 "$bench_py" --world-size 2 --shape all >"$tmpdir/tp2.log" 2>"$tmpdir/tp2.err" || status=1
python3 "$bench_py" --world-size 8 --shape all >"$tmpdir/tp8.log" 2>"$tmpdir/tp8.err" || status=1

if [[ "$status" -ne 0 ]]; then
  for tp in 2 8; do
    if [[ -s "$tmpdir/tp${tp}.err" ]]; then
      printf 'ERROR: TP=%s benchmark stderr tail:\n' "$tp" >&2
      tail -n 80 "$tmpdir/tp${tp}.err" >&2
    fi
  done
fi

read -r ark_ms parse_status < <(python3 - "$tmpdir/tp2.log" "$tmpdir/tp8.log" "$status" "$head_sha" "$target_ms" "$source_root" "$ARK_ROOT" <<'PY'
import os
import re
import sys

log_paths = [(2, sys.argv[1]), (8, sys.argv[2])]
status = int(sys.argv[3])
head_sha = sys.argv[4]
target_ms = float(sys.argv[5])
source_root = os.path.realpath(sys.argv[6])
ark_root = os.path.realpath(sys.argv[7])
expected_ark_roots = [
    os.path.join(source_root, "python"),
    os.path.join(source_root, "build", "python"),
    ark_root,
    os.path.join(ark_root, "python"),
]


def is_under(path, roots):
    for root in roots:
        try:
            if os.path.commonpath([path, root]) == root:
                return True
        except ValueError:
            continue
    return False

# Gate both the new prefill route and the decode-sized packet fallback within a
# bounded regression window, preserving the all-reduce dispatch contract.
decode_baselines_ms = {2: 0.0588, 8: 0.0637}
decode_limit_factor = 1.20
errors = []
prefill_values = []
all_ark_paths = set()

if head_sha == "unknown":
    errors.append("source checkout SHA is unknown")
if status:
    errors.append("one or more benchmark workers failed")

for world_size, path in log_paths:
    text = open(path, encoding="utf-8").read()
    if f"BENCH_SHA sha={head_sha}" not in text:
        errors.append(f"TP={world_size}: missing or stale BENCH_SHA")

    prefill_gate = re.findall(
        r"PERF_GATE name=allreduce_prefill\s+"
        r"ark_ms=([0-9.]+)\s+sglang_ms=([0-9.]+)\s+ratio=([0-9.]+)",
        text,
    )
    if len(prefill_gate) != 1:
        errors.append(f"TP={world_size}: expected one allreduce_prefill line")
    else:
        ark_ms = float(prefill_gate[0][0])
        sglang_ms = float(prefill_gate[0][1])
        if abs(sglang_ms - target_ms) > 1e-4:
            errors.append(f"TP={world_size}: wrong target {sglang_ms:.4f}")
        if ark_ms >= 999999.0:
            errors.append(f"TP={world_size}: sentinel prefill latency")
        if ark_ms > target_ms:
            errors.append(
                f"TP={world_size}: prefill ark_ms {ark_ms:.4f} > "
                f"{target_ms:.4f}"
            )
        prefill_values.append(ark_ms)

    bench_results = re.findall(
        rf"BENCH_RESULT shape=(decode|prefill) world_size={world_size}\s+"
        r"max_rank=([0-9]+)\s+route=([a-z_]+)\s+ark_path=(\S+)\s+"
        r"latency_us=([0-9.]+)",
        text,
    )
    by_shape = {
        shape: (int(rank), route, ark_path, float(us))
        for shape, rank, route, ark_path, us in bench_results
    }
    ark_paths = {
        os.path.realpath(ark_path)
        for _, _, ark_path, _ in by_shape.values()
        if ark_path
    }
    all_ark_paths.update(ark_paths)
    if len(ark_paths) != 1 or "" in ark_paths:
        errors.append(f"TP={world_size}: missing or inconsistent ark_path")

    if "prefill" not in by_shape:
        errors.append(f"TP={world_size}: missing prefill BENCH_RESULT")
    elif by_shape["prefill"][1] != "prefill":
        errors.append(
            f"TP={world_size}: prefill route {by_shape['prefill'][1]} "
            "is not prefill"
        )

    if "decode" not in by_shape:
        errors.append(f"TP={world_size}: missing decode BENCH_RESULT")
        continue
    _, decode_route, _, decode_us = by_shape["decode"]
    decode_ms = decode_us / 1000.0
    decode_limit = decode_baselines_ms[world_size] * decode_limit_factor
    if decode_route != "packet":
        errors.append(
            f"TP={world_size}: decode route {decode_route} is not packet"
        )
    if decode_ms >= 999999.0:
        errors.append(f"TP={world_size}: sentinel decode latency")
    if decode_ms > decode_limit:
        errors.append(
            f"TP={world_size}: decode ark_ms {decode_ms:.4f} > "
            f"{decode_limit:.4f}"
        )

all_ark_paths = {os.path.realpath(path) for path in all_ark_paths if path}
if len(all_ark_paths) != 1:
    errors.append("missing or inconsistent ark_path across TP runs")
else:
    ark_path = next(iter(all_ark_paths))
    if not is_under(ark_path, expected_ark_roots):
        errors.append(
            "ark_path is outside expected checkout/build roots: "
            f"{ark_path}"
        )

for error in errors:
    print(f"ERROR: {error}", file=sys.stderr)
if errors or len(prefill_values) != 2:
    print("999999.0000 1")
else:
    print(f"{max(prefill_values):.4f} 0")
PY
)
ratio=$(python3 - "$ark_ms" "$target_ms" <<'PY'
import sys

print(f"{float(sys.argv[1]) / float(sys.argv[2]):.4f}")
PY
)
printf 'PERF_GATE name=allreduce_prefill ark_ms=%s sglang_ms=%s ratio=%s\n' "$ark_ms" "$target_ms" "$ratio"
python3 - "$ark_ms" "$target_ms" "$status" "$parse_status" <<'PY'
import sys

ark_ms = float(sys.argv[1])
target_ms = float(sys.argv[2])
status = int(sys.argv[3])
parse_status = int(sys.argv[4])
if status or parse_status or ark_ms > target_ms:
    raise SystemExit(1)
PY
