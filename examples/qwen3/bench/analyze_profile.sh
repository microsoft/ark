#!/usr/bin/env bash
# Analyze a torch.profiler Chrome-trace JSON for Qwen3-8B per-component budget.
#
# Usage:
#   bash analyze_profile.sh <trace.json> [--tp 8]
#
# Steps:
#   1. Run trace_analyzer.py (torch-profiler skill) for kernel/comm/gap sections.
#   2. Run classify_kernels.py to produce the per-component latency budget.
#
# Prerequisites:
#   - Python 3.10+
#   - trace_analyzer.py path set via TRACE_ANALYZER env var, or auto-detected
#     from the torch-profiler skill directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Arguments ----------------------------------------------------------------

if [ $# -lt 1 ]; then
    echo "Usage: $0 <trace.json> [--tp 8]" >&2
    exit 1
fi

TRACE_FILE="$1"
shift

TP=8
while [ $# -gt 0 ]; do
    case "$1" in
        --tp) if [ $# -lt 2 ]; then echo "Error: --tp requires a value" >&2; exit 1; fi; TP="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [ ! -f "${TRACE_FILE}" ]; then
    echo "Error: trace file not found: ${TRACE_FILE}" >&2
    exit 1
fi

# --- Locate trace_analyzer.py -------------------------------------------------

TRACE_ANALYZER="${TRACE_ANALYZER:-}"
if [ -z "${TRACE_ANALYZER}" ]; then
    # Try common locations
    for candidate in \
        "${SCRIPT_DIR}/../../../.pi/skills/torch-profiler/scripts/trace_analyzer.py" \
        "${HOME}/.pi/skills/torch-profiler/scripts/trace_analyzer.py"; do
        if [ -f "${candidate}" ]; then
            TRACE_ANALYZER="${candidate}"
            break
        fi
    done
fi

# --- Step 1: trace_analyzer.py (if available) ---------------------------------

OUTPUT_DIR="$(dirname "${TRACE_FILE}")/analysis"
mkdir -p "${OUTPUT_DIR}"

if [ -n "${TRACE_ANALYZER}" ] && [ -f "${TRACE_ANALYZER}" ]; then
    echo "=== Trace Analyzer: kernel breakdown ==="
    python3 "${TRACE_ANALYZER}" "${TRACE_FILE}" --section kernels --top-n 30 \
        | tee "${OUTPUT_DIR}/kernels.txt"
    echo ""

    echo "=== Trace Analyzer: communication ==="
    python3 "${TRACE_ANALYZER}" "${TRACE_FILE}" --section comm \
        | tee "${OUTPUT_DIR}/comm.txt"
    echo ""

    echo "=== Trace Analyzer: GPU gaps ==="
    python3 "${TRACE_ANALYZER}" "${TRACE_FILE}" --section gaps \
        | tee "${OUTPUT_DIR}/gaps.txt"
    echo ""

    echo "=== Trace Analyzer: full summary ==="
    python3 "${TRACE_ANALYZER}" "${TRACE_FILE}" --full \
        -o "${OUTPUT_DIR}/full_analysis.md"
    echo "Full analysis saved to ${OUTPUT_DIR}/full_analysis.md"
    echo ""
else
    echo "Warning: trace_analyzer.py not found. Set TRACE_ANALYZER env var." >&2
    echo "Skipping trace_analyzer.py step." >&2
    echo ""
fi

# --- Step 2: per-component classification -------------------------------------

echo "=== Per-component classification (TP=${TP}) ==="
python3 "${SCRIPT_DIR}/classify_kernels.py" "${TRACE_FILE}" --tp "${TP}" \
    | tee "${OUTPUT_DIR}/component_budget.md"

echo ""
echo "Analysis complete. Results in: ${OUTPUT_DIR}/"
