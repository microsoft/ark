#!/usr/bin/env bash
# Launch SGLang server with Qwen3-8B on A100 GPUs.
#
# Usage:
#   ./launch_sglang.sh          # TP=8 (default)
#   ./launch_sglang.sh 1        # TP=1
#   ./launch_sglang.sh 8        # TP=8
#
# Prerequisites:
#   - Docker with NVIDIA runtime
#   - 8×A100-80GB (for TP=8) or 1×A100-80GB (for TP=1)
#   - HuggingFace cache at $HF_HOME (default: ~/.cache/huggingface)

set -euo pipefail

# --- Configuration -----------------------------------------------------------

TP="${1:-8}"
MODEL="Qwen/Qwen3-8B"
PORT=30000
CONTAINER_NAME="sglang-qwen3-bench"

# Pinned SGLang image — CUDA 12.x, A100-compatible.
# Update DIGEST after first pull: docker inspect --format='{{index .RepoDigests 0}}' <image>
IMAGE_TAG="lmsysorg/sglang:v0.4.6.post1-cu124"
IMAGE_DIGEST="TBD (run: docker inspect --format='{{index .RepoDigests 0}}' ${IMAGE_TAG})"

HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

# --- Validate -----------------------------------------------------------------

if [[ "${TP}" != "1" && "${TP}" != "8" ]]; then
    echo "Error: TP must be 1 or 8, got '${TP}'" >&2
    exit 1
fi

# --- Pull image ---------------------------------------------------------------

echo "Pulling ${IMAGE_TAG} ..."
docker pull "${IMAGE_TAG}"

# --- Stop any existing container ----------------------------------------------

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Stopping existing container '${CONTAINER_NAME}' ..."
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1
fi

# --- Launch server ------------------------------------------------------------

echo "Launching SGLang server: model=${MODEL}, tp=${TP}, port=${PORT}"
docker run -d \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    --ipc=host \
    --network=host \
    -v "${HF_HOME}:/root/.cache/huggingface" \
    -e HF_HOME=/root/.cache/huggingface \
    "${IMAGE_TAG}" \
    python -m sglang.launch_server \
        --model "${MODEL}" \
        --tp "${TP}" \
        --port "${PORT}" \
        --mem-fraction-static 0.85 \
        --trust-remote-code

echo ""
echo "Container '${CONTAINER_NAME}' started."
echo "Waiting for server to be ready on port ${PORT} ..."

# --- Wait for server readiness ------------------------------------------------

MAX_WAIT=300
ELAPSED=0
while ! curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; do
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    if [ "${ELAPSED}" -ge "${MAX_WAIT}" ]; then
        echo "Error: server did not become ready within ${MAX_WAIT}s." >&2
        echo "Check logs: docker logs ${CONTAINER_NAME}" >&2
        exit 1
    fi
done

echo "Server is ready (waited ${ELAPSED}s)."
echo ""
echo "Run the benchmark:"
echo "  python measure_baseline.py --port ${PORT}"
