# Reproducing the Qwen3-8B SGLang Baseline

Target node: `mscclpp-a100-dev` (8× A100-SXM4-80GB).

## Prerequisites

- Docker with NVIDIA runtime (`nvidia-docker2` or `--gpus` support).
- HuggingFace model cache with network access (model downloads on first run).
- Python 3.8+ with `requests` package on the host (for `measure_baseline.py`).

## Step 1: Clone and navigate

```bash
git clone <ark-repo-url>
cd ark/examples/qwen3/bench
```

## Step 2: Run TP=8 (matched-regime comparison)

TP=8 for an 8B model amplifies kernel-launch and communication overhead.
This is the matched-regime comparison for the ARK target, not SGLang's
best config.

```bash
# Launch server (TP=8, default)
./launch_sglang.sh 8

# Wait for "Server is ready" message, then measure:
python measure_baseline.py --port 30000 --trials 5 --output baseline_results_tp8.json

# Record results in BASELINE.md under "TP=8" section.
# Save the image digest:
docker inspect --format='{{index .RepoDigests 0}}' lmsysorg/sglang:v0.4.6.post1-cu124

# Stop server
docker rm -f sglang-qwen3-bench
```

## Step 3: Run TP=1 (SGLang natural best-latency config)

TP=1 is the lowest-overhead config for an 8B model and represents
SGLang's best achievable latency for this model size.

```bash
# Launch server (TP=1)
./launch_sglang.sh 1

# Wait for "Server is ready" message, then measure:
python measure_baseline.py --port 30000 --trials 5 --output baseline_results_tp1.json

# Record results in BASELINE.md under "TP=1" section.

# Stop server
docker rm -f sglang-qwen3-bench
```

## Step 4: Record GPU clocks

`measure_baseline.py` captures GPU clocks automatically via `nvidia-smi`.
Copy the clock values from the script output into BASELINE.md.

Alternatively, capture manually:

```bash
nvidia-smi --query-gpu=index,clocks.gr,clocks.mem,clocks.max.gr,clocks.max.mem \
    --format=csv,noheader,nounits
```

## Step 5: Fill in BASELINE.md

Update `BASELINE.md` with:
- Image digest (from `docker inspect`).
- GPU clocks (from script output or `nvidia-smi`).
- TTFT and decode per-token latency for both TP=1 and TP=8.

## Pinned image

| Field     | Value |
|-----------|-------|
| Image     | `lmsysorg/sglang:v0.4.6.post1-cu124` |
| CUDA      | 12.4 |
| Digest    | TBD (record after first pull) |

## Server flags

All runs use:

```
--model Qwen/Qwen3-8B
--tp <1 or 8>
--port 30000
--mem-fraction-static 0.85
--trust-remote-code
```

## Measurement script flags

```
python measure_baseline.py --port 30000 --trials 5 --warmup 3 --prompt-tokens 2048
```

Key behaviors:
- Uses `/generate` endpoint with `ignore_eos: true` (not `/v1/chat/completions`).
- Temperature 0.0 (greedy decoding).
- Reports median over 5 trials.
- Writes results JSON; output path is configurable via `--output` (default: `baseline_results.json`).

## Troubleshooting

**Server does not start within 300s:**
Check container logs: `docker logs sglang-qwen3-bench`. Common causes:
model download in progress (first run), OOM, or CUDA driver mismatch.

**`requests.ConnectionError`:**
Server is not ready. Wait longer or check `docker logs`.

**Low output token count despite max_new_tokens=128:**
Ensure `ignore_eos: true` is set. The script uses the `/generate` endpoint
which correctly respects this flag.
