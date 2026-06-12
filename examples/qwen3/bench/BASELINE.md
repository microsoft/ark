# SGLang Qwen3-8B Baseline Latency

> **Note:** Values marked "TBD" will be filled after running benchmarks on the target hardware.

Model: `Qwen/Qwen3-8B` (~16 GB fp16)

## Image

| Field          | Value |
|----------------|-------|
| Image tag      | `lmsysorg/sglang:v0.4.6.post1-cu124` |
| Image digest   | TBD (run on mscclpp-a100-dev) |
| CUDA version   | 12.4 |
| SGLang version | 0.4.6.post1 |

## Hardware

| Field             | Value |
|-------------------|-------|
| Node              | mscclpp-a100-dev |
| GPUs              | 8× NVIDIA A100-SXM4-80GB |
| GPU clocks (gr)   | TBD (run on mscclpp-a100-dev) |
| GPU clocks (mem)  | TBD (run on mscclpp-a100-dev) |
| Interconnect      | NVLink + NVSwitch |

## TP=1 — SGLang natural best-latency config

Single-GPU inference. For an 8B model (~16 GB fp16), TP=1 is the
lowest-overhead configuration and represents SGLang's best achievable
latency for this model size.

### Server flags

```
python -m sglang.launch_server \
    --model Qwen/Qwen3-8B \
    --tp 1 \
    --port 30000 \
    --mem-fraction-static 0.85 \
    --trust-remote-code
```

### Results

| Metric                          | Value |
|---------------------------------|-------|
| Prefill TTFT (prompt=2048, gen=1) | TBD (run on mscclpp-a100-dev) |
| Decode per-token (prompt=2048, gen=128) | TBD (run on mscclpp-a100-dev) |
| Trials                          | 5 |

## TP=8 — Matched-regime comparison

TP=8 for an 8B model is the ARK-favorable regime: it minimizes per-GPU
compute and amplifies kernel-launch and communication overhead
(hypothesis 1). This matches the parallelism used by the ARK target
but is **not** SGLang's natural best config for this model size.

### Server flags

```
python -m sglang.launch_server \
    --model Qwen/Qwen3-8B \
    --tp 8 \
    --port 30000 \
    --mem-fraction-static 0.85 \
    --trust-remote-code
```

### Results

| Metric                          | Value |
|---------------------------------|-------|
| Prefill TTFT (prompt=2048, gen=1) | TBD (run on mscclpp-a100-dev) |
| Decode per-token (prompt=2048, gen=128) | TBD (run on mscclpp-a100-dev) |
| Trials                          | 5 |

## Methodology

- Endpoint: `/generate` with `ignore_eos: true` (not `/v1/chat/completions`).
- Prompt: ~2048 tokens (repeating pattern).
- Warmup: 3 requests discarded before measurement.
- TTFT: time for prompt=2048, max_new_tokens=1 (single request, not streaming).
- Decode per-token: total_time / max(output_tokens, 1) for prompt=2048, max_new_tokens=128.
- Reported value: median over 5 trials.
- Temperature: 0.0 (greedy).
