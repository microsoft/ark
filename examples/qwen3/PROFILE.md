# Qwen3-8B Per-Component Latency Profile

Per-component latency breakdown of SGLang Qwen3-8B (TP=8, batch=1) on
8×A100-80GB.  Profiled with `torch.profiler` via `profile_sglang.py`.

## Configuration

| Parameter       | Value                                   |
|-----------------|-----------------------------------------|
| Model           | Qwen/Qwen3-8B                           |
| SGLang image    | lmsysorg/sglang:v0.4.6.post1-cu124      |
| Hardware        | 8×A100-80GB (SXM4)                      |
| TP              | 8                                        |
| Batch           | 1                                        |
| Prompt length   | 2048 tokens                              |
| Generate length | 128 tokens                               |
| Profiler        | torch.profiler (wait=2, warmup=1, active=3) |

## Prefill Phase (prompt=2048, max_new_tokens=1)

| Component      | Kernel time (ms) | % of total | ARK target | Q-item |
|----------------|-----------------|------------|------------|--------|
| gemm_mlp       | TBD             | TBD        | TBD        | Q5     |
| gemm_attention | TBD             | TBD        | TBD        | Q4     |
| attention      | TBD             | TBD        | TBD        | Q6     |
| norms_rope     | TBD             | TBD        | TBD        | Q7     |
| nccl           | TBD             | TBD        | TBD        | Q8     |
| embed_lm_head  | TBD             | TBD        | TBD        | —      |
| other          | TBD             | TBD        | TBD        | —      |
| **Total**      | **TBD**         | **100%**   |            |        |

## Decode Phase (prompt=2048, max_new_tokens=128)

| Component      | Kernel time (ms) | % of total | ARK target | Q-item |
|----------------|-----------------|------------|------------|--------|
| gemm_mlp       | TBD             | TBD        | TBD        | Q5     |
| gemm_attention | TBD             | TBD        | TBD        | Q4     |
| attention      | TBD             | TBD        | TBD        | Q6     |
| norms_rope     | TBD             | TBD        | TBD        | Q7     |
| nccl           | TBD             | TBD        | TBD        | Q8     |
| embed_lm_head  | TBD             | TBD        | TBD        | —      |
| other          | TBD             | TBD        | TBD        | —      |
| **Total**      | **TBD**         | **100%**   |            |        |

## Q4–Q8 Re-ranking

Re-rank Q4–Q8 by descending kernel time once numbers land.  The component
that consumes the most GPU time gets highest optimization priority.

| Priority | Q-item | Component | Prefill % | Decode % |
|----------|--------|-----------|-----------|----------|
| 1        | TBD    | TBD       | TBD       | TBD      |
| 2        | TBD    | TBD       | TBD       | TBD      |
| 3        | TBD    | TBD       | TBD       | TBD      |
| 4        | TBD    | TBD       | TBD       | TBD      |
| 5        | TBD    | TBD       | TBD       | TBD      |

Numbers and final ordering filled out-of-band after profiling on
`mscclpp-a100-dev`.  See [reproduce_profile.md](bench/reproduce_profile.md)
for repro steps.
