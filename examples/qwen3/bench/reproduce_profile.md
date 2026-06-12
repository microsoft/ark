# Reproducing the Qwen3-8B Profile

## Prerequisites

- 8×A100-80GB node (e.g., `mscclpp-a100-dev`)
- Docker with NVIDIA runtime
- HuggingFace cache with `Qwen/Qwen3-8B` weights (or network access to pull)

## Steps

### 1. Launch SGLang server

Start the SGLang server manually with the pinned image
(`lmsysorg/sglang:v0.4.6.post1-cu124`), TP=8, and the `Qwen/Qwen3-8B` model.
The `launch_sglang.sh` script is not included in this changeset.

### 2. Run the profiling harness

```bash
# Profile both prefill and decode (5 trials each)
python profile_sglang.py --port 30000 --phase both --trials 5

# Or profile each phase separately
python profile_sglang.py --port 30000 --phase prefill --trials 5
python profile_sglang.py --port 30000 --phase decode  --trials 5
```

Traces are saved server-side at `/tmp/sglang_profile/` (configurable via
`--output-dir`).

### 3. Copy traces from the container

```bash
CONTAINER=sglang-qwen3-bench
docker cp "${CONTAINER}:/tmp/sglang_profile/" ./traces/
```

### 4. Analyze traces

```bash
# Full analysis (trace_analyzer.py + per-component classifier)
bash analyze_profile.sh ./traces/<trace_file>.json --tp 8

# Or run the classifier directly
python3 classify_kernels.py traces/<trace_file>.json --tp 8
```

### 5. Fill PROFILE.md

Copy the per-component budget numbers into `../PROFILE.md`.  Re-rank Q4–Q8
by descending total kernel time.

## Expected output structure

```
traces/
├── analysis/
│   ├── kernels.txt           # Top-N kernel breakdown
│   ├── comm.txt              # NCCL communication analysis
│   ├── gaps.txt              # GPU idle gap analysis
│   ├── full_analysis.md      # Complete trace_analyzer output
│   └── component_budget.md   # Per-component latency budget
└── *.pt.trace.json           # Raw Chrome-trace files
```

## Notes

- The profiler adds overhead; do not compare profiled latencies with the
  Q1 baseline numbers.  The profile is for *relative component breakdown*,
  not absolute latency.
- `record_shapes=True` increases trace size.  Profile rank 0 only for
  manageable file sizes.
- SGLang's `/start_profile` and `/stop_profile` endpoints control the
  server-side `torch.profiler`.  The profiler schedule (wait=2, warmup=1,
  active=3) is configured server-side.
