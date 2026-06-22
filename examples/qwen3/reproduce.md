# Qwen3 reproduce skeleton and evidence inventory

This document is a reproduce skeleton, not a final comparison report. It lists the Qwen3 evidence that can be rerun from this branch and the evidence that is still missing.

## Scope

- ARK-only evidence belongs in this repo under `examples/qwen3/`.
- The SGLang baseline and profiler harness stay local-only outside this repo.
- Fixed SGLang numbers below are target context only. This branch does not include commands that launch, drive, or profile SGLang.
- A missing `PERF_GATE` line is missing evidence. It is not a pass.
- This branch makes no end-to-end speedup claim.

## Hardware and build assumptions

Use the same topology as ARK CUDA CI:

- 8×A100-80GB.
- CUDA-capable ARK build container.
- Repository root checked out at the branch SHA under review.
- Qwen3 component gates run after the normal ARK build.

From the repository root, build ARK with the project workflow:

```bash
mkdir -p build
cd build
cmake ..
make -j"$(nproc)" ut ark_py
export ARK_ROOT="$PWD"
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
```

## SGLang baseline context

Local-only SGLang measurements from 2026-06-15 define the targets. They are not rerunnable from this branch.

| Config | Prefill TTFT | Decode/token | Context |
|--------|-------------:|-------------:|---------|
| TP=8 matched regime | 47.41 ms | 4.26 ms | Primary comparison target. |
| TP=1 SGLang best | 52.31 ms | 12.98 ms | Reference only. |

The local TP=8 profile was decode-dominated and communication-bound:

| Component | SGLang profile budget | Share | Status on this branch |
|-----------|----------------------:|------:|-----------------------|
| nccl / comm | 214.69 ms | 77.3% | Missing branch evidence. |
| gemm | 38.08 ms | 13.7% | Missing branch evidence. |
| attention | 20.93 ms | 7.5% | Missing branch evidence. |
| norms_rope | 3.40 ms | 1.2% | Missing branch evidence. |

## ARK-only evidence inventory at this branch SHA

Current branch artifacts under `examples/qwen3/`:

| Artifact | Evidence type | Rerun command from `build/` | Status |
|----------|---------------|-----------------------------|--------|
| `examples/qwen3/reproduce.md` | Reproduce skeleton and evidence inventory | Not applicable. | Present. |
| `examples/qwen3/bench_*.py` | ARK component perf gates | None available on this branch. | Missing. |
| `examples/qwen3/test_*.py` | Torch-reference equivalence tests | None available on this branch. | Missing. |

Inventory check from `build/`:

```bash
find ../examples/qwen3 -maxdepth 1 \( -name 'bench_*.py' -o -name 'test_*.py' \) -print | sort
```

Expected result for this branch: no Qwen3 benchmark or equivalence-test files are listed.

## Rerun slots for future ARK-only evidence

When a component benchmark lands on this branch, rerun it from `build/` and require exactly one machine-readable line:

```bash
python3 ../examples/qwen3/bench_<component>.py
# PERF_GATE name=<component> ark_ms=<float> sglang_ms=<float> ratio=<float>
```

When a component equivalence test lands on this branch, rerun only that component from `build/`:

```bash
python3 -m pytest -q ../examples/qwen3/test_<component>.py
```

These slots are placeholders until the branch contains the named files. They do not create evidence by themselves.

## Missing final-comparison prerequisites

The final Qwen3 ARK vs SGLang comparison remains blocked.

- Q13 end-to-end prefill/decode sweep evidence is unavailable.
- Q12A-PRE-VPR still needs a strict clean-environment A100 `PERF_GATE name=kv_cache_slot` rerun.
- PR #270 is unmerged, so its all-reduce evidence is not branch evidence here.
- Required Qwen3 component implementations, equivalence tests, and ARK-only perf gates are missing or blocked on this branch.
- No branch artifact proves a full ARK Qwen3 op graph, TP=8 decode path, or ≥3.5× end-to-end speedup.

A future final report must cite rerunnable repo artifacts at its branch SHA for every ARK correctness and performance claim.
