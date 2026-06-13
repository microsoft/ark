#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Microbenchmark: ARK SwiGLU MLP vs torch eager SwiGLUMLP.

Torch-only pipeline (matmul + F.silu·gate). ARK silu·gate deferred to
upstream fix (same composed-graph planner bug class as Q4).

Shapes: S=2048 (prefill) and S=1 (decode) at Qwen3-8B dimensions.
Run out-of-band on A100:  ``python -m examples.qwen3.bench_mlp``
"""

import torch

from .qwen3_config import Qwen3Config
from .qwen3_ref import SwiGLUMLP
from .ark_mlp import ark_swiglu_mlp
from .microbench import microbench

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def _run(seq_len, label):
    cfg = Qwen3Config()  # 8B defaults
    torch.manual_seed(42)
    mlp = SwiGLUMLP(cfg).cuda().half()

    B = 1
    x = torch.randn(
        B, seq_len, cfg.hidden_dim, device="cuda", dtype=torch.float16
    )

    # --- Torch eager ---
    def run_torch():
        with torch.no_grad():
            mlp(x)

    torch_res = microbench(
        run_torch,
        use_cuda_graph=False,
        flush_l2=False,
    )

    # --- ARK (torch-only fallback) ---
    gate_w = mlp.gate_proj.weight.detach()
    up_w = mlp.up_proj.weight.detach()
    down_w = mlp.down_proj.weight.detach()

    def run_ark():
        with torch.no_grad():
            ark_swiglu_mlp(x, gate_w, up_w, down_w, cfg).eval()

    ark_res = microbench(
        run_ark,
        use_cuda_graph=False,
        flush_l2=False,
    )

    return label, torch_res, ark_res


def main():
    print("NOTE: torch-only (ARK silu·gate deferred to upstream fix / Q10).")
    print(
        f"{'Shape':<20} {'Torch (us)':>16} {'ARK-wrap (us)':>20} {'Speedup':>10}"
    )
    print("-" * 70)
    for seq, label in [(2048, "prefill S=2048"), (1, "decode  S=1")]:
        name, t, a = _run(seq, label)
        sp = t["mean_us"] / a["mean_us"] if a["mean_us"] > 0 else float("nan")
        print(
            f"{name:<20} "
            f"{t['mean_us']:>10.1f} ± {t['std_us']:<5.1f}"
            f"{a['mean_us']:>14.1f} ± {a['std_us']:<5.1f}"
            f"{sp:>8.2f}x"
        )


if __name__ == "__main__":
    main()
