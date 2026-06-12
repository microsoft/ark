#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Microbenchmark: ARK GQA attention vs torch SDPA.

Shapes: S=2048 (prefill) and S=1 (decode) at Qwen3-8B dimensions.
Run out-of-band on A100:  ``python -m examples.qwen3.bench_attention``
"""

import torch

from .qwen3_config import Qwen3Config
from .qwen3_ref import GQAAttention, precompute_rope_freqs
from .ark_attention import ark_gqa_attention, precompute_ark_rope_freqs
from .microbench import microbench

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def _torch_sdpa(x, attn, rope_freqs, mask):
    """Run torch GQAAttention forward (eager, no compile)."""
    with torch.no_grad():
        return attn(x, rope_freqs, mask)


def _run(seq_len, label):
    cfg = Qwen3Config()  # 8B defaults
    torch.manual_seed(42)
    attn = GQAAttention(cfg).cuda().half()
    rope_freqs = precompute_rope_freqs(
        cfg.head_dim, cfg.max_seq_len, cfg.rope_theta
    ).to("cuda")

    B = 1
    x = torch.randn(
        B, seq_len, cfg.hidden_dim, device="cuda", dtype=torch.float16
    )
    mask = torch.full(
        (seq_len, seq_len), float("-inf"), device="cuda", dtype=torch.float16
    )
    mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)

    # --- Torch ---
    torch_res = microbench(
        lambda: _torch_sdpa(x, attn, rope_freqs, mask),
        use_cuda_graph=False,
        flush_l2=False,
    )

    # --- ARK ---
    import ark

    ark_rf = precompute_ark_rope_freqs(
        cfg.head_dim, cfg.max_seq_len, cfg.rope_theta
    ).cuda()[:, :, :seq_len, :]

    # Build the ARK graph once, outside the timed region.
    ark.init()
    ark_result = ark_gqa_attention(
        x,
        attn.q_proj.weight.detach(),
        attn.k_proj.weight.detach(),
        attn.v_proj.weight.detach(),
        attn.o_proj.weight.detach(),
        attn.qk_norm.q_norm.weight.detach().half(),
        attn.qk_norm.k_norm.weight.detach().half(),
        ark_rf,
        mask,
        cfg,
    )

    ark_res = microbench(
        lambda: ark_result.eval(),
        use_cuda_graph=False,
        flush_l2=False,
    )

    return label, torch_res, ark_res


def main():
    print(f"{'Shape':<20} {'Torch (us)':>16} {'ARK (us)':>16} {'Speedup':>10}")
    print("-" * 66)
    for seq, label in [(2048, "prefill S=2048"), (1, "decode  S=1")]:
        name, t, a = _run(seq, label)
        sp = t["mean_us"] / a["mean_us"] if a["mean_us"] > 0 else float("nan")
        print(
            f"{name:<20} "
            f"{t['mean_us']:>10.1f} ± {t['std_us']:<5.1f}"
            f"{a['mean_us']:>10.1f} ± {a['std_us']:<5.1f}"
            f"{sp:>8.2f}x"
        )


if __name__ == "__main__":
    main()
