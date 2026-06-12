#!/usr/bin/env python3
"""Classify GPU kernel names into per-component latency buckets.

Component buckets
-----------------
  attention       FlashInfer / FlashAttention / cuDNN SDPA kernels
  gemm_attention  GEMM kernels for Q/K/V/O projections
  gemm_mlp        GEMM kernels for gate/up/down projections (SwiGLU MLP)
  nccl            NCCL all-reduce / reduce-scatter / all-gather
  norms_rope      RMSNorm, RoPE, QK-norm, SiLU, elementwise fused ops
  embed_lm_head   Embedding lookup and lm_head GEMM (vocab-sized dim)
  other           Kernel-launch gaps, CPU overhead, unclassified

Classification table (kernel name patterns)
-------------------------------------------
  Pattern                           Component
  ─────────────────────────────────  ────────────────
  flash_*, fmha*, flashinfer*       attention
  cudnn*sdpa*                       attention
  cutlass_*, cublas*, sm{N}_xmma*   gemm_* (by shape)
  ncclDevKernel_*, nccl_*           nccl
  rms_norm*, rmsnorm*, layernorm*   norms_rope
  fused_*norm*                      norms_rope
  silu*, gelu*, elementwise*        norms_rope
  rotary_*, rope_*                  norms_rope
  (everything else)                 other

GEMM shape disambiguation (Qwen3-8B, TP=8)
-------------------------------------------
  Attention projections  K/N ∈ {512, 128, 768}   (Q, K/V, fused QKV per GPU)
  MLP projections        K/N ∈ {1792, 3584}      (gate/up per GPU, fused)
  lm_head                K/N = 151936 or /TP     (vocab dimension)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Component bucket names
# ---------------------------------------------------------------------------

ATTENTION = "attention"
GEMM_ATTENTION = "gemm_attention"
GEMM_MLP = "gemm_mlp"
NCCL = "nccl"
NORMS_ROPE = "norms_rope"
EMBED_LM_HEAD = "embed_lm_head"
OTHER = "other"

ALL_COMPONENTS = [
    ATTENTION,
    GEMM_ATTENTION,
    GEMM_MLP,
    NCCL,
    NORMS_ROPE,
    EMBED_LM_HEAD,
    OTHER,
]

# ---------------------------------------------------------------------------
# Qwen3-8B model dimensions
# ---------------------------------------------------------------------------

HIDDEN = 4096
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
INTERMEDIATE = 14336
VOCAB = 151936

# ---------------------------------------------------------------------------
# Kernel name patterns → component (first match wins)
# ---------------------------------------------------------------------------

_KERNEL_PATTERNS: list[tuple[str, str]] = [
    # Attention kernels
    (r"flash_", ATTENTION),
    (r"fmha", ATTENTION),
    (r"flashinfer", ATTENTION),
    (r"cudnn.*sdpa", ATTENTION),
    (r"sdpa_", ATTENTION),
    # NCCL
    (r"ncclDevKernel", NCCL),
    (r"nccl_", NCCL),
    (r"ncclKernel", NCCL),
    # Norms / RoPE / activations / elementwise
    (r"rms_norm", NORMS_ROPE),
    (r"rmsnorm", NORMS_ROPE),
    (r"layernorm", NORMS_ROPE),
    (r"layer_norm", NORMS_ROPE),
    (r"fused_.*norm", NORMS_ROPE),
    (r"rotary_", NORMS_ROPE),
    (r"rope_", NORMS_ROPE),
    (r"silu", NORMS_ROPE),
    (r"gelu", NORMS_ROPE),
    (r"elementwise", NORMS_ROPE),
]

_COMPILED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(pat, re.IGNORECASE), comp) for pat, comp in _KERNEL_PATTERNS
]

# GEMM kernel name patterns (for shape-based disambiguation)
_GEMM_PATTERNS: list[re.Pattern[str]] = [
    re.compile(pat, re.IGNORECASE)
    for pat in [
        r"cutlass_",
        r"cublas",
        r"sm\d+_xmma",
        r"gemm",
        r"ampere_",
        r"turing_",
    ]
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_gemm_kernel(name: str) -> bool:
    """Return True if *name* matches a known GEMM kernel pattern."""
    return any(p.search(name) for p in _GEMM_PATTERNS)


def _classify_gemm_by_shape(
    shapes: list[list[int]] | None,
    tp: int = 8,
) -> str:
    """Disambiguate a GEMM kernel into attention / MLP / embed by shapes.

    *shapes* comes from ``record_shapes=True`` in torch.profiler — a list of
    input-tensor shapes.  We flatten all dimension values and look for
    component-distinctive sizes.
    """
    if not shapes:
        return OTHER

    # Vocab-related dimensions
    vocab_dims = {VOCAB, VOCAB // tp}

    # Attention projection dimensions (per-GPU, excluding HIDDEN which is
    # shared with MLP and therefore non-distinctive)
    q_dim = HIDDEN // tp  # 512 for TP=8
    kv_dim = (NUM_KV_HEADS * HEAD_DIM) // tp  # 128 for TP=8
    fused_qkv = q_dim + 2 * kv_dim  # 768 for TP=8
    attn_dims = {q_dim, kv_dim, fused_qkv}

    # MLP projection dimensions (per-GPU)
    mlp_dim = INTERMEDIATE // tp  # 1792 for TP=8
    fused_gate_up = 2 * mlp_dim  # 3584 for TP=8
    mlp_dims = {mlp_dim, fused_gate_up}

    # Flatten all shape dimensions
    all_dims: set[int] = set()
    for shape in shapes:
        all_dims.update(shape)

    # Check for vocab dimension first (embed / lm_head)
    if all_dims & vocab_dims:
        return EMBED_LM_HEAD

    # MLP dimensions are more distinctive than attention; check first
    if all_dims & mlp_dims:
        return GEMM_MLP

    # Attention dimensions
    if all_dims & attn_dims:
        return GEMM_ATTENTION

    return OTHER


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_kernel(
    name: str,
    shapes: list[list[int]] | None = None,
    tp: int = 8,
) -> str:
    """Classify a single kernel into a component bucket.

    Args:
        name: CUDA kernel name from the profiler trace.
        shapes: Optional list of tensor shapes (from ``record_shapes=True``).
        tp: Tensor-parallelism degree (default 8).

    Returns:
        Component bucket string (one of :data:`ALL_COMPONENTS`).
    """
    # Non-GEMM patterns first (cheaper regex, unambiguous)
    for pattern, component in _COMPILED_PATTERNS:
        if pattern.search(name):
            return component

    # GEMM kernels need shape-based disambiguation
    if _is_gemm_kernel(name):
        return _classify_gemm_by_shape(shapes, tp)

    return OTHER


@dataclass
class ComponentBudget:
    """Aggregated latency budget for one component."""

    component: str
    total_us: float = 0.0
    kernel_count: int = 0
    kernel_names: set[str] = field(default_factory=set)

    @property
    def total_ms(self) -> float:
        return self.total_us / 1000.0

    def pct_of(self, total_us: float) -> float:
        """Return this component's share of *total_us* as a percentage."""
        if total_us <= 0:
            return 0.0
        return 100.0 * self.total_us / total_us


def classify_trace_events(
    events: list[dict[str, Any]],
    tp: int = 8,
) -> dict[str, ComponentBudget]:
    """Classify a list of Chrome-trace events into component budgets.

    Each event dict must have ``name`` (str) and ``dur`` (µs, float).
    Optional ``args.shapes`` enables GEMM disambiguation.

    Returns:
        Dict mapping component name → :class:`ComponentBudget`.
    """
    budgets = {comp: ComponentBudget(component=comp) for comp in ALL_COMPONENTS}

    for ev in events:
        name = ev.get("name", "")
        dur_us = ev.get("dur", 0.0)
        shapes = None
        if "args" in ev and "shapes" in ev["args"]:
            shapes = ev["args"]["shapes"]

        comp = classify_kernel(name, shapes=shapes, tp=tp)
        budgets[comp].total_us += dur_us
        budgets[comp].kernel_count += 1
        budgets[comp].kernel_names.add(name)

    return budgets


def format_budget_table(
    budgets: dict[str, ComponentBudget],
    phase: str,
) -> str:
    """Format component budgets as a Markdown table sorted by descending time."""
    total_us = sum(b.total_us for b in budgets.values())
    lines = [
        f"## {phase}",
        "",
        "| Component | Kernel time (ms) | % of total | ARK target | Q-item |",
        "|-----------|-----------------|------------|------------|--------|",
    ]
    sorted_budgets = sorted(
        budgets.values(), key=lambda b: b.total_us, reverse=True
    )
    for b in sorted_budgets:
        lines.append(
            f"| {b.component} | {b.total_ms:.2f} "
            f"| {b.pct_of(total_us):.1f}% | TBD | TBD |"
        )
    lines.append(
        f"| **Total** | **{total_us / 1000:.2f}** | **100%** | | |"
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Load a Chrome-trace JSON, classify kernel events, and print a budget table."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Classify GPU kernels from a Chrome-trace JSON into component buckets.",
    )
    parser.add_argument("trace", help="Path to Chrome-trace JSON file")
    parser.add_argument("--tp", type=int, default=8, help="Tensor-parallelism degree (default 8)")
    args = parser.parse_args(argv)

    with open(args.trace) as f:
        trace = json.load(f)

    events = [
        e
        for e in trace.get("traceEvents", [])
        if e.get("ph") == "X" and e.get("cat", "") == "kernel"
    ]

    budgets = classify_trace_events(events, tp=args.tp)
    print(format_budget_table(budgets, "GPU Kernel Budget"))


if __name__ == "__main__":
    main()
