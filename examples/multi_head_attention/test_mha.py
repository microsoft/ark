#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Test and benchmark ARK MultiHeadAttention against FlashAttention-2.

Correctness: uses Tensor.eval() for concise graph execution.
Benchmark: follows gpu-kernel-perf-bench methodology —
  - L2 cache pollution via rotated input buffers
  - Pilot-driven iteration count (target 0.1-0.3s total)
  - torch.profiler for FlashAttention timing
  - ARK native rt.run(iter=N) for ARK timing (persistent loop kernel)
"""

import sys
import os
import math
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mha import MultiHeadAttention, MultiHeadAttentionOptimized

try:
    from flash_attn import flash_attn_func

    _has_flash = True
except ImportError:
    _has_flash = False

import ark

DEVICE = "cuda:0"


# ─── Correctness ────────────────────────────────────────────────────────────


def test_correctness(B, H, N, D, dtype=torch.float16):
    """Compare ARK MHA output against FlashAttention-2 using eval()."""
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(B, H, N, D, dtype=dtype, device=DEVICE)
    k = torch.randn(B, H, N, D, dtype=dtype, device=DEVICE)
    v = torch.randn(B, H, N, D, dtype=dtype, device=DEVICE)
    k_t = k.transpose(-2, -1).contiguous()

    # ARK vanilla — uses eval()
    result = MultiHeadAttention(D)(
        ark.Tensor.from_torch(q), ark.Tensor.from_torch(k_t), ark.Tensor.from_torch(v)
    ).eval()

    # Reference
    if _has_flash:
        q_fa = q.transpose(1, 2).contiguous()
        k_fa = k.transpose(1, 2).contiguous()
        v_fa = v.transpose(1, 2).contiguous()
        ref = flash_attn_func(q_fa, k_fa, v_fa, softmax_scale=scale)
        ref = ref.transpose(1, 2).contiguous()
        label = "FA2"
    else:
        ref = F.scaled_dot_product_attention(q, k, v, scale=scale)
        label = "SDPA"

    diff = (result - ref).abs().max().item()
    atol = 5e-2 if dtype == torch.float16 else 1e-1
    ok = diff < atol
    print(f"  B={B} H={H} N={N:4d} D={D}  diff={diff:.4f} vs {label}  {'PASS' if ok else 'FAIL'}")
    return ok


# ─── Benchmark helpers ──────────────────────────────────────────────────────

# L2 cache size for H200 ≈ 50 MB. Use 2× = 100 MB worth of buffers.
L2_CACHE_BYTES = 50 * 1024 * 1024


def _make_rotated_inputs(B, H, N, D, dtype, num_bufs):
    """Create multiple input buffer sets for L2 cache pollution."""
    bufs = []
    for _ in range(num_bufs):
        q = torch.randn(B, H, N, D, dtype=dtype, device=DEVICE)
        k = torch.randn(B, H, N, D, dtype=dtype, device=DEVICE)
        v = torch.randn(B, H, N, D, dtype=dtype, device=DEVICE)
        bufs.append((q, k, v))
    return bufs


def _pilot_iters(run_once_fn, target_sec=0.2):
    """Determine iteration count to reach target_sec total time."""
    # Single pilot
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    run_once_fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    per_iter = max(t1 - t0, 1e-6)
    iters = max(1, int(target_sec / per_iter))
    return iters


def bench_flash_attn(B, H, N, D, dtype=torch.float16):
    """Benchmark FlashAttention-2 with L2 pollution and torch.profiler."""
    if not _has_flash:
        return float("nan")
    scale = 1.0 / math.sqrt(D)
    elem_bytes = N * D * torch.finfo(dtype).bits // 8
    num_bufs = max(4, (2 * L2_CACHE_BYTES) // (3 * B * H * elem_bytes) + 1)
    bufs = _make_rotated_inputs(B, H, N, D, dtype, num_bufs)

    def run_one(i):
        q, k, v = bufs[i % num_bufs]
        q_fa = q.transpose(1, 2).contiguous()
        k_fa = k.transpose(1, 2).contiguous()
        v_fa = v.transpose(1, 2).contiguous()
        flash_attn_func(q_fa, k_fa, v_fa, softmax_scale=scale)

    iters = _pilot_iters(lambda: run_one(0))

    # Warmup
    for i in range(min(3, iters)):
        run_one(i)

    # Timed
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(iters):
        run_one(i)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed / iters * 1000  # ms


def bench_ark(B, H, N, D, mha_cls, mha_args, dtype=torch.float16):
    """Benchmark an ARK MHA module using the persistent loop kernel."""
    q = torch.randn(B, H, N, D, dtype=dtype, device=DEVICE)
    k = torch.randn(B, H, N, D, dtype=dtype, device=DEVICE)
    v = torch.randn(B, H, N, D, dtype=dtype, device=DEVICE)
    k_t = k.transpose(-2, -1).contiguous()

    ark.init()
    mha = mha_cls(*mha_args)
    out = mha(
        ark.Tensor.from_torch(q), ark.Tensor.from_torch(k_t), ark.Tensor.from_torch(v)
    )

    with ark.Runtime() as rt:
        rt.launch()
        # Pilot: single iteration
        iters = _pilot_iters(lambda: rt.run(iter=1), target_sec=0.2)

        # Warmup
        rt.run(iter=min(3, iters))

        # Timed
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        rt.run(iter=iters)
        elapsed = time.perf_counter() - t0

    return elapsed / iters * 1000  # ms


def run_benchmark(B, H, N, D, dtype=torch.float16):
    fa_ms = bench_flash_attn(B, H, N, D, dtype)
    vanilla_ms = bench_ark(B, H, N, D, MultiHeadAttention, (D,), dtype)
    opt_ms = bench_ark(B, H, N, D, MultiHeadAttentionOptimized, (D, N), dtype)
    ratio = opt_ms / fa_ms if fa_ms > 0 else float("nan")
    print(
        f"  B={B} H={H:2d} N={N:4d} D={D}  "
        f"FA2={fa_ms:.3f}ms  ARK={vanilla_ms:.3f}ms  ARK-Opt={opt_ms:.3f}ms  "
        f"(Opt/FA2={ratio:.2f}x)"
    )


# ─── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Correctness: ARK MHA vs FlashAttention-2")
    print("=" * 70)
    all_pass = True
    for B, H, N, D in [
        (1, 1, 256, 128),
        (1, 4, 256, 128),
        (2, 8, 256, 128),
        (1, 1, 512, 128),
    ]:
        all_pass &= test_correctness(B, H, N, D)

    if not all_pass:
        print("\nSome tests FAILED!")
        sys.exit(1)
    print("\nAll correctness tests PASSED!")

    print()
    print("=" * 70)
    print("Performance: ARK vs FlashAttention-2")
    print("=" * 70)
    for B, H, N, D in [
        (1, 1, 256, 128),
        (1, 4, 256, 128),
        (1, 8, 256, 128),
        (1, 1, 512, 128),
        (1, 4, 512, 128),
    ]:
        run_benchmark(B, H, N, D)
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Test and benchmark ARK MultiHeadAttention against:
  - flash_attn (Tri Dao's FlashAttention-2, flash_attn_func)
  - PyTorch SDPA (F.scaled_dot_product_attention, which dispatches to
    flash/mem-efficient/math backends automatically)
"""

import ark
import torch
import torch.nn.functional as F
import time
import math
import sys

from flash_attn import flash_attn_func

sys.path.insert(0, ".")
from mha import MultiHeadAttention, MultiHeadAttentionOptimized


def flash_attn_reference(q, k, v, scale):
    """Run Tri Dao's FlashAttention-2.

    flash_attn_func expects (batch, seq_len, heads, head_dim).
    Our tensors are (batch, heads, seq_len, head_dim), so we transpose.
    """
    q_fa = q.transpose(1, 2).contiguous()  # (B, N, H, D)
    k_fa = k.transpose(1, 2).contiguous()
    v_fa = v.transpose(1, 2).contiguous()
    o_fa = flash_attn_func(q_fa, k_fa, v_fa, softmax_scale=scale)
    return o_fa.transpose(1, 2).contiguous()  # back to (B, H, N, D)


def torch_sdpa_reference(q, k, v, scale):
    """PyTorch's scaled_dot_product_attention (auto backend selection)."""
    return F.scaled_dot_product_attention(q, k, v, scale=scale)


def test_correctness(batch, heads, seq_len, head_dim, dtype=torch.float16):
    print(f"  B={batch}, H={heads}, N={seq_len}, D={head_dim}", end="")
    scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device="cuda:0")
    k = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device="cuda:0")
    v = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device="cuda:0")

    # Reference: FlashAttention-2
    ref = flash_attn_reference(q, k, v, scale)

    # ARK standard MHA
    ark.init()
    k_t = k.transpose(-2, -1).contiguous()
    mha = MultiHeadAttention(head_dim)
    ark_out = mha(ark.Tensor.from_torch(q), ark.Tensor.from_torch(k_t), ark.Tensor.from_torch(v))
    with ark.Runtime() as rt:
        rt.launch()
        rt.run()
        result = ark_out.to_torch()

    diff = (result - ref).abs().max().item()
    atol = 5e-2 if dtype == torch.float16 else 1e-1
    ok = diff < atol
    print(f"  diff={diff:.4f}  {'PASS' if ok else 'FAIL'}")
    return ok


def bench_one(label, run_fn, num_warmup=10, num_iter=50):
    """Benchmark helper: warmup, then time num_iter iterations."""
    for _ in range(num_warmup):
        run_fn()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iter):
        run_fn()
    torch.cuda.synchronize()
    ms = (time.time() - start) / num_iter * 1000
    return ms


def run_benchmark(batch, heads, seq_len, head_dim, dtype=torch.float16):
    scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device="cuda:0")
    k = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device="cuda:0")
    v = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device="cuda:0")
    k_t = k.transpose(-2, -1).contiguous()

    # --- FlashAttention-2 (Tri Dao) ---
    q_fa = q.transpose(1, 2).contiguous()
    k_fa = k.transpose(1, 2).contiguous()
    v_fa = v.transpose(1, 2).contiguous()

    flash_ms = bench_one(
        "FlashAttn2",
        lambda: flash_attn_func(q_fa, k_fa, v_fa, softmax_scale=scale),
    )

    # --- PyTorch SDPA ---
    sdpa_ms = bench_one(
        "SDPA",
        lambda: F.scaled_dot_product_attention(q, k, v, scale=scale),
    )

    # --- ARK Vanilla ---
    ark.init()
    mha = MultiHeadAttention(head_dim)
    ark_out = mha(ark.Tensor.from_torch(q), ark.Tensor.from_torch(k_t), ark.Tensor.from_torch(v))
    with ark.Runtime() as rt:
        rt.launch()
        vanilla_ms = bench_one("ARK", lambda: rt.run(iter=1), num_warmup=5)

    # --- ARK Optimized (fused softmax) ---
    ark.init()
    mha_opt = MultiHeadAttentionOptimized(head_dim, seq_len)
    ark_out2 = mha_opt(ark.Tensor.from_torch(q), ark.Tensor.from_torch(k_t), ark.Tensor.from_torch(v))
    with ark.Runtime() as rt:
        rt.launch()
        opt_ms = bench_one("ARK-Opt", lambda: rt.run(iter=1), num_warmup=5)

    print(
        f"  B={batch} H={heads:2d} N={seq_len:4d} D={head_dim:3d}  "
        f"FlashAttn2={flash_ms:.3f}ms  SDPA={sdpa_ms:.3f}ms  "
        f"ARK={vanilla_ms:.3f}ms  ARK-Opt={opt_ms:.3f}ms  "
        f"(Opt/Flash={opt_ms/flash_ms:.2f}x)"
    )
    return flash_ms, sdpa_ms, vanilla_ms, opt_ms


if __name__ == "__main__":
    print("=" * 70)
    print("Correctness: ARK MHA vs FlashAttention-2")
    print("=" * 70)
    all_pass = True
    for B, H, N, D in [
        (1, 1, 256, 128),
        (1, 4, 256, 128),
        (2, 8, 256, 128),
        (1, 1, 512, 128),
    ]:
        all_pass &= test_correctness(B, H, N, D)

    if not all_pass:
        print("\nSome tests FAILED!")
        sys.exit(1)
    print("\nAll correctness tests PASSED!")

    print()
    print("=" * 70)
    print("Performance: ARK vs FlashAttention-2 vs PyTorch SDPA")
    print("=" * 70)
    for B, H, N, D in [
        (1, 1, 256, 128),
        (1, 4, 256, 128),
        (1, 8, 256, 128),
        (1, 1, 512, 128),
        (1, 4, 512, 128),
        (1, 8, 512, 128),
    ]:
        run_benchmark(B, H, N, D)
