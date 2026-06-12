# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Microbenchmark helper: CUDA-graph capture, L2 flush, steady-state timing.

Follows the gpu-kernel-perf-bench methodology:
- L2 cache pollution buffer sized to 2x L2 cache.
- CUDA-graph capture for launch-overhead elimination.
- Pilot iteration tuning targeting 0.1-0.3 s total.
- cuda.Event timing for all measurements (pilot, calibration, and measured runs).
- Returns structured dict: mean_us, std_us, n_iters.
"""

from typing import Callable, Dict

import torch


def _l2_flush_buffer(device: torch.device) -> torch.Tensor:
    """Allocate a buffer exceeding 2x typical L2 cache (128 MB covers A100's 40 MB)."""
    nbytes = 128 * 1024 * 1024  # 128 MB
    return torch.empty(nbytes // 4, dtype=torch.float32, device=device)


def _flush_l2(buf: torch.Tensor) -> None:
    """Touch the L2-flush buffer to evict cached data."""
    buf.zero_()


def _determine_iters(
    fn: Callable[[], None],
    target_secs: float = 0.2,
    device: torch.device = None,
) -> int:
    """Pilot run: find iteration count for ~target_secs total execution time."""
    if device is None:
        device = torch.device("cuda")

    # Warm up
    fn()
    torch.cuda.synchronize(device)

    # Time a single call
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize(device)
    single_ms = start.elapsed_time(end)

    if single_ms <= 0:
        return 100

    n = max(1, int(target_secs * 1000 / single_ms))
    return n


def microbench(
    fn: Callable[[], None],
    device: torch.device = None,
    n_iters: int = None,
    use_cuda_graph: bool = True,
    flush_l2: bool = True,
) -> Dict[str, float]:
    """Benchmark a CUDA callable and return timing statistics.

    Args:
        fn: Zero-argument callable that performs the GPU work.
        device: CUDA device. Defaults to cuda:0.
        n_iters: Override iteration count (else auto-tuned via pilot).
        use_cuda_graph: Capture fn into a CUDA graph for replay.
        flush_l2: Flush L2 cache between graph replays.

    Returns:
        Dict with keys: mean_us, std_us, n_iters.
    """
    if device is None:
        device = torch.device("cuda")

    # Pilot: determine iteration count
    if n_iters is None:
        n_iters = _determine_iters(fn, device=device)
    n_iters = max(n_iters, 1)

    flush_buf = _l2_flush_buffer(device) if flush_l2 else None

    if use_cuda_graph:
        # Warm-up run for CUDA graph capture
        torch.cuda.synchronize(device)
        fn()
        torch.cuda.synchronize(device)

        # Capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()

        # Determine per-graph batch to keep each replay > 1 ms
        graph.replay()
        torch.cuda.synchronize(device)
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        graph.replay()
        end_ev.record()
        torch.cuda.synchronize(device)
        replay_ms = start_ev.elapsed_time(end_ev)

        per_graph = max(1, int(1.0 / max(replay_ms, 1e-6)))
        # With L2 flush, each replay must start cold.
        if flush_l2:
            per_graph = 1
            n_replays = n_iters
        else:
            n_replays = max(1, n_iters // per_graph)

        replay_fn = graph.replay
    else:
        per_graph = 1
        n_replays = n_iters
        replay_fn = fn

    # Warm-up replay (not measured)
    for _ in range(per_graph):
        replay_fn()
    torch.cuda.synchronize(device)

    # Measured runs with cuda.Event timing
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    times_us: list[float] = []
    for _ in range(n_replays):
        if flush_l2 and flush_buf is not None:
            _flush_l2(flush_buf)
            torch.cuda.synchronize(device)
        start_ev.record()
        for _ in range(per_graph):
            replay_fn()
        end_ev.record()
        torch.cuda.synchronize(device)
        times_us.append(start_ev.elapsed_time(end_ev) * 1000.0)  # ms → us

    mean_us = sum(times_us) / len(times_us) / per_graph
    if len(times_us) > 1:
        variance = sum((t / per_graph - mean_us) ** 2 for t in times_us) / (
            len(times_us) - 1
        )
        std_us = variance**0.5
    else:
        std_us = 0.0

    return {
        "mean_us": mean_us,
        "std_us": std_us,
        "n_iters": n_replays * per_graph,
    }
