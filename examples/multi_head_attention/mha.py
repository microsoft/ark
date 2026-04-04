# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Multi-Head Attention implemented as an ARK Module using composed small ops.

Two versions:
1. MultiHeadAttention — standard (non-flash) attention for correctness baseline
2. FlashMultiHeadAttention — online softmax (flash attention) algorithm

Both use ARK's operator composition with PlannerContext for scheduling.
"""

import ark
import math


class MultiHeadAttention(ark.Module):
    """Standard multi-head attention: O = softmax(Q @ K^T / sqrt(d)) @ V

    Args:
        head_dim: dimension per head (used for scaling).
        causal: whether to apply causal masking (not yet supported).
    """

    def __init__(self, head_dim: int, causal: bool = False):
        super().__init__()
        self.scale = 1.0 / math.sqrt(head_dim)
        self.causal = causal

    def forward(self, q, r_k, v):
        """
        Args:
            q:   (batch, heads, seq_len, head_dim)  — query
            r_k: (batch, heads, head_dim, seq_len)  — key, already transposed
            v:   (batch, heads, seq_len, head_dim)  — value

        Returns:
            o: (batch, heads, seq_len, head_dim)
        """
        # S = Q @ K^T  -> (batch, heads, seq_len, seq_len)
        s = ark.matmul(q, r_k)

        # Scale: S = S * (1 / sqrt(d))
        s = ark.mul(s, self.scale)

        # Softmax along last axis
        # max
        m = ark.reduce_max(s, axis=-1)
        s = ark.sub(s, m)
        s = ark.exp(s)
        l = ark.reduce_sum(s, axis=-1)
        p = ark.div(s, l)

        # O = P @ V  -> (batch, heads, seq_len, head_dim)
        o = ark.matmul(p, v)
        return o


class MultiHeadAttentionOptimized(ark.Module):
    """Tile-fused MHA: merges matmul, softmax, and output matmul into
    aligned tile tasks using PlannerContext.

    Key insight: ARK's matmul is tile-based (e.g., [128, N] per task).
    By configuring softmax ops to use the same tile grid — each task
    processes the same row-block — all ops can be fused into one task
    with sync=False. This eliminates ALL inter-op sync barriers.

    The tile alignment is:
    - matmul(Q, K^T):    [TileM, N] tiles of S matrix
    - softmax(S):        [TileM, N] tiles (full-row reduction per tile)
    - matmul(P, V):      [TileM, D] tiles of output

    All ops share the same number of tasks = batch*heads * ceil(N/TileM).

    Args:
        head_dim: dimension per head.
        seq_len: sequence length.
        tile_m: row-block size for tiling (must divide seq_len).
    """

    def __init__(self, head_dim: int, seq_len: int = 256, tile_m: int = 128):
        super().__init__()
        self.scale = 1.0 / math.sqrt(head_dim)
        self.seq_len = seq_len
        self.tile_m = tile_m

    def forward(self, q, r_k, v):
        shape = q.shape()
        N = shape[-2]
        D = shape[-1]
        batch_heads = 1
        for d in shape[:-2]:
            batch_heads *= d
        TM = self.tile_m
        S = self.seq_len  # = N for self-attention
        num_tasks = batch_heads * (N // TM)

        # Fuse matmul(Q,K^T) + softmax into one task per row-block.
        # All ops use NumWarps=8 and tile height=TM to produce matching
        # task counts. The key fix: reduce ops now use Tile=[TM,1] to
        # match the matmul's tile grid.
        with ark.PlannerContext(
            sync=False,
            warp_range=[0, 8],
            sram_range=[0, 147456],
        ):
            # Matmul Q[TM,D] @ K^T[D,S] -> S[TM,S]
            with ark.PlannerContext(
                config={
                    "NumWarps": 8,
                    "SramBytes": 147456,
                    "Tile": [TM, S],
                },
            ):
                s = ark.matmul(q, r_k)

            # scale — element-wise, tile matches matmul
            with ark.PlannerContext(
                config={
                    "NumWarps": 8, "SramBytes": 0,
                    "Tile": [TM, S], "NumTasks": num_tasks,
                },
            ):
                s = ark.mul(s, self.scale)

            # reduce_max — NOW with Tile=[TM,1] to match task count
            with ark.PlannerContext(
                config={
                    "NumWarps": 8, "SramBytes": 256,
                    "ImplType": "WarpWise",
                    "Tile": [TM, 1],
                },
            ):
                m = ark.reduce_max(s, axis=-1)

            # sub + exp
            with ark.PlannerContext(
                config={
                    "NumWarps": 8, "SramBytes": 0,
                    "Tile": [TM, S], "NumTasks": num_tasks,
                },
            ):
                s = ark.sub(s, m)
                s = ark.exp(s)

            # reduce_sum — Tile=[TM,1]
            with ark.PlannerContext(
                config={
                    "NumWarps": 8, "SramBytes": 256,
                    "ImplType": "WarpWise",
                    "Tile": [TM, 1],
                },
            ):
                l = ark.reduce_sum(s, axis=-1)

            # div
            with ark.PlannerContext(
                config={
                    "NumWarps": 8, "SramBytes": 0,
                    "Tile": [TM, S], "NumTasks": num_tasks,
                },
            ):
                p = ark.div(s, l)

        # Matmul P @ V — separate task (different SRAM requirement)
        o = ark.matmul(p, v)

        return o
