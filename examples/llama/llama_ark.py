# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = (
        256  # make SwiGLU hidden layer size multiple of large power of 2
    )
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(ark.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = ark.Parameter(ark.tensor([dim], ark.FP32))

    def _norm(self, x):
        # x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        square = ark.mul(x, x)
        square_sum = ark.reduce_mean(square, axis=square.ndims() - 1)
        return ark.div(x, ark.sqrt(square_sum))

    def forward(self, x):
        output = self._norm(x)
        return ark.mul(output, self.weight)


class Linear(ark.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = ark.Parameter(ark.tensor([out_dim, in_dim], ark.FP32))

    def forward(self, x):
        return ark.matmul(x, self.weight, transpose_b=True)


class Silu(ark.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return ark.mul(x, ark.sigmoid(x))


class FeedForward(ark.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * (
            (hidden_dim + multiple_of - 1) // multiple_of
        )

        self.w1 = Linear(dim, hidden_dim)
        self.w2 = Linear(hidden_dim, dim)
        self.w3 = Linear(dim, hidden_dim)

    def forward(self, x):
        # self.w2(F.silu(self.w1(x)) * self.w3(x))
        x1 = self.w1(x)
        x1 = Silu()(x1)
        x2 = self.w3(x)
        x3 = ark.mul(x1, x2)
        x4 = self.w2(x3)
        return x4