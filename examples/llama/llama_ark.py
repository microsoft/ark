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
        self.weight = ark.Parameter(ark.tensor([dim], ark.FP16))

    def _norm(self, x):
        # x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        square = ark.mul(x, x)
        square_sum = ark.reduce_mean(square, axis=square.ndims() - 1)
        return ark.div(x, ark.sqrt(square_sum))

    def forward(self, x):
        output = self._norm(x)
        return ark.mul(output, self.weight)
