# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Qwen3-8B model configuration as a parameterized dataclass."""

from dataclasses import dataclass


@dataclass
class Qwen3Config:
    """Qwen3 model configuration with 8B defaults.

    All fields are overridable. For example, a 32B variant is a one-liner:
        Qwen3Config(n_layers=64, hidden_dim=5120, n_q_heads=40, n_kv_heads=8,
                     intermediate_dim=15360)
    """

    n_layers: int = 36
    hidden_dim: int = 4096
    n_q_heads: int = 32
    n_kv_heads: int = 8
    head_dim: int = 128
    intermediate_dim: int = 12288
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e6
    max_seq_len: int = 4096
    dtype: str = "float16"
