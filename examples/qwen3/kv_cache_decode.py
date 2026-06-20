# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Graph-reused ARK KV-cache decode proof for Qwen3 attention.

The graph writes the current token's K/V into preallocated external cache
buffers, orders the later cache read with identity dependencies, applies a
prefix mask, and computes one-token grouped-query attention. Callers precreate
all torch CUDA tensors before launching the ARK runtime; no torch GPU work is
required while the runtime is launched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch


@dataclass(frozen=True)
class KVCacheDecodeConfig:
    """Static graph shape for one-token GQA decode."""

    max_seq_len: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype = torch.float16

    def __post_init__(self):
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.num_q_heads <= 0:
            raise ValueError("num_q_heads must be positive")
        if self.num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError("num_q_heads must be divisible by num_kv_heads")
        if self.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("dtype must be torch.float16 or torch.bfloat16")

    @property
    def group_size(self) -> int:
        return self.num_q_heads // self.num_kv_heads

    @property
    def ark_q_rows(self) -> int:
        return max(self.group_size, 64)


QWEN3_DECODE_CONFIG = KVCacheDecodeConfig(
    max_seq_len=640,
    num_q_heads=32,
    num_kv_heads=8,
    head_dim=128,
)


_SMALL_TEST_CONFIG = KVCacheDecodeConfig(
    max_seq_len=64,
    num_q_heads=8,
    num_kv_heads=2,
    head_dim=64,
)


def _ark_dtype(dtype: torch.dtype):
    import ark

    if dtype == torch.float16:
        return ark.fp16
    if dtype == torch.bfloat16:
        return ark.bf16
    raise ValueError(f"unsupported dtype: {dtype}")


def _check_tensor(
    name: str,
    tensor: torch.Tensor,
    shape: Sequence[int],
    dtype: torch.dtype,
    device: torch.device,
):
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(
            f"{name} shape must be {tuple(shape)}, got {tuple(tensor.shape)}"
        )
    if tensor.dtype != dtype:
        raise ValueError(f"{name} dtype must be {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} device must be {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def check_decode_tensors(
    config: KVCacheDecodeConfig,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    mask: torch.Tensor,
    output: torch.Tensor,
    position: int,
):
    """Validate tensors that will be bound to the ARK decode graph."""

    if not q.is_cuda:
        raise ValueError("q must be a CUDA tensor")
    if not (0 <= position < config.max_seq_len):
        raise ValueError(
            f"position must be in [0, {config.max_seq_len}), got {position}"
        )
    device = q.device
    dtype = config.dtype
    _check_tensor("q", q, (config.num_q_heads, config.head_dim), dtype, device)
    _check_tensor("k", k, (config.num_kv_heads, config.head_dim), dtype, device)
    _check_tensor("v", v, (config.num_kv_heads, config.head_dim), dtype, device)
    _check_tensor(
        "k_cache",
        k_cache,
        (config.num_kv_heads, config.max_seq_len, config.head_dim),
        dtype,
        device,
    )
    _check_tensor(
        "v_cache",
        v_cache,
        (config.num_kv_heads, config.max_seq_len, config.head_dim),
        dtype,
        device,
    )
    _check_tensor("mask", mask, (1, config.max_seq_len), dtype, device)
    _check_tensor(
        "output", output, (config.num_q_heads, config.head_dim), dtype, device
    )


def make_prefix_mask(
    config: KVCacheDecodeConfig, position: int, device: torch.device
) -> torch.Tensor:
    """Return a mask with zeros through *position* and -inf-like values after."""

    if not (0 <= position < config.max_seq_len):
        raise ValueError(
            f"position must be in [0, {config.max_seq_len}), got {position}"
        )
    mask = torch.full(
        (1, config.max_seq_len), -10000.0, dtype=config.dtype, device=device
    )
    mask[:, : position + 1] = 0.0
    return mask.contiguous()


def torch_gqa_decode_reference(
    config: KVCacheDecodeConfig,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    position: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch reference for one-token GQA decode on CPU or CUDA tensors."""

    if not (0 <= position < config.max_seq_len):
        raise ValueError(
            f"position must be in [0, {config.max_seq_len}), got {position}"
        )
    next_k_cache = k_cache.clone()
    next_v_cache = v_cache.clone()
    next_k_cache[:, position, :] = k
    next_v_cache[:, position, :] = v

    scale = 1.0 / math.sqrt(config.head_dim)
    outs = []
    for kv_head in range(config.num_kv_heads):
        q0 = kv_head * config.group_size
        q1 = q0 + config.group_size
        q_group = q[q0:q1].float()
        k_prefix = next_k_cache[kv_head, : position + 1].float()
        v_prefix = next_v_cache[kv_head, : position + 1].float()
        scores = (q_group @ k_prefix.transpose(-2, -1)) * scale
        attn = torch.softmax(scores, dim=-1)
        outs.append(attn @ v_prefix)

    output = torch.cat(outs, dim=0).to(config.dtype)
    return output, next_k_cache, next_v_cache


class KVCacheDecodeGraph:
    """One compiled ARK graph for repeated one-token KV-cache decode runs."""

    def __init__(self, config: KVCacheDecodeConfig):
        self.config = config
        self.runtime = None
        self._build_graph()

    def _build_graph(self):
        import ark

        ark.init()
        dtype = _ark_dtype(self.config.dtype)
        cfg = self.config
        scale = 1.0 / math.sqrt(cfg.head_dim)

        self.q_group = []
        self.k_token = []
        self.v_token = []
        self.k_slot = []
        self.v_slot = []
        self.k_cache = []
        self.v_cache = []
        self.mask = []
        self.output_group = []

        for kv_head in range(cfg.num_kv_heads):
            q_group = ark.placeholder(
                [cfg.ark_q_rows, cfg.head_dim], dtype, name=f"q_group_{kv_head}"
            )
            k_token = ark.placeholder(
                [1, cfg.head_dim], dtype, name=f"k_token_{kv_head}"
            )
            v_token = ark.placeholder(
                [1, cfg.head_dim], dtype, name=f"v_token_{kv_head}"
            )
            k_slot = ark.placeholder(
                [1, cfg.head_dim], dtype, name=f"k_slot_{kv_head}"
            )
            v_slot = ark.placeholder(
                [1, cfg.head_dim], dtype, name=f"v_slot_{kv_head}"
            )
            k_cache = ark.placeholder(
                [cfg.max_seq_len, cfg.head_dim],
                dtype,
                name=f"k_cache_{kv_head}",
            )
            v_cache = ark.placeholder(
                [cfg.max_seq_len, cfg.head_dim],
                dtype,
                name=f"v_cache_{kv_head}",
            )
            mask = ark.placeholder(
                [1, cfg.max_seq_len], dtype, name=f"mask_{kv_head}"
            )
            output_group = ark.placeholder(
                [cfg.group_size, cfg.head_dim],
                dtype,
                name=f"output_group_{kv_head}",
            )

            updated_k = ark.copy(
                k_token, output=k_slot, name=f"write_k_{kv_head}"
            )
            updated_v = ark.copy(
                v_token, output=v_slot, name=f"write_v_{kv_head}"
            )
            k_after = ark.identity(
                k_cache, deps=[updated_k, updated_v], name=f"k_after_{kv_head}"
            )
            v_after = ark.identity(
                v_cache, deps=[updated_k, updated_v], name=f"v_after_{kv_head}"
            )

            scores = ark.matmul(
                q_group,
                k_after,
                transpose_other=True,
                name=f"scores_{kv_head}",
            )
            scores = ark.mul(scores, scale, name=f"scale_{kv_head}")
            scores = ark.add(scores, mask, name=f"mask_scores_{kv_head}")
            probs = ark.softmax(scores, name=f"softmax_{kv_head}")
            context = ark.matmul(probs, v_after, name=f"context_{kv_head}")
            ark.copy(
                context[: cfg.group_size, :],
                output=output_group,
                name=f"write_output_{kv_head}",
            )

            self.q_group.append(q_group)
            self.k_token.append(k_token)
            self.v_token.append(v_token)
            self.k_slot.append(k_slot)
            self.v_slot.append(v_slot)
            self.k_cache.append(k_cache)
            self.v_cache.append(v_cache)
            self.mask.append(mask)
            self.output_group.append(output_group)

    def bindings(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        mask: torch.Tensor,
        output: torch.Tensor,
        position: int,
    ) -> Dict:
        """Return placeholder bindings for one decode position."""

        check_decode_tensors(
            self.config, q, k, v, k_cache, v_cache, mask, output, position
        )
        cfg = self.config
        bindings = {}
        for kv_head in range(cfg.num_kv_heads):
            q0 = kv_head * cfg.group_size
            q1 = q0 + cfg.group_size
            q_group = torch.zeros(
                cfg.ark_q_rows, cfg.head_dim, dtype=cfg.dtype, device=q.device
            )
            q_group[: cfg.group_size, :] = q[q0:q1, :]
            bindings[self.q_group[kv_head]] = q_group
            bindings[self.k_token[kv_head]] = k[kv_head : kv_head + 1, :]
            bindings[self.v_token[kv_head]] = v[kv_head : kv_head + 1, :]
            bindings[self.k_slot[kv_head]] = k_cache[
                kv_head, position : position + 1, :
            ]
            bindings[self.v_slot[kv_head]] = v_cache[
                kv_head, position : position + 1, :
            ]
            bindings[self.k_cache[kv_head]] = k_cache[kv_head, :, :]
            bindings[self.v_cache[kv_head]] = v_cache[kv_head, :, :]
            bindings[self.mask[kv_head]] = mask
            bindings[self.output_group[kv_head]] = output[q0:q1, :]
        return bindings

    def launch(self, bindings: Dict, device_id: int | None = None):
        """Compile and launch the ARK runtime in non-loop mode."""

        import ark

        if device_id is None:
            first_tensor = next(iter(bindings.values()))
            device_id = first_tensor.device.index or 0
        self.runtime = ark.Runtime()
        self.runtime.launch(
            device_id=device_id, loop_mode=False, tensor_mappings=bindings
        )

    def run(self, bindings: Dict | None = None):
        """Run one decode token with optional placeholder rebinding."""

        if self.runtime is None:
            raise RuntimeError("runtime is not launched")
        if bindings is None:
            self.runtime.run(iter=1)
        else:
            self.runtime.run(iter=1, tensor_mappings=bindings)

    def stop(self):
        """Stop the ARK runtime if it is launched."""

        if self.runtime is not None and self.runtime.launched():
            self.runtime.stop()


def run_decode_sequence(
    config: KVCacheDecodeConfig,
    q_tokens: torch.Tensor,
    k_tokens: torch.Tensor,
    v_tokens: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    positions: Sequence[int],
    masks: Sequence[torch.Tensor] | None = None,
) -> torch.Tensor:
    """Run several decode positions through one compiled ARK graph."""

    if len(positions) == 0:
        raise ValueError("positions must be non-empty")
    steps = len(positions)
    device = q_tokens.device
    _check_tensor(
        "q_tokens",
        q_tokens,
        (steps, config.num_q_heads, config.head_dim),
        config.dtype,
        device,
    )
    _check_tensor(
        "k_tokens",
        k_tokens,
        (steps, config.num_kv_heads, config.head_dim),
        config.dtype,
        device,
    )
    _check_tensor(
        "v_tokens",
        v_tokens,
        (steps, config.num_kv_heads, config.head_dim),
        config.dtype,
        device,
    )
    outputs = torch.empty(
        (steps, config.num_q_heads, config.head_dim),
        dtype=config.dtype,
        device=device,
    )
    if masks is None:
        masks = [make_prefix_mask(config, pos, device) for pos in positions]
    elif len(masks) != steps:
        raise ValueError("masks length must match positions length")

    graph = KVCacheDecodeGraph(config)
    bindings = [
        graph.bindings(
            q_tokens[i],
            k_tokens[i],
            v_tokens[i],
            k_cache,
            v_cache,
            masks[i],
            outputs[i],
            positions[i],
        )
        for i in range(steps)
    ]

    torch.cuda.synchronize(device)
    graph.launch(bindings[0])
    try:
        graph.run()
        for binding in bindings[1:]:
            graph.run(binding)
    finally:
        graph.stop()
    return outputs


def make_positions(count: int, max_seq_len: int) -> List[int]:
    """Return *count* decode positions capped by *max_seq_len*."""

    if count <= 0:
        raise ValueError("count must be positive")
    if count > max_seq_len:
        raise ValueError("count cannot exceed max_seq_len")
    return list(range(count))


def clone_cpu_tensors(tensors: Iterable[torch.Tensor]) -> List[torch.Tensor]:
    """Detach and clone tensors on CPU for reference work after ARK stops."""

    return [tensor.detach().cpu().clone() for tensor in tensors]
