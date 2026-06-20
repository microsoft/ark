# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Equivalence tests for graph-reused ARK KV-cache decode."""

import os
import sys

import pytest
import torch

try:
    from .kv_cache_decode import (
        _SMALL_TEST_CONFIG,
        KVCacheDecodeConfig,
        check_decode_tensors,
        make_prefix_mask,
        run_decode_sequence,
        torch_gqa_decode_reference,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from kv_cache_decode import (  # type: ignore
        _SMALL_TEST_CONFIG,
        KVCacheDecodeConfig,
        check_decode_tensors,
        make_prefix_mask,
        run_decode_sequence,
        torch_gqa_decode_reference,
    )

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for ARK decode"
)


def _make_cpu_inputs(config, steps, seed=1234):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    q = (
        torch.randn(
            steps,
            config.num_q_heads,
            config.head_dim,
            generator=gen,
            dtype=torch.float32,
        )
        * 0.2
    ).to(config.dtype)
    k = (
        torch.randn(
            steps,
            config.num_kv_heads,
            config.head_dim,
            generator=gen,
            dtype=torch.float32,
        )
        * 0.2
    ).to(config.dtype)
    v = (
        torch.randn(
            steps,
            config.num_kv_heads,
            config.head_dim,
            generator=gen,
            dtype=torch.float32,
        )
        * 0.2
    ).to(config.dtype)
    return q, k, v


def _reference_sequence(config, q_tokens, k_tokens, v_tokens, positions):
    k_cache = torch.zeros(
        config.num_kv_heads,
        config.max_seq_len,
        config.head_dim,
        dtype=config.dtype,
    )
    v_cache = torch.zeros_like(k_cache)
    outputs = []
    for i, position in enumerate(positions):
        out, k_cache, v_cache = torch_gqa_decode_reference(
            config,
            q_tokens[i],
            k_tokens[i],
            v_tokens[i],
            k_cache,
            v_cache,
            position,
        )
        outputs.append(out)
    return torch.stack(outputs), k_cache, v_cache


def test_kv_cache_decode_two_positions_matches_torch_reference():
    """One compiled graph updates and reuses the same external cache buffers."""

    config = _SMALL_TEST_CONFIG
    positions = [0, 1]
    q_cpu, k_cpu, v_cpu = _make_cpu_inputs(config, len(positions))
    expected, expected_k_cache, expected_v_cache = _reference_sequence(
        config, q_cpu, k_cpu, v_cpu, positions
    )

    device = torch.device("cuda:0")
    q = q_cpu.to(device)
    k = k_cpu.to(device)
    v = v_cpu.to(device)
    k_cache = torch.zeros(
        config.num_kv_heads,
        config.max_seq_len,
        config.head_dim,
        dtype=config.dtype,
        device=device,
    )
    v_cache = torch.zeros_like(k_cache)
    masks = [
        make_prefix_mask(config, position, device) for position in positions
    ]

    outputs = run_decode_sequence(
        config, q, k, v, k_cache, v_cache, positions, masks=masks
    )

    outputs_cpu = outputs.detach().cpu()
    k_cache_cpu = k_cache.detach().cpu()
    v_cache_cpu = v_cache.detach().cpu()

    torch.testing.assert_close(outputs_cpu, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(k_cache_cpu, expected_k_cache, rtol=0, atol=0)
    torch.testing.assert_close(v_cache_cpu, expected_v_cache, rtol=0, atol=0)

    stale_k_cache = torch.zeros_like(expected_k_cache)
    stale_v_cache = torch.zeros_like(expected_v_cache)
    stale_out, _, _ = torch_gqa_decode_reference(
        config,
        q_cpu[1],
        k_cpu[1],
        v_cpu[1],
        stale_k_cache,
        stale_v_cache,
        position=1,
    )
    assert not torch.allclose(outputs_cpu[1], stale_out, rtol=2e-2, atol=2e-2)


def test_kv_cache_decode_representative_later_prefix():
    """The same graph handles a later active prefix through the mask binding."""

    config = _SMALL_TEST_CONFIG
    positions = [3, 4]
    q_cpu, k_cpu, v_cpu = _make_cpu_inputs(config, len(positions), seed=5678)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(9012)
    seed_k = (
        torch.randn(
            config.num_kv_heads,
            config.max_seq_len,
            config.head_dim,
            generator=gen,
            dtype=torch.float32,
        )
        * 0.2
    ).to(config.dtype)
    seed_v = (
        torch.randn(
            config.num_kv_heads,
            config.max_seq_len,
            config.head_dim,
            generator=gen,
            dtype=torch.float32,
        )
        * 0.2
    ).to(config.dtype)
    expected_k_cache = seed_k.clone()
    expected_v_cache = seed_v.clone()
    expected_outputs = []
    for i, position in enumerate(positions):
        out, expected_k_cache, expected_v_cache = torch_gqa_decode_reference(
            config,
            q_cpu[i],
            k_cpu[i],
            v_cpu[i],
            expected_k_cache,
            expected_v_cache,
            position,
        )
        expected_outputs.append(out)
    expected_outputs = torch.stack(expected_outputs)

    device = torch.device("cuda:0")
    q = q_cpu.to(device)
    k = k_cpu.to(device)
    v = v_cpu.to(device)
    k_cache = seed_k.to(device)
    v_cache = seed_v.to(device)
    masks = [
        make_prefix_mask(config, position, device) for position in positions
    ]

    outputs = run_decode_sequence(
        config, q, k, v, k_cache, v_cache, positions, masks=masks
    )

    torch.testing.assert_close(
        outputs.detach().cpu(), expected_outputs, rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        k_cache.detach().cpu(), expected_k_cache, rtol=0, atol=0
    )
    torch.testing.assert_close(
        v_cache.detach().cpu(), expected_v_cache, rtol=0, atol=0
    )


def test_kv_cache_decode_shape_dtype_checks():
    """Validation rejects non-GQA shapes and malformed runtime bindings."""

    with pytest.raises(ValueError, match="divisible"):
        KVCacheDecodeConfig(
            max_seq_len=8, num_q_heads=3, num_kv_heads=2, head_dim=64
        )

    config = _SMALL_TEST_CONFIG
    device = torch.device("cuda:0")
    q = torch.empty(
        config.num_q_heads, config.head_dim, dtype=config.dtype, device=device
    )
    k = torch.empty(
        config.num_kv_heads, config.head_dim, dtype=config.dtype, device=device
    )
    v = torch.empty_like(k)
    k_cache = torch.empty(
        config.num_kv_heads,
        config.max_seq_len,
        config.head_dim,
        dtype=config.dtype,
        device=device,
    )
    v_cache = torch.empty_like(k_cache)
    mask = torch.empty(1, config.max_seq_len, dtype=config.dtype, device=device)
    output = torch.empty_like(q)

    with pytest.raises(ValueError, match="position"):
        check_decode_tensors(
            config, q, k, v, k_cache, v_cache, mask, output, config.max_seq_len
        )

    bad_q = torch.empty(
        config.num_q_heads,
        config.head_dim + 1,
        dtype=config.dtype,
        device=device,
    )
    with pytest.raises(ValueError, match="q shape"):
        check_decode_tensors(
            config, bad_q, k, v, k_cache, v_cache, mask, output, 0
        )
