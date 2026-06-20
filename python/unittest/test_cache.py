# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Model-level tests for fixed-layout KV-cache slot primitives."""

import pytest

import ark


def test_kv_cache_slot_constructs_fixed_layout():
    """kv_cache_slot returns one slot-shaped tensor for a fixed cache."""
    ark.init()
    cache = ark.placeholder([4, 2, 3], ark.fp16)
    token = ark.tensor([2, 3], ark.fp16)
    position = ark.placeholder([1], ark.int32)

    slot = ark.kv_cache_slot(cache, token, position)

    assert slot.shape() == [2, 3]
    assert slot.dtype() == ark.fp16


def test_kv_cache_slot_rejects_token_shape_mismatch():
    """The current-token tensor must match one cache slot exactly."""
    ark.init()
    cache = ark.placeholder([4, 2, 3], ark.fp16)
    token = ark.tensor([2, 4], ark.fp16)
    position = ark.placeholder([1], ark.int32)

    with pytest.raises(ark.ModelError):
        ark.kv_cache_slot(cache, token, position)


def test_kv_cache_slot_rejects_non_contiguous_cache():
    """Only contiguous [max_seq, ...slot_shape] external cache is supported."""
    ark.init()
    cache = ark.placeholder(
        [4, 2, 3], ark.fp16, strides=[5, 2, 3], padded_shape=[4, 2, 3]
    )
    token = ark.tensor([2, 3], ark.fp16)
    position = ark.placeholder([1], ark.int32)

    with pytest.raises(ark.ModelError):
        ark.kv_cache_slot(cache, token, position)


def test_kv_cache_slot_rejects_non_int32_position():
    """Runtime-selected position state is an external INT32 scalar."""
    ark.init()
    cache = ark.placeholder([4, 2, 3], ark.fp16)
    token = ark.tensor([2, 3], ark.fp16)
    position = ark.placeholder([1], ark.fp32)

    with pytest.raises(ark.ModelError):
        ark.kv_cache_slot(cache, token, position)
