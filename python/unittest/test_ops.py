# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json

import pytest

from common import ark, pytest_ark


@pytest_ark()
def test_ops_add():
    a = ark.tensor([64, 64], ark.fp16)
    b = ark.tensor([64, 64], ark.fp16)
    c = ark.add(a, b)
    assert c.shape() == [64, 64]

    d = ark.add(a, 1.0)
    assert d.shape() == [64, 64]

    e = ark.add(1.0, a)
    assert e.shape() == [64, 64]

    f = ark.add(1.0, 1.0)
    assert f == 2.0


@pytest_ark()
def test_ops_cast():
    a = ark.tensor([64, 64], ark.fp16)
    b = ark.cast(a, ark.fp32)
    assert b.shape() == [64, 64]
    assert b.dtype() == ark.fp32


@pytest_ark()
def test_ops_constant():
    a = ark.constant(1.0, [64, 64])
    assert a.shape() == [64, 64]


@pytest_ark()
def test_ops_copy():
    a = ark.tensor([64, 64], ark.fp16)
    b = ark.copy(a)
    assert b.shape() == [64, 64]


@pytest_ark()
def test_ops_div():
    a = ark.tensor([64, 64], ark.fp16)
    b = ark.tensor([64, 64], ark.fp16)
    c = ark.div(a, b)
    assert c.shape() == [64, 64]

    d = ark.div(a, 1.0)
    assert d.shape() == [64, 64]


@pytest_ark()
def test_ops_embedding():
    a = ark.tensor([64, 64], ark.int32)
    b = ark.tensor([100, 4096], ark.fp16)
    c = ark.embedding(a, b)
    assert c.shape() == [64, 64, 4096]


@pytest_ark()
def test_ops_exp():
    a = ark.tensor([64, 64], ark.fp16)
    b = ark.exp(a)
    assert b.shape() == [64, 64]


@pytest_ark()
def test_ops_gelu():
    a = ark.tensor([64, 64], ark.fp16)
    b = ark.gelu(a)
    assert b.shape() == [64, 64]


@pytest_ark()
def test_ops_identity():
    a = ark.tensor([64, 64], ark.fp16)
    b = ark.identity(a)
    assert b.shape() == [64, 64]


@pytest_ark()
def test_ops_matmul():
    a = ark.tensor([64, 64], ark.fp16)
    b = ark.tensor([64, 64], ark.fp16)
    c = ark.matmul(a, b)
    assert c.shape() == [64, 64]


@pytest_ark()
def test_ops_mul():
    a = ark.tensor([64, 64], ark.fp16)
    b = ark.tensor([64, 64], ark.fp16)
    c = ark.mul(a, b)
    assert c.shape() == [64, 64]

    d = ark.mul(a, 1.0)
    assert d.shape() == [64, 64]


@pytest_ark()
def test_ops_reduce_max():
    a = ark.tensor([64, 32], ark.fp16)
    b = ark.reduce_max(a, axis=0, keepdims=True)
    assert b.shape() == [1, 32]

    c = ark.reduce_max(a, axis=1, keepdims=True)
    assert c.shape() == [64, 1]

    d = ark.reduce_max(a, axis=0, keepdims=False)
    assert d.shape() == [32]

    e = ark.reduce_max(a, axis=1, keepdims=False)
    assert e.shape() == [64]


@pytest_ark()
def test_ops_reduce_mean():
    a = ark.tensor([64, 32], ark.fp16)
    b = ark.reduce_mean(a, axis=0, keepdims=True)
    assert b.shape() == [1, 32]

    c = ark.reduce_mean(a, axis=1, keepdims=True)
    assert c.shape() == [64, 1]

    d = ark.reduce_mean(a, axis=0, keepdims=False)
    assert d.shape() == [32]

    e = ark.reduce_mean(a, axis=1, keepdims=False)
    assert e.shape() == [64]


@pytest_ark()
def test_ops_reduce_sum():
    a = ark.tensor([64, 32], ark.fp16)
    b = ark.reduce_sum(a, axis=0, keepdims=True)
    assert b.shape() == [1, 32]

    c = ark.reduce_sum(a, axis=1, keepdims=True)
    assert c.shape() == [64, 1]

    d = ark.reduce_sum(a, axis=0, keepdims=False)
    assert d.shape() == [32]

    e = ark.reduce_sum(a, axis=1, keepdims=False)
    assert e.shape() == [64]


@pytest_ark()
def test_ops_relu():
    a = ark.tensor([64, 64], ark.fp16)
    b = ark.relu(a)
    assert b.shape() == [64, 64]


@pytest_ark()
def test_ops_reshape():
    a = ark.tensor([64, 64], ark.fp16)
    b = ark.reshape(a, [64, 64, 1])
    assert b.shape() == [64, 64, 1]


@pytest_ark()
def test_ops_rope():
    a = ark.tensor([64, 64], ark.fp16)
    b = ark.tensor([64, 64], ark.fp16)
    c = ark.rope(a, b)
    assert c.shape() == [64, 64]


@pytest_ark()
def test_ops_rsqrt():
    a = ark.tensor([64, 64], ark.fp16)
    b = ark.rsqrt(a)
    assert b.shape() == [64, 64]


@pytest_ark()
def test_ops_sharding():
    a = ark.tensor([64, 64], ark.fp16)
    b = ark.sharding(a, axis=0, dim_per_shard=2)
    assert len(b) == 32
    assert b[0].shape() == [2, 64]


@pytest_ark()
def test_ops_sigmoid():
    a = ark.tensor([64, 64], ark.fp16)
    b = ark.sigmoid(a)
    assert b.shape() == [64, 64]


@pytest_ark()
def test_ops_sqrt():
    a = ark.tensor([64, 64], ark.fp16)
    b = ark.sqrt(a)
    assert b.shape() == [64, 64]


@pytest_ark()
def test_ops_sub():
    a = ark.tensor([64, 64], ark.fp16)
    b = ark.tensor([64, 64], ark.fp16)
    c = ark.sub(a, b)
    assert c.shape() == [64, 64]

    d = ark.sub(a, 1.0)
    assert d.shape() == [64, 64]


@pytest_ark()
def test_ops_transpose():
    a = ark.tensor([64, 32], ark.fp16)
    b = ark.transpose(a, [1, 0])
    assert b.shape() == [32, 64]


def _op_names(op_type):
    model = ark.Model.get_model().compress()
    data = json.loads(model.serialize())
    return [
        node["Op"]["Name"]
        for node in data["Nodes"]
        if node["Op"]["Type"] == op_type
    ]


@pytest_ark()
def test_ops_all_reduce_packet():
    ark.set_world_size(2)
    a = ark.tensor([1024], ark.fp16)
    b = ark.all_reduce_packet(a, rank=0, world_size=2)
    assert b.shape() == [1024]
    assert "all_reduce_packet" in _op_names("AllReducePacketFused")


@pytest_ark()
def test_ops_all_reduce_qwen3_decode_routes_to_packet():
    ark.set_world_size(2)
    a = ark.tensor([1, 4096], ark.fp16)
    b = ark.all_reduce(a, rank=0, world_size=2)
    assert b.shape() == [1, 4096]
    names = _op_names("AllReducePacketFused")
    assert "all_reduce_packet" in names
    assert "all_reduce_prefill" not in names


@pytest_ark()
def test_ops_all_reduce_qwen3_prefill_routes_to_prefill():
    ark.set_world_size(2)
    a = ark.tensor([2048, 4096], ark.fp16)
    b = ark.all_reduce(a, rank=0, world_size=2)
    assert b.shape() == [2048, 4096]
    assert "all_reduce_prefill" in _op_names("AllReducePacketFused")


@pytest_ark()
def test_ops_all_reduce_qwen3_rank1_prefill_routes_to_prefill():
    ark.set_rank(1)
    ark.set_world_size(2)
    a = ark.tensor([2048, 4096], ark.fp16)
    b = ark.all_reduce(a, rank=1, world_size=2)
    assert b.shape() == [2048, 4096]
    assert "all_reduce_prefill" in _op_names("AllReducePacketFused")


@pytest_ark()
def test_ops_all_reduce_prefill_api_and_rejection():
    ark.set_world_size(2)
    a = ark.tensor([2048 * 4096], ark.fp16)
    b = ark.all_reduce_prefill(a, rank=0, world_size=2, output=a)
    assert b.shape() == [2048 * 4096]
    assert "all_reduce_prefill" in _op_names("AllReducePacketFused")

    with pytest.raises(ark.ModelError):
        ark.all_reduce_prefill(
            ark.tensor([4097], ark.fp16), rank=0, world_size=2
        )


@pytest_ark()
def test_ops_all_reduce_unsupported_shape_falls_back_to_ring():
    ark.set_world_size(2)
    a = ark.tensor([4097], ark.fp16)
    b = ark.all_reduce(a, rank=0, world_size=2)
    assert b.shape() == [4097]
    assert _op_names("AllReducePacketFused") == []
    assert _op_names("Send")
    assert _op_names("Recv")
