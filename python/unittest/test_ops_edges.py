# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for ops.py error/edge branches not covered by test_ops.py."""

from common import ark, pytest_ark
import pytest


@pytest_ark()
def test_reshape_invalid_shape_type():
    """reshape raises InvalidUsageError when shape is not a list/tuple."""
    a = ark.tensor([64, 64], ark.fp16)
    with pytest.raises(ark.InvalidUsageError):
        ark.reshape(a, 64)


@pytest_ark()
def test_reshape_exceeds_4d():
    """reshape raises InvalidUsageError for > 4 dimensions."""
    a = ark.tensor([2, 2, 2, 2], ark.fp16)
    with pytest.raises(ark.InvalidUsageError):
        ark.reshape(a, [1, 2, 2, 2, 2])


@pytest_ark()
def test_transpose_invalid_perm_type():
    """transpose raises InvalidUsageError when perm is not a list/tuple."""
    a = ark.tensor([64, 32], ark.fp16)
    with pytest.raises(ark.InvalidUsageError):
        ark.transpose(a, 0)


@pytest_ark()
def test_transpose_exceeds_4d():
    """transpose raises InvalidUsageError for perm > 4 dimensions."""
    a = ark.tensor([2, 2, 2, 2], ark.fp16)
    with pytest.raises(ark.InvalidUsageError):
        ark.transpose(a, [0, 1, 2, 3, 4])


@pytest_ark()
def test_identity_invalid_dep():
    """identity raises InvalidUsageError if deps contain non-Tensor."""
    a = ark.tensor([64], ark.fp16)
    with pytest.raises(ark.InvalidUsageError):
        ark.identity(a, deps=["not_a_tensor"])


@pytest_ark()
def test_add_two_scalars_returns_float():
    """add with two float scalars returns float sum."""
    result = ark.add(2.5, 3.5)
    assert result == 6.0


@pytest_ark()
def test_add_two_scalars_with_output():
    """add with two float scalars and an output tensor uses copy."""
    out = ark.tensor([1], ark.fp32)
    result = ark.add(2.0, 3.0, output=out)
    assert result.shape() == [1]


@pytest_ark()
def test_ops_noop():
    """noop does not return a value."""
    a = ark.tensor([64], ark.fp16)
    result = ark.noop(a)
    assert result is None


@pytest_ark()
def test_ops_copy_scalar():
    """copy accepts a scalar value."""
    out = ark.copy(42.0)
    assert out.shape() == [1]


@pytest_ark()
def test_ops_softmax_shape():
    """softmax composite op produces correct shape."""
    a = ark.tensor([4, 64], ark.fp16)
    out = ark.softmax(a)
    assert out.shape() == [4, 64]


@pytest_ark()
def test_ops_layernorm_shape():
    """layernorm composite op produces correct shape."""
    a = ark.tensor([4, 64], ark.fp16)
    out = ark.layernorm(a)
    assert out.shape() == [4, 64]


@pytest_ark()
def test_ops_mean_alias():
    """mean is an alias for reduce_mean."""
    from ark.ops import mean

    a = ark.tensor([4, 64], ark.fp16)
    out = mean(a, axis=1, keepdims=True)
    assert out.shape() == [4, 1]


@pytest_ark()
def test_ops_ones_zeros():
    """ones and zeros produce correct shapes."""
    o = ark.ones([8, 8])
    assert o.shape() == [8, 8]
    z = ark.zeros([8, 8], ark.fp16)
    assert z.shape() == [8, 8]


@pytest_ark()
def test_ops_send_recv():
    """send/send_done/recv produce tensors (model-only, no actual comm)."""
    ark.set_world_size(2)
    a = ark.tensor([64], ark.fp16)
    s = ark.send(a, remote_rank=1, tag=0)
    assert s.shape() == [64]
    sd = ark.send_done(s)
    assert sd.shape() == [64]

    out = ark.tensor([64], ark.fp16)
    r = ark.recv(out, remote_rank=0, tag=0)
    assert r.shape() == [64]


@pytest_ark()
def test_ops_all_reduce():
    """all_reduce produces correct shape."""
    ark.set_world_size(2)
    a = ark.tensor([1024], ark.fp16)
    out = ark.all_reduce(a, rank=0, world_size=2)
    assert out.shape() == [1024]
