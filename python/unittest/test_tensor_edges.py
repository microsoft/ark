# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for tensor.py edge/error branches."""

from common import ark, pytest_ark
import pytest


@pytest_ark()
def test_tensor_shape_strides_nelems_dtype():
    """Basic accessors on a fresh tensor."""
    t = ark.tensor([4, 64], ark.fp16)
    assert t.shape() == [4, 64]
    assert t.nelems() == 4 * 64
    assert t.dtype() == ark.fp16
    assert isinstance(t.strides(), list)


@pytest_ark()
def test_tensor_getitem_int_index():
    """Integer indexing returns a 1-element slice along indexed dim."""
    t = ark.tensor([4, 64], ark.fp16)
    s = t[2]
    # Only the indexed dimension is reflected in the result shape
    assert s.shape() == [1]


@pytest_ark()
def test_tensor_getitem_slice():
    """Slice indexing."""
    t = ark.tensor([8, 32], ark.fp16)
    s = t[2:6, :16]
    assert s.shape() == [4, 16]


@pytest_ark()
def test_tensor_getitem_negative_step():
    """Slice with step=-1."""
    t = ark.tensor([8, 32], ark.fp16)
    s = t[5:1:-1]
    # step -1 swaps start/stop: becomes [2:6] → shape [4]
    assert s.shape() == [4]


@pytest_ark()
def test_tensor_getitem_invalid_step():
    """Slice with step != 1 or -1 raises UnsupportedError."""
    t = ark.tensor([8, 32], ark.fp16)
    with pytest.raises(ark.UnsupportedError):
        t[::2]


@pytest_ark()
def test_tensor_getitem_too_many_dims():
    """Indexing with more dims than tensor raises InvalidUsageError."""
    t = ark.tensor([8, 32], ark.fp16)
    with pytest.raises(ark.InvalidUsageError):
        t[0, 0, 0]


@pytest_ark()
def test_tensor_getitem_invalid_type():
    """Indexing with non-int/non-slice raises InvalidUsageError."""
    t = ark.tensor([8, 32], ark.fp16)
    with pytest.raises(ark.InvalidUsageError):
        t["bad"]


@pytest_ark()
def test_tensor_hash_eq():
    """Tensor hash and equality."""
    t1 = ark.tensor([64], ark.fp16)
    t2 = ark.tensor([64], ark.fp16)
    # Same tensor object hashes/equals itself
    assert t1 == t1
    assert hash(t1) == hash(t1)
    # Different tensors are not equal
    assert t1 != t2


@pytest_ark()
def test_tensor_eq_non_tensor():
    """Tensor != non-Tensor object."""
    t = ark.tensor([64], ark.fp16)
    assert t != "not a tensor"
    assert t != 42


@pytest_ark()
def test_cpp_tensor_invalid_shape():
    """_cpp_tensor raises when shape is not list/tuple."""
    from ark.tensor import _cpp_tensor

    with pytest.raises(ark.InvalidUsageError):
        _cpp_tensor(shape=64)


@pytest_ark()
def test_cpp_tensor_invalid_strides():
    """_cpp_tensor raises when strides is not list/tuple."""
    from ark.tensor import _cpp_tensor

    with pytest.raises(ark.InvalidUsageError):
        _cpp_tensor(shape=[64], strides=64)


@pytest_ark()
def test_cpp_tensor_invalid_offsets():
    """_cpp_tensor raises when offsets is not list/tuple."""
    from ark.tensor import _cpp_tensor

    with pytest.raises(ark.InvalidUsageError):
        _cpp_tensor(shape=[64], offsets="bad")


@pytest_ark()
def test_cpp_tensor_invalid_padded_shape():
    """_cpp_tensor raises when padded_shape is not list/tuple."""
    from ark.tensor import _cpp_tensor

    with pytest.raises(ark.InvalidUsageError):
        _cpp_tensor(shape=[64], padded_shape=128)


@pytest_ark()
def test_cpp_tensor_exceeds_4d():
    """_cpp_tensor raises ValueError for > 4D shape."""
    from ark.tensor import _cpp_tensor

    with pytest.raises(ValueError):
        _cpp_tensor(shape=[1, 2, 3, 4, 5])


@pytest_ark()
def test_tensor_is_external():
    """Regular tensor is not external."""
    t = ark.tensor([64], ark.fp16)
    assert not t.is_external()


@pytest_ark()
def test_parameter_basic():
    """Parameter creation and attributes."""
    p = ark.parameter([32, 32], ark.fp32)
    assert isinstance(p, ark.Parameter)
    assert p.shape() == [32, 32]
    assert p.dtype() == ark.fp32
    assert p.requires_grad is False
