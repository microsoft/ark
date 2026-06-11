# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for data_type.py edge branches."""

from common import ark, pytest_ark
import pytest


@pytest_ark(need_torch=True)
def test_data_type_from_torch():
    """DataType.from_torch for known types."""
    import torch

    assert ark.DataType.from_torch(torch.float32) == ark.fp32
    assert ark.DataType.from_torch(torch.float16) == ark.fp16
    assert ark.DataType.from_torch(torch.bfloat16) == ark.bf16
    assert ark.DataType.from_torch(torch.int32) == ark.int32
    assert ark.DataType.from_torch(torch.int8) == ark.int8
    assert ark.DataType.from_torch(torch.uint8) == ark.uint8


@pytest_ark(need_torch=True)
def test_data_type_from_torch_unknown():
    """DataType.from_torch raises ValueError for unsupported type."""
    import torch

    with pytest.raises(ValueError):
        ark.DataType.from_torch(torch.float64)


@pytest_ark()
def test_data_type_bf16_torch_type():
    """bf16 to_torch returns bfloat16."""
    import torch

    assert ark.bf16.to_torch() == torch.bfloat16


@pytest_ark()
def test_data_type_bf16_numpy_none():
    """bf16 has no numpy equivalent."""
    assert ark.bf16.to_numpy() is None
