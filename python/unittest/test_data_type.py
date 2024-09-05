# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from common import ark, pytest_ark
import pytest
import numpy as np


@pytest_ark()
def test_data_type_from_numpy():
    assert ark.DataType.from_numpy(np.dtype(np.float32)) == ark.fp32
    assert ark.DataType.from_numpy(np.dtype(np.float16)) == ark.fp16
    assert ark.DataType.from_numpy(np.dtype(np.int32)) == ark.int32
    assert ark.DataType.from_numpy(np.dtype(np.uint32)) == ark.uint32
    assert ark.DataType.from_numpy(np.dtype(np.int8)) == ark.int8
    assert ark.DataType.from_numpy(np.dtype(np.uint8)) == ark.uint8

    with pytest.raises(ark.error.InvalidUsageError):
        ark.DataType.from_numpy(None)


@pytest_ark()
def test_data_type_from_name():
    assert ark.DataType.from_name("fp32") == ark.fp32
    assert ark.DataType.from_name("fp16") == ark.fp16
    assert ark.DataType.from_name("int32") == ark.int32
    assert ark.DataType.from_name("uint32") == ark.uint32
    assert ark.DataType.from_name("int8") == ark.int8
    assert ark.DataType.from_name("uint8") == ark.uint8

    with pytest.raises(ark.error.InvalidUsageError):
        ark.DataType.from_name("unknown")


@pytest_ark()
def test_data_type_from_ctype():
    assert ark.DataType.from_ctype(ark.core.FP32) == ark.fp32
    assert ark.DataType.from_ctype(ark.core.FP16) == ark.fp16
    assert ark.DataType.from_ctype(ark.core.INT32) == ark.int32
    assert ark.DataType.from_ctype(ark.core.UINT32) == ark.uint32
    assert ark.DataType.from_ctype(ark.core.INT8) == ark.int8
    assert ark.DataType.from_ctype(ark.core.UINT8) == ark.uint8

    with pytest.raises(ark.error.InvalidUsageError):
        ark.DataType.from_ctype(None)


@pytest_ark()
def test_data_type_to_numpy():
    assert ark.fp32.to_numpy() == np.float32
    assert ark.fp16.to_numpy() == np.float16
    assert ark.int32.to_numpy() == np.int32
    assert ark.uint32.to_numpy() == np.uint32
    assert ark.int8.to_numpy() == np.int8
    assert ark.uint8.to_numpy() == np.uint8


@pytest_ark()
def test_data_type_ctype():
    assert ark.fp32.ctype() == ark.core.FP32
    assert ark.fp16.ctype() == ark.core.FP16
    assert ark.int32.ctype() == ark.core.INT32
    assert ark.uint32.ctype() == ark.core.UINT32
    assert ark.int8.ctype() == ark.core.INT8
    assert ark.uint8.ctype() == ark.core.UINT8


@pytest_ark()
def test_data_type_element_size():
    assert ark.fp32.element_size() == 4
    assert ark.fp16.element_size() == 2
    assert ark.int32.element_size() == 4
    assert ark.uint32.element_size() == 4
    assert ark.int8.element_size() == 1
    assert ark.uint8.element_size() == 1
