# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from ._ark_core import _TensorType, _FP32, _FP16, _INT32, _BYTE


class DataType:
    @staticmethod
    def from_numpy(np_type: np.dtype) -> "DataType":
        if np_type == np.float32:
            return fp32
        elif np_type == np.float16:
            return fp16
        elif np_type == np.int32:
            return int32
        elif np_type == np.uint8:
            return byte
        else:
            raise NotImplementedError

    @staticmethod
    def from_ttype(ttype: _TensorType) -> "DataType":
        if ttype == _FP32:
            return fp32
        elif ttype == _FP16:
            return fp16
        elif ttype == _INT32:
            return int32
        elif ttype == _BYTE:
            return byte
        else:
            raise NotImplementedError

    @staticmethod
    def from_str(type_str: str) -> "DataType":
        if type_str == "fp32":
            return fp32
        elif type_str == "fp16":
            return fp16
        elif type_str == "int32":
            return int32
        elif type_str == "byte":
            return byte
        else:
            raise NotImplementedError

    @staticmethod
    def to_numpy() -> np.dtype:
        ...

    @staticmethod
    def ttype() -> _TensorType:
        ...


class fp32(DataType):
    @staticmethod
    def to_numpy() -> np.float32:
        return np.float32

    @staticmethod
    def ttype() -> _TensorType:
        return _FP32


class fp16(DataType):
    @staticmethod
    def to_numpy() -> np.float16:
        return np.float16

    @staticmethod
    def ttype() -> _TensorType:
        return _FP16


class int32(DataType):
    @staticmethod
    def to_numpy() -> np.int32:
        return np.int32

    @staticmethod
    def ttype() -> _TensorType:
        return _INT32


class byte(DataType):
    @staticmethod
    def to_numpy() -> np.uint8:
        return np.uint8

    @staticmethod
    def ttype() -> _TensorType:
        return _BYTE
