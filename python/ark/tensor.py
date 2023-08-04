# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from ._ark_core import _Tensor, TensorType
from . import executor
import numpy as np
import logging


class Tensor:
    def __init__(self, _tensor: _Tensor):
        """
        Initializes a new instance of the Tensor class.
        Args:
            _tensor (_ark_core._Tensor): The underlying _Tensor object.
        """
        self._tensor = _tensor
        self.shape = _tensor.shape
        self.is_parameter = False

    def offset(self, i0=0, i1=0, i2=0, i3=0):
        """
        Returns the offset of the tensor at the specified indices.
        """
        return self._tensor.offset(i0, i1, i2, i3)

    def size(self):
        """
        Returns the number of elements in the tensor excluding padding.
        """
        return self._tensor.size()

    def ndims(self):
        """
        Returns the number of dimensions in the tensor.
        """
        return self._tensor.ndims()

    def padded_shape(self):
        """
        Returns the shape of the tensor including padding.
        """
        return self._tensor.padded_shape()

    def type_bytes(self):
        """
        Returns the number of bytes of each element in the tensor.
        """
        return self._tensor.type_bytes()

    def shape_bytes(self):
        """
        Returns the number of bytes of the tensor.
        """
        return self._tensor.shape_bytes()

    def ldims_bytes(self):
        """
        Returns the number of bytes of the TensorBuf.
        """
        return self._tensor.ldims_bytes()

    def offset_bytes(self, i0=0, i1=0, i2=0, i3=0):
        """
        Returns the offset of the tensor at the specified indices in bytes.
        """
        return self._tensor.offset_bytes(i0, i1, i2, i3)

    def tensor_type(self):
        """
        Returns the type of the tensor.
        """
        return self._tensor.type

    def to_numpy(self, ndarray: np.ndarray = None):
        """
        Copy a tensor from device to host. If dst is None, a new numpy array will be created.
        """
        # Create a new numpy array if dst is None
        if ndarray is None:
            np_type = None
            if self.tensor_type() == TensorType.FP32:
                np_type = np.float32
            elif self.tensor_type() == TensorType.FP16:
                np_type = np.float16
            else:
                logging.error("Unsupported tensor type")
                raise TypeError("Unsupported tensor type")
            ndarray = np.empty(self.shape, dtype=np_type)
        executor.Executor.get_global_executor().tensor_memcpy_device_to_host(
            ndarray, self._tensor
        )
        return ndarray

    def from_numpy(self, ndarray: np.ndarray):
        """
        Copies the tensor from a host numpy array to the device.
        """
        executor.Executor.get_global_executor().tensor_memcpy_host_to_device(
            self._tensor, ndarray
        )
        return self


def Parameter(
    tensor: Tensor,
) -> Tensor:
    """
    Set the tensor as a parameter.

    Args:
        tensor (Tensor): The tensor to set as a parameter.

    Returns:
        Tensor: The input tensor marked as a parameter.
    """
    tensor.is_parameter = True
    return tensor


FP16 = TensorType.FP16
FP32 = TensorType.FP32
INT32 = TensorType.INT32
BYTE = TensorType.BYTE
