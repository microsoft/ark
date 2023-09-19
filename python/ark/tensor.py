# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from typing import List

from ._ark_core import _Dims, _Tensor, _TensorBuf
from .data_type import DataType


Dims = _Dims
TensorBuf = _TensorBuf


class Tensor:
    def __init__(self, _tensor: _Tensor):
        """
        Initializes a new instance of the Tensor class.
        Args:
            _tensor (_ark_core._Tensor): The underlying _Tensor object.
        """
        self._tensor = _tensor
        self.is_parameter = False

    def shape(self) -> List[int]:
        """
        Returns the shape of the tensor.
        """
        return self._tensor.shape

    def ldims(self) -> List[int]:
        """
        Returns the ldims of the tensor.
        """
        return self._tensor.ldims

    def offset(self, i0=0, i1=0, i2=0, i3=0) -> int:
        """
        Returns the offset of the tensor at the specified indices.
        """
        return self._tensor.offset(i0, i1, i2, i3)

    def size(self) -> int:
        """
        Returns the number of elements in the tensor excluding padding.
        """
        return self._tensor.size()

    def ndims(self) -> int:
        """
        Returns the number of dimensions in the tensor.
        """
        return self._tensor.ndims()

    def type_bytes(self) -> int:
        """
        Returns the number of bytes of each element in the tensor.
        """
        return self._tensor.type_bytes()

    def shape_bytes(self) -> int:
        """
        Returns the number of bytes of the tensor.
        """
        return self._tensor.shape_bytes()

    def ldims_bytes(self) -> int:
        """
        Returns the number of bytes of the TensorBuf.
        """
        return self._tensor.ldims_bytes()

    def offset_bytes(self, i0=0, i1=0, i2=0, i3=0) -> int:
        """
        Returns the offset of the tensor at the specified indices in bytes.
        """
        return self._tensor.offset_bytes(i0, i1, i2, i3)

    def dtype(self) -> DataType:
        """
        Returns the type of the tensor.
        """
        return DataType.from_ttype(self._tensor.type)

    def clear(self):
        self._tensor.clear()

    def to_numpy(self, ndarray: np.ndarray = None) -> np.ndarray:
        """
        Copy a tensor from device to host. If `ndarray` is None,
        a new numpy array will be created. If the tensor is not allocated,
        an empty numpy array without the data buffer will be returned.
        """
        np_type = self.dtype().to_numpy()
        if not self._tensor.is_alloced():
            return np.ndarray(self.shape(), dtype=np_type, buffer=None)
        if ndarray is None:
            ndarray = np.zeros(self.shape(), dtype=np_type)
        elif not ndarray.flags["C_CONTIGUOUS"]:
            raise ValueError("ndarray is not contiguous in memory")
        elif ndarray.shape != self.shape():
            raise ValueError("ndarray shape does not match the tensor")
        elif ndarray.dtype != np_type:
            raise ValueError("ndarray dtype does not match the tensor")
        self._tensor.read(ndarray)
        return ndarray

    def from_numpy(self, ndarray: np.ndarray) -> "Tensor":
        """
        Copies the tensor from a host numpy array to the device.
        """
        if not self._tensor.is_alloced():
            raise RuntimeError(
                "Tensor is not allocated yet. `Tensor.from_numpy()` is "
                "usable only after you call `Runtime.launch()`."
            )
        ndarray = ndarray.astype(self.dtype().to_numpy())
        if not ndarray.flags["C_CONTIGUOUS"]:
            ndarray = np.ascontiguousarray(ndarray)
        self._tensor.write(ndarray)
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
