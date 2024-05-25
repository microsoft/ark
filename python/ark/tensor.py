# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from typing import List

from _ark_core import _Dims, _Tensor, _NullTensor
from .data_type import DataType
from .runtime import Runtime

NullTensor = _NullTensor


class Dims(_Dims):
    pass


class Tensor:
    def __init__(self, _tensor: _Tensor):
        """
        Initializes a new instance of the Tensor class.
        Args:
            _tensor (_ark_core._Tensor): The underlying _Tensor object.
        """
        self._tensor = _tensor

    def shape(self) -> List[int]:
        """
        Returns the shape of the tensor.
        """
        return self._tensor.shape().vector()

    def strides(self) -> List[int]:
        """
        Returns the strides of the tensor.
        """
        return self._tensor.strides().vector()

    def nelems(self) -> int:
        """
        Returns the number of elements in the tensor.
        """
        return self._tensor.shape().nelems()

    def dtype(self) -> DataType:
        """
        Returns the type of the tensor.
        """
        return DataType.from_ctype(self._tensor.data_type())

    def to_numpy(self, ndarray: np.ndarray = None) -> np.ndarray:
        """
        Copy a tensor from device to host. If `ndarray` is None,
        a new numpy array will be created. If the tensor is not allocated,
        an empty numpy array without the data buffer will be returned.
        """
        np_type = self.dtype().to_numpy()
        rt = Runtime.get_runtime()
        if not rt.launched():
            return np.ndarray(self.shape(), dtype=np_type, buffer=None)
        if ndarray is None:
            ndarray = np.zeros(self.shape(), dtype=np_type)
        elif not ndarray.flags["C_CONTIGUOUS"]:
            raise ValueError("ndarray is not contiguous in memory")
        elif ndarray.shape != self.shape():
            raise ValueError("ndarray shape does not match the tensor")
        elif ndarray.dtype != np_type:
            raise ValueError("ndarray dtype does not match the tensor")
        elif ndarray.nbytes != self.nelems() * self.dtype().element_size():
            raise ValueError("ndarray size does not match the tensor")
        rt.executor.tensor_read(self._tensor, ndarray)
        return ndarray

    def from_numpy(self, ndarray: np.ndarray) -> "Tensor":
        """
        Copies the tensor from a host numpy array to the device.
        """
        rt = Runtime.get_runtime()
        if not rt.launched():
            raise RuntimeError(
                "Tensor is not allocated yet. `Tensor.from_numpy()` is "
                "usable only after you call `Runtime.launch()`."
            )
        ndarray = ndarray.astype(self.dtype().to_numpy())
        if not ndarray.flags["C_CONTIGUOUS"]:
            ndarray = np.ascontiguousarray(ndarray)
        if ndarray.nbytes != self.nelems() * self.dtype().element_size():
            raise ValueError("ndarray size does not match the tensor")
        rt.executor.tensor_write(self._tensor, ndarray)
        return self


class Parameter(Tensor):
    """
    A tensor as a parameter.
    """

    def __init__(self, _tensor: _Tensor):
        """
        Initializes a new instance of the Parameter class.
        """
        super().__init__(_tensor)
