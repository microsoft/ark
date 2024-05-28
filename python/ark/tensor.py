# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from typing import Callable, List, Union, Type

from _ark_core import _Dims, _Tensor, _NullTensor
from .data_type import DataType
from .runtime import Runtime
from .model import Model

try:
    import torch

    _no_torch = False
except ImportError:
    from . import torch_mock as torch

    _no_torch = True

NullTensor = _NullTensor


class Dims(_Dims):
    pass


Initializer = Type[Callable[[], Union[torch.Tensor, np.ndarray]]]


class Tensor:
    def __init__(self, _tensor: _Tensor, initializer: Initializer = None):
        """
        Initializes a new instance of the Tensor class.
        Args:
            _tensor (_ark_core._Tensor): The underlying _Tensor object.
        """
        self._tensor = _tensor
        self.initializer: Initializer = initializer

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

    def to_torch(self, tensor: torch.Tensor = None) -> torch.Tensor:
        """ """
        if _no_torch:
            raise ImportError("torch is not available")
        torch_type = self.dtype().to_torch()
        if tensor is None:
            return torch.from_numpy(self.to_numpy())
        elif tensor.shape != self.shape():
            raise ValueError("torch tensor shape does not match the tensor")
        elif tensor.dtype != torch_type:
            raise ValueError("torch tensor dtype does not match the tensor")
        elif not tensor.is_contiguous():
            raise ValueError("torch tensor is not contiguous in memory")
        elif tensor.numel() != self.nelems():
            raise ValueError("torch tensor size does not match the tensor")
        tensor.copy_(torch.from_numpy(self.to_numpy()))
        return tensor

    @staticmethod
    def from_numpy(ndarray: np.ndarray):
        return Tensor(
            Model.get_model().tensor(
                Dims(list(ndarray.shape)),
                DataType.from_numpy(ndarray.dtype).ctype(),
                Dims(),
                Dims(),
                Dims(),
                "",
            ),
            lambda: ndarray,
        )

    @staticmethod
    def from_torch(tensor: torch.Tensor):
        return Tensor(
            Model.get_model().tensor(
                Dims(list(tensor.shape)),
                DataType.from_torch(tensor.dtype).ctype(),
                Dims(),
                Dims(),
                Dims(),
                "",
            ),
            lambda: tensor,
        )

    def copy(self, data: Union[np.ndarray, torch.Tensor]) -> "Tensor":
        """
        Copies the tensor from a host numpy array to the device.
        """
        rt = Runtime.get_runtime()
        if not rt.launched():
            raise RuntimeError(
                "Tensor is not allocated yet. `Tensor.from_numpy()` is "
                "usable only after you call `Runtime.launch()`."
            )
        if isinstance(data, torch.Tensor):
            if data.dtype != self.dtype().to_torch():
                raise ValueError("data dtype does not match the tensor")
            if not data.is_contiguous():
                data = data.contiguous()
            if data.numel() != self.nelems():
                raise ValueError("data size does not match the tensor")
            rt.executor.tensor_write(
                self._tensor,
                data.data_ptr(),
                data.numel() * data.element_size(),
            )
        elif isinstance(data, np.ndarray):
            if data.dtype != self.dtype().to_numpy():
                raise ValueError("data dtype does not match the tensor")
            if not data.flags["C_CONTIGUOUS"]:
                data = np.ascontiguousarray(data)
            if data.nbytes != self.nelems() * self.dtype().element_size():
                raise ValueError("data size does not match the tensor")
            rt.executor.tensor_write(self._tensor, data)
        else:
            raise ValueError("data must be a numpy array or a torch tensor")
        return self

    def initialize(self) -> "Tensor":
        """
        Initializes the tensor.
        """
        if self.initializer is not None:
            data = self.initializer()
            self.copy(data)
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
