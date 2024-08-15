# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from typing import Callable, List, Union, Type

from ._ark_core import _Dims, _Tensor, _NullTensor
from .data_type import DataType
from .runtime import Runtime

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
    def __init__(
        self,
        _tensor: _Tensor,
        initializer: Initializer = None,
        requires_grad: bool = False,
    ):
        """
        Initializes a new instance of the Tensor class.
        Args:
            _tensor (_ark_core._Tensor): The underlying _Tensor object.
            initializer (Initializer): The initializer for the Tensor.
            requires_grad (bool): Whether the tensor requires gradient. Defaults to True.
        """
        self._tensor = _tensor
        self.initializer: Initializer = initializer
        self.requires_grad = requires_grad
    
    def __hash__(self):
        return self._tensor.id()
    
    def __eq__(self, other):
        if not isinstance(other, Tensor):
            return False
        return self._tensor.id() == other._tensor.id()


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

    def to_numpy(
        self, ndarray: np.ndarray = None, stream: int = 0
    ) -> np.ndarray:
        """
        Copy a tensor from device to host. If `ndarray` is None,
        a new numpy array will be created. If the tensor is not allocated,
        an empty numpy array without the data buffer will be returned.
        """
        np_type = self.dtype().to_numpy()
        if np_type is None:
            raise ValueError(
                f"Tensor data type {self.dtype().__name__} is not supported by numpy."
            )
        rt = Runtime.get_runtime()
        if not rt.launched():
            raise RuntimeError(
                "Tensor is not allocated yet. `Tensor.to_numpy()` is "
                "usable only after you call `Runtime.launch()`."
            )
        elif ndarray is None:
            ndarray = np.zeros(self.shape(), dtype=np_type)
        elif not ndarray.flags["C_CONTIGUOUS"]:
            raise ValueError("ndarray is not contiguous in memory")
        elif ndarray.shape != self.shape():
            raise ValueError("ndarray shape does not match the tensor")
        elif ndarray.dtype != np_type:
            raise ValueError("ndarray dtype does not match the tensor")
        elif ndarray.nbytes != self.nelems() * self.dtype().element_size():
            raise ValueError("ndarray size does not match the tensor")
        rt.executor.tensor_read(self._tensor, ndarray, stream)
        return ndarray

    def from_numpy(self, ndarray: np.ndarray, stream: int = 0) -> "Tensor":
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
        rt.executor.tensor_write(self._tensor, ndarray, stream)
        return self

    def to_dlpack(self):
        """
        Returns a DLPack tensor that shares the same memory with the device tensor.
        """
        rt = Runtime.get_runtime()
        if not rt.launched():
            raise RuntimeError(
                "Tensor is not allocated yet. `Tensor.to_dlpack()` is "
                "usable only after you call `Runtime.launch()`."
            )
        return rt.executor.tensor_to_dlpack(self._tensor)

    def to_torch(self) -> torch.Tensor:
        """
        Returns a torch tensor that shares the same memory with the device tensor.
        """
        if _no_torch:
            raise ImportError("torch is not available")
        dl_capsule = self.to_dlpack()
        torch_view = torch.utils.dlpack.from_dlpack(dl_capsule)
        # Keep dl_capsule alive not to free the memory
        torch_view.__ark_buffer__ = dl_capsule
        return torch_view

    def copy(
        self, data: Union[np.ndarray, torch.Tensor], stream: int = 0
    ) -> "Tensor":
        """
        Copies data into this tensor. The data type may differ,
        but the size must match.
        """
        rt = Runtime.get_runtime()
        if not rt.launched():
            raise RuntimeError(
                "Tensor is not allocated yet. `Tensor.copy()` is "
                "usable only after you call `Runtime.launch()`."
            )
        tensor_bytes = self.nelems() * self.dtype().element_size()
        if isinstance(data, torch.Tensor):
            if not data.is_contiguous():
                data = data.contiguous()
            if data.numel() * data.element_size() != tensor_bytes:
                raise ValueError("data size does not match the tensor")
            rt.executor.tensor_write(
                self._tensor,
                data.data_ptr(),
                tensor_bytes,
                stream,
                data.device.type == "cuda",
            )
            data.requires_grad = self.requires_grad
            if isinstance(self, Parameter):
                self.torch_param = data
        elif isinstance(data, np.ndarray):
            if not data.flags["C_CONTIGUOUS"]:
                data = np.ascontiguousarray(data)
            if data.nbytes != tensor_bytes:
                raise ValueError("data size does not match the tensor")
            rt.executor.tensor_write(self._tensor, data, stream)
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

    def __init__(
        self,
        tensor: _Tensor,
        from_torch: bool,
    ):
        """
        Initializes a new instance of the Parameter class.
        Args:
            _tensor (_ark_core._Tensor): The underlying _Tensor object.
            from_torch: Indicates if the Parameter is tied to a torch.nn.Paramter
        """
        if not _no_torch and from_torch:
            _tensor = tensor._tensor
            self.torch_param = tensor
            self.staged_tensor = None
            Tensor.__init__(
                self,
                _tensor,
                requires_grad=tensor.requires_grad,
            )
        elif isinstance(tensor, _Tensor):
            _tensor = tensor
            self.torch_param = None
            self.staged_tensor = None
            Tensor.__init__(self, _tensor, requires_grad=False)
        else:
            raise TypeError(
                "tensor must be an ARK tensor or a torch.nn.Parameter"
            )

    def update_gradient(self, ark_tensor: Tensor):
        """
        Stages an ARK tensor to be used for updating the gradient of its associated parameter.
        """
        if _no_torch:
            raise ImportError("torch is not available")
        if self.torch_param is None:
            raise ValueError(
                "there is no PyTorch parameter associated with this ARK parameter"
            )
        if not self.torch_param.requires_grad:
            raise ValueError("parameter does not require gradient updates")
        if ark_tensor is None or not isinstance(ark_tensor, Tensor):
            raise ValueError("cannot use non-ARK tensor to update ARK gradient")
        self.staged_tensor = ark_tensor
