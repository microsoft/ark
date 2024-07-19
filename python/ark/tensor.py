# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from typing import Callable, List, Union, Type

from ._ark_core import _Dims, _Tensor, _NullTensor
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
    def __init__(
        self,
        _tensor: _Tensor,
        initializer: Initializer = None,
        runtime_id: int = -1,
    ):
        """
        Initializes a new instance of the Tensor class.
        Args:
            _tensor (_ark_core._Tensor): The underlying _Tensor object.
            intializer (Initializer): The initializer for the Tensor.
            runtime_id (int): The ID of the Runtime to use. Defaults to -1, which is the default Runtime.
        """
        self._tensor = _tensor
        self.initializer: Initializer = initializer
        self.runtime_id = runtime_id

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
        rt = Runtime.get_runtime(self.runtime_id)
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

    def to_torch(
        self, tensor: torch.Tensor = None, stream: int = 0
    ) -> torch.Tensor:
        """ """
        if _no_torch:
            raise ImportError("torch is not available")
        rt = Runtime.get_runtime(self.runtime_id)
        if not rt.launched():
            raise RuntimeError(
                "Tensor is not allocated yet. `Tensor.to_torch()` is "
                "usable only after you call `Runtime.launch()`."
            )
        torch_type = self.dtype().to_torch()
        if tensor is None:
            dev_name = f"cuda:{rt.executor.device_id()}"
            tensor = torch.zeros(
                self.shape(), dtype=torch_type, device=torch.device(dev_name)
            )
        elif list(tensor.shape) != self.shape():
            raise ValueError(
                f"torch tensor shape {list(tensor.shape)} "
                f"does not match the tensor {self.shape()}"
            )
        elif tensor.dtype != torch_type:
            raise ValueError(
                f"torch tensor dtype {tensor.dtype} "
                f"does not match the tensor {torch_type}"
            )
        elif not tensor.is_contiguous():
            raise ValueError("torch tensor is not contiguous in memory")
        elif tensor.numel() != self.nelems():
            raise ValueError(
                f"torch tensor size {tensor.numel()} "
                f"does not match the tensor {self.nelems()}"
            )
        tensor_bytes = self.nelems() * self.dtype().element_size()
        rt.executor.tensor_read(
            self._tensor, tensor.data_ptr(), tensor_bytes, stream, True
        )
        return tensor

    def get_torch_view(self) -> torch.Tensor:
        """
        Returns a torch tensor that shares the same memory with the device tensor.
        """
        if _no_torch:
            raise ImportError("torch is not available")
        rt = Runtime.get_runtime(self.runtime_id)
        if not rt.launched():
            raise RuntimeError(
                "Tensor is not allocated yet. `Tensor.get_torch_view()` is "
                "usable only after you call `Runtime.launch()`."
            )
        dl_tensor = rt.executor.get_dl_tensor(self._tensor)
        torch_view = torch.utils.dlpack.from_dlpack(dl_tensor)
        return torch_view

    def from_numpy(self, ndarray: np.ndarray, stream: int = 0) -> "Tensor":
        """
        Copies the tensor from a host numpy array to the device.
        """
        rt = Runtime.get_runtime(self.runtime_id)
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

    @staticmethod
    def from_torch(tensor: torch.Tensor, runtime_id: int = -1) -> "Tensor":
        """
        Returns an ARK tensor that shares the same memory with the torch tensor.
        """
        if _no_torch:
            raise ImportError("torch is not available")
        elif not tensor.is_contiguous():
            raise ValueError("Torch tensor must be contiguous.")
        elif tensor.device.type == "cpu":
            raise ValueError("Torch tensor must be on a device.")
        ark_dtype = DataType.from_torch(tensor.dtype)
        dl_capsule = torch.utils.dlpack.to_dlpack(tensor)
        ark_tensor = _Tensor(dl_capsule, ark_dtype.ctype())
        return Tensor(ark_tensor, runtime_id=runtime_id)

    def copy(self, data: Union[np.ndarray, torch.Tensor]) -> "Tensor":
        """
        Copies data into this tensor. The data type may differ,
        but the size must match.
        """
        rt = Runtime.get_runtime(self.runtime_id)
        if not rt.launched():
            raise RuntimeError(
                "Tensor is not allocated yet. `Tensor.from_numpy()` is "
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
                data.device.type == "cuda",
            )
        elif isinstance(data, np.ndarray):
            if not data.flags["C_CONTIGUOUS"]:
                data = np.ascontiguousarray(data)
            if data.nbytes != tensor_bytes:
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

    def __init__(
        self, tensor: Union[_Tensor, "torch.nn.Parameter"], runtime_id: int = -1
    ):
        """
        Initializes a new instance of the Parameter class.
        """
        if not _no_torch and isinstance(tensor, torch.nn.Parameter):
            ark_tensor = Tensor.from_torch(tensor)
            core_tensor = ark_tensor._tensor
            self.torch_param = tensor
            self.staged_tensor = None
        elif isinstance(tensor, _Tensor):
            core_tensor = tensor
            self.torch_param = None
        else:
            raise TypeError(
                "tensor must be an ARK tensor or a torch.nn.Parameter"
            )

        super().__init__(core_tensor, runtime_id=runtime_id)
        self.runtime_id = runtime_id

    def update_gradient(self, ark_tensor: Tensor):
        """
        Stages an ARK tensor to be used for updating the gradient of its associated PyTorch parameter.
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
