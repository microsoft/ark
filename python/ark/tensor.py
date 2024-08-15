# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from typing import Callable, Iterable, List, Union, Type

from ._ark_core import _Dims, _Tensor, _NullTensor
from .data_type import DataType, fp32
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

    @staticmethod
    def from_dlpack(ext_tensor) -> "Tensor":
        """
        Copies the tensor from a DLPack tensor to the device.
        """
        # return Tensor(_Tensor(ext_tensor))
        raise NotImplementedError("from_dlpack is not implemented yet")

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

    @staticmethod
    def from_torch(tensor: torch.Tensor) -> "Tensor":
        """
        Returns an ARK tensor that shares the same memory with the torch tensor.
        """
        if _no_torch:
            raise ImportError("torch is not available")
        elif not tensor.is_contiguous():
            raise ValueError("Torch tensor must be contiguous.")
        elif tensor.device.type == "cpu":
            raise ValueError("Torch tensor must be on a device.")
        # TODO: support strides and offsets
        ark_tensor = Tensor(
            _cpp_tensor(
                shape=list(tensor.shape),
                dtype=DataType.from_torch(tensor.dtype),
                data=tensor.data_ptr(),
            )
        )
        # Share ownership of the memory with the torch tensor
        ark_tensor.__torch_buffer__ = tensor
        return ark_tensor

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


class Parameter(Tensor, torch.nn.Parameter):
    """
    A tensor as a parameter.
    """

    def __init__(
        self,
        tensor: Union[_Tensor, "torch.nn.Parameter"],
    ):
        """
        Initializes a new instance of the Parameter class.
        """
        if not _no_torch and isinstance(tensor, torch.nn.Parameter):
            ark_tensor = Tensor.from_torch(tensor)
            core_tensor = ark_tensor._tensor
            self.torch_param = tensor
            self.staged_tensor = None
            Tensor.__init__(
                self,
                core_tensor,
                requires_grad=tensor.requires_grad,
            )
        elif isinstance(tensor, _Tensor):
            core_tensor = tensor
            self.torch_param = None
            self.staged_tensor = None
            Tensor.__init__(self, core_tensor, requires_grad=False)
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


def _is_list_or_tuple(obj):
    return isinstance(obj, list) or isinstance(obj, tuple)


def _cpp_tensor(
    shape: Iterable[int],
    dtype: DataType = fp32,
    strides: Iterable[int] = [],
    offsets: Iterable[int] = [],
    padded_shape: Iterable[int] = [],
    rank: int = -1,
    data: int = None,
    name: str = "",
) -> Tensor:
    if not _is_list_or_tuple(shape):
        raise ValueError("shape should be a list or tuple of integers")
    if not _is_list_or_tuple(strides):
        raise ValueError("strides should be a list or tuple of integers")
    if not _is_list_or_tuple(offsets):
        raise ValueError("offsets should be a list or tuple of integers")
    if not _is_list_or_tuple(padded_shape):
        raise ValueError("padded_shape should be a list or tuple of integers")
    # only support tensors with up to 4 dimensions
    if (
        len(shape) > 4
        or len(strides) > 4
        or len(offsets) > 4
        or len(padded_shape) > 4
    ):
        raise ValueError("Only support tensors with up to 4 dimensions")
    if data is not None:
        cpp_tensor = Model.get_model().placeholder(
            Dims(shape),
            dtype.ctype(),
            Dims(strides),
            Dims(offsets),
            Dims(padded_shape),
            rank,
            data,
            name,
        )
    else:
        cpp_tensor = Model.get_model().tensor(
            Dims(shape),
            dtype.ctype(),
            Dims(strides),
            Dims(offsets),
            Dims(padded_shape),
            rank,
            name,
        )
    return cpp_tensor
