# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from typing import Callable, Iterable, List, Union, Type

from . import log
from .core import CoreDims, CoreTensor, NullTensor
from .torch import torch, _no_torch
from .data_type import DataType, fp32
from .executor import Executor
from .model import Model

__all__ = ["Dims", "Tensor", "Parameter", "NullTensor"]


class Dims(CoreDims):
    pass


Initializer = Type[Callable[[], Union[torch.Tensor, np.ndarray]]]


class Tensor:
    def __init__(
        self,
        _tensor: CoreTensor,
        initializer: Initializer = None,
        requires_grad: bool = False,
    ):
        """
        Initializes a new instance of the Tensor class.
        Args:
            _tensor (core.CoreTensor): The underlying _Tensor object.
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

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        new_args = []
        for arg in args:
            if isinstance(arg, Tensor):
                new_args.append(Tensor.to_torch(arg))
            else:
                new_args.append(arg)
        new_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, Tensor):
                new_kwargs[key] = Tensor.to_torch(value)
            else:
                new_kwargs[key] = value
        return func(*new_args, **new_kwargs)

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

    def data_ptr(self) -> int:
        """
        Returns the underlying data pointer.
        """
        return Executor.get().tensor_address(self._tensor)

    def is_external(self) -> bool:
        """
        Returns true if the tensor's data is not managed by ARK.
        """
        return self._tensor.is_external()

    def _raise_if_no_data(self):
        if self.data_ptr() != 0:
            return
        if self.is_external():
            raise log.InvalidUsageError(
                "Tried to access data of an external tensor that does not "
                "have data set. This is likely because this tensor is a "
                "placeholder and you have not set the data."
            )
        raise log.InvalidUsageError(
            "Tried to access data of a tensor that is not allocated yet. "
            "This is likely due to either you have not called "
            "`Runtime.launch()` for the model or the tensor is unused "
            "in the model."
        )

    def to_numpy(
        self, ndarray: np.ndarray = None, stream: int = 0
    ) -> np.ndarray:
        """
        Copy a tensor from device to host. If `ndarray` is None,
        a new numpy array will be created. If the tensor is not allocated,
        an empty numpy array without the data buffer will be returned.
        """
        self._raise_if_no_data()
        np_type = self.dtype().to_numpy()
        if np_type is None:
            raise log.InvalidUsageError(
                f"Tensor data type {self.dtype().__name__} is not supported by numpy."
            )
        if ndarray is None:
            ndarray = np.zeros(self.shape(), dtype=np_type)
        elif not ndarray.flags["C_CONTIGUOUS"]:
            raise log.InvalidUsageError("ndarray is not contiguous in memory")
        elif ndarray.shape != self.shape():
            raise log.InvalidUsageError(
                "ndarray shape does not match the tensor"
            )
        elif ndarray.dtype != np_type:
            raise log.InvalidUsageError(
                "ndarray dtype does not match the tensor"
            )
        elif ndarray.nbytes != self.nelems() * self.dtype().element_size():
            raise log.InvalidUsageError(
                "ndarray size does not match the tensor"
            )
        Executor.get().tensor_read(self._tensor, ndarray, stream)
        return ndarray

    def from_numpy(self, ndarray: np.ndarray, stream: int = 0) -> "Tensor":
        """
        Copies the tensor from a host numpy array to the device.
        """
        self._raise_if_no_data()
        ndarray = ndarray.astype(self.dtype().to_numpy())
        if not ndarray.flags["C_CONTIGUOUS"]:
            ndarray = np.ascontiguousarray(ndarray)
        if ndarray.nbytes != self.nelems() * self.dtype().element_size():
            raise log.InvalidUsageError(
                "ndarray size does not match the tensor"
            )
        Executor.get().tensor_write(self._tensor, ndarray, stream)
        return self

    def to_dlpack(self):
        """
        Returns a DLPack tensor that shares the same memory with the device tensor.
        """
        self._raise_if_no_data()
        return Executor.get().tensor_to_dlpack(self._tensor)

    @staticmethod
    def from_dlpack(ext_tensor) -> "Tensor":
        """
        Copies the tensor from a DLPack tensor to the device.
        """
        raise log.UnsupportedError("from_dlpack is not implemented yet")

    def to_torch(self) -> torch.Tensor:
        """
        Returns a torch tensor that shares the same memory with the device tensor.
        """
        if _no_torch:
            raise log.SystemError("torch is not available")
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
            raise log.SystemError("torch is not available")
        elif not tensor.is_contiguous():
            raise log.InvalidUsageError("Torch tensor must be contiguous.")
        elif tensor.device.type == "cpu":
            raise log.InvalidUsageError("Torch tensor must be on a device.")
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
        self._raise_if_no_data()
        tensor_bytes = self.nelems() * self.dtype().element_size()
        if isinstance(data, torch.Tensor):
            if not data.is_contiguous():
                data = data.contiguous()
            if data.numel() * data.element_size() != tensor_bytes:
                raise log.InvalidUsageError(
                    "data size does not match the tensor"
                )
            Executor.get().tensor_write(
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
                raise log.InvalidUsageError(
                    "data size does not match the tensor"
                )
            Executor.get().tensor_write(self._tensor, data, stream)
        else:
            raise log.InvalidUsageError(
                "data must be a numpy array or a torch tensor"
            )
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
        tensor: CoreTensor,
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
        elif isinstance(tensor, CoreTensor):
            _tensor = tensor
            self.torch_param = None
            self.staged_tensor = None
            Tensor.__init__(self, _tensor, requires_grad=False)
        else:
            raise log.InvalidUsageError(
                "tensor must be an ARK tensor or a torch.nn.Parameter"
            )

    def update_gradient(self, ark_tensor: Tensor):
        """
        Stages an ARK tensor to be used for updating the gradient of its associated parameter.
        """
        if _no_torch:
            raise log.SystemError("torch is not available")
        if self.torch_param is None:
            raise log.InvalidUsageError(
                "there is no PyTorch parameter associated with this ARK parameter"
            )
        if not self.torch_param.requires_grad:
            raise log.InvalidUsageError(
                "parameter does not require gradient updates"
            )
        if ark_tensor is None or not isinstance(ark_tensor, Tensor):
            raise log.InvalidUsageError(
                "cannot use non-ARK tensor to update ARK gradient"
            )
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
        raise log.InvalidUsageError(
            "shape should be a list or tuple of integers"
        )
    if not _is_list_or_tuple(strides):
        raise log.InvalidUsageError(
            "strides should be a list or tuple of integers"
        )
    if not _is_list_or_tuple(offsets):
        raise log.InvalidUsageError(
            "offsets should be a list or tuple of integers"
        )
    if not _is_list_or_tuple(padded_shape):
        raise log.InvalidUsageError(
            "padded_shape should be a list or tuple of integers"
        )
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
