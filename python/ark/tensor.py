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

    def write(self, buf: np.ndarray):
        """
        Copy contiguous data from a host buffer to the given tensor's (possibly
        non-contiguous) data range.

        For example, say the tensor is a 2D float tensor with shape [2, 3],
        ldims [2, 4], offs [0, 0], and pads [1, 1], then the data in the host
        buffer is 0, 1, ..., 5. After writing, the data in the tensor will be:

            [[0, 1, 2, ?],
             [3, 4, 5, ?]]

        where ? means the original unmodified value.
        """
        if not isinstance(buf, np.ndarray):
            logging.error("buf is not a numpy array")
            raise TypeError("buf is not a numpy array")
        # Check if buf is contiguous is memory
        if not buf.flags["C_CONTIGUOUS"]:
            logging.debug(
                "Warning: buf is not contiguous in memory, copy to a contiguous array"
            )
            buf = np.ascontiguousarray(buf)
        self._tensor.write(buf)

    def read(self, buf: np.ndarray):
        """
        Copy (possibly non-contiguous) data from a tensor on GPU to a contiguous
        host buffer.

        The given number of bytes is copied, in order of appearance
        on the memory. This function assumes that `buf` is large enough to hold
        the data. For example, say the tensor is a 2D float tensor with shape
        [2, 3], ldims [2, 4], offs [0, 0], and pads [1, 1], then the data in the
        tensor is:

            [[0, 1, 2, 3],
             [4, 5, 6, 7]]

        After read, the data in the host buffer will be 0, 1, 2, 4, 5, 6.
        """
        if not isinstance(buf, np.ndarray):
            logging.error("buf is not a numpy array")
            raise TypeError("buf is not a numpy array")
        if not buf.flags["C_CONTIGUOUS"]:
            logging.error("buf is not contiguous in memory")
            raise ValueError("buf is not contiguous in memory")
        self._tensor.read(buf)

    def clear(self):
        self._tensor.clear()

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
        self.read(ndarray)
        return ndarray

    def from_numpy(self, ndarray: np.ndarray):
        """
        Copies the tensor from a host numpy array to the device.
        """
        if self.tensor_type == TensorType.FP32:
            ndarray = ndarray.astype(np.float32)
        elif self.tensor_type == TensorType.FP16:
            ndarray = ndarray.astype(np.float16)
        self.write(ndarray)
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
