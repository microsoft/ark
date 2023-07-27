# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from ._ark_core import _Tensor
from .executor import Executor
import numpy as np

# A wrapper class for _Tensor that provides a more pythonic interface
class Tensor():
    def __init__(self, _tensor : _Tensor):
        self._tensor = _tensor
    def __init__(self, dims, tensor_type):
        pass

    def shape(self):
        pass
        # ark_dims = self._tensor.shape()
        # shape = []
        # for i in range(ark_dims.ndims()):
        #     shape.append(ark_dims[i])
        # return shape

    def to(self, device, np_tensor=None):
        if device == "cpu":
            return self.cpu()
        elif device == "cuda":
            if np_tensor == None:
                raise RuntimeError("Usage: tensor.to('cuda', np_tensor)")
            return self.initiate(np_tensor)
        else:
            raise RuntimeError("Unknown device: " + device, "Only cpu and cuda are supported")
    def cpu(self):
        np_array = np.empty(self.shape(), dtype=np.float16)
        return Executor.get_executor().tensor_memcpy_device_to_host(np_array, self._tensor)

    def initiate(self, np_tensor):
        return Executor.get_executor().tensor_memcpy_host_to_device(self._tensor, np_tensor)
