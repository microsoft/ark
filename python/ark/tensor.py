# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from ._ark_core import _Tensor
from .executor import Executor
import numpy as np

class Tensor(_Tensor):
    def __init__(self, np_tensor):
        self.np_tensor = np_tensor

    def shape(self):
        ark_dims = super().shape()
        shape = []
        for i in range(ark_dims.ndims()):
            shape.append(ark_dims[i])
        return shape
    def to(self, device):
        if device == "cpu":
            return self.cpu()
        elif device == "cuda":
            if self.np_tensor == None:
                raise RuntimeError("Tensor is not initiated with a numpy array, cannot move to cuda")
            return self.cuda()
        else:
            raise RuntimeError("Unknown device: " + device, "Only cpu and cuda are supported")
    def cpu(self):
        np_array = np.empty(self.shape(), dtype=np.float16)
        return Executor.get_executor().tensor_memcpy_device_to_host(np_array, self)
    
    def cuda(self):
        return Executor.get_executor().tensor_memcpy_host_to_device(self, self.np_tensor)

