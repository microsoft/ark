# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from ._ark_core import _Executor
import numpy as np


class Executor(_Executor):
    # static list of executors
    executors = []
    def __init__(self, gpu_id, rank, world_size, model, name, num_warps_per_sm = 16):
        super().__init__(gpu_id, rank, world_size, model, name, num_warps_per_sm = 16)
        self.executors.append(self)
        
    def tensor_memcpy_host_to_device(self, dst, src):
        # check if src is contiguous is memory
        if not src.flags['C_CONTIGUOUS']:
            print("Warning: src is not contiguous in memory, copy to a contiguous array")
            src = np.ascontiguousarray(src)
        super().tensor_memcpy_host_to_device(dst, src)
    def tensor_memcpy_device_to_host(self, dst, src):
        super().tensor_memcpy_device_to_host(dst, src)

    @staticmethod
    def get_executor(executor_id=0):
        # get the global executor
        if executor_id >= len(Executor.executors):
            raise RuntimeError("Executor of id " + str(executor_id) + " is not initialized")
        return Executor.executors[executor_id]

def tensor_memcpy_host_to_device(dst, src, executor_id=0):
    Executor.get_executor(executor_id).tensor_memcpy_host_to_device(dst, src)
    return dst

def tensor_memcpy_device_to_host(dst, src, executor_id=0):
    Executor.get_executor(executor_id).tensor_memcpy_device_to_host(dst, src)
    return dst

def tensor_from_device(src, executor_id=0):
    np_shape = []
    ark_dims = src.shape()
    for i in range(src.ndim()):
        np_shape.append(ark_dims[i])
    dst = np.empty(np_shape, dtype=src.dtype)
    Executor.get_executor(executor_id).tensor_memcpy_device_to_host(dst, src)
    return dst

