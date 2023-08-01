# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from ._ark_core import _Executor, TensorType
import numpy as np
from .model import Model


class Executor(_Executor):
    # static list of executors
    global_executor = None

    def __init__(
        self,
        gpu_id: int,
        rank: int,
        world_size: int,
        model: Model,
        name: str,
        num_warps_per_sm: int = 16,
    ):
        super().__init__(
            gpu_id, rank, world_size, model, name, num_warps_per_sm=16
        )

    def tensor_memcpy_host_to_device(self, dst, src: np.ndarray):
        assert isinstance(src, np.ndarray), "src should be a numpy array"
        # check if src is contiguous is memory
        if not src.flags["C_CONTIGUOUS"]:
            print(
                "Warning: src is not contiguous in memory, copy to a contiguous array"
            )
            src = np.ascontiguousarray(src)
        super().tensor_memcpy_host_to_device(dst, src)

    def tensor_memcpy_device_to_host(self, dst: np.ndarray, src):
        assert isinstance(dst, np.ndarray), "dst should be a numpy array"
        assert dst.flags["C_CONTIGUOUS"], "dst is not contiguous in memory"
        super().tensor_memcpy_device_to_host(dst, src)

    @staticmethod
    def get_executor():
        # get the global executor
        if Executor.global_executor is None:
            raise RuntimeError("Executor is not initialized")
        return Executor.global_executor


def tensor_memcpy_host_to_device(dst, src):
    Executor.get_executor().tensor_memcpy_host_to_device(dst, src)
    return dst


def tensor_memcpy_device_to_host(dst, src):
    if dst is None:
        np_type = None
        if src.tensor_type() == TensorType.FP32:
            np_type = np.float32
        elif src.tensor_type() == TensorType.FP16:
            np_type = np.float16
        else:
            raise RuntimeError("Unsupported tensor type")
        dst = np.empty(src.shape, dtype=np_type)
    Executor.get_executor().tensor_memcpy_device_to_host(dst, src)
    return dst
