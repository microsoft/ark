# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from ._ark_core import _Executor, _Tensor
from .model import Model
import numpy as np
import logging


class Executor(_Executor):
    # Global Executor
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

    def tensor_memcpy_host_to_device(self, dst: _Tensor, src: np.ndarray):
        if not isinstance(src, np.ndarray):
            logging.error("src is not a numpy array")
            raise TypeError("src is not a numpy array")
        # Check if src is contiguous is memory
        if not src.flags["C_CONTIGUOUS"]:
            logging.debug(
                "Warning: src is not contiguous in memory, copy to a contiguous array"
            )
            src = np.ascontiguousarray(src)
        super().tensor_memcpy_host_to_device(dst, src)

    def tensor_memcpy_device_to_host(self, dst: np.ndarray, src: _Tensor):
        if not isinstance(dst, np.ndarray):
            logging.error("dst is not a numpy array")
            raise TypeError("dst is not a numpy array")
        if not dst.flags["C_CONTIGUOUS"]:
            logging.error("dst is not contiguous in memory")
            raise ValueError("dst is not contiguous in memory")
        super().tensor_memcpy_device_to_host(dst, src)

    @staticmethod
    def get_global_executor():
        # Get the global executor
        if Executor.global_executor is None:
            logging.error("Executor is not initialized")
            raise RuntimeError("Executor is not initialized")
        return Executor.global_executor
