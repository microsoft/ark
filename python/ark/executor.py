# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from ._ark_core import _Executor, _Tensor
import numpy as np
import logging


class Executor(_Executor):
    """
    Convenience class for executing a model.
    """

    global_executor = None

    def __init__(
        self,
        gpu_id: int,
        rank: int,
        world_size: int,
        model,
        name: str,
        num_warps_per_sm: int = 16,
    ):
        super().__init__(
            gpu_id, rank, world_size, model, name, num_warps_per_sm
        )

    @staticmethod
    def get_global_executor():
        """
        Get the global executor
        """
        if Executor.global_executor is None:
            logging.error("Executor is not initialized")
            raise RuntimeError("Executor is not initialized")
        return Executor.global_executor
