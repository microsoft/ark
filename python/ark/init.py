# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from ._ark_core import _init
import os
from .model import Model
from .executor import Executor

rank, world_size = os.environ.get("RANK", 0), os.environ.get("WORLD_SIZE", 1)


def init():
    """
    Initialize the runtime and create a global model that will record
    all the operations.
    """
    if rank == 0:
        """
        Only the first process should call _init() to prevent one
        process deleting the shared memory created by another process.
        """
        _init()
    Model.global_model = Model(rank)


def launch():
    Executor(rank, rank, world_size, Model.global_model, "Executor")
    Executor.get_executor(0).compile()
    Executor.get_executor(0).launch()


def run(iter=1):
    Executor.get_executor(0).run(iter)
    Executor.get_executor(0).stop()
