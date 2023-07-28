# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from ._ark_core import _init
import os
from .model import Model
from .executor import Executor


class config:
    global_config = None

    def __init__(self, rank=-1, world_size=1):
        rank = rank
        world_size = world_size


def init(config=None):
    """
    Initialize the runtime and create a global model that will record
    all the operations.
    """
    if config is not None:
        config.global_config = config
        if config.rank != -1:
            Model.global_model = Model(config.rank)
        """
        Only the first process should call _init() to prevent one
        process deleting the shared memory created by another process.
        """
        if config.rank == 0:
            _init()
    _init()


def launch():
    rank = config.global_config.rank
    world_size = config.global_config.world_size
    Executor(rank, rank, world_size, Model.global_model, "Executor")
    Executor.get_executor().compile()
    Executor.get_executor().launch()


def run(iter=1):
    Executor.get_executor().run(iter)
    Executor.get_executor().stop()
