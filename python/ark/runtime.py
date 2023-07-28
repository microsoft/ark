# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from .model import Model
from .executor import Executor


class config:
    global_config = None

    def __init__(self, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size


def init_model(rank: int = 0, world_size: int = 1):
    """
    Initialize the ARK runtime and create a global model that will record
    all the operations.
    """
    config.global_config = config(rank, world_size)
    Model.global_model = Model(rank)


def launch():
    rank = config.global_config.rank
    world_size = config.global_config.world_size
    Executor(rank, rank, world_size, Model.global_model, "Executor")
    Executor.get_executor().compile()
    Executor.get_executor().launch()


def run(iter=1):
    Executor.get_executor().run(iter)
    Executor.get_executor().stop()
