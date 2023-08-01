# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from .model import Model
from .executor import Executor
from ._ark_core import _init

# Use a global variable to track the state of the ARK runtime
ArkRuntimeState = ("start", "init_model", "launch", "run", "destroy")


class ARKConfig:
    global_config = None

    def __init__(self, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        self.ark_runtime_state = ArkRuntimeState[0]


def init():
    """
    Init an ARK program. Call this function to clean up the shared memory directory. This is useful when the previous run crashed, as this forces to remove locks generated by previous runs. This may crash other ARK processes running on the same machine, if there are any.
    """
    ARKConfig.global_config = None
    Model.global_model = None
    Executor.global_executor = None
    _init()


def init_model(rank: int = 0, world_size: int = 1):
    """
    Initialize the ARK runtime and create a global model that will record
    all the operations.
    """
    if ARKConfig.global_config is None:
        ARKConfig.global_config = ARKConfig(rank, world_size)
    assert (
        ARKConfig.global_config.ark_runtime_state == "start"
    ), "ARK runtime is already initialized"
    if Model.global_model is None:
        Model.global_model = Model(rank)
    ARKConfig.global_config.ark_runtime_state = "init_model"


def launch():
    """
    Create an executor and schedule the ARK model. The scheduler will generate
    the CUDA kernels. The GPU context and the connection between GPUs will be
    initialized. The executor will compile the cuda kernels and launch the ARK runtime.
    """
    assert (
        ARKConfig.global_config.ark_runtime_state == "init_model"
    ), "ARK runtime is not initialized"
    ARKConfig.global_config.ark_runtime_state = "launch"
    rank = ARKConfig.global_config.rank
    world_size = ARKConfig.global_config.world_size
    Executor.global_executor = Executor(
        rank, rank, world_size, Model.global_model, "Executor"
    )
    Executor.get_executor().compile()
    Executor.get_executor().launch()


def run(iter=1):
    """
    Run the ARK program for iter iterations and wait for the kernel to finish.
    """
    assert (
        ARKConfig.global_config.ark_runtime_state == "launch"
    ), "ARK runtime is not launched"
    ARKConfig.global_config.ark_runtime_state = "run"
    Executor.get_executor().run(iter)
    Executor.get_executor().stop()
    ARKConfig.global_config.ark_runtime_state = "launch"


def destroy():
    """
    Destroy the ARK runtime and release all the resources.
    """
    ARKConfig.global_config.ark_runtime_state = "destroy"
    Executor.global_executor = None
    Model.global_model = None
    ARKConfig.global_config = None
