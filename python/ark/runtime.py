# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .model import Model
from .executor import Executor
import logging
from enum import Enum

# Use a global variable to track the state of the ARK runtime


class RuntimeState(Enum):
    init = 0
    launch = 1
    run = 2
    stop = 3
    destroy = 4


class Runtime:
    global_runtime = None

    def __init__(self, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        self.ark_runtime_state = RuntimeState.init
        Model.global_model = Model(rank)

    def __del__(self):
        """
        Destroy the ARK runtime and release all the resources.
        """
        if (
            self.ark_runtime_state == RuntimeState.run
            or self.ark_runtime_state == RuntimeState.launch
        ):
            self.stop()
        self.ark_runtime_state = RuntimeState.destroy
        Executor.global_executor = None
        Model.global_model = None

    @staticmethod
    def get_global_runtime():
        """
        Get the global ARK runtime.
        """
        if Runtime.global_runtime is None:
            logging.error("ARK runtime is not initialized")
            raise RuntimeError("ARK runtime is not initialized")
        return Runtime.global_runtime

    def launch(self):
        """
        Create an executor and schedule the ARK model. The scheduler will generate
        the CUDA kernels. The GPU context and the connection between GPUs will be
        initialized. The executor will compile the cuda kernels and launch the ARK runtime.
        """
        if (
            self.ark_runtime_state != RuntimeState.init
            and self.ark_runtime_state != RuntimeState.stop
        ):
            logging.warn(
                "ARK runtime is not initialized or already launched, skip launching"
            )
            return
        self.ark_runtime_state = RuntimeState.launch
        Executor.global_executor = Executor(
            self.rank,
            self.rank,
            self.world_size,
            Model.get_global_model(),
            "Executor",
        )
        Executor.get_global_executor().compile()
        Executor.get_global_executor().launch()

    def run(self, iter=1, async_run=False):
        """
        Run the ARK program for iter iterations and wait for the kernel to finish.
        """
        if self.ark_runtime_state != RuntimeState.launch:
            logging.error("ARK runtime is not launched")
            raise RuntimeError("ARK runtime is not launched")
        self.ark_runtime_state = RuntimeState.run
        Executor.get_global_executor().run(iter)
        if not async_run:
            self.stop()

    def wait(self):
        """
        Wait for the kernel to finish.
        """
        if self.ark_runtime_state != RuntimeState.run:
            logging.warn("ARK runtime is not running, skip waiting")
            return
        Executor.get_global_executor().wait()
        self.ark_runtime_state = RuntimeState.launch

    def stop(self):
        """
        Stop the model and return the elapsed time in milliseconds.
        Once this is called, we need to call `launch()` again to run the model again.
        """
        if (
            self.ark_runtime_state != RuntimeState.run
            and self.ark_runtime_state != RuntimeState.launch
        ):
            logging.warn(
                "ARK runtime is not running or launched, skip stopping"
            )
            return
        Executor.get_global_executor().stop()
        self.ark_runtime_state = RuntimeState.stop
