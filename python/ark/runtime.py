# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from enum import Enum

from ._ark_core import _Executor
from .model import Model


class _RuntimeStateType(Enum):
    """
    Runtime state types.
    """

    init = 0
    launch = 1
    run = 2
    stop = 3
    destroy = 4


class Runtime:
    """
    Convenience class for running a model.
    """

    def __init__(self):
        self.executor: _Executor = None
        self.state: _RuntimeStateType = _RuntimeStateType.init

    def __del__(self):
        """
        Destroy the ARK runtime and release all the resources.
        """
        if (
            self.state == _RuntimeStateType.run
            or self.state == _RuntimeStateType.launch
        ):
            self.stop()
        self.executor = None
        self.state = _RuntimeStateType.destroy

    def launch(self, num_warps_per_sm: int = 16):
        """
        Create an executor and schedule the ARK model. The scheduler will generate
        the CUDA kernels. The GPU context and the connection between GPUs will be
        initialized. The executor will compile the cuda kernels and launch the ARK runtime.
        """
        if (
            self.state != _RuntimeStateType.init
            and self.state != _RuntimeStateType.stop
        ):
            logging.warn(
                "Runtime is not initialized or already launched, skip launching"
            )
            return
        # If the RuntimeState is init, we need to create a new executor and
        # compile the kernels
        if self.state == _RuntimeStateType.init:
            self.executor = _Executor(
                Model.get_rank(),
                Model.get_world_size(),
                Model.get_model(),
                "DefaultRuntime",
                num_warps_per_sm,
            )
            self.executor.compile()
        self.executor.launch()
        self.state = _RuntimeStateType.launch

    def run(self, iter=1, non_blocking=False):
        """
        Run the ARK program for iter iterations and wait for the kernel to finish.
        """
        if self.state != _RuntimeStateType.launch:
            logging.error("ARK runtime is not launched")
            raise RuntimeError("ARK runtime is not launched")
        self.state = _RuntimeStateType.run
        self.executor.run(iter)
        if not non_blocking:
            self.wait()

    def wait(self):
        """
        Wait for the kernel to finish.
        """
        if self.state != _RuntimeStateType.run:
            logging.warn("ARK runtime is not running, skip waiting")
            return
        self.executor.wait()
        self.state = _RuntimeStateType.launch

    def stop(self) -> float:
        """
        Stop the model and return the elapsed time in milliseconds.
        Once this is called, we need to call `launch()` again to run the model again.
        """
        if (
            self.state != _RuntimeStateType.run
            and self.state != _RuntimeStateType.launch
        ):
            logging.warn(
                "ARK runtime is not running or launched, skip stopping"
            )
            return
        elapsed = self.executor.stop()
        self.state = _RuntimeStateType.stop
        return elapsed
