# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from enum import Enum
from typing import NewType

from ._ark_core import _Executor, _DefaultPlanner
from .model import Model

_RuntimeState = NewType("_RuntimeState", None)


class Runtime:
    """
    Convenience class for running a model.
    """

    class State(Enum):
        """
        Runtime states.
        """

        Init = 0
        LaunchedNotRunning = 1
        Running = 2

    @staticmethod
    def get_runtime():
        """
        Get the runtime.
        """
        if _RuntimeState.runtime is None:
            _RuntimeState.runtime = Runtime()
        return _RuntimeState.runtime

    def __init__(self):
        if _RuntimeState.runtime is not None:
            logging.error("Runtime is already created")
            raise RuntimeError("Runtime is already created")
        self.executor: _Executor = None
        self.state: Runtime.State = Runtime.State.Init
        _RuntimeState.runtime = self

    def __del__(self):
        """
        Destroy the ARK runtime and release all the resources.
        """
        if self.launched():
            self.stop()
        self.executor = None

    def launched(self) -> bool:
        """
        Check if the runtime is launched.
        """
        return (
            self.state == Runtime.State.LaunchedNotRunning
            or self.state == Runtime.State.Running
        )

    def running(self) -> bool:
        """
        Check if the runtime is running.
        """
        return self.state == Runtime.State.Running

    def launch(
        self,
        rank: int = 0,
        world_size: int = 1,
        gpu_id: int = 0,
        plan: str = "",
        plan_path: str = "",
    ):
        """
        Create an executor and schedule the ARK model. The scheduler will generate
        the CUDA kernels. The GPU context and the connection between GPUs will be
        initialized. The executor will compile the cuda kernels and launch the ARK runtime.
        """
        if self.launched():
            logging.warn("Runtime is already launched, skip launching")
            return
        if plan == "" and plan_path == "":
            plan = _DefaultPlanner(Model.get_model(), gpu_id).plan(indent=2)
            # Write plan to a file
            with open("plan.json", "w") as f:
                f.write(plan)
        else:
            with open(plan_path, "r") as f:
                plan = f.read()
        # If the RuntimeState is init, we need to create a new executor and
        # compile the kernels
        if self.state == Runtime.State.Init:
            self.executor = _Executor(
                rank,
                world_size,
                gpu_id,
                "ArkRuntime",
                plan,
            )
            self.executor.compile()
        self.executor.launch()
        self.state = Runtime.State.LaunchedNotRunning

    def run(self, iter=1, non_blocking=False):
        """
        Run the ARK program for iter iterations and wait for the kernel to finish.
        """
        if self.state != Runtime.State.LaunchedNotRunning:
            logging.error("ARK runtime is not launched")
            raise RuntimeError("ARK runtime is not launched")
        self.state = Runtime.State.Running
        self.executor.run(iter)
        if not non_blocking:
            self.wait()

    def wait(self):
        """
        Wait for the kernel to finish.
        """
        if self.state != Runtime.State.Running:
            logging.warn("ARK runtime is not running, skip waiting")
            return
        self.executor.wait()
        self.state = Runtime.State.LaunchedNotRunning

    def stop(self) -> float:
        """
        Stop the model and return the elapsed time in milliseconds.
        Once this is called, we need to call `launch()` again to run the model again.
        """
        if not self.launched():
            logging.warn("ARK runtime is never launched, skip stopping")
            return
        elapsed = self.executor.stop()
        self.state = Runtime.State.LaunchedNotRunning
        return elapsed


class _RuntimeState:
    """
    The _RuntimeState class is used to store the state of the model.
    """

    runtime: Runtime = None
