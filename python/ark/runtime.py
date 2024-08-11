# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from enum import Enum

from _ark_core import _Executor
from .planner import Planner, Plan


class _RuntimeState:
    """
    The _RuntimeState class is used to store the state of the model.
    """

    runtime = None


class Executor(_Executor):
    pass


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

    def __init__(self):
        self.executor: Executor = None
        self.state: Runtime.State = Runtime.State.Init
        _RuntimeState.runtime = self

    def get_state(self) -> "Runtime.State":
        """
        Get the runtime state.
        """
        return self.state

    @staticmethod
    def get_runtime() -> "Runtime":
        """
        Get the runtime.
        If the runtime does not exist, create a new runtime.
        """
        if _RuntimeState.runtime is None:
            _RuntimeState.runtime = Runtime()
        return _RuntimeState.runtime

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reset()

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

    def add_plan(self, plan: Plan):
        """
        Add a plan to the executor.
        """
        if self.executor is None:
            raise RuntimeError("Executor is not initialized")
        self.executor.add_plan(str(plan))

    def launch(
        self,
        plan: Plan = None,
        device_id: int = 0,
        stream: int = 0,
        loop_mode: bool = True,
    ):
        """
        Create an executor and schedule the ARK model. The scheduler will generate
        the CUDA kernels. The GPU context and the connection between GPUs will be
        initialized. The executor will compile the cuda kernels and launch the ARK runtime.
        """
        plan = Planner(device_id).plan() if plan is None else plan
        if self.launched():
            # If the Runtime state is already launched and we are adding another plan
            # to the executor, we compile the new kernel and launch the executor again.
            self.executor.add_plan(str(plan))
            self.executor.compile()
            self.executor.launch()
            return

        # If the RuntimeState is init, we need to create a new executor and
        # compile the kernels
        if self.state == Runtime.State.Init:
            if self.executor is not None:
                if not self.executor.destroyed():
                    logging.warning(
                        f"Runtime has already been launched. Destroying the old executor"
                    )
                    self.executor.destroy()
            self.executor = Executor(
                device_id,
                stream,
                "ArkRuntime",
                str(plan),
                loop_mode,
            )
            self.executor.compile()
        self.executor.launch()
        self.state = Runtime.State.LaunchedNotRunning

    def run(self, iter=1, non_blocking=False):
        """
        Run the ARK program for iter iterations and wait for the kernel to finish.
        """
        if self.state != Runtime.State.LaunchedNotRunning:
            logging.error(f"ARK runtime is not launched")
            raise RuntimeError(f"ARK runtime is not launched")
        self.state = Runtime.State.Running
        self.executor.run(iter)
        if not non_blocking:
            self.wait()

    def barrier(self):
        """
        Barrier for all ranks.
        """
        if self.state != Runtime.State.LaunchedNotRunning:
            logging.error("ARK runtime is not launched")
            raise RuntimeError("ARK runtime is not launched")
        self.executor.barrier()

    def wait(self):
        """
        Wait for the kernel to finish.
        """
        if self.state != Runtime.State.Running:
            logging.warning(f"ARK runtime is not running, skip waiting")
            return
        self.executor.wait()
        self.state = Runtime.State.LaunchedNotRunning

    def stop(self) -> float:
        """
        Stop the model and return the elapsed time in milliseconds.
        Once this is called, we need to call `launch()` again to run the model again.
        """
        if not self.launched():
            logging.warning(f"ARK runtime is never launched, skip stopping")
            return
        elapsed = self.executor.stop()
        self.state = Runtime.State.LaunchedNotRunning
        return elapsed

    def reset(self, persist=False):
        """
        Reset the runtime.
        """
        if self.launched():
            self.stop()
        if persist:
            return
        if self.executor is not None:
            if not self.executor.destroyed():
                self.executor.destroy()
            self.executor = None
        self.state = Runtime.State.Init
