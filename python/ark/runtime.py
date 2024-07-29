# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from enum import Enum
from typing import Callable

from _ark_core import _Executor, _DefaultPlanner
from .model import Model


class _RuntimeState:
    """
    The _RuntimeState class is used to store the state of the model.
    """

    runtime = None
    executor = None


class DefaultPlanner(_DefaultPlanner):
    def __init__(self, gpu_id: int = 0):
        compressed = Model.get_model().compress()
        super().__init__(compressed, gpu_id)

    def install_config_rule(self, rule: Callable[[str, str], str]):
        """
        Install a configuration rule.

        Args:
            rule: A function that takes an operator description and a target
            architecture name and returns a configuration description.
        """
        super().install_config_rule(rule)

    def plan(self, pretty: bool = True) -> str:
        """
        Generate an execution plan.

        Args:
            pretty: Whether to generate a pretty plan.
        """
        return super().plan(pretty)


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

    @staticmethod
    def get_runtime() -> "Runtime":
        """
        Get the runtime.
        """
        if _RuntimeState.runtime is None:
            _RuntimeState.runtime = Runtime()
        return _RuntimeState.runtime

    def __init__(self):
        self.executor: Executor = None
        self.state: Runtime.State = Runtime.State.Init
        _RuntimeState.runtime = self

    def __del__(self):
        self.reset()

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

    def launch(
        self,
        gpu_id: int = 0,
        plan: str = "",
        plan_path: str = "",
        stream: int = 0,
        loop_mode: bool = True,
    ):
        """
        Create an executor and schedule the ARK model. The scheduler will generate
        the CUDA kernels. The GPU context and the connection between GPUs will be
        initialized. The executor will compile the cuda kernels and launch the ARK runtime.
        """
        if self.launched():
            # If the Runtime state is already launched and we are adding another plan
            # to the executor, we compile the new kernel and launch the executor again.
            if not plan:
                plan = DefaultPlanner(gpu_id).plan()
            self.executor.add_plan(str(plan))
            self.executor.compile()
            self.executor.launch()
            return
        if not plan:
            if not plan_path:
                plan = DefaultPlanner(gpu_id).plan()
            else:
                with open(plan_path, "r") as f:
                    plan = f.read()
        # If the RuntimeState is init, we need to create a new executor and
        # compile the kernels
        if self.state == Runtime.State.Init:
            if _RuntimeState.executor is not None:
                if not _RuntimeState.executor.destroyed():
                    logging.warn("Destroying an old executor")
                    _RuntimeState.executor.destroy()

            _RuntimeState.executor = Executor(
                gpu_id,
                stream,
                "ArkRuntime",
                plan,
                loop_mode,
            )
            self.executor = _RuntimeState.executor
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

    def reset(self, persist=False):
        """
        Reset the runtime. If persist is True, keep the executor alive to run
        additional plans. If persist is False, destroy the executor.
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
