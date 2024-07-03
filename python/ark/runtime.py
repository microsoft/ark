# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from enum import Enum
from typing import Dict, List

from ._ark_core import _Executor
from .model import Model
from .planner import DefaultPlanner, Plan


class _RuntimeState:
    """
    The _RuntimeState class is used to store the state of the model.
    """

    runtime: Dict[int, "Runtime"] = {}

    @staticmethod
    def reset_all():
        """
        Resets all runtimes.
        """
        runtime_ids = list(_RuntimeState.runtime.keys())
        for runtime_id in runtime_ids:
            _RuntimeState.runtime[runtime_id].reset()

    @staticmethod
    def delete_all():
        """
        Deletes all runtimes.
        """
        runtime_ids = list(_RuntimeState.runtime.keys())
        for runtime_id in runtime_ids:
            _RuntimeState.runtime[runtime_id].reset(delete=True)

    @staticmethod
    def print_runtime_states():
        """
        Print runtimes and their corresponding states.
        """
        print(f"{'Runtime ID':<12} | {'Status':<20}")
        print(f"{'-'*12} | {'-'*20}")
        for runtime_id, runtime in _RuntimeState.runtime.items():
            runtime_id = "-1(Default)" if runtime_id == -1 else runtime_id
            print(f"{runtime_id:<12} | {runtime.state:<20}")


class Executor(_Executor):
    def __init__(self, plan: Plan, device_id: int, name: str):
        super().__init__(plan.rank, plan.world_size, device_id, name, str(plan))


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

    def __init__(self, runtime_id: int = -1):
        self.runtime_id = runtime_id
        self.executor: Executor = None
        self.state: Runtime.State = Runtime.State.Init
        _RuntimeState.runtime[runtime_id] = self

    def get_state(self) -> "Runtime.State":
        """
        Get the runtime state.
        """
        return self.state

    @staticmethod
    def exists(runtime_id: int) -> bool:
        """
        Check if a runtime exists with the given ID.
        """
        return runtime_id in _RuntimeState.runtime

    @staticmethod
    def get_all_ids() -> List[int]:
        """
        Get a list of all existing runtime IDs.
        """
        return list(_RuntimeState.runtime.keys())

    @staticmethod
    def get_runtime(runtime_id=-1) -> "Runtime":
        """
        Get the runtime by ID. If runtime_id is not provided, use a default ID of -1.
        If the runtime does not exist, create a new runtime with the given ID.
        """
        if runtime_id not in _RuntimeState.runtime:
            _RuntimeState.runtime[runtime_id] = Runtime(runtime_id)
        return _RuntimeState.runtime[runtime_id]

    @staticmethod
    def see_runtime_statuses() -> "Dict[int, Runtime]":
        """
        Returns the runtime dictionary containing all of the runtimes.
        """
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

    def launch(
        self,
        plan: Plan = None,
        device_id: int = 0,
    ):
        """
        Create an executor and schedule the ARK model. The scheduler will generate
        the CUDA kernels. The GPU context and the connection between GPUs will be
        initialized. The executor will compile the cuda kernels and launch the ARK runtime.
        """
        if self.launched():
            logging.warn(
                f"Runtime {self.runtime_id} is already launched, skip launching"
            )
            return
        if not plan:
            plan = DefaultPlanner(device_id).plan()
        # If the RuntimeState is init, we need to create a new executor and
        # compile the kernels
        if self.state == Runtime.State.Init:
            if self.executor is not None:
                if not self.executor.destroyed():
                    logging.warn(
                        f"Runtime {self.runtime_id}, has already been launched. Destroying the old executor"
                    )
                    self.executor.destroy()
            self.executor = Executor(
                plan,
                device_id,
                "ArkRuntime",
            )
            self.executor.compile()
        self.executor.launch()
        self.state = Runtime.State.LaunchedNotRunning

    def run(self, iter=1, non_blocking=False):
        """
        Run the ARK program for iter iterations and wait for the kernel to finish.
        """
        if self.state != Runtime.State.LaunchedNotRunning:
            logging.error(f"ARK runtime {self.runtime_id} is not launched")
            raise RuntimeError(f"ARK runtime {self.runtime_id} is not launched")
        self.state = Runtime.State.Running
        self.executor.run(iter)
        if not non_blocking:
            self.wait()

    def wait(self):
        """
        Wait for the kernel to finish.
        """
        if self.state != Runtime.State.Running:
            logging.warn(
                f"ARK runtime {self.runtime_id} is not running, skip waiting"
            )
            return
        self.executor.wait()
        self.state = Runtime.State.LaunchedNotRunning

    def stop(self) -> float:
        """
        Stop the model and return the elapsed time in milliseconds.
        Once this is called, we need to call `launch()` again to run the model again.
        """
        if not self.launched():
            logging.warn(
                f"ARK runtime {self.runtime_id} is never launched, skip stopping"
            )
            return
        elapsed = self.executor.stop()
        self.state = Runtime.State.LaunchedNotRunning
        return elapsed

    def reset(self, delete=False):
        """
        Reset the runtime. If delete is True, delete the runtime associated with the runtime_id.
        """
        if self.launched():
            self.stop()
        if self.executor is not None:
            if not self.executor.destroyed():
                self.executor.destroy()
            self.executor = None
        self.state = Runtime.State.Init
        if delete:
            del _RuntimeState.runtime[self.runtime_id]

    @staticmethod
    def reset_all_runtimes():
        """
        Reset all runtimes.
        """
        _RuntimeState.reset_all()

    @staticmethod
    def delete_all_runtimes():
        """
        Delete all runtimes.
        """
        _RuntimeState.delete_all()

    @staticmethod
    def print_runtime_states():
        """
        Print runtimes and their corresponding states.
        """
        _RuntimeState.print_runtime_states()
