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
    def __init__(
        self,
        device_id: int,
        stream: int,
        name: str,
        plan: Plan,
        loop_mode: bool = True,
    ):
        super().__init__(device_id, stream, name, str(plan), loop_mode)


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
        self.executor_map: Dict[int, Executor] = {}
        self.executor_states: Dict[int, Runtime.State] = {}
        _RuntimeState.runtime[runtime_id] = self

    @property
    def state(self) -> "Runtime.State":
        """
        Returns the runtime state of the default executor.
        """
        return self.executor_states.get(-1, Runtime.State.Init)

    @property
    def executor(self) -> "Executor":
        """
        Returns the default executor.
        """
        return self.executor_map.get(-1, None)

    def get_state(self, executor_id: int = -1) -> "Runtime.State":
        """
        Returns the runtime state of the executor with the given ID.
        """
        return self.executor_states.get(executor_id, Runtime.State.Init)

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

    def launched(self, executor_id: int = -1) -> bool:
        """
        Check if the runtime is launched.
        """
        return self.get_state(executor_id=executor_id) in {
            Runtime.State.LaunchedNotRunning,
            Runtime.State.Running
        }

    def running(self, executor_id: int = -1) -> bool:
        """
        Check if the runtime is running.
        """
        return self.get_state(executor_id) == Runtime.State.Running 

    def launch(
        self,
        plan: Plan = None,
        device_id: int = 0,
        stream: int = 0,
        loop_mode: bool = True,
        executor_id: int = -1
    ):
        """
        Create an executor and schedule the ARK model. The scheduler will generate
        the CUDA kernels. The GPU context and the connection between GPUs will be
        initialized. The executor will compile the cuda kernels and launch the ARK runtime.
        """
        if self.launched(executor_id=executor_id):
            logging.warning(
                f"{('Default executor' if executor_id == -1 else f'Executor {executor_id}')}"
                f" of runtime {self.runtime_id} is already launched, skip launching"
            )
            return
        if plan is None:
            plan = DefaultPlanner(device_id).plan()
        # If the RuntimeState is init, we need to create a new executor and
        # compile the kernels
        if self.get_state(executor_id=executor_id) == Runtime.State.Init:
            if executor_id in self.executor_map:
                if not self.executor_map[executor_id].destroyed():
                    executor_desc = (
                        "Default Executor"
                        if executor_id == -1
                        else f"Executor {executor_id}"
                    )
                    logging.warning(
                        f'''{executor_desc} associated with runtime {self.runtime_id}, 
                            has already been launched. Destroying the old executor'''
                    )
                    self.executor_map[executor_id].destroy()
            executor = Executor(
                device_id,
                stream,
                "ArkRuntime",
                plan,
                loop_mode,
            )
            self.executor_map[executor_id] = executor
            executor.compile()
        self.executor_map[executor_id].launch()
        self.executor_states[executor_id] = Runtime.State.LaunchedNotRunning

    def run(self, iter=1, non_blocking=False, executor_id: int = -1):
        """
        Run the ARK program for iter iterations and wait for the kernel to finish.
        """
        if self.get_state(executor_id) != Runtime.State.LaunchedNotRunning:
            executor_desc = (
                        "Default Executor"
                        if executor_id == -1
                        else f"Executor {executor_id}"
                    )
            logging.error(f"{executor_desc} of ARK runtime {self.runtime_id} is not launched")
            raise RuntimeError(f"{executor_desc} of ARK runtime {self.runtime_id} is not launched")
        self.executor_states[executor_id] = Runtime.State.Running
        self.executor_map[executor_id].run(iter)
        if not non_blocking:
            self.wait(executor_id)

    def wait(self, executor_id: int = -1):
        """
        Wait for the kernel to finish.
        """
        if self.get_state(executor_id) != Runtime.State.Running:
            executor_desc = (
                        "Default Executor"
                        if executor_id == -1
                        else f"Executor {executor_id}"
                    )
            logging.warning(
                f"{executor_desc} associated with runtime {self.runtime_id} is not running, skip waiting"
            )
            return
        self.executor_map[executor_id].wait()
        self.executor_states[executor_id] = Runtime.State.LaunchedNotRunning

    def stop(self, executor_id = -1) -> float:
        """
        Stop the model and return the elapsed time in milliseconds.
        Once this is called, we need to call `launch()` again to run the model again.
        """
        if not self.launched(executor_id):
            executor_desc = (
                        "Default Executor"
                        if executor_id == -1
                        else f"Executor {executor_id}"
                    )
            logging.warning(
                f"{executor_desc} assocaited with runtime {self.runtime_id} is never launched, skip stopping"
            )
            return
        elapsed = self.executor_map[executor_id].stop()
        self.executor_states[executor_id] = Runtime.State.LaunchedNotRunning
        return elapsed
    
    def reset_executor(self, executor_id: int = -1, persist=False):
        """
        Reset a specific executor associated with the runtime.
        """
        if self.launched(executor_id):
            self.stop(executor_id)
        if persist:
            return
        if executor_id in self.executor_map:
            if not self.executor_map[executor_id].destroyed():
                self.executor_map[executor_id].destroy()
            del self.executor_map[executor_id]
            del self.executor_states[executor_id]
        

    def reset(self, delete=False):
        """
        Reset the runtime. If delete is True, delete the runtime associated with the runtime_id.
        """
        for executor_id in list(self.executor_map.keys()):
            if self.launched(executor_id):
                self.stop(executor_id)
            if (executor_id in self.executor_map and
                not self.executor_map[executor_id].destroyed()):
                self.executor_map[executor_id].destroy()
        self.executor_map.clear()
        self.executor_states.clear()
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
