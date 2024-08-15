# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from enum import Enum

from ._ark_core import _Executor
from .planner import Planner, Plan
from typing import Dict
try:
    import torch

    _no_torch = False
except ImportError:
    from . import torch_mock as torch

    _no_torch = True


class _RuntimeState:
    """
    The _RuntimeState class is used to store the state of the model.
    """

    runtime = None


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
        self.executor: _Executor = _Executor()
        self.state: Runtime.State = Runtime.State.Init
        self.loop_mode = True
        _RuntimeState.runtime = self

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

    def launch(
        self,
        plan: Plan = None,
        device_id: int = 0,
        stream: int = 0,
        loop_mode: bool = True,
        tensor_mappings: Dict = {}
    ):
        """
        Create an executor and schedule the ARK model. The scheduler will generate
        the CUDA kernels. The GPU context and the connection between GPUs will be
        initialized. The executor will compile the cuda kernels and launch the ARK runtime.
        """
        if device_id < 0:
            logging.error(f"Invalid device_id: {device_id}")
            raise ValueError(f"Invalid device_id: {device_id}")
        plan = Planner(device_id).plan() if plan is None else plan
        plan_str = str(plan)
        if self.launched():
            # Stop the current running model
            self.stop()
        
        for ark_tensor in tensor_mappings:
            torch_tensor = tensor_mappings[ark_tensor]
            if not isinstance(torch_tensor, torch.Tensor):
                raise ValueError("Must bind PyTorch tensor")
            tensor_mappings[ark_tensor] = torch_tensor.data_ptr()

        # Recompile if the previous launch was not compiled with the same info
        # or if this is the first launch
        if (
            plan_str != self.executor.plan()
            or device_id != self.executor.device_id()
        ):
            self.executor.compile(plan_str, device_id, tensor_mappings)
        self.executor.launch(stream, loop_mode)
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

    def reset(self):
        """
        Reset the runtime.
        """
        if self.launched():
            self.stop()
        self.executor.destroy()
        self.executor = _Executor()
        self.state = Runtime.State.Init


__all__ = ["Runtime"]
