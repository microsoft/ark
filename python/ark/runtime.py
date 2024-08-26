# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from enum import Enum

from ._ark_core import _Executor
from .torch import torch
from .planner import Planner, Plan
from .model import Model
from typing import Dict


class _RuntimeState:
    """
    The _RuntimeState class is used to store the state of the model.
    """

    runtime = None


class Runtime:
    """
    Convenience class for running a model.
    """

    _loop_mode: bool = True

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
        device_id: int = -1,
        stream: int = 0,
        loop_mode: bool = True,
        tensor_mappings: Dict = {},
    ):
        """
        Create an executor and schedule the ARK model. The scheduler will generate
        the CUDA kernels. The GPU context and the connection between GPUs will be
        initialized. The executor will compile the cuda kernels and launch the ARK runtime.
        """
        if device_id == -1:
            device_id = Model.get_device_id()
        elif device_id < 0:
            logging.error(f"Invalid device_id: {device_id}")
            raise ValueError(f"Invalid device_id: {device_id}")
        plan = Planner(device_id).plan() if plan is None else plan
        plan_str = str(plan)
        if self.launched():
            # Stop the current running model
            self.stop()
        for ark_tensor in list(tensor_mappings.keys()):
            torch_tensor = tensor_mappings[ark_tensor]
            if not isinstance(torch_tensor, torch.Tensor):
                raise ValueError("Must bind PyTorch tensor")
            internal_ark_tensor = ark_tensor._tensor
            tensor_mappings[internal_ark_tensor] = torch_tensor.data_ptr()
            del tensor_mappings[ark_tensor]
        # Recompile if the previous launch was not compiled with the same info
        # or if this is the first launch
        if (
            plan_str != self.executor.plan()
            or device_id != self.executor.device_id()
        ):
            self.executor.compile(plan_str, device_id)
        self.executor.launch(tensor_mappings, stream, loop_mode)
        self.state = Runtime.State.LaunchedNotRunning
        Runtime._loop_mode = loop_mode

    def run(self, iter=1, non_blocking=False, tensor_mappings={}):
        """
        Run the ARK program for iter iterations and wait for the kernel to finish.
        """
        if Runtime._loop_mode and tensor_mappings:
            raise ValueError(
                "`loop_mode` argument when calling `runtime.launch` "
                "must be set to false in order to pass non-empty "
                "tensor mappings in `runtime.run`."
            )
        if self.state != Runtime.State.LaunchedNotRunning:
            logging.error(f"ARK runtime is not launched")
            raise RuntimeError(f"ARK runtime is not launched")
        self.state = Runtime.State.Running
        ph_map = {}
        for ark_tensor in list(tensor_mappings.keys()):
            t = tensor_mappings[ark_tensor]
            ph_map[ark_tensor._tensor] = t.data_ptr()
        self.executor.run(iter, ph_map)
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
