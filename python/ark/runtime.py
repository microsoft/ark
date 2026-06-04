# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum

from . import log
from .tensor import Tensor
from .torch import torch
from .executor import Executor
from .planner import Planner, Plan
from .model import Model
from typing import Dict

__all__ = ["Runtime"]


class Runtime:
    """
    Convenience class for running a model.
    """

    class StateCode(Enum):
        """
        Runtime state code.
        """

        Init = 0
        LaunchedNotRunning = 1
        Running = 2

    def __init__(self):
        self.loop_mode: bool = True
        self.state: Runtime.StateCode = Runtime.StateCode.Init
        self.stream: int = 0
        self.record: bool = False

    def _normalize_tensor_mappings(self, tensor_mappings: Dict) -> Dict:
        normalized = {}
        for ark_tensor, torch_tensor in tensor_mappings.items():
            if not isinstance(torch_tensor, torch.Tensor):
                raise log.InvalidUsageError("Must bind PyTorch tensor")
            normalized[ark_tensor._tensor] = torch_tensor.data_ptr()
        return normalized

    def __enter__(self) -> "Runtime":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.launched():
            self.stop()

    def __del__(self):
        if self.launched():
            self.stop()

    def launched(self) -> bool:
        """
        Check if the runtime is launched.
        """
        return (
            self.state == Runtime.StateCode.LaunchedNotRunning
            or self.state == Runtime.StateCode.Running
        )

    def running(self) -> bool:
        """
        Check if the runtime is running.
        """
        return self.state == Runtime.StateCode.Running

    def launch(
        self,
        plan: Plan = None,
        device_id: int = -1,
        stream: int = 0,
        loop_mode: bool = True,
        record: bool = False,
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
            raise log.InvalidUsageError(f"Invalid device_id: {device_id}")
        plan = Planner(device_id).plan() if plan is None else plan
        plan_str = str(plan)
        if self.launched():
            # Stop the current running model
            self.stop()
        tensor_mappings = self._normalize_tensor_mappings(tensor_mappings)
        # Recompile if the previous launch was not compiled with the same info
        # or if this is the first launch
        exe = Executor.get()
        if plan_str != exe.plan() or device_id != exe.device_id():
            exe.compile(plan_str, device_id)
        exe.launch(tensor_mappings, stream, loop_mode, record)
        self.state = Runtime.StateCode.LaunchedNotRunning
        self.loop_mode = loop_mode
        self.stream = exe.stream()
        self.record = record

    def run(
        self,
        iter: int = 1,
        non_blocking: bool = False,
        tensor_mappings: Dict[Tensor, torch.Tensor] = {},
    ):
        """
        Run the ARK program for iter iterations and wait for the kernel to finish.
        """
        if self.loop_mode and tensor_mappings:
            raise log.InvalidUsageError(
                "`loop_mode` argument when calling `runtime.launch` "
                "must be set to false in order to pass non-empty "
                "tensor mappings in `runtime.run`."
            )
        if self.state != Runtime.StateCode.LaunchedNotRunning:
            raise log.InvalidUsageError(f"ARK runtime is not launched")
        tensor_mappings = self._normalize_tensor_mappings(tensor_mappings)
        exe = Executor.get()
        if tensor_mappings and not self.loop_mode:
            exe.stop()
            exe.launch(tensor_mappings, self.stream, False, self.record)
            tensor_mappings = {}
        self.state = Runtime.StateCode.Running
        exe.run(iter, tensor_mappings)
        if not non_blocking:
            self.wait()

    def barrier(self):
        """
        Barrier for all ranks.
        """
        if self.state != Runtime.StateCode.LaunchedNotRunning:
            raise log.InvalidUsageError("ARK runtime is not launched")
        Executor.get().barrier()

    def wait(self):
        """
        Wait for the kernel to finish.
        """
        if self.state != Runtime.StateCode.Running:
            log.WARN(f"ARK runtime is not running, skip waiting")
            return
        Executor.get().wait()
        self.state = Runtime.StateCode.LaunchedNotRunning

    def stop(self) -> float:
        """
        Stop the model and return the elapsed time in milliseconds.
        Once this is called, we need to call `launch()` again to run the model again.
        """
        if not self.launched():
            log.WARN(f"ARK runtime is never launched, skip stopping")
            return -1
        elapsed = Executor.get().stop()
        self.state = Runtime.StateCode.LaunchedNotRunning
        return elapsed
