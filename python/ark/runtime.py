# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .model import Model
from .executor import Executor
from .tensor import Tensor
from ._ark_core import TensorType
import logging
import numpy as np
from enum import Enum

# Use a global variable to track the state of the ARK runtime


class ARKRuntimeState(Enum):
    init = 0
    launch = 1
    run = 2
    stop = 3
    destroy = 4


class ARKRuntime:
    global_runtime = None

    def __init__(self, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        self.ark_runtime_state = ARKRuntimeState.init
        Model.global_model = Model(rank)

    def __del__(self):
        """
        Destroy the ARK runtime and release all the resources.
        """
        if (
            self.ark_runtime_state == ARKRuntimeState.run
            or self.ark_runtime_state == ARKRuntimeState.launch
        ):
            self.stop()
        self.ark_runtime_state = ARKRuntimeState.destroy
        Executor.global_executor = None
        Model.global_model = None

    @staticmethod
    def get_global_runtime():
        """
        Get the global ARK runtime.
        """
        if ARKRuntime.global_runtime is None:
            logging.error("ARK runtime is not initialized")
            raise RuntimeError("ARK runtime is not initialized")
        return ARKRuntime.global_runtime

    def launch(self):
        """
        Create an executor and schedule the ARK model. The scheduler will generate
        the CUDA kernels. The GPU context and the connection between GPUs will be
        initialized. The executor will compile the cuda kernels and launch the ARK runtime.
        """
        if (
            self.ark_runtime_state != ARKRuntimeState.init
            and self.ark_runtime_state != ARKRuntimeState.stop
        ):
            logging.warn(
                "ARK runtime is not initialized or already launched, skip launching"
            )
            return
        self.ark_runtime_state = ARKRuntimeState.launch
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
        if self.ark_runtime_state != ARKRuntimeState.launch:
            logging.error("ARK runtime is not launched")
            raise RuntimeError("ARK runtime is not launched")
        self.ark_runtime_state = ARKRuntimeState.run
        Executor.get_global_executor().run(iter)
        if not async_run:
            self.stop()

    def wait(self):
        """
        Wait for the kernel to finish.
        """
        if self.ark_runtime_state != ARKRuntimeState.run:
            logging.warn("ARK runtime is not running, skip waiting")
            return
        Executor.get_global_executor().wait()
        self.ark_runtime_state = ARKRuntimeState.launch

    def stop(self):
        """
        Stop the model and return the elapsed time in milliseconds.
        Once this is called, we need to call `launch()` again to run the model again.
        """
        if (
            self.ark_runtime_state != ARKRuntimeState.run
            and self.ark_runtime_state != ARKRuntimeState.launch
        ):
            logging.warn(
                "ARK runtime is not running or launched, skip stopping"
            )
            return
        Executor.get_global_executor().stop()
        self.ark_runtime_state = ARKRuntimeState.stop

    def tensor_memcpy_host_to_device(self, dst: Tensor, src: np.ndarray):
        """
        Copy a tensor from host to device. Used for initializing the tensor on device.
        """
        # Check the current ARK runtime status
        if (
            self.ark_runtime_state != ARKRuntimeState.launch
            and self.ark_runtime_state != ARKRuntimeState.stop
        ):
            logging.error("ARK runtime is not launched")
            raise RuntimeError("ARK runtime is not launched")
        Executor.get_global_executor().tensor_memcpy_host_to_device(
            dst._tensor, src
        )
        return dst

    def tensor_memcpy_device_to_host(self, dst: np.ndarray, src: Tensor):
        """
        Copy a tensor from device to host. If dst is None, a new numpy array will be created.
        """
        # Check the current ARK runtime status
        if (
            self.ark_runtime_state != ARKRuntimeState.launch
            and self.ark_runtime_state != ARKRuntimeState.stop
        ):
            logging.error("ARK runtime is not launched")
            raise RuntimeError("ARK runtime is not launched")
        # Create a new numpy array if dst is None
        if dst is None:
            np_type = None
            if src.tensor_type() == TensorType.FP32:
                np_type = np.float32
            elif src.tensor_type() == TensorType.FP16:
                np_type = np.float16
            else:
                logging.error("Unsupported tensor type")
                raise TypeError("Unsupported tensor type")
            dst = np.empty(src.shape, dtype=np_type)
        Executor.get_global_executor().tensor_memcpy_device_to_host(
            dst, src._tensor
        )
        return dst


def init(rank: int = 0, world_size: int = 1):
    """
    Initialize the ARK runtime and create a global model that will record
    all the operations.
    """
    ARKRuntime.global_runtime = ARKRuntime(rank, world_size)


def launch():
    """
    Create an executor and schedule the ARK model. The scheduler will generate
    the CUDA kernels. The GPU context and the connection between GPUs will be
    initialized. The executor will compile the cuda kernels and launch the ARK runtime.
    """
    ARKRuntime.get_global_runtime().launch()


def run(iter: int = 1, async_run: bool = False):
    """
    Run the ARK program for iter iterations and wait for the kernel to finish.
    """
    ARKRuntime.get_global_runtime().run(iter, async_run)


def wait():
    """
    Wait for the kernel to finish.
    """
    ARKRuntime.get_global_runtime().wait()


def stop():
    """
    Stop the model and return the elapsed time in milliseconds.
    Once this is called, we need to call `launch()` again to run the model again.
    """
    ARKRuntime.get_global_runtime().stop()


def tensor_memcpy_host_to_device(dst: Tensor, src: np.ndarray):
    """
    Copy a tensor from host to device. Used for initializing the tensor on device.
    """
    return ARKRuntime.get_global_runtime().tensor_memcpy_host_to_device(
        dst, src
    )


def tensor_memcpy_device_to_host(dst: np.ndarray, src: Tensor):
    """
    Copy a tensor from device to host. If dst is None, a new numpy array will be created.
    """
    return ARKRuntime.get_global_runtime().tensor_memcpy_device_to_host(
        dst, src
    )
