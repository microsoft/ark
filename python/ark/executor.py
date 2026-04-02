# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .core import CoreExecutor


__all__ = ["Executor"]


class ExecutorState:
    executor: CoreExecutor = None


class Executor:
    @staticmethod
    def get() -> CoreExecutor:
        if ExecutorState.executor is None:
            ExecutorState.executor = CoreExecutor()
        return ExecutorState.executor

    @staticmethod
    def reset() -> None:
        if ExecutorState.executor is None:
            return
        ExecutorState.executor.destroy()
        ExecutorState.executor = None
