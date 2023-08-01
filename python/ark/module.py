# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, Dict, Callable, Any
import numpy as np
from ._ark_core import _Model, Tensor
from .executor import Executor


class Module:
    """
    Base class for all neural network modules.
    """

    # The submodules of the module.
    _sub_modules: Dict[str, Optional["Module"]]
    # The parameters of the module.
    _parameters: Dict[str, Optional["Tensor"]]
    # The gradient computed at backward stage. A map from parameter to gradient.
    _grads: Dict[Optional["Tensor"], Optional["Tensor"]]

    def __init__(self):
        self._sub_modules = dict()
        self._parameters = dict()

    # Adds a child module to the current module.
    def register_module(self, name: str, module: Optional["Module"]) -> None:
        self._sub_modules[name] = module

    # Adds a parameter to the module.
    def register_parameter(self, name: str, param: Optional["Tensor"]) -> None:
        self._parameters[name] = param

    def __setattr__(self, __name: str, __value: Any) -> None:
        if isinstance(__value, Module):
            self.register_module(__name, __value)
        elif isinstance(__value, Tensor):
            self.register_parameter(__name, __value)
        super().__setattr__(__name, __value)

    # Loads a model from a state_dict and copy the parameters to the device GPU.
    # Must be called after the executor is launched.
    def load_state_dict(self, state_dict, prefix="", executor: Executor = None):
        print("Loading model from state_dict")
        if executor is None:
            executor = Executor.get_executor()
        for name, module in self._sub_modules.items():
            if module is not None:
                module.load_state_dict(
                    state_dict, prefix=prefix + name + ".", executor=executor
                )
        for name, param in self._parameters.items():
            executor.tensor_memcpy_host_to_device(
                param, state_dict[prefix + name]
            )

    # Copies the parameters from the device GPU to the host and saves the model to a state_dict.
    # Must be called after the executor is launched.
    def state_dict(self, prefix="", executor: Executor = None):
        if executor is None:
            executor = Executor.get_executor()
        state_dict = {}
        for name, module in self._sub_modules.items():
            if module is not None:
                state_dict.update(
                    module.state_dict(
                        prefix=prefix + name + ".", executor=executor
                    )
                )
        for name, param in self._parameters.items():
            param_np = np.zeros(param.shape, dtype=np.float16)
            executor.tensor_memcpy_device_to_host(param_np, param)
            state_dict[prefix + name] = param_np
        return state_dict

    forward: Callable[..., Any] = NotImplemented
    backward: Callable[..., Any] = NotImplemented

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)
