# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, Callable, Any
import numpy as np
from .tensor import Tensor
from .executor import Executor
import logging


class Module:
    """
    Base class for all neural network modules.
    """

    # The submodules of the module.
    sub_modules: Dict[str, "Module"]
    # The parameters of the module.
    parameters: Dict[str, Tensor]

    def __init__(self):
        self.sub_modules = dict()
        self.parameters = dict()

    # Adds a child module to the current module.
    def register_module(self, name: str, module: "Module") -> None:
        if not isinstance(module, Module):
            logging.error("module must be a Module")
        self.sub_modules[name] = module

    # Adds a parameter to the module.
    def register_parameter(self, name: str, param: Tensor) -> None:
        if not isinstance(param, Tensor):
            logging.error("param must be a Tensor")
        self.parameters[name] = param

    def __setattr__(self, __name: str, __value: Any) -> None:
        """
        When setting an attribute, if the attribute is a Module, add it to
        the sub_modules. If the attribute is a Tensor and this Tensor is a
        parameter, add it to the parameters.
        """
        if isinstance(__value, Module):
            self.register_module(__name, __value)
        elif isinstance(__value, Tensor):
            if __value.is_parameter:
                self.register_parameter(__name, __value)
        super().__setattr__(__name, __value)

    def load_state_dict(self, state_dict, prefix="", executor: Executor = None):
        """
        Loads a model from a state_dict and copy the parameters to the device GPU.
        Must be called after the executor is launched.
        """
        print("Loading model from state_dict")
        if executor is None:
            executor = Executor.get_executor()
        for name, module in self.sub_modules.items():
            if module is not None:
                module.load_state_dict(
                    state_dict, prefix=prefix + name + ".", executor=executor
                )
        for name, param in self.parameters.items():
            executor.tensor_memcpy_host_to_device(
                param._tensor, state_dict[prefix + name]
            )

    def state_dict(self, prefix="", executor: Executor = None):
        """
        Copies the parameters from the device GPU to the host and saves the model to a state_dict.
        Must be called after the executor is launched.
        """
        if executor is None:
            executor = Executor.get_executor()
        state_dict = {}
        for name, module in self.sub_modules.items():
            if module is not None:
                state_dict.update(
                    module.state_dict(
                        prefix=prefix + name + ".", executor=executor
                    )
                )
        for name, param in self.parameters.items():
            param_np = np.zeros(param.shape, dtype=np.float16)
            executor.tensor_memcpy_device_to_host(param_np, param._tensor)
            state_dict[prefix + name] = param_np
        return state_dict

    forward: Callable[..., Any] = NotImplemented
    backward: Callable[..., Any] = NotImplemented

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)
