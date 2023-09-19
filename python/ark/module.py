# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import numpy as np
from typing import Any, Dict
from .tensor import Tensor


class Module:
    """
    Base class for all neural network modules.
    """

    def __init__(self):
        # The submodules of the module.
        self.sub_modules: dict[str, "Module"] = dict()
        # The parameters of the module.
        self.parameters: dict[str, Tensor] = dict()

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

    def __call__(self, *args: Any, **kwargs: Any):
        return self.forward(*args, **kwargs)

    def register_module(self, name: str, module: "Module") -> None:
        """Adds a child module to the current module."""
        if not isinstance(module, Module):
            logging.error("module must be a Module")
            raise TypeError("module must be a Module")
        self.sub_modules[name] = module

    def register_parameter(self, name: str, param: Tensor) -> None:
        """Adds a parameter to the module."""
        if not isinstance(param, Tensor):
            logging.error("param must be a Tensor")
            raise TypeError("param must be a Tensor")
        self.parameters[name] = param

    def load_state_dict(self, state_dict, prefix=""):
        """
        Loads a model from a state_dict and copy the parameters to the device GPU.
        Must be called after the executor is launched.
        """
        logging.info("Loading model from state_dict")
        for name, module in self.sub_modules.items():
            if module is not None:
                module.load_state_dict(state_dict, prefix=prefix + name + ".")
        for name, param in self.parameters.items():
            param.from_numpy(state_dict[prefix + name])

    def state_dict(self, prefix="") -> Dict[str, np.ndarray]:
        """
        Copies the parameters from the device GPU to the host and saves the model to a state_dict.
        Must be called after the executor is launched.
        """
        state_dict = {}
        for name, module in self.sub_modules.items():
            if module is not None:
                state_dict.update(module.state_dict(prefix=prefix + name + "."))
        for name, param in self.parameters.items():
            param_np = param.to_numpy()
            state_dict[prefix + name] = param_np
        return state_dict

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def backward(self, *args: Any, **kwargs: Any) -> Any:
        ...
