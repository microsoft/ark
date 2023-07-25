# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pickle
import Dict
from typing import Optional, Dict, Callable, Any
import numpy as np
from ._ark_core import Model, Executor, Tensor 

class Module:
    _modules: Dict[str, Optional['Module']]
    _parameters: Dict[str, Optional['Tensor']]
    def __init__(self,model):
        self._modules = Dict[str, Optional['Module']]()
        self._Tensors = Dict[str, Optional['Tensor']]()
        self._model = model
        super().__init__()

    # Adds a child module to the current module.
    def register_module(self, name: str, module: Optional['Module']) -> None:
        self._modules[name] = module

    # Adds a parameter to the module.
    def register_parameter(self, name: str, param: Optional['Tensor']) -> None:
        self._parameters[name] = param

    def __setattr__(self, __name: str, __value: Any) -> None:
        if isinstance(__value, Module):
            self.register_module(__name, __value)
        elif isinstance(__value, Tensor):
            self.register_parameter(__name, __value)
        else:
            super().__setattr__(__name, __value)

    # Loads a model from a state_dict and copy the parameters to the device GPU.
    # Must be called after the executor is launched.
    def load_state_dict(self,executor, state_dict, prefix=''):
        for name, module in self._modules.items():
            if module is not None:
                module.load_state_dict(executor, state_dict, prefix=prefix + name + '.')
        for name, param in self._parameters.items():
            executor.tensor_memcpy_host_to_device(param, state_dict[prefix + name])

    # Copies the parameters from the device GPU to the host and saves the model to a state_dict.
    # Must be called after the executor is launched.
    def state_dict(self, executor):
        state_dict = {}
        for name, module in self._modules.items():
            if module is not None:
                state_dict.update(module.state_dict())
        for name, param in self._parameters.items():
            param_np = np.zeros(param.shape, dtype=np.float16)
            executor.tensor_memcpy_device_to_host(param_np, param)
            state_dict[name] = param_np
        return state_dict

    forward: Callable[..., Any] = NotImplemented
    backward: Callable[..., Any] = NotImplemented