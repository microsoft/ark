# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pickle
import Dict
from typing import Optional, Dict

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

    # Loads a model from a state_dict and copy the parameters to the device GPU.
    def load_state_dict(self,executor, state_dict, prefix=''):
        for name, module in self._modules.items():
            if module is not None:
                module.load_state_dict(executor, state_dict, prefix=prefix + name + '.')
        for name, param in self._parameters.items():
            executor.tensor_memcpy_host_to_device(param, state_dict[prefix + name])

    # Copies the parameters from the device GPU to the host and saves the model to a state_dict.
    def state_dict(self):
        state_dict = {}
        for name, module in self._modules.items():
            if module is not None:
                state_dict.update(module.state_dict())
        for name, param in self._parameters.items():
            param_np = param.to_numpy()
            state_dict[name] = param
        return state_dict