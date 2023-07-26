# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, Dict, Callable, Any
import numpy as np
from ._ark_core import _Tensor
from .tensor import Tensor 
from .executor import Executor
from .model import Model
class Module:
    """
    Base class for all neural network modules.
    Usage:
    class TestModel(ark.Module):
        def __init__(self, model):
            super(TestModel, self).__init__(model)
            self.weight_1 = model.tensor(
                ark.Dims(64, 128), ark.TensorType.FP16
            )
            self.weight_2 = model.tensor(
                ark.Dims(128, 64), ark.TensorType.FP16
            )
    
        def forward(self, inputs):
            middle_result = self.model.matmul(inputs, self.weight_1, is_relu=True)
            middle_result1 = self.model.matmul(middle_result, self.weight_2)
            output = self.model.add(middle_result1, inputs)
            output_layernorm = self.model.layernorm(output)
            return output_layernorm
    """

    # The submodules of the module.
    _modules: Dict[str, Optional['Module']]
    # The parameters of the module.
    _parameters: Dict[str, Optional['Tensor']]
    def __init__(self, model):
        self._modules = dict()
        self._parameters = dict()
        self.model = model

    # Adds a child module to the current module.
    def register_module(self, name: str, module: Optional['Module']) -> None:
        self._modules[name] = module

    # Adds a parameter to the module.
    def register_parameter(self, name: str, param: Optional['Tensor']) -> None:
        self._parameters[name] = param

    def __setattr__(self, __name: str, __value: Any) -> None:
        if isinstance(__value, Module):
            self.register_module(__name, __value)
        elif isinstance(__value, Tensor) or isinstance(__value, _Tensor):
            self.register_parameter(__name, __value)
        super().__setattr__(__name, __value)

    # Loads a model from a state_dict and copy the parameters to the device GPU.
    # Must be called after the executor is launched.
    def load_state_dict(self, state_dict, prefix='', executor=None):
        print("Loading model from state_dict", self._parameters)
        if executor is None:
            executor = Executor.get_executor()
        for name, module in self._modules.items():
            if module is not None:
                module.load_state_dict(self.executor, state_dict, prefix=prefix + name + '.')
        for name, param in self._parameters.items():
            print("Loading parameter: " + prefix + name)
            executor.tensor_memcpy_host_to_device(param, state_dict[prefix + name])

    # Copies the parameters from the device GPU to the host and saves the model to a state_dict.
    # Must be called after the executor is launched.
    def state_dict(self, executor = None):
        if executor is None:
            executor = Executor.get_executor()
        state_dict = {}
        for name, module in self._modules.items():
            if module is not None:
                state_dict.update(module.state_dict())
        for name, param in self._parameters.items():
            ark_shape = param.shape()
            np_shape = []
            for i in range(param.ndims()):
                np_shape.append(ark_shape[i])
            param_np = np.zeros(np_shape, dtype=np.float16)
            executor.tensor_memcpy_device_to_host(param_np, param)
            state_dict[name] = param_np
        return state_dict

    forward: Callable[..., Any] = NotImplemented
    backward: Callable[..., Any] = NotImplemented

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

