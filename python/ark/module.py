# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import numpy as np
from typing import Any, Dict, List, Union
from .tensor import Tensor, Parameter
from .runtime import Runtime, DefaultPlanner

try:
    import torch

    _no_torch = False
except ImportError:
    from . import torch_mock as torch

    _no_torch = True


class Module:
    """
    Base class for all neural network modules.
    """

    def __init__(self):
        # The submodules of the module.
        self.sub_modules: dict[str, "Module"] = dict()
        # The parameters of the module.
        self.parameters: dict[str, Parameter] = dict()

    def __setattr__(self, __name: str, __value: Any) -> None:
        """
        When setting an attribute, if the attribute is a Module, add it to
        the sub_modules. If the attribute is a Tensor and this Tensor is a
        parameter, add it to the parameters.
        """
        if isinstance(__value, Module):
            self.register_module(__name, __value)
        elif isinstance(__value, Parameter):
            self.register_parameter(__name, __value)
        super().__setattr__(__name, __value)

    def __call__(self, *args: Any, **kwargs: Any):
        return self.forward(*args, **kwargs)

    def register_module(self, name: str, module: "Module") -> None:
        """Adds a child module to the current module."""
        if not isinstance(module, Module):
            raise TypeError("module must be a Module")
        self.sub_modules[name] = module

    def register_parameter(self, name: str, param: Parameter) -> None:
        """Adds a parameter to the module."""
        if not isinstance(param, Parameter):
            raise TypeError("param must be a Parameter")
        self.parameters[name] = param

    def params_dict(self, prefix="") -> Dict[str, Parameter]:
        params_dict = {}
        for name, module in self.sub_modules.items():
            if module is not None:
                params_dict.update(
                    module.params_dict(prefix=prefix + name + ".")
                )
        for name, param in self.parameters.items():
            params_dict[prefix + name] = param
        return params_dict

    def load_state_dict(
        self,
        state_dict: Dict[str, Union[np.ndarray, torch.Tensor]],
        prefix: str = "",
    ):
        """
        Loads a model from a state_dict and copy the parameters to the device GPU.
        Must be called after the executor is launched.
        """
        logging.info("Loading model from state_dict")

        all_keys = set(state_dict.keys())
        pd = self.params_dict(prefix)
        for name, param in pd.items():
            data = state_dict.get(name, None)
            if isinstance(data, np.ndarray):
                param.from_numpy(data)
            elif isinstance(data, torch.Tensor):
                param.from_torch(data)
            else:
                continue
            all_keys.remove(name)
        if all_keys:
            logging.warning(
                f"{len(all_keys)} unused parameter(s) in state_dict"
            )

    def state_dict(
        self, prefix: str = "", mode: str = "numpy"
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        Copies the parameters from the device GPU to the host and saves the
        model to a state_dict.
        Must be called after the executor is launched.
        """
        if mode == "numpy":
            return {
                k: v.to_numpy() for k, v in self.params_dict(prefix).items()
            }
        elif mode == "torch":
            return {
                k: v.to_torch() for k, v in self.params_dict(prefix).items()
            }
        raise ValueError(f"Unsupported mode: {mode}")

    def forward(self, *args: Any, **kwargs: Any) -> Any: ...

    def backward(self, *args: Any, **kwargs: Any) -> Any: ...

    def initialize(self):
        for param in self.parameters.values():
            param.initialize()
        for module in self.sub_modules.values():
            module.initialize()


def _recursive_ark_to_torch(object):
    if isinstance(object, Tensor):
        return object.to_torch()
    if isinstance(object, dict):
        return {k: _recursive_ark_to_torch(v) for k, v in object.items()}
    if isinstance(object, list):
        return [_recursive_ark_to_torch(v) for v in object]
    return object


class RuntimeModule(Module):
    def __init__(self):
        if _no_torch:
            raise ImportError("torch is not available")
        super().__init__()
        self.built_forward = False
        self.built_backward = False
        self.forward_input_tensor_args: List[Tensor] = []
        self.forward_input_tensor_kwargs: Dict[str, Tensor] = {}
        self.forward_output = None
        self.backward_tensor_args = []
        self.backward_tensor_kwargs = {}

    def build_forward(self, *args: Any, **kwargs: Any) -> Any: ...

    def build_backward(self, *args: Any, **kwargs: Any) -> Any: ...

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if not self.built_forward:
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    self.forward_input_tensor_args.append(
                        Tensor.from_torch(arg)
                    )
            for key, value in kwargs.items():
                if isinstance(value, torch.Tensor):
                    self.forward_input_tensor_kwargs[key] = Tensor.from_torch(
                        value
                    )
            self.forward_output = self.build_forward(
                *self.forward_input_tensor_args,
                **self.forward_input_tensor_kwargs,
            )
            self.built_forward = True

        with Runtime.get_runtime() as rt:
            rt.launch(plan=DefaultPlanner().plan())
            for arg in self.forward_input_tensor_args:
                arg.initialize()
            for value in self.forward_input_tensor_kwargs.values():
                value.initialize()

            rt.run()
            return _recursive_ark_to_torch(self.forward_output)
