# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import numpy as np
from typing import Any, Dict, List, Union
from .tensor import Tensor, Parameter
from .runtime import Runtime, DefaultPlanner
from .ops import tensor
from .data_type import DataType
import ark

try:
    import torch
    _no_torch = False
except ImportError:
    from . import torch_mock as torch

    _no_torch = True


class Module(torch.nn.Module):
    """
    Base class for all neural network modules.
    """

    def __init__(self):
        super().__init__()
        # The submodules of the module.
        self.sub_modules: dict[str, "Module"] = dict()
        # The parameters of the module.
        self.parameters: dict[str, Parameter] = dict()
        # ARK parameters that are not yet initialized.
        self.deffered_parmas: dict[str, Parameter] = dict()

    def __setattr__(self, __name: str, __value: Any) -> None:
        """
        When setting an attribute, if the attribute is a Module, add it to
        the sub_modules. If the attribute is a Tensor and this Tensor is a
        parameter, add it to the parameters. If the attribute is a
        torch.nn.Parameter, convert it to an ARK Parameter before adding.
        """
        if isinstance(__value, Module):
            self.register_module(__name, __value)
        elif isinstance(__value, Parameter):
            self.register_parameter(__name, __value)
        elif not _no_torch and isinstance(__value, torch.nn.Parameter):
            __value = Parameter(__value)
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
        if param.torch_param is None:
            self.deffered_parmas[name] = param

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
            if data is None:
                continue
            param.copy(data)
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

    def forward(self, *args: Any, **kwargs: Any):
        return _ARKFunction.apply(self, *args, **kwargs)

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
        self.forward_input_args = []
        self.forward_input_kwargs = {}
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
                        tensor(
                            list(arg.shape),
                            DataType.from_torch(arg.dtype),
                        )
                    )
                    self.forward_input_args.append(
                        self.forward_input_tensor_args[-1]
                    )
                else:
                    self.forward_input_args.append(arg)
            for key, value in kwargs.items():
                if isinstance(value, torch.Tensor):
                    self.forward_input_tensor_kwargs[key] = tensor(
                        list(value.shape),
                        DataType.from_torch(value.dtype),
                    )
                    self.forward_input_kwargs[key] = (
                        self.forward_input_tensor_kwargs[key]
                    )
                else:
                    self.forward_input_kwargs[key] = value
            self.forward_output = self.build_forward(
                *self.forward_input_args,
                **self.forward_input_kwargs,
            )
            self.built_forward = True

        with Runtime.get_runtime() as rt:
            rt.launch(plan=DefaultPlanner().plan())
            for tns, arg in zip(self.forward_input_tensor_args, args):
                tns.copy(arg)
            for key, value in self.forward_input_tensor_kwargs.items():
                value.copy(kwargs[key])

            rt.run()
            return _recursive_ark_to_torch(self.forward_output)

class _ARKFunction(torch.autograd.Function):
    """
    Facilitates the integration of ARK modules with PyTorch's
    autograd system by defining custom forward and backward passes that
    utilize the user's defined ARK module.
    """
    @staticmethod
    def forward(ctx, ark_module, input):
        """
        Returns a PyTorch tensor that is the result
        of the forward pass of the ARK module.
        """
        ark.init(keep_runtime=True)
        ctx.ark_module = ark_module
        input_ark = Tensor.from_torch(input)
        output = ark_module.forward(input_ark)
        rt = Runtime.get_runtime()
        forward_plan = DefaultPlanner().plan()
        rt.launch(plan=forward_plan)
        rt.run()
        output = output.get_torch_view()
        rt.reset(persist=True)
        return output
        
    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        Converts the gradient outputs to ARK format, computes the gradients for the input
        and parameters using the ARK module backwards pass, and updates the gradients of the corresponding
        PyTorch parameters.
        """
        ark.init(keep_runtime=True)
        ark_grad_outputs = [Tensor.from_torch(grad) for grad in grad_outputs]
        grad_input, *grad_weights = ctx.ark_module.backward(*ark_grad_outputs)
        params_dict = ctx.ark_module.params_dict()
        rt = Runtime.get_runtime()
        backward_plan = DefaultPlanner().plan()
        rt.launch(plan=backward_plan)
        rt.run()
        output = grad_input.get_torch_view()
        for _, param in params_dict.items():
            if param.staged_tensor is not None:
                pytorch_grad = param.staged_tensor.get_torch_view()
                param.torch_param.grad = pytorch_grad
        rt.reset(persist=True)
        return (output, None)

class RuntimeModule(torch.nn.Module):
    """
    Wraps an ARK module to be used as a PyTorch autograd function.
    """
    def __init__(self, ark_module):
        super().__init__()
        self.ark_module = ark_module

    def forward(self, input):
        return _ARKFunction.apply(input, self.ark_module)