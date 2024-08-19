# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import numpy as np
from typing import Any, Dict, Union
from .tensor import Parameter
from .torch import torch, _no_torch
from .runtime import Runtime
from .model import Model
from .data_type import DataType
from .ops import placeholder


class Module:
    """
    Base class for all neural network modules.
    """

    def __init__(self):
        super().__init__()
        # The submodules of the module.
        self.sub_modules: dict[str, "Module"] = dict()
        # The parameters of the module.
        self.parameters: dict[str, Parameter] = dict()

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
            shape, dtype = list(__value.shape), DataType.from_torch(
                __value.dtype
            )
            __value = Parameter(placeholder(shape, dtype, data=__value), True)
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
        stream: int = 0,
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
            param.copy(data, stream=stream)
            all_keys.remove(name)
        if all_keys:
            logging.warning(
                f"{len(all_keys)} unused parameter(s) in state_dict"
            )

    def state_dict(
        self,
        prefix: str = "",
        mode: str = "numpy",
        stream: int = 0,
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        Copies the parameters from the device GPU to the host and saves the
        model to a state_dict.
        Must be called after the executor is launched.
        """
        if mode == "numpy":
            return {
                k: v.to_numpy(stream=stream)
                for k, v in self.params_dict(prefix).items()
            }
        elif mode == "torch":
            return {
                k: v.to_torch(stream=stream)
                for k, v in self.params_dict(prefix).items()
            }
        raise ValueError(f"Unsupported mode: {mode}")

    def forward(self, *args: Any, **kwargs: Any) -> Any: ...

    def backward(self, *args: Any, **kwargs: Any) -> Any: ...

    def initialize(self):
        for param in self.parameters.values():
            param.initialize()
        for module in self.sub_modules.values():
            module.initialize()


class _Function(torch.autograd.Function):
    """
    Facilitates the integration of ARK modules with PyTorch's
    autograd system by defining custom forward and backward passes that
    utilize the user's defined ARK module.
    """

    @staticmethod
    def forward(ctx, ark_module, *args, **kwargs):
        """
        Returns a PyTorch tensor that is the result
        of the forward pass of the ARK module.
        """
        Model.reset()
        ctx.ark_module = ark_module
        input_args, input_kwargs = [], {}
        input_requires_grad = 0
        for arg in args:
            if isinstance(arg, torch.Tensor):
                shape, dtype = list(arg.shape), DataType.from_torch(arg.dtype)
                input_args.append(placeholder(shape, dtype, data=arg))
                if arg.requires_grad:
                    input_requires_grad += 1
            else:
                input_args.append(arg)
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                shape, dtype = list(arg.shape), DataType.from_torch(arg.dtype)
                input_kwargs[k] = placeholder(shape, dtype, data=v)
                if v.requires_grad:
                    input_requires_grad += 1
            else:
                input_kwargs[k] = v
        ctx.num_inp_grad = input_requires_grad
        output = ark_module.forward(*input_args, **input_kwargs)
        rt = Runtime.get_runtime()
        rt.launch()
        rt.run()
        rt.stop()
        output = output.to_torch()
        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        Converts the gradient outputs to ARK format, computes the gradients for the input
        and parameters using the ARK module backwards pass, and updates the gradients of the corresponding
        PyTorch parameters.
        """
        Model.reset()
        # i think we should support placeholder initialization
        # with just pytorch tensor
        ark_grad_outputs = []
        for grad in grad_outputs:
            shape, dtype = list(grad.shape), DataType.from_torch(grad.dtype)
            ark_grad_outputs.append(placeholder(shape, dtype, data=grad))
        grads = ctx.ark_module.backward(*ark_grad_outputs)
        grad_inputs, grad_weights = (
            grads[: ctx.num_inp_grad],
            grads[ctx.num_inp_grad :],
        )
        params_dict = ctx.ark_module.params_dict()
        rt = Runtime.get_runtime()
        rt.launch()
        rt.run()
        rt.stop()
        grad_inputs = [grad.to_torch() for grad in grad_inputs]
        for _, param in params_dict.items():
            if param.staged_tensor is not None:
                pytorch_grad = param.staged_tensor.to_torch()
                param.torch_param.grad = pytorch_grad
        return (None, *grad_inputs)


class RuntimeModule(torch.nn.Module):
    """
    Wraps an ARK module to be used as a PyTorch autograd function.
    """

    def __init__(self, ark_module):
        super().__init__()
        self.ark_module = ark_module

    def forward(self, *args, **kwargs):
        return _Function.apply(self.ark_module, *args, **kwargs)
