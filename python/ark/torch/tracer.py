# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

try:
    import torch
    import torch.nn as nn
    from torch import _dynamo as torchdynamo
except ImportError:
    raise ImportError("torch is required to use this module")

import operator
from typing import List

from ..tensor import Tensor
from ..runtime import Runtime
from ..model import Model
from ..data_type import DataType
from .. import ops


__all__ = ["tracer"]


class Tracer:
    def __init__(self, cls: torch.nn.Module):
        self.cls = cls
        self.placeholders = {}
        self.outputs = []

    def __call__(self, *args, **kwargs):
        print("Tracer called")

        def call_wrapper(instance, *args, **kwargs):
            @torchdynamo.optimize(self.trace)
            def call(*args, **kwargs):
                return instance(*args, **kwargs)

            return call(*args, **kwargs)

        def forward_ark(instance, *args, **kwargs):
            rt = Runtime.get_runtime()
            if not rt.launched():
                rt.launch(loop_mode=False)
            ph_map = {}
            for idx, arg in enumerate(args):
                ph_map[self.placeholders[("args", idx)]] = arg
            # TODO: support kwargs
            rt.run(tensor_mappings=ph_map)
            # TODO: support other formats of output
            if len(self.outputs) == 1:
                return self.outputs[0]
            return self.outputs

        # instance = self.cls(*args, **kwargs)
        self.cls.__call__ = call_wrapper
        self.cls.forward_ark = forward_ark
        return self.cls(*args, **kwargs)

    def trace(
        self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
    ):
        print("Compiler called")
        Model.reset()
        ark_tensors = {}
        cur_device: torch.device = None
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                fake_tensor = node.meta["grapharg"].fake_tensor
                t = ops.placeholder(
                    shape=fake_tensor.shape,
                    dtype=DataType.from_torch(fake_tensor.dtype),
                    name=node.name,
                )
                ark_tensors[node.name] = t
                source = node._dynamo_source
                local_name = source.base.local_name
                index = source.index
                self.placeholders[(local_name, index)] = t
            elif node.op == "get_attr":
                pass
            elif node.op == "call_function":
                if node.target == torch.nn.functional.silu:
                    t = ark_tensors.get(node.args[0].name, None)
                    if t is None:
                        raise ValueError(
                            f"Input tensor {node.args[0].name} not found"
                        )
                    t = ops.mul(t, ops.sigmoid(t))
                    ark_tensors[node.name] = t

                elif node.target == operator.mul:
                    t1 = ark_tensors.get(node.args[0].name, None)
                    t2 = ark_tensors.get(node.args[1].name, None)
                    if t1 is None:
                        raise ValueError(
                            f"Input tensor {node.args[0].name} not found"
                        )
                    if t2 is None:
                        raise ValueError(
                            f"Input tensor {node.args[1].name} not found"
                        )
                    t = ops.mul(t1, t2)
                    ark_tensors[node.name] = t
                else:
                    raise ValueError(f"Unsupported function {node.target}")
            elif node.op == "call_module":
                module = gm._modules[node.target]
                if isinstance(module, nn.Linear):
                    t = ark_tensors.get(node.args[0].name, None)
                    if t is None:
                        raise ValueError(
                            f"Input tensor {node.args[0].name} not found"
                        )

                    if cur_device is None:
                        cur_device = module.weight.device
                    elif cur_device != module.weight.device:
                        raise ValueError(
                            "All parameters must be on the same device. "
                            f"Expected {cur_device}, got {module.weight.device}"
                        )

                    weight = Tensor.from_torch(module.weight)
                    t = ops.matmul(t, weight, transpose_other=True)
                    if module.bias is not None:
                        bias = Tensor.from_torch(module.bias)
                        t = ops.add(t, bias)
                    ark_tensors[node.name] = t
                else:
                    raise ValueError(f"Unsupported module {module}")

            elif node.op == "output":
                t = ark_tensors.get(node.args[0][0].name, None)
                if t is None:
                    raise ValueError(
                        f"Input tensor {node.args[0][0].name} not found"
                    )
                self.outputs.append(t)

        Model.set_device_id(cur_device.index)
        return gm.forward


def tracer(cls: torch.nn.Module):
    return Tracer(cls)
