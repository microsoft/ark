# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

try:
    import torch
except ImportError:
    raise ImportError("torch is required to use this module")

import logging
from typing import List, Dict, Optional, Callable, Any

from ..planner import Planner, Plan
from ..tensor import Tensor
from ..runtime import Runtime
from ..model import Model
from .. import ops


__all__ = ["tracer"]


def handle_aten_add_scalar(
    node: torch.fx.node.Node, tensors: Dict[str, Tensor]
) -> Tensor:
    """Element-wise addition with a scalar"""
    t = tensors[node.args[0].name]
    value = node.args[1]
    return ops.add(t, value, name=node.name)


def handle_aten_add_tensor(
    node: torch.fx.node.Node, tensors: Dict[str, Tensor]
) -> Tensor:
    """Element-wise subtraction"""
    t1 = tensors[node.args[0].name]
    t2 = tensors[node.args[1].name]
    return ops.add(t1, t2, name=node.name)


def handle_aten_sub_tensor(
    node: torch.fx.node.Node, tensors: Dict[str, Tensor]
) -> Tensor:
    """Element-wise subtraction"""
    t1 = tensors[node.args[0].name]
    t2 = tensors[node.args[1].name]
    return ops.sub(t1, t2, name=node.name)


def handle_aten_mul_tensor(
    node: torch.fx.node.Node, tensors: Dict[str, Tensor]
) -> Tensor:
    """Element-wise multiplication"""
    t1 = tensors[node.args[0].name]
    t2 = tensors[node.args[1].name]
    return ops.mul(t1, t2, name=node.name)


def handle_aten_t(
    node: torch.fx.node.Node, tensors: Dict[str, Tensor]
) -> Tensor:
    """Transpose"""
    t = tensors[node.args[0].name]
    perm = list(range(len(t.shape())))
    if len(perm) < 2:
        raise ValueError(f"Expected at least 2 dimensions, got {len(perm)}")
    perm[-2], perm[-1] = perm[-1], perm[-2]
    return ops.transpose(t, perm=perm, name=node.name)


def handle_aten_mm(
    node: torch.fx.node.Node, tensors: Dict[str, Tensor]
) -> Tensor:
    """Matrix multiplication"""
    input = tensors[node.args[0].name]
    weight = tensors[node.args[1].name]
    return ops.matmul(input, weight, name=node.name)


def handle_aten_addmm(
    node: torch.fx.node.Node, tensors: Dict[str, Tensor]
) -> Tensor:
    """Matrix multiplication followed by addition"""
    bias = tensors[node.args[0].name]
    input = tensors[node.args[1].name]
    weight = tensors[node.args[2].name]
    t = ops.matmul(input, weight)
    t = ops.add(t, bias, name=node.name)
    return t


def handle_aten_silu(
    node: torch.fx.node.Node, tensors: Dict[str, Tensor]
) -> Tensor:
    """Sigmoid Linear Unit"""
    t = tensors[node.args[0].name]
    return ops.mul(t, ops.sigmoid(t), name=node.name)


def handle_aten_sum_dim_intlist(
    node: torch.fx.node.Node, tensors: Dict[str, Tensor]
) -> Tensor:
    """Sum with specified dimensions"""
    if len(node.args[1]) != 1:
        raise NotImplementedError("Multiple dimensions are not supported")
    t = tensors[node.args[0].name]
    axis = node.args[1][0]
    keepdims = node.args[2]
    return ops.reduce_sum(t, axis=axis, keepdims=keepdims, name=node.name)


def handle_aten_view(
    node: torch.fx.node.Node, tensors: Dict[str, Tensor]
) -> Tensor:
    """Reshape"""
    t = tensors[node.args[0].name]
    shape = node.args[1]
    return ops.reshape(t, shape, name=node.name)


def handle_aten_sigmoid(
    node: torch.fx.node.Node, tensors: Dict[str, Tensor]
) -> Tensor:
    """Sigmoid"""
    t = tensors[node.args[0].name]
    return ops.sigmoid(t, name=node.name)


def handle_aten_empty_like(
    node: torch.fx.node.Node, tensors: Dict[str, Tensor]
) -> Tensor:
    """Create an empty tensor with the same shape"""
    t = tensors[node.args[0].name]
    new_t = ops.tensor(t.shape(), dtype=t.dtype())
    new_t = ops.identity(new_t, deps=[t], name=node.name)
    return new_t


def handle_aten_fill_scalar(
    node: torch.fx.node.Node, tensors: Dict[str, Tensor]
) -> Tensor:
    """Fill a tensor with a scalar value"""
    t = tensors[node.args[0].name]
    value = node.args[1]
    return ops.copy(value, t, name=node.name)


def handle_aten_mse_loss(
    node: torch.fx.node.Node, tensors: Dict[str, Tensor]
) -> Tensor:
    """Mean Squared Error loss"""
    input = tensors[node.args[0].name]
    target = tensors[node.args[1].name]
    t = ops.sub(input, target)
    t = ops.mul(t, t)
    t = ops.reshape(t, [-1])
    t = ops.reduce_mean(t, axis=0)
    return t


def handle_aten_mse_loss_backward(
    node: torch.fx.node.Node, tensors: Dict[str, Tensor]
) -> Tensor:
    """Backward pass for Mean Squared Error loss"""
    grad_output = tensors[node.args[0].name]
    input = tensors[node.args[1].name]
    target = tensors[node.args[2].name]
    grad_input = ops.sub(input, target)
    grad_input = ops.mul(grad_input, grad_output)
    grad_input = ops.mul(grad_input, 2.0 / grad_input.shape()[0])
    return grad_input


_REGISTRY_FUNCTION_HANDLER: Dict[str, Callable] = {
    "aten::add.Scalar": handle_aten_add_scalar,
    "aten::add.Tensor": handle_aten_add_tensor,
    "aten::sub.Tensor": handle_aten_sub_tensor,
    "aten::mul.Tensor": handle_aten_mul_tensor,
    "aten::t": handle_aten_t,
    "aten::mm": handle_aten_mm,
    "aten::addmm": handle_aten_addmm,
    "aten::silu": handle_aten_silu,
    "aten::sum.dim_IntList": handle_aten_sum_dim_intlist,
    "aten::view": handle_aten_view,
    "aten::sigmoid": handle_aten_sigmoid,
    "aten::empty_like": handle_aten_empty_like,
    "aten::fill.Scalar": handle_aten_fill_scalar,
    "aten::mse_loss": handle_aten_mse_loss,
    "aten::mse_loss_backward": handle_aten_mse_loss_backward,
}


class Tracer:
    def __init__(self):
        self.tensors: Dict[str, Tensor] = {}
        self.params: Optional[List[torch.nn.Parameter]] = None
        self.params_idx: int = 0
        self.inputs_fw: List[Tensor] = []
        self.inputs_bw: List[Tensor] = []
        self.outputs_fw: List[Tensor] = []
        self.outputs_bw: List[Tensor] = []
        self.plan_fw: Optional[Plan] = None
        self.plan_bw: Optional[Plan] = None
        self.device: Optional[torch.device] = None
        self.failed: bool = False
        self.launched_fw: bool = False
        self.launched_bw: bool = False

    def __call__(self, target: Callable) -> Callable:
        is_module = issubclass(target, torch.nn.Module)
        is_function = callable(target) and not isinstance(target, type)
        if not is_module and not is_function:
            raise ValueError("Tracer can only be applied to a subclass of `torch.nn.Module` or a function")
        if is_function:
            return torch._dynamo.optimize(self.autograd_trace_)(target)

        target.forward_torch = target.forward

        def forward_wrapper(instance: torch.nn.Module, *args, **kwargs) -> Any:
            if self.plan_fw is None:
                return instance.forward_torch(*args, **kwargs)
            rt = Runtime.get_runtime()
            if not self.launched_fw:
                rt.launch(
                    plan=self.plan_fw,
                    device_id=self.device.index,
                    loop_mode=False,
                )
                self.launched_fw = True
                self.launched_bw = False

            ph_map = {ph: data for ph, data in zip(self.inputs_fw, args)}
            rt.run(tensor_mappings=ph_map)
            # TODO: how to get the output tensor(s)?
            return self.outputs_fw[0]

        def backward_wrapper(instance: torch.nn.Module, *args, **kwargs):
            if self.plan_bw is None:
                return instance.forward_torch(*args, **kwargs)
            rt = Runtime.get_runtime()
            if not self.launched_bw:
                rt.launch(
                    plan=self.plan_bw,
                    device_id=self.device.index,
                    loop_mode=False,
                )
                self.launched_bw = True
                self.launched_fw = False

            ph_map = {ph: data for ph, data in zip(self.inputs_bw, args)}
            rt.run(tensor_mappings=ph_map)
            for i, param in enumerate(self.params):
                param.grad = self.outputs_bw[i].to_torch()

        def call_wrapper(instance: torch.nn.Module, *args, **kwargs) -> Any:
            if self.params is None:
                params = []
                for _, param in instance.named_parameters(remove_duplicate=False):
                    params.append(param)
                for _, param in instance.named_buffers(remove_duplicate=False):
                    params.append(param)
                self.params = params

            @torch._dynamo.optimize(self.autograd_trace_)
            def call(*args, **kwargs):
                return instance.forward_torch(*args, **kwargs)

            return call(*args, **kwargs)

        target.forward_ark = forward_wrapper
        target.backward_ark = backward_wrapper
        target.__call__ = call_wrapper
        return target

    def autograd_trace_(
        self, gm: torch.nn.Module, _: List[torch.Tensor]
    ) -> Callable:
        def fw_compiler(gm: torch.fx.GraphModule, _):
            logging.info("==== FW Starts ====")
            return self.autograd_trace_impl_(gm, _, True)

        def bw_compiler(gm: torch.fx.GraphModule, _):
            logging.info("==== BW Starts ====")
            return self.autograd_trace_impl_(gm, _, False)

        return torch._dynamo.backends.common.aot_autograd(
            fw_compiler=fw_compiler, bw_compiler=bw_compiler
        )(gm, _)

    def autograd_trace_impl_(
        self, gm: torch.fx.GraphModule, _: List[torch.Tensor], is_fw: bool
    ) -> Callable:

        def run(args) -> Any:
            Model.reset()
            if not self.failed:
                for node in gm.graph.nodes:
                    logging.info(node.format_node(), node.meta)
                    if not self.handle_node_(node, is_fw):
                        print(f"Failed to handle node {node.format_node()}")
                        self.failed = True
                        break
                if not self.failed:
                    Model.set_device_id(self.device.index)
                    if is_fw:
                        self.plan_fw = Planner(self.device.index).plan()
                    else:
                        self.plan_bw = Planner(self.device.index).plan()
            return torch.fx.Interpreter(gm).boxed_run(args)

        run._boxed_call = True
        return run

    def handle_node_(self, node: torch.fx.node.Node, is_fw: bool) -> bool:
        if node.op == "placeholder":
            t = self.tensors.get(node.name, None)
            if t is not None:
                return True
            meta = node.meta["tensor_meta"]
            if len(self.params) > self.params_idx:
                # placeholder for parameter
                param = self.params[self.params_idx]
                self.params_idx += 1
                if param.dtype != meta.dtype:
                    raise ValueError(
                        f"Expected dtype {meta.dtype}, got {param.dtype}"
                    )
                if self.device is None:
                    if param.device.type != "cuda":
                        raise ValueError(
                            f"Expected device cuda, got {param.device.type}"
                        )
                    self.device = param.device
                elif self.device != param.device:
                    raise ValueError(
                        "All parameters must be on the same device. "
                        f"Expected {self.device}, got {param.device}"
                    )
                data = param.data_ptr()
            else:
                # no more parameter -- remainings are inputs
                data = 0
            t = ops.placeholder(
                shape=meta.shape,
                dtype=ops.DataType.from_torch(meta.dtype),
                name=node.name,
                data=data,
            )
            self.tensors[node.name] = t
            if data == 0:
                if is_fw:
                    self.inputs_fw.append(t)
                else:
                    self.inputs_bw.append(t)
        elif node.op == "output":
            outputs_list = self.outputs_fw if is_fw else self.outputs_bw
            if outputs_list:
                raise ValueError("Multiple output nodes are unexpected")
            for out in node.args[0]:
                if isinstance(out, torch.fx.node.Node):
                    if out.name not in self.tensors:
                        raise ValueError(f"Output tensor {out.name} not found")
                    outputs_list.append(self.tensors[out.name])
                else:
                    outputs_list.append(out)
        elif node.op == "call_function":
            target_name = node.target.name()
            if target_name not in _REGISTRY_FUNCTION_HANDLER:
                logging.warning(
                    f"Unsupported function {target_name}. Usage: {node.format_node()}"
                )
                return False
            t = _REGISTRY_FUNCTION_HANDLER[target_name](node, self.tensors)
            self.tensors[node.name] = t
        else:
            raise ValueError(f"Unexpected node {node.format_node()}")
        return True


def tracer(target: Callable):
    return Tracer()(target)
