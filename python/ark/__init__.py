# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

ark_root = os.environ.get("ARK_ROOT", None)
if ark_root is None:
    os.environ["ARK_ROOT"] = os.path.abspath(os.path.dirname(__file__))

from ._ark_core import (
    init,
    srand,
    rand,
    Dims,
    TensorBuf,
    TensorType,
    Tensor,
)

from .runtime import init_model, launch, run

from .module import Module
from .executor import (
    Executor,
    tensor_memcpy_device_to_host,
    tensor_memcpy_host_to_device,
)
from .serialize import (
    save,
    load,
    convert_state_dict,
    convert_state_dict_to_numpy,
)
from .trainer import Optimizer, Trainer
from .model import (
    Model,
    tensor,
    reshape,
    identity,
    sharding,
    reduce_sum,
    reduce_mean,
    reduce_max,
    layernorm,
    softmax,
    transpose,
    matmul,
    im2col,
    conv2d,
    max_pool,
    scale,
    relu,
    gelu,
    add,
    mul,
    send,
    send_done,
    recv,
    send_mm,
    recv_mm,
    all_gather,
    all_reduce,
)


__all__ = [
    "init",
    "srand",
    "rand",
    "Dims",
    "TensorBuf",
    "TensorType",
    "Tensor",
    "init_model",
    "launch",
    "run",
    "Module",
    "Executor",
    "tensor_memcpy_device_to_host",
    "tensor_memcpy_host_to_device",
    "save",
    "load",
    "convert_state_dict",
    "convert_state_dict_to_numpy",
    "Optimizer",
    "Trainer",
    "Model",
    "tensor",
    "reshape",
    "identity",
    "sharding",
    "reduce_sum",
    "reduce_mean",
    "reduce_max",
    "layernorm",
    "softmax",
    "transpose",
    "matmul",
    "im2col",
    "conv2d",
    "max_pool",
    "scale",
    "relu",
    "gelu",
    "add",
    "mul",
    "send",
    "send_done",
    "recv",
    "send_mm",
    "recv_mm",
    "all_gather",
    "all_reduce",
]
