# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

ark_root = os.environ.get("ARK_ROOT", None)
if ark_root is None:
    os.environ["ARK_ROOT"] = os.path.abspath(os.path.dirname(__file__))

from . import _ark_core

__version__ = _ark_core.version()

from ._ark_core import (
    init,
    srand,
    rand,
    Dims,
    TensorBuf,
    TensorType,
)

from .runtime import Runtime

from .tensor import Tensor, Parameter, FP16, FP32, INT32, BYTE

from .module import Module
from .executor import Executor
from .serialize import (
    save,
    load,
)

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
    max_pool,
    scale,
    relu,
    gelu,
    sigmoid,
    exp,
    sqrt,
    rope,
    add,
    sub,
    mul,
    div,
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
    "Parameter",
    "FP16",
    "FP32",
    "INT32",
    "BYTE",
    "Runtime",
    "Module",
    "Executor",
    "save",
    "load",
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
    "max_pool",
    "scale",
    "relu",
    "gelu",
    "sigmoid",
    "exp",
    "sqrt",
    "rope",
    "add",
    "sub",
    "mul",
    "div",
    "send",
    "send_done",
    "recv",
    "send_mm",
    "recv_mm",
    "all_gather",
    "all_reduce",
]
