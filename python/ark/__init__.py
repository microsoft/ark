# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

if os.environ.get("ARK_ROOT", None) is None:
    os.environ["ARK_ROOT"] = os.path.abspath(os.path.dirname(__file__))

from . import _ark_core

__version__ = _ark_core.version()


def version():
    """Returns the version of ARK."""
    return __version__


from .runtime import Runtime
from .data_type import DataType, fp32, fp16, int32, byte
from .tensor import Dims, Tensor, TensorBuf, Parameter
from .module import Module
from .serialize import save, load
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
    rmsnorm,
    softmax,
    transpose,
    matmul,
    im2col,
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
    embedding,
)


def init():
    """Initializes ARK."""
    _ark_core.init()
    Model.reset()


def srand(seed):
    """Sets the seed for random number generation."""
    _ark_core.srand(seed)


def set_rank(rank):
    """Sets the rank of the current process."""
    Model.set_rank(rank)


def set_world_size(world_size):
    """Sets the world size of the current process."""
    Model.set_world_size(world_size)
