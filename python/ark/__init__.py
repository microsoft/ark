# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import os

if os.environ.get("ARK_ROOT", None) is None:
    os.environ["ARK_ROOT"] = os.path.abspath(os.path.dirname(__file__))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _ark_core
from .model import Model


__version__ = _ark_core.version()


def version():
    """Returns the version of ARK."""
    return __version__


def srand(seed):
    """Sets the seed for random number generation."""
    _ark_core.srand(seed)


def set_rank(rank):
    """Sets the rank of the current process."""
    Model.set_rank(rank)


def set_world_size(world_size):
    """Sets the world size of the current process."""
    Model.set_world_size(world_size)


from .init import init
from .tensor import Dims, Tensor, Parameter
from .module import Module
from .runtime import Runtime
from .serialize import save, load
from .data_type import (
    DataType,
    fp16,
    fp32,
    int32,
    uint32,
    int8,
    uint8,
    byte,
)
from .ops import (
    tensor,
    parameter,
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
    # im2col,
    exp,
    sqrt,
    rsqrt,
    rope,
    relu,
    gelu,
    sigmoid,
    add,
    sub,
    mul,
    div,
    # send,
    # send_done,
    # recv,
    # all_gather,
    # local_all_gather,
    # local_reduce_scatter,
    all_reduce,
    # local_all_reduce,
    # local_all_reduce_packet,
    embedding,
    cast,
    constant,
    ones,
    zeros,
)
from .planner import *
from .error import *
