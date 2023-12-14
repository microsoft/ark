# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

if os.environ.get("ARK_ROOT", None) is None:
    os.environ["ARK_ROOT"] = os.path.abspath(os.path.dirname(__file__))

import ctypes

ctypes.CDLL(os.environ["ARK_ROOT"] + "/lib/libmscclpp.so")

from . import _ark_core
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
from .tensor import Dims, Tensor, TensorBuf, Parameter
from .model import Model, _REGISTRY_OPERATOR
from .module import Module
from .runtime import Runtime
from .serialize import save, load


# Read the version.
__version__ = _ark_core.version()

# Import operators.
for op_name, op_func in _REGISTRY_OPERATOR.items():
    globals()[op_name] = op_func


def version():
    """Returns the version of ARK."""
    return __version__


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
