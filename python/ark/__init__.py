# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

if os.environ.get("ARK_ROOT", None) is None:
    os.environ["ARK_ROOT"] = os.path.abspath(os.path.dirname(__file__))

from .core import version
from .model import Model


__version__ = version()


def version():
    """Returns the version of ARK."""
    return __version__


def set_rank(rank):
    """Sets the rank of the current process."""
    Model.set_rank(rank)


def set_world_size(world_size):
    """Sets the world size of the current process."""
    Model.set_world_size(world_size)


from .init import init
from .tensor import Dims, Tensor, Parameter
from .module import Module
from .runtime import *
from .serialize import save, load
from .data_type import *
from .profiler import Profiler
from .ops import *
from .planner import *
from .error import *
