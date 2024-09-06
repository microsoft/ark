# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from . import core
from .model import Model
from .executor import Executor

__all__ = ["init"]


def init():
    """Initializes ARK."""
    Executor.reset()
    Model.reset()
    core.init()
