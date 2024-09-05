# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from . import core
from .model import Model
from .runtime import RuntimeState

__all__ = ["init"]


def init():
    """Initializes ARK."""
    Model.reset()
    if RuntimeState.runtime is not None:
        del RuntimeState.runtime
        RuntimeState.runtime = None
    core.init()
