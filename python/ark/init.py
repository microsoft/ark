# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from . import _ark_core
from .model import Model
from .runtime import _RuntimeState


def init():
    """Initializes ARK."""
    Model.reset()
    if _RuntimeState.runtime:
        _RuntimeState.delete_all()
    _ark_core.init()
