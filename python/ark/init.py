# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from . import _ark_core
from .model import Model
from .runtime import _RuntimeState


def init(keep_runtime: bool = False):
    """Initializes ARK."""
    Model.reset()
    if not keep_runtime and _RuntimeState.runtime:
        _RuntimeState.delete_all()
        _ark_core.init()
