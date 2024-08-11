# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import _ark_core
from .model import Model
from .runtime import _RuntimeState


def init():
    """Initializes ARK."""
    Model.reset()
    if _RuntimeState.runtime is not None:
        del _RuntimeState.runtime
        _RuntimeState.runtime = None
    _ark_core.init()
