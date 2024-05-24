# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import _ark_core
from .model import Model
from .runtime import _RuntimeState


def init():
    """Initializes ARK."""
    Model.reset()
    if _RuntimeState.executor is not None:
        if not _RuntimeState.executor.destroyed():
            _RuntimeState.executor.destroy()
    _ark_core.init()
