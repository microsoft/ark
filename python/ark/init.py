# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import _ark_core
from .model import Model
from .runtime import _RuntimeState


def init(keep_runtime: bool = False):
    """Initializes ARK."""
    Model.reset()
    if not keep_runtime and _RuntimeState.executor is not None:
        if not _RuntimeState.executor.destroyed():
            _RuntimeState.executor.destroy()
    _ark_core.init()
