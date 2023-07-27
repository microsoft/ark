# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

ark_root = os.environ.get("ARK_ROOT", None)
if ark_root is None:
    os.environ["ARK_ROOT"] = os.path.abspath(os.path.dirname(__file__))

from ._ark_core import (
    init,
    srand,
    rand,
    Dims,
    TensorBuf,
    TensorType,
    Tensor,
    _Model,
)
from .module import Module
from .executor import (
    Executor,
    tensor_memcpy_device_to_host,
    tensor_memcpy_host_to_device,
)
from .serialize import (
    save,
    load,
    convert_state_dict_to_pytorch,
    convert_state_dict_to_numpy,
)
from .trainer import optimizer, trainer
from .model import Model

# Convert all functions in Model to global functions
for name in dir(_Model):
    if not name.startswith("__") and callable(getattr(_Model, name)):
        func = getattr(_Model, name)

        def wrapper(*args, **kwargs):
            result = func(Model.global_model, *args, **kwargs)
            return result

        globals()[name] = wrapper

__all__ = [
    "init",
    "srand",
    "rand",
    "Dims",
    "Tensor",
    "TensorBuf",
    "TensorType",
    "Model",
    "Executor",
    "Module",
    "save",
    "load",
]
