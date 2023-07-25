# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

ark_root = os.environ.get("ARK_ROOT", None)
if ark_root is None:
    os.environ["ARK_ROOT"] = os.path.abspath(os.path.dirname(__file__))

from ._ark_core import init, srand, rand, Dims,Model, Executor, Tensor, TensorBuf, TensorType
from .module import Module
from .serialize import save_state_dict, load_state_dict

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
    "save_state_dict",
    "load_state_dict",
]