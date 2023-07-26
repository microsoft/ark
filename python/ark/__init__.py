# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

ark_root = os.environ.get("ARK_ROOT", None)
if ark_root is None:
    os.environ["ARK_ROOT"] = os.path.abspath(os.path.dirname(__file__))

from ._ark_core import init, srand, rand, Dims, Tensor, TensorBuf, TensorType, Model
from .module import Module
from .executor import Executor, tensor_memcpy_device_to_host, tensor_memcpy_host_to_device
from .serialize import save, load, convert_state_dict_to_pytorch, convert_state_dict_to_numpy

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