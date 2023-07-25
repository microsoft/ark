# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

ark_root = os.environ.get("ARK_ROOT", None)
if ark_root is None:
    os.environ["ARK_ROOT"] = os.path.abspath(os.path.dirname(__file__))

from ._ark_core import init, srand, rand, Dims, Executor, Model, Tensor, TensorBuf, TensorType

__all__ = [
    "init",
    "srand",
    "rand",
    "Dims",
    "Executor",
    "Model",
    "Tensor",
    "TensorBuf",
    "TensorType",
    "Model",
]