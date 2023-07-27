# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from ._ark_core import _Model, _Tensor
from .tensor import Tensor


class Model(_Model):
    # a global model object
    global_model = None
    def __init__(self, rank: int = 0):
        super().__init__(rank)
        Model.global_model = self


