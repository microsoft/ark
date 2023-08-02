# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from ._ark_core import _Tensor


class Tensor:
    def __init__(self, _tensor: _Tensor):
        self._tensor = _tensor
