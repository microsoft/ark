# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from ._ark_core import _Tensor


class Tensor:
    def __init__(self, _tensor: _Tensor):
        self._tensor = _tensor
        self.shape = _tensor.shape
        self.is_parameter = False


def Parameter(
    tensor: Tensor,
) -> Tensor:
    """
    Set the tensor as a parameter.
    """
    tensor.is_parameter = True
    return tensor
