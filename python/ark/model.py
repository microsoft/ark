# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from ._ark_core import _Model

class Model(_Model):
    def __init__(self, rank=0):
        super().__init__(rank)

