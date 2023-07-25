# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pickle
import Dict
from typing import Optional, Dict

from ._ark_core import Model, Executor, Tensor

class Module:
    _modules: Dict[str, Optional['Module']]
    _parameters: Dict[str, Optional['Tensor']]
    def __init__(self,model):
        self._modules = Dict[str, Optional['Module']]()
        self._Tensors = Dict[str, Optional['Tensor']]()
        self._model = model
        super().__init__()


    def load_state_dict(self,executor, state_dict, prefix=''):
        for name, module in self._modules.items():
            if module is not None:
                module.load_state_dict(executor, state_dict, prefix=prefix + name + '.')