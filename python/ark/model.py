# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from ._ark_core import _Model, _Tensor
from .tensor import Tensor

# Decorator to convert the model function to a global function
def _convert_tensor_type_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before function")
        # convert all _Tensor in input args and kwargs to Tensor
        args = [Tensor(arg) if isinstance(arg, _Tensor) else arg for arg in args]
        kwargs = {k: Tensor(v) if isinstance(v, _Tensor) else v for k, v in kwargs.items()}

        result = Model.global_model.func(*args, **kwargs)
        if isinstance(result, _Tensor):
            result = Tensor(result)
        return result
    return wrapper

class Model(_Model):
    # a global model object
    global_model = None
    def __init__(self, rank: int = 0):
        super().__init__(rank)
        Model.global_model = self


# Convert all functions in Model to global functions
for name in dir(_Model):
    if not name.startswith("__") and callable(getattr(_Model, name)):
        func = getattr(_Model, name)
        print("Converting function: ", name)
        globals()[name] = _convert_tensor_type_decorator(func)