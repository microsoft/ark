# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from ._ark_core import _Model, _Tensor
from .tensor import Tensor

# def _convert_tensor_type_decorator(func):
#     def wrapper(*args, **kwargs):
#         print("Before function")
#         result = func(*args, **kwargs)
#         if isinstance(result, _Tensor):
#             return type(Tensor)(result)
#         return result
#     return wrapper

class Model(_Model):
    pass


# # Add decorator to all functions in Model
# for name in dir(Model):
#     if not name.startswith("__") and callable(getattr(Model, name)):
#         func = getattr(Model, name)
#         setattr(Model, name, _convert_tensor_type_decorator(func))