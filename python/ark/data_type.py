# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy
import string
from . import _ark_core


_REGISTRY_DATA_TYPE = {
    "fp32": {"np": numpy.float32, "doc": """32-bit floating point."""},
    "fp16": {"np": numpy.float16, "doc": """16-bit floating point."""},
    "bf16": {"np": None, "doc": """bfloat16 floating point."""},
    "int32": {"np": numpy.int32, "doc": """32-bit signed integer."""},
    "uint32": {"np": numpy.uint32, "doc": """32-bit unsigned integer."""},
    "int8": {"np": numpy.int8, "doc": """8-bit signed integer."""},
    "uint8": {"np": numpy.uint8, "doc": """8-bit unsigned integer."""},
    "byte": {
        "np": numpy.ubyte,
        "doc": """
Represent the data as bytes, supposed to be untyped binary.

Unlike other data types, casting to/from `byte` from/to another data type
is considered as reinterpretation of the data, instead of conversion.
""",
    },
}


class DataType:
    @staticmethod
    def from_numpy(np_type: numpy.dtype) -> "DataType":
        for type_name, reg in _REGISTRY_DATA_TYPE.items():
            if reg["np"] == np_type:
                return DataType.from_name(type_name)
        raise ValueError(
            f"Undefined conversion from numpy data type {np_type}"
            f" to ark data type."
        )

    @staticmethod
    def from_name(type_name: str) -> "DataType":
        return globals()[type_name]

    @staticmethod
    def from_ttype(ttype: _ark_core._TensorType) -> "DataType":
        return DataType.from_name(ttype.name())

    @staticmethod
    def to_numpy() -> numpy.dtype:
        """Return the corresponding numpy data type."""
        ...

    @staticmethod
    def ttype() -> _ark_core._TensorType:
        """Return the corresponding tensor type."""
        ...

    @staticmethod
    def element_size() -> int:
        """Return the size of the data type in bytes."""
        ...


_DATA_TYPE_TEMPLATE = string.Template(
    """
class $type_name(DataType):
    @staticmethod
    def to_numpy() -> numpy.dtype:
        return _REGISTRY_DATA_TYPE[__class__.__name__]["np"]

    @staticmethod
    def ttype() -> _ark_core._TensorType:
        return getattr(_ark_core, "_" + __class__.__name__.upper())

    @staticmethod
    def element_size() -> int:
        return __class__.ttype().bytes()
"""
)

for type_name, reg in _REGISTRY_DATA_TYPE.items():
    exec(_DATA_TYPE_TEMPLATE.substitute(type_name=type_name))
    globals()[type_name].__doc__ = reg["doc"]
