# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy
from .torch import torch
from . import core
from . import log

__all__ = [
    "DataType",
    "fp16",
    "fp32",
    "int32",
    "uint32",
    "int8",
    "uint8",
]

REGISTRY_DATA_TYPE = {
    "fp32": {"np": numpy.float32, "torch": torch.float32},
    "fp16": {"np": numpy.float16, "torch": torch.float16},
    "bf16": {"np": None, "torch": torch.bfloat16},
    "int32": {"np": numpy.int32, "torch": torch.int32},
    "uint32": {"np": numpy.uint32, "torch": None},
    "int8": {"np": numpy.int8, "torch": torch.int8},
    "uint8": {"np": numpy.uint8, "torch": torch.uint8},
}


class MetaDataType(type):
    def __new__(cls, name, bases, attrs):
        new_class = super().__new__(cls, name, bases, attrs)
        if name in REGISTRY_DATA_TYPE:
            reg = REGISTRY_DATA_TYPE[name]
            new_class.to_numpy = staticmethod(lambda: reg["np"])
            new_class.to_torch = staticmethod(lambda: reg["torch"])
            new_class.ctype = staticmethod(lambda: getattr(core, name.upper()))
            new_class.element_size = staticmethod(
                lambda: new_class.ctype().bytes()
            )
        return new_class


class DataType(metaclass=MetaDataType):
    """
    Represent the data type of a tensor.
    """

    @staticmethod
    def from_numpy(np_type: numpy.dtype) -> "DataType":
        """
        Return the corresponding ark data type.

        Parameters:
            np_type (numpy.dtype): The numpy data type.

        Returns:
            DataType: The corresponding ark data type.

        Raises:
            InvalidUsageError: If there is no defined conversion from numpy data type to ark data type.
        """
        if not isinstance(np_type, numpy.dtype):
            raise log.InvalidUsageError(
                f"Expected a numpy data type, but got {type(np_type)}"
            )
        for type_name, reg in REGISTRY_DATA_TYPE.items():
            if reg["np"] == np_type:
                return DataType.from_name(type_name)
        raise log.InvalidUsageError(
            f"Undefined conversion from numpy data type {np_type}"
            f" to ark data type."
        )

    @staticmethod
    def from_torch(torch_type: torch.dtype) -> "DataType":
        """
        Return the corresponding ark data type.

        Parameters:
            torch_type (torch.dtype): The torch data type.

        Returns:
            DataType: The corresponding ark data type.

        Raises:
            ValueError: If there is no defined conversion from torch data type to ark data type.
        """
        for type_name, reg in REGISTRY_DATA_TYPE.items():
            if reg["torch"] == torch_type:
                return DataType.from_name(type_name)
        raise ValueError(
            f"Undefined conversion from torch data type {torch_type}"
            f" to ark data type."
        )

    @staticmethod
    def from_name(type_name: str) -> "DataType":
        """
        Return the corresponding ark data type.

        Parameters:
            type_name (str): The name of the data type.

        Returns:
            DataType: The corresponding ark data type.

        Raises:
            ValueError: If the data type is not defined.
        """
        ret = globals().get(type_name, None)
        if ret is None:
            raise log.InvalidUsageError(f"Undefined data type {type_name}")
        return ret

    @staticmethod
    def from_ctype(ctype: core.CoreDataType) -> "DataType":
        """
        Return the corresponding ark data type.

        Parameters:
            ctype (core.CoreDataType): The cpp type.

        Returns:
            DataType: The corresponding ark data type.

        Raises:
            ValueError: If the data type is not defined.
        """
        if not isinstance(ctype, core.CoreDataType):
            raise log.InvalidUsageError(
                f"Expected a core data type, but got {type(ctype)}"
            )
        return DataType.from_name(ctype.name().lower())

    @staticmethod
    def to_numpy() -> numpy.dtype:
        """
        Return the corresponding numpy data type.

        Returns:
            numpy.dtype: The corresponding numpy data type.
        """
        ...

    @staticmethod
    def to_torch() -> torch.dtype:
        """
        Return the corresponding torch data type.

        Returns:
            torch.dtype: The corresponding torch data type.
        """
        ...

    @staticmethod
    def ctype() -> core.CoreDataType:
        """
        Return the corresponding cpp type.

        Returns:
            core.CoreDataType: The corresponding cpp type.
        """
        ...

    @staticmethod
    def element_size() -> int:
        """
        Return the size of the data type in bytes.

        Returns:
            int: The size of the data type in bytes.
        """
        ...


class fp32(DataType):
    """32-bit floating point."""

    ...


class fp16(DataType):
    """16-bit floating point."""

    ...


class bf16(DataType):
    """bfloat16 floating point."""

    ...


class int32(DataType):
    """32-bit signed integer."""

    ...


class uint32(DataType):
    """32-bit unsigned integer."""

    ...


class int8(DataType):
    """8-bit signed integer."""

    ...


class uint8(DataType):
    """8-bit unsigned integer."""

    ...


class byte(DataType):
    """
    Represent the data as bytes, supposed to be untyped binary.

    Unlike other data types, casting to/from `byte` from/to another data type
    is considered as reinterpretation of the data, instead of conversion.
    """

    ...
