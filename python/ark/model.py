# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import NewType
from ._ark_core import _Model

_ModelState = NewType("_ModelState", None)


class Model(_Model):
    @staticmethod
    def get_model():
        """
        Get the underlying model.
        """
        if _ModelState.model is None:
            _ModelState.model = Model(_ModelState.rank, _ModelState.world_size)
        return _ModelState.model

    @staticmethod
    def get_rank():
        """
        Get the rank of the model.
        """
        return _ModelState.rank

    @staticmethod
    def get_world_size():
        """
        Get the world size of the model.
        """
        return _ModelState.world_size

    @staticmethod
    def set_rank(rank: int):
        """
        Set the rank of the model.
        """
        _ModelState.rank = rank

    @staticmethod
    def set_world_size(world_size: int):
        """
        Set the world size of the model.
        """
        _ModelState.world_size = world_size

    @staticmethod
    def reset():
        """
        Reset the model state.
        """
        _ModelState.model = None
        _ModelState.rank = 0
        _ModelState.world_size = 1

    def compress(self) -> "Model":
        """
        Compress the model.
        """
        return super().compress()

    def serialize(self, pretty: bool = True) -> str:
        """
        Serialize the model.

        Args:
            pretty: Whether to pretty print the model.

        Returns:
            The serialized model.
        """
        return super().serialize(pretty)


class _ModelState:
    """
    The _ModelState class is used to store the state of the model.
    """

    model: Model = None
    rank: int = 0
    world_size: int = 1
