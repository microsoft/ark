# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from ._ark_core import _Model


class _ModelState:
    """
    The _ModelState class is used to store the state of the model.
    """

    model: _Model = None
    rank: int = 0
    world_size: int = 1


class Model:
    """
    Defines static methods to handle _ModelState.
    """

    @staticmethod
    def get_model():
        """
        Get the underlying model.
        """
        if _ModelState.model is None:
            _ModelState.model = _Model(_ModelState.rank)
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
