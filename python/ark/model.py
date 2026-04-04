# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import NewType
from contextlib import contextmanager
from . import log
from .core import CoreModel


__all__ = ["Model", "set_model", "current_model", "use_model"]

ModelState = NewType("ModelState", None)


class Model(CoreModel):
    @staticmethod
    def get_model() -> "Model":
        """
        Get the underlying model.
        """
        if ModelState.model is None:
            ModelState.model = Model(ModelState.rank, ModelState.world_size)
        return ModelState.model

    @staticmethod
    def get_rank():
        """
        Get the rank of the model.
        """
        return ModelState.rank

    @staticmethod
    def get_world_size():
        """
        Get the world size of the model.
        """
        return ModelState.world_size

    @staticmethod
    def get_device_id():
        """
        Get the device id.
        """
        return ModelState.device_id

    @staticmethod
    def set_rank(rank: int):
        """
        Set the rank of the model.
        """
        ModelState.rank = rank

    @staticmethod
    def set_world_size(world_size: int):
        """
        Set the world size of the model.
        """
        ModelState.world_size = world_size

    @staticmethod
    def set_device_id(device_id: int):
        """
        Set the device id.
        """
        if device_id < 0:
            raise log.InvalidUsageError("device_id must be non-negative")
        ModelState.device_id = device_id

    @staticmethod
    def reset():
        """
        Reset the model state.
        """
        ModelState.model = None
        ModelState.rank = 0
        ModelState.world_size = 1

    def __init__(self, rank: int = 0, world_size: int = 1):
        """
        Initialize the model.

        Args:
            rank: The rank of the model.
            world_size: The world size of the model.
        """
        super().__init__(rank, world_size)

    def __str__(self) -> str:
        return self.serialize()

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


class ModelState:
    """
    The ModelState class is used to store the state of the model.
    """

    model: Model = None
    rank: int = 0
    world_size: int = 1
    device_id: int = 0


def set_model(model: Model) -> None:
    """Set the current active model. All subsequent ARK ops will be added to this model.

    Similar to ``torch.cuda.set_stream()``.

    Args:
        model: The model to set as the current active model.
    """
    ModelState.model = model


def current_model() -> Model:
    """Return the current active model, creating one if none exists.

    Similar to ``torch.cuda.current_stream()``.

    Returns:
        The current active model.
    """
    return Model.get_model()


@contextmanager
def use_model(model: Model):
    """Context manager to temporarily switch the active model.

    All ARK ops within the ``with`` block are added to the given model.
    On exit, the previous model is restored.

    Similar to ``torch.cuda.stream()``.

    Example::

        m1 = ark.Model()
        m2 = ark.Model()
        with ark.use_model(m1):
            a = ark.add(x, y)    # added to m1
        with ark.use_model(m2):
            b = ark.mul(x, y)    # added to m2

    Args:
        model: The model to use within the context.
    """
    prev = ModelState.model
    ModelState.model = model
    try:
        yield model
    finally:
        ModelState.model = prev
