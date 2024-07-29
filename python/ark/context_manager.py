# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from .model import Model
from ._ark_core import _ContextManager


class ContextManager(_ContextManager):
    def __init__(self, **kwargs):
        context_map = {key: json.dumps(value) for key, value in kwargs.items()}
        super().__init__(Model.get_model(), context_map)

    def __enter__(self) -> "ContextManager":
        """
        Enter the context manager.
        """
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        """
        Exit the context manager.
        """
        del self
