# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from typing import List, Dict, Any
from .model import Model
from ._ark_core import _PlanManager


class PlanManager(_PlanManager):
    def __init__(self, **kwargs):
        """
        Plan manager for specifying the parallelization and tiling configuration of the operators in the context.

        Args:
            processor_range (List[int], optional): The range of processors to be used. Defaults to None.
            warp_range (List[int], optional): The range of warps to be used. Defaults to None.
            sram_range (List[int], optional): The range of SRAMs to be used. Defaults to None.
            sync (bool, optional): Whether to synchronize the execution. Defaults to True.
            config (Dict[str, Any], optional): The configuration for the operators. Defaults to None.
        """
        super().__init__(Model.get_model(), json.dumps(kwargs))

    def __enter__(self) -> "PlanManager":
        """
        Enter the plan manager.
        """
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        """
        Exit the plan manager.
        """
        del self
