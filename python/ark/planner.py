# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from typing import Callable, List, Dict, Any

from _ark_core import _Planner, _PlannerContext
from .model import Model


class PlannerContext(_PlannerContext):
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
        super().__init__(Model.get_model())
        prange: List[int] = kwargs.get("processor_range", None)
        wrange: List[int] = kwargs.get("warp_range", None)
        srange: List[int] = kwargs.get("sram_range", None)
        sync: bool = kwargs.get("sync", True)
        config: Dict[str, Any] = kwargs.get("config", None)

        if prange is not None:
            self.processor_range(*prange)
        if wrange is not None:
            self.warp_range(*wrange)
        if srange is not None:
            self.sram_range(*srange)
        if sync is False:
            self.sync(sync)
        if config is not None:
            self.config(json.dumps(config))

    def __enter__(self) -> "PlannerContext":
        """
        Enter the plan manager.
        """
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        """
        Exit the plan manager.
        """
        del self


class Planner(_Planner):
    def __init__(self, gpu_id: int = 0):
        compressed = Model.get_model().compress()
        super().__init__(compressed, gpu_id)

    def install_config_rule(self, rule: Callable[[str, str], str]):
        """
        Install a configuration rule.

        Args:
            rule: A function that takes an operator description and a target
            architecture name and returns a configuration description.
        """
        super().install_config_rule(rule)

    def plan(self, pretty: bool = True) -> str:
        """
        Generate an execution plan.

        Args:
            pretty: Whether to generate a pretty plan.
        """
        return super().plan(pretty)


__all__ = ["PlannerContext", "Planner"]
