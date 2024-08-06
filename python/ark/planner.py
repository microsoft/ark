# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import json
from typing import Callable, Dict, List, Any

from _ark_core import _Planner, _PlannerContext
from .model import Model


def idnt(indent):
    return " " * indent


def dquote(s):
    return '"' + s + '"'


def denser_json_obj(obj, key, level, indent, indent_step, ret=""):
    if len(obj) == 0:
        if key:
            return ret + idnt(indent) + dquote(key) + ": {}"
        else:
            return ret + idnt(indent) + "{}"
    ret += idnt(indent)
    if key:
        ret += dquote(key) + ": {\n"
    else:
        ret += "{\n"
    num_item = len(obj)
    for k, v in obj.items():
        is_obj_or_arr = isinstance(v, dict) or isinstance(v, list)
        is_num_arr = isinstance(v, list) and v and isinstance(v[0], int)
        if level <= 0 or not is_obj_or_arr or is_num_arr:
            ret += (
                idnt(indent + indent_step)
                + dquote(k)
                + ": "
                + json.dumps(v, separators=(",", ":"))
            )
        elif isinstance(v, dict):
            ret += denser_json_obj(
                v, k, level - 1, indent + indent_step, indent_step
            )
        elif isinstance(v, list):
            ret += denser_json_arr(
                v, k, level - 1, indent + indent_step, indent_step
            )
        num_item -= 1
        if num_item > 0:
            ret += ",\n"
        else:
            ret += "\n"
    ret += idnt(indent) + "}"
    return ret


def denser_json_arr(obj, key, level, indent, indent_step, ret=""):
    if len(obj) == 0:
        if key:
            return ret + idnt(indent) + dquote(key) + ": []"
        else:
            return ret + idnt(indent) + "[]"
    ret += idnt(indent)
    if key:
        ret += dquote(key) + ": [\n"
    else:
        ret += "[\n"
    num_item = len(obj)
    for v in obj:
        is_obj_or_arr = isinstance(v, dict) or isinstance(v, list)
        is_num_arr = (
            isinstance(v, list)
            and v
            and (isinstance(v[0], int) or isinstance(v[0], float))
        )
        if level <= 0 or not is_obj_or_arr or is_num_arr:
            ret += idnt(indent + indent_step) + json.dumps(
                v, separators=(",", ":")
            )
        elif isinstance(v, dict):
            ret += denser_json_obj(
                v, "", level - 1, indent + indent_step, indent_step
            )
        elif isinstance(v, list):
            ret += denser_json_arr(
                v, "", level - 1, indent + indent_step, indent_step
            )
        num_item -= 1
        if num_item > 0:
            ret += ",\n"
        else:
            ret += "\n"
    ret += idnt(indent) + "]"
    return ret


def denser_json(obj, level, indent_step=2):
    if isinstance(obj, dict):
        return denser_json_obj(obj, "", level, 0, indent_step, "")
    elif isinstance(obj, list):
        return denser_json_arr(obj, "", level, 0, indent_step, "")
    return json.dumps(obj, indent=indent_step)


class Plan:
    def __init__(self, plan: Dict[str, Any]):
        if plan is None:
            plan = {}
            plan["Rank"] = 0
            plan["WorldSize"] = 1
            plan["Architecture"] = "ANY"
            plan["NumProcessors"] = 1
            plan["NumWarpsPerProcessor"] = 1
            plan["TaskInfos"] = []
            plan["ProcessorGroups"] = []
        else:
            plan = copy.deepcopy(plan)
        self.plan = plan

    def __str__(self) -> str:
        return denser_json(self.plan, 5)

    @property
    def rank(self) -> int:
        return self.plan["Rank"]

    @property
    def world_size(self) -> int:
        return self.plan["WorldSize"]

    @property
    def architecture(self) -> str:
        return self.plan["Architecture"]

    @property
    def num_processors(self) -> int:
        return self.plan["NumProcessors"]

    @property
    def num_warps_per_processor(self) -> int:
        return self.plan["NumWarpsPerProcessor"]

    @property
    def task_infos(self) -> List[Dict[str, Any]]:
        return self.plan["TaskInfos"]

    @property
    def processor_groups(self) -> List[Dict[str, Any]]:
        return self.plan["ProcessorGroups"]

    @staticmethod
    def from_str(plan_str: str) -> "Plan":
        plan = json.loads(plan_str)
        return Plan(plan)

    @staticmethod
    def from_file(file_path: str) -> "Plan":
        with open(file_path, "r") as f:
            plan = json.load(f)
        return Plan(plan)


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
    def __init__(self, device_id: int = 0):
        compressed = Model.get_model().compress()
        super().__init__(compressed, device_id)

    def install_config_rule(self, rule: Callable[[str, str], str]):
        """
        Install a configuration rule.

        Args:
            rule: A function that takes an operator description and a target
            architecture name and returns a configuration description.
        """
        super().install_config_rule(rule)

    def plan(self) -> Plan:
        """
        Generate an execution plan.
        """
        return Plan.from_str(super().plan(pretty=False))


__all__ = ["Plan", "PlannerContext", "Planner"]
