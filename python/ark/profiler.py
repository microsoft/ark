# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import time

from .runtime import Runtime
from .planner import Plan


def timeit(plan: Plan, iter: int):
    with Runtime() as rt:
        rt.launch(plan=plan)
        start_time = time.time()
        rt.run(iter=iter)
        end_time = time.time()
        return (end_time - start_time) / iter


class Profiler:
    def __init__(self, plan: Plan):
        self.plan = plan

    def run(self, iter: int = 1000, profile_processor_groups: bool = False):
        sys.stderr.write(f"End-to-end: {timeit(self.plan, iter):.6f} seconds/iter\n")

        if not profile_processor_groups:
            return
        num_processor_groups = len(self.plan.processor_groups)
        new_plan = {
            "Rank": self.plan.rank,
            "WorldSize": self.plan.world_size,
            "Architecture": self.plan.architecture,
            "NumProcessors": self.plan.num_processors,
            "NumWarpsPerProcessor": self.plan.num_warps_per_processor,
            "TaskInfos": self.plan.task_infos,
            "ProcessorGroups": [None],
        }
        for i in range(num_processor_groups):
            new_plan["ProcessorGroups"][0] = self.plan.processor_groups[i]
            lat_per_iter = timeit(Plan(new_plan), iter)
            sys.stderr.write(
                f"Processor group {i}: {lat_per_iter:.6f} seconds/iter\n"
            )
