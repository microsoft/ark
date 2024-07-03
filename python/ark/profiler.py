# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import time

from .runtime import Runtime
from .planner import Plan


class Profiler:
    def __init__(self, plan: Plan):
        self.plan = plan

    def run(self):
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
            with Runtime() as rt:
                rt.launch(plan=str(new_plan))
                start_time = time.time()
                iter = 1000
                rt.run(iter=iter)
                end_time = time.time()
                sys.stderr.write(
                    f"Processor group {i} runtime: {(end_time - start_time)/iter:.6f} seconds/iter\n"
                )
