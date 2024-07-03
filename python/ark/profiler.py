# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import sys
import time
from .runtime import Runtime


class Profiler:
    def __init__(self, plan: str):
        self.plan = json.loads(plan)

    def run(self):
        num_processor_groups = len(self.plan["ProcessorGroups"])
        new_plan = {
            "Rank": self.plan["Rank"], "WorldSize": self.plan["WorldSize"],
            "NumProcessors": self.plan["NumProcessors"],
            "NumWarpsPerProcessor": self.plan["NumWarpsPerProcessor"],
            "TaskInfos": self.plan["TaskInfos"],
            "ProcessorGroups": [{}]}
        for i in range(num_processor_groups):
            new_plan["ProcessorGroups"][0] = self.plan["ProcessorGroups"][i]
            with Runtime() as rt:
                rt.launch(plan=json.dumps(new_plan))
                start_time = time.time()
                iter = 1000
                rt.run(iter=iter)
                end_time = time.time()
                sys.stderr.write(f"Processor group {i} runtime: {(end_time - start_time)/iter:.6f} seconds/iter\n")
