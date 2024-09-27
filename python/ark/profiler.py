# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import time
from typing import Optional, List

from .runtime import Runtime
from .planner import Plan


def timeit(plan: Plan, iter: int, loop_mode: bool):
    with Runtime() as rt:
        rt.launch(plan=plan, loop_mode=loop_mode)
        start_time = time.time()
        rt.run(iter=iter)
        end_time = time.time()
        return (end_time - start_time) / iter


class Profiler:
    def __init__(self, plan: Plan):
        self.plan = plan

    def run(
        self,
        iter: int = 1000,
        loop_mode: bool = True,
        profile_processor_groups: bool = False,
        target_processor_groups: Optional[List[int]] = None,
    ):
        if target_processor_groups is None:
            sys.stderr.write(
                f"End-to-end: {timeit(self.plan, iter, loop_mode):.6f} seconds/iter\n"
            )

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
            if target_processor_groups is not None and i not in target_processor_groups:
                continue
            new_plan["ProcessorGroups"][0] = self.plan.processor_groups[i]
            lat_per_iter = timeit(Plan(new_plan), iter, loop_mode)
            sys.stderr.write(
                f"Processor group {i}: {lat_per_iter:.6f} seconds/iter\n"
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ARK Profiler")
    parser.add_argument(
        "--iter",
        type=int,
        default=1000,
        help="Number of iterations to run for each measurement",
    )
    parser.add_argument(
        "--loop_mode",
        action="store_true",
        help="Use loop mode to measure end-to-end latency",
    )
    parser.add_argument(
        "--profile_processor_groups",
        action="store_true",
        help="Profile processor groups",
    )
    parser.add_argument(
        "--target_processor_groups",
        type=str,
        help="Target processor groups to profile",
    )
    parser.add_argument("--plan", type=str, help="Path to the plan file", required=True)
    args = parser.parse_args()

    target_processor_groups = None
    if args.target_processor_groups is not None:
        target_processor_groups = list(map(int, args.target_processor_groups.split(",")))

    plan = Plan.from_file(args.plan)
    profiler = Profiler(plan)
    profiler.run(
        iter=args.iter,
        loop_mode=args.loop_mode,
        profile_processor_groups=args.profile_processor_groups,
        target_processor_groups=target_processor_groups,
    )
