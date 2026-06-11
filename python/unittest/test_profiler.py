# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import subprocess
import sys

from common import ark, pytest_ark
import pytest


def test_profiler_cli_help():
    """Test that `python -m ark.profiler --help` exits 0 and shows usage."""
    result = subprocess.run(
        [sys.executable, "-m", "ark.profiler", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "ARK Profiler" in result.stdout


def test_profiler_cli_missing_plan():
    """Test that `python -m ark.profiler` without --plan exits non-zero."""
    result = subprocess.run(
        [sys.executable, "-m", "ark.profiler"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--plan" in result.stderr


@pytest_ark()
def test_profiler_non_loop_mode():
    """Test profiler in non-loop (record) mode."""
    t = ark.tensor([64, 64], ark.fp16)
    out = ark.mul(t, 2.0)

    plan = ark.Planner().plan()
    profiler = ark.Profiler(plan)

    profiler.run(iter=10, loop_mode=False)


@pytest_ark()
def test_profiler_processor_groups():
    """Test profiler with per-processor-group profiling in non-loop mode."""
    t = ark.tensor([64, 64], ark.fp16)
    out = ark.add(t, 1.0)

    plan = ark.Planner().plan()
    profiler = ark.Profiler(plan)

    profiler.run(
        iter=10,
        loop_mode=False,
        profile_processor_groups=True,
    )


@pytest_ark()
def test_profiler_target_processor_groups():
    """Test profiler targeting specific processor groups."""
    t = ark.tensor([64, 64], ark.fp16)
    out = ark.add(t, 1.0)

    plan = ark.Planner().plan()
    profiler = ark.Profiler(plan)

    profiler.run(
        iter=10,
        loop_mode=False,
        profile_processor_groups=True,
        target_processor_groups=[0],
    )


@pytest_ark()
def test_timeit():
    """Test the standalone timeit function."""
    t = ark.tensor([64, 64], ark.fp16)
    out = ark.add(t, 1.0)

    plan = ark.Planner().plan()

    elapsed = ark.profiler.timeit(plan, iter=10, loop_mode=False)
    assert elapsed > 0, f"Expected positive elapsed time, got {elapsed}"
    assert isinstance(elapsed, float)


@pytest_ark()
def test_profiler_plan_attributes():
    """Test that Plan object has expected attributes for the profiler."""
    t = ark.tensor([64, 64], ark.fp16)
    out = ark.add(t, 1.0)

    plan = ark.Planner().plan()

    assert plan.rank == 0
    assert plan.world_size == 1
    assert isinstance(plan.architecture, str)
    assert plan.num_processors > 0
    assert plan.num_warps_per_processor > 0
    assert isinstance(plan.task_infos, list)
    assert isinstance(plan.processor_groups, list)
    assert len(plan.processor_groups) > 0
