# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import subprocess
import sys
from unittest.mock import patch, MagicMock

from common import ark, pytest_ark
import pytest

try:
    from ark import core as _ark_core  # noqa: F401

    _has_ark_core = True
except ImportError:
    _has_ark_core = False


@pytest.mark.skipif(
    not _has_ark_core, reason="native _ark_core extension not available"
)
def test_profiler_cli_help():
    """Test that `python -m ark.profiler --help` exits 0 and shows usage."""
    result = subprocess.run(
        [sys.executable, "-m", "ark.profiler", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "ARK Profiler" in result.stdout


@pytest.mark.skipif(
    not _has_ark_core, reason="native _ark_core extension not available"
)
def test_profiler_cli_missing_plan():
    """Test that `python -m ark.profiler` without --plan exits with code 2.
    Validates that --plan is configured as a required argument."""
    result = subprocess.run(
        [sys.executable, "-m", "ark.profiler"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "--plan" in result.stderr


def test_profiler_main_arg_parsing():
    """Test main() parses args and delegates to Profiler.run."""
    mock_plan = MagicMock()
    mock_profiler = MagicMock()

    with patch("ark.profiler.Plan") as MockPlan, patch(
        "ark.profiler.Profiler"
    ) as MockProfiler:
        MockPlan.from_file.return_value = mock_plan
        MockProfiler.return_value = mock_profiler

        ark.profiler.main(
            [
                "--plan",
                "test.json",
                "--iter",
                "5",
                "--loop_mode",
                "--profile_processor_groups",
                "--target_processor_groups",
                "0,1",
            ]
        )

        MockPlan.from_file.assert_called_once_with("test.json")
        MockProfiler.assert_called_once_with(mock_plan)
        mock_profiler.run.assert_called_once_with(
            iter=5,
            loop_mode=True,
            profile_processor_groups=True,
            target_processor_groups=[0, 1],
        )


def test_profiler_main_defaults():
    """Test main() uses default args when only --plan is given."""
    mock_plan = MagicMock()
    mock_profiler = MagicMock()

    with patch("ark.profiler.Plan") as MockPlan, patch(
        "ark.profiler.Profiler"
    ) as MockProfiler:
        MockPlan.from_file.return_value = mock_plan
        MockProfiler.return_value = mock_profiler

        ark.profiler.main(["--plan", "plan.json"])

        MockPlan.from_file.assert_called_once_with("plan.json")
        MockProfiler.assert_called_once_with(mock_plan)
        mock_profiler.run.assert_called_once_with(
            iter=1000,
            loop_mode=False,
            profile_processor_groups=False,
            target_processor_groups=None,
        )


def test_profiler_main_target_groups_whitespace():
    """Test that target_processor_groups handles whitespace and empty segments."""
    mock_plan = MagicMock()
    mock_profiler = MagicMock()

    with patch("ark.profiler.Plan") as MockPlan, patch(
        "ark.profiler.Profiler"
    ) as MockProfiler:
        MockPlan.from_file.return_value = mock_plan
        MockProfiler.return_value = mock_profiler

        ark.profiler.main(
            ["--plan", "t.json", "--target_processor_groups", "0, 1"]
        )

        mock_profiler.run.assert_called_once_with(
            iter=1000,
            loop_mode=False,
            profile_processor_groups=False,
            target_processor_groups=[0, 1],
        )


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
