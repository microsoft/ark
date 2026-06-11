# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for planner.py edge branches — Plan creation, from_str, from_file errors."""

from common import ark, pytest_ark
import json
import os
import tempfile
import pytest


@pytest_ark()
def test_plan_default_init():
    """Plan(None) creates a valid default plan."""
    plan = ark.Plan(None)
    assert plan.rank == 0
    assert plan.world_size == 1
    assert plan.architecture == "ANY"
    assert plan.num_processors == 1
    assert plan.num_warps_per_processor == 1
    assert plan.task_infos == []
    assert plan.processor_groups == []


@pytest_ark()
def test_plan_str():
    """Plan.__str__ produces valid JSON."""
    plan = ark.Plan(None)
    s = str(plan)
    parsed = json.loads(s)
    assert parsed["Rank"] == 0


@pytest_ark()
def test_plan_from_str_valid():
    """Plan.from_str with valid JSON."""
    data = {
        "Rank": 0,
        "WorldSize": 1,
        "Architecture": "ANY",
        "NumProcessors": 1,
        "NumWarpsPerProcessor": 1,
        "TaskInfos": [],
        "ProcessorGroups": [],
    }
    plan = ark.Plan.from_str(json.dumps(data))
    assert plan.rank == 0
    assert plan.world_size == 1


@pytest_ark()
def test_plan_from_str_invalid_json():
    """Plan.from_str raises InvalidUsageError on bad JSON."""
    with pytest.raises(ark.InvalidUsageError):
        ark.Plan.from_str("not valid json {{{")


@pytest_ark()
def test_plan_from_file_valid():
    """Plan.from_file loads a valid JSON file."""
    data = {
        "Rank": 0,
        "WorldSize": 2,
        "Architecture": "ANY",
        "NumProcessors": 4,
        "NumWarpsPerProcessor": 8,
        "TaskInfos": [],
        "ProcessorGroups": [],
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(data, f)
        path = f.name

    try:
        plan = ark.Plan.from_file(path)
        assert plan.world_size == 2
        assert plan.num_processors == 4
    finally:
        os.unlink(path)


@pytest_ark()
def test_plan_from_file_not_found():
    """Plan.from_file raises InvalidUsageError for missing file."""
    with pytest.raises(ark.InvalidUsageError):
        ark.Plan.from_file("/tmp/nonexistent_ark_plan_12345.json")


@pytest_ark()
def test_plan_from_file_invalid_json():
    """Plan.from_file raises InvalidUsageError for invalid JSON content."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        f.write("not valid json")
        path = f.name

    try:
        with pytest.raises(ark.InvalidUsageError):
            ark.Plan.from_file(path)
    finally:
        os.unlink(path)


@pytest_ark()
def test_planner_context_warp_range():
    """PlannerContext with warp_range kwarg."""
    t = ark.tensor([64, 64], ark.fp16)
    with ark.PlannerContext(warp_range=[0, 4]):
        ark.add(t, 1.0)

    plan = ark.Planner().plan()
    assert len(plan.processor_groups) >= 1


@pytest_ark()
def test_planner_context_sram_range():
    """PlannerContext with sram_range kwarg."""
    t = ark.tensor([64, 64], ark.fp16)
    with ark.PlannerContext(sram_range=[0, 1024]):
        ark.add(t, 1.0)

    plan = ark.Planner().plan()
    assert len(plan.processor_groups) >= 1


@pytest_ark()
def test_planner_context_config():
    """PlannerContext with config dict kwarg does not raise on entry."""
    t = ark.tensor([64, 64], ark.fp16)
    # Just verify entering the context with a config does not crash;
    # we don't run the planner since invalid configs will error there.
    with ark.PlannerContext(config={"NumWarps": 4, "Tile": [64, 64]}):
        ark.add(t, 1.0)
    # Success: context entered and exited without error


@pytest_ark()
def test_planner_context_dump():
    """PlannerContext.dump returns a JSON string."""
    t = ark.tensor([64, 64], ark.fp16)
    with ark.PlannerContext(processor_range=[0, 8]) as ctx:
        s = ctx.dump()
        assert isinstance(s, str)
        # Should be valid JSON
        parsed = json.loads(s)
        assert isinstance(parsed, (dict, list))
