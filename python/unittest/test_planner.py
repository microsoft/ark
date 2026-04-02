# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from common import ark, pytest_ark


@pytest_ark()
def test_planner_processor_range():
    input_tensor = ark.tensor([64, 64], ark.fp16)
    other_tensor = ark.tensor([64, 64], ark.fp16)

    with ark.PlannerContext(processor_range=[0, 128]):
        with ark.PlannerContext(processor_range=[0, 8], sync=False):
            ark.add(input_tensor, other_tensor)
        with ark.PlannerContext(processor_range=[8, 16], sync=False):
            ark.add(input_tensor, other_tensor)

    plan = ark.Planner().plan()

    pg = plan.processor_groups
    assert len(pg) == 1
    assert pg[0]["ResourceGroups"][0]["ProcessorRange"] == [0, 8]
    assert pg[0]["ResourceGroups"][1]["ProcessorRange"] == [8, 16]


@pytest_ark()
def test_planner_sync():
    input_tensor = ark.tensor([64, 64], ark.fp16)
    other_tensor = ark.tensor([64, 64], ark.fp16)

    with ark.PlannerContext(sync=False):
        with ark.PlannerContext():
            ark.add(input_tensor, other_tensor)
        with ark.PlannerContext():
            ark.add(input_tensor, other_tensor)

    plan = ark.Planner().plan()

    pg = plan.processor_groups
    assert len(pg) == 1
