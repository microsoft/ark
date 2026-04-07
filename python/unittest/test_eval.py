# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Test that Tensor.eval() correctly reuses compiled plans and recompiles
when the model graph changes."""

import pytest
import torch
import ark

DEVICE = "cuda:0"


@pytest.fixture(autouse=True)
def _ark_init():
    """Reset ARK state before each test."""
    ark.init()


def _get_compiled_plan():
    """Return the plan string currently compiled in the executor."""
    from ark.executor import Executor

    return Executor.get().plan()


def test_eval_same_structure_produces_correct_results():
    """Two eval() calls on same-shaped graphs should both produce correct results.
    Note: the plan strings may differ (different tensor IDs), but the executor's
    file-level compile cache avoids redundant nvcc invocations."""
    a = torch.ones(64, dtype=torch.float32, device=DEVICE) * 3.0
    b = torch.ones(64, dtype=torch.float32, device=DEVICE) * 4.0

    r1 = ark.add(a, b).eval()
    assert torch.allclose(r1, a + b)

    r2 = ark.add(a, b).eval()
    assert torch.allclose(r2, a + b)


def test_eval_recompile_on_different_graph():
    """A different graph should produce a different plan → triggers recompile."""
    a = torch.ones(64, dtype=torch.float32, device=DEVICE) * 2.0
    b = torch.ones(64, dtype=torch.float32, device=DEVICE) * 3.0

    # Graph 1: add
    r1 = ark.add(a, b).eval()
    plan1 = _get_compiled_plan()
    assert torch.allclose(r1, a + b)

    # Graph 2: mul (different op → different plan)
    r2 = ark.mul(a, b).eval()
    plan2 = _get_compiled_plan()
    assert torch.allclose(r2, a * b)

    assert (
        plan1 != plan2
    ), "Different graph structure should produce a different plan"


def test_eval_recompile_on_graph_update():
    """Building more ops on top of a previously eval'd graph should
    recompile and produce correct results."""
    ark.init()
    a = torch.ones(64, dtype=torch.float32, device=DEVICE) * 2.0
    b = torch.ones(64, dtype=torch.float32, device=DEVICE) * 3.0

    # Step 1: build c = a + b, eval
    c = ark.add(a, b)
    r1 = c.eval()
    plan1 = _get_compiled_plan()
    assert torch.allclose(r1, a + b)

    # Step 2: extend the SAME graph with d = c + a, eval
    # c is still a valid ARK tensor in the same model
    d = ark.add(c, a)
    r2 = d.eval()
    plan2 = _get_compiled_plan()
    assert torch.allclose(r2, (a + b) + a)

    # The plan must have changed (graph grew from 1 op to 2 ops)
    assert (
        plan1 != plan2
    ), "Extending the graph should produce a different plan and recompile"


def test_eval_with_torch_stream():
    """eval() with a torch.cuda.Stream should correctly interleave with
    torch operations on the same stream across multiple iterations."""
    s = torch.cuda.Stream()
    x = torch.ones(64, dtype=torch.float32, device=DEVICE)

    for i in range(5):
        # Reset ARK model each iteration so eval() only runs the single add op
        ark.init()
        # torch op on the stream: x = x * 2
        with torch.cuda.stream(s):
            x = x * 2
        # ARK op on the same stream: x = x + 1
        x = ark.add(x, 1.0).eval(stream=s)

    s.synchronize()
    # Each iteration: x = x * 2 + 1
    # Starting from 1: 3, 7, 15, 31, 63
    expected = torch.full((64,), 63.0, dtype=torch.float32, device=DEVICE)
    assert torch.allclose(x, expected)


def test_eval_chain_with_intermediate_read():
    """Build a chain of dependent ARK ops, eval() the final tensor,
    then verify an intermediate tensor also has the correct value."""
    a = torch.ones(64, dtype=torch.float32, device=DEVICE) * 2.0

    # Chain: b = a + 3 -> c = b * 4 -> d = c - 1
    b = ark.add(a, 3.0)
    c = ark.mul(b, 4.0)
    d = ark.sub(c, 1.0)

    # Only eval the final tensor
    result = d.eval()

    # Final: (2+3)*4 - 1 = 19
    assert torch.allclose(result, torch.full((64,), 19.0, device=DEVICE))

    # Intermediate b should also be materialized: 2+3 = 5
    b_val = b.to_torch()
    assert torch.allclose(b_val, torch.full((64,), 5.0, device=DEVICE))

    # Intermediate c should also be materialized: 5*4 = 20
    c_val = c.to_torch()
    assert torch.allclose(c_val, torch.full((64,), 20.0, device=DEVICE))
