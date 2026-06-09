# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from common import ark, pytest_ark


@pytest_ark(need_torch=True)
def test_eval_basic():
    """Test basic Tensor.eval() — compile, run, return torch tensor."""
    import torch

    x = torch.ones(64, dtype=torch.float32, device="cuda:0") * 3.0
    out = ark.add(x, 2.0)

    result = out.eval()

    assert isinstance(result, torch.Tensor)
    expected = torch.ones(64, dtype=torch.float32, device="cuda:0") * 5.0
    assert torch.allclose(result, expected)


@pytest_ark(need_torch=True)
def test_eval_chain():
    """Test eval on a chained computation."""
    import torch

    x = torch.ones(64, dtype=torch.float16, device="cuda:0") * 4.0
    y = ark.mul(x, 2.0)
    z = ark.add(y, 1.0)

    result = z.eval()

    assert isinstance(result, torch.Tensor)
    expected = torch.ones(64, dtype=torch.float16, device="cuda:0") * 9.0
    assert torch.allclose(result.cpu(), expected.cpu(), atol=1e-2)


@pytest_ark(need_torch=True)
def test_eval_relu():
    """Test eval with relu op."""
    import torch

    x = torch.tensor(
        [-1.0, 0.0, 1.0, 2.0], dtype=torch.float32, device="cuda:0"
    )
    out = ark.relu(x)

    result = out.eval()

    expected = torch.tensor(
        [0.0, 0.0, 1.0, 2.0], dtype=torch.float32, device="cuda:0"
    )
    assert torch.allclose(result, expected)


@pytest_ark(need_torch=True)
def test_eval_matmul():
    """Test eval with matmul."""
    import torch

    a = torch.ones(4, 64, dtype=torch.float16, device="cuda:0")
    b = torch.ones(64, 8, dtype=torch.float16, device="cuda:0")
    out = ark.matmul(a, b)

    result = out.eval()

    assert result.shape == (4, 8)
    expected = torch.full((4, 8), 64.0, dtype=torch.float16, device="cuda:0")
    assert torch.allclose(result, expected, atol=1e-1)


@pytest_ark(need_torch=True)
def test_eval_independent_calls():
    """Test that independent eval calls on different graphs produce correct results."""
    import torch

    x = torch.ones(64, dtype=torch.float32, device="cuda:0") * 2.0
    out = ark.add(x, 3.0)

    result1 = out.eval()
    assert torch.allclose(
        result1,
        torch.ones(64, dtype=torch.float32, device="cuda:0") * 5.0,
    )

    # Second eval on a different graph should also work
    with ark.use_model(None):
        x2 = torch.ones(64, dtype=torch.float32, device="cuda:0") * 10.0
        out2 = ark.add(x2, 1.0)
        result2 = out2.eval()
        assert torch.allclose(
            result2,
            torch.ones(64, dtype=torch.float32, device="cuda:0") * 11.0,
        )
