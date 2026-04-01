# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from common import ark, pytest_ark
import numpy as np


@pytest_ark()
def test_placeholder_is_external():
    """Test that placeholder tensors are marked as external."""
    t_placeholder = ark.placeholder([64], ark.fp32)
    assert t_placeholder.is_external(), "Placeholder tensor should be external"

    t_regular = ark.tensor([64], ark.fp32)
    assert not t_regular.is_external(), (
        "Regular tensor should not be external"
    )


@pytest_ark(need_torch=True)
def test_placeholder_immediate_binding():
    """Test placeholder tensor with torch data bound at model creation."""
    import torch

    torch_data = torch.arange(64, dtype=torch.float32, device="cuda:0")
    t = ark.placeholder([64], ark.fp32, data=torch_data)
    out = ark.add(t, 1.0)

    with ark.Runtime() as rt:
        rt.launch()
        rt.run()

        result = out.to_numpy()

    expected = torch_data.cpu().numpy() + 1.0
    assert np.allclose(result, expected), (
        f"max diff: {np.max(np.abs(result - expected))}"
    )


@pytest_ark(need_torch=True)
def test_placeholder_scalar_add():
    """Test placeholder with scalar addition on non-aligned shape."""
    import torch

    torch_data = torch.arange(10, dtype=torch.float32, device="cuda:0").reshape(10, 1)
    t = ark.placeholder([10, 1], ark.fp32, data=torch_data)
    out = ark.add(t, 5.0)

    with ark.Runtime() as rt:
        rt.launch()
        rt.run()

        result = out.to_numpy()

    expected = torch_data.cpu().numpy() + 5.0
    assert np.allclose(result, expected), (
        f"max diff: {np.max(np.abs(result - expected))}"
    )


@pytest_ark(need_torch=True)
def test_placeholder_multiple():
    """Test multiple placeholder tensors in the same model."""
    import torch

    torch_a = torch.ones(64, dtype=torch.float32, device="cuda:0") * 2.0
    torch_b = torch.ones(64, dtype=torch.float32, device="cuda:0") * 3.0

    a = ark.placeholder([64], ark.fp32, data=torch_a)
    b = ark.placeholder([64], ark.fp32, data=torch_b)
    out = ark.add(a, b)

    with ark.Runtime() as rt:
        rt.launch()
        rt.run()

        result = out.to_numpy()

    expected = torch_a.cpu().numpy() + torch_b.cpu().numpy()
    assert np.allclose(result, expected), (
        f"max diff: {np.max(np.abs(result - expected))}"
    )


@pytest_ark(need_torch=True)
def test_placeholder_fp16():
    """Test placeholder with fp16 data type."""
    import torch

    torch_data = torch.ones(128, dtype=torch.float16, device="cuda:0") * 4.0
    t = ark.placeholder([128], ark.fp16, data=torch_data)
    out = ark.mul(t, 0.5)

    with ark.Runtime() as rt:
        rt.launch()
        rt.run()

        result = out.to_numpy()

    expected = torch_data.cpu().numpy() * 0.5
    assert np.allclose(result, expected, atol=1e-2), (
        f"max diff: {np.max(np.abs(result - expected))}"
    )


@pytest_ark(need_torch=True)
def test_placeholder_from_torch():
    """Test creating a placeholder from a torch tensor via Tensor.from_torch()."""
    import torch

    torch_tensor = torch.arange(64, dtype=torch.float32, device="cuda:0")

    ark_tensor = ark.Tensor.from_torch(torch_tensor)
    out = ark.add(ark_tensor, 10.0)

    with ark.Runtime() as rt:
        rt.launch()
        rt.run()

        result = out.to_numpy()

    expected = torch_tensor.cpu().numpy() + 10.0
    assert np.allclose(result, expected), (
        f"max diff: {np.max(np.abs(result - expected))}"
    )


@pytest_ark(need_torch=True)
def test_placeholder_tensor_mappings_launch():
    """Test delayed binding with torch tensor via tensor_mappings at launch."""
    import torch

    t = ark.placeholder([256], ark.fp32)
    out = ark.mul(t, 3.0)

    torch_input = torch.ones(256, dtype=torch.float32, device="cuda:0") * 7.0

    with ark.Runtime() as rt:
        rt.launch(tensor_mappings={t: torch_input})
        rt.run()

        result = out.to_numpy()

    expected = torch_input.cpu().numpy() * 3.0
    assert np.allclose(result, expected), (
        f"max diff: {np.max(np.abs(result - expected))}"
    )


@pytest_ark(need_torch=True)
def test_placeholder_runtime_rebinding():
    """Test rebinding placeholder to different data between run() calls."""
    import torch

    t = ark.placeholder([64], ark.fp32)
    out = ark.add(t, 1.0)

    with ark.Runtime() as rt:
        input1 = torch.ones(64, dtype=torch.float32, device="cuda:0") * 5.0
        rt.launch(loop_mode=False, tensor_mappings={t: input1})
        rt.run()
        result1 = out.to_numpy()

        # Rebind to different data and run again
        input2 = torch.ones(64, dtype=torch.float32, device="cuda:0") * 10.0
        rt.run(tensor_mappings={t: input2})
        result2 = out.to_numpy()

    assert np.allclose(result1, 6.0), f"Run 1: expected 6.0, got {result1[:5]}"
    assert np.allclose(result2, 11.0), (
        f"Run 2: expected 11.0, got {result2[:5]}"
    )
