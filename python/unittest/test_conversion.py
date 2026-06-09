# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from common import ark, pytest_ark
import numpy as np


@pytest_ark(need_torch=True)
def test_conversion_torch_to_ark_fp32():
    """Test converting a torch fp32 tensor to ARK and back."""
    import torch

    torch_data = torch.arange(64, dtype=torch.float32, device="cuda:0")
    ark_tensor = ark.Tensor.from_torch(torch_data)

    assert ark_tensor.shape() == [64]
    assert ark_tensor.dtype() == ark.fp32

    out = ark.add(ark_tensor, 1.0)

    with ark.Runtime() as rt:
        rt.launch()
        rt.run()
        result = out.to_numpy()

    expected = torch_data.cpu().numpy() + 1.0
    assert np.allclose(result, expected)


@pytest_ark(need_torch=True)
def test_conversion_torch_to_ark_fp16():
    """Test converting a torch fp16 tensor to ARK and back."""
    import torch

    torch_data = torch.ones(128, dtype=torch.float16, device="cuda:0") * 3.0
    ark_tensor = ark.Tensor.from_torch(torch_data)

    assert ark_tensor.shape() == [128]
    assert ark_tensor.dtype() == ark.fp16

    out = ark.mul(ark_tensor, 2.0)

    with ark.Runtime() as rt:
        rt.launch()
        rt.run()
        result = out.to_numpy()

    expected = torch_data.cpu().numpy() * 2.0
    assert np.allclose(result, expected, atol=1e-2)


@pytest_ark(need_torch=True)
def test_conversion_ark_to_torch():
    """Test converting an ARK tensor result to torch."""
    import torch

    torch_input = torch.ones(64, dtype=torch.float32, device="cuda:0") * 5.0
    ark_input = ark.Tensor.from_torch(torch_input)
    out = ark.add(ark_input, 10.0)

    with ark.Runtime() as rt:
        rt.launch()
        rt.run()
        torch_result = out.to_torch()

    assert isinstance(torch_result, torch.Tensor)
    expected = torch.ones(64, dtype=torch.float32, device="cuda:0") * 15.0
    assert torch.allclose(torch_result, expected)


@pytest_ark(need_torch=True)
def test_conversion_ensure_ark_passthrough():
    """Test that _ensure_ark passes through ARK tensors unchanged."""
    from ark.ops import _ensure_ark

    t = ark.tensor([64], ark.fp32)
    assert _ensure_ark(t) is t


@pytest_ark(need_torch=True)
def test_conversion_ensure_ark_converts_torch():
    """Test that _ensure_ark converts torch tensors to ARK tensors."""
    import torch
    from ark.ops import _ensure_ark

    torch_t = torch.ones(64, dtype=torch.float32, device="cuda:0")
    ark_t = _ensure_ark(torch_t)

    assert isinstance(ark_t, ark.Tensor)
    assert ark_t.shape() == [64]


@pytest_ark(need_torch=True)
def test_conversion_ops_accept_torch():
    """Test that ops accept torch tensors directly via _ensure_ark."""
    import torch

    a = torch.ones(64, dtype=torch.float32, device="cuda:0") * 2.0
    b = torch.ones(64, dtype=torch.float32, device="cuda:0") * 3.0

    out = ark.add(a, b)

    with ark.Runtime() as rt:
        rt.launch()
        rt.run()
        result = out.to_numpy()

    expected = np.ones(64, dtype=np.float32) * 5.0
    assert np.allclose(result, expected)
