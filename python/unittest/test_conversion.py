# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from common import ark, pytest_ark
import numpy as np
import pytest
from typing import Callable

try:
    import torch

    _no_torch = False
except ImportError:
    _no_torch = True

# ARK to Torch tests


def initialize_tensor(dimensions, dtype):
    tensor = ark.tensor(dimensions, dtype)
    tensor_host = np.random.rand(*dimensions).astype(dtype.to_numpy())
    return tensor, tensor_host


# Test function to validate the integrity of the PyTorch view of the ARK tensor,
# including its data and attributes such as shape and data type.
@pytest_ark(need_torch=True)
@pytest.mark.parametrize("num_dims,size", [(1, 5), (1, 1024), (2, 5), (2, 32)])
@pytest.mark.parametrize("dtype", [ark.fp16, ark.fp32])
def test_values_fixed_dims(num_dims: int, size: int, dtype: ark.DataType):
    import torch

    dimensions = [size] * num_dims

    input_tensor, input_tensor_host = initialize_tensor(dimensions, dtype)
    other_tensor, other_tensor_host = initialize_tensor(dimensions, dtype)
    output_tensor = ark.add(input_tensor, other_tensor)

    with ark.Runtime() as rt:
        rt.launch()

        input_tensor.from_numpy(input_tensor_host)
        other_tensor.from_numpy(other_tensor_host)

        input_view = input_tensor.to_torch()
        other_view = other_tensor.to_torch()
        output_view = output_tensor.to_torch()

        rt.run()

        input_view_numpy = input_view.cpu().numpy()
        other_view_numpy = other_view.cpu().numpy()
        output_view_numpy = output_view.cpu().numpy()

        output_tensor_host = output_tensor.to_numpy()

    assert np.allclose(input_tensor_host, input_view_numpy)
    assert np.allclose(other_tensor_host, other_view_numpy)
    assert np.allclose(output_tensor_host, output_view_numpy)


# Function to check if there is a difference between two arrays at a specific index
def check_diff(input_tensor_host, input_view_numpy, value, index):
    mask = np.ones(input_tensor_host.shape, dtype=bool)
    mask[index] = False
    if not np.allclose(input_tensor_host[mask], input_view_numpy[mask]):
        print("Difference found at index: ", index)
        return False
    if input_view_numpy[index] != value:
        print(input_view_numpy[index], value)
        return False
    return True


# Test function to check if changes to the torch views are reflected in the original tensors
@pytest_ark(need_torch=True)
@pytest.mark.parametrize("dtype", [ark.fp16, ark.fp32])
def test_ark_to_torch_aliasing(dtype: ark.DataType):
    import torch

    dimensions = [4, 4]
    input_tensor, input_tensor_host = initialize_tensor(dimensions, dtype)
    other_tensor, other_tensor_host = initialize_tensor(dimensions, dtype)
    output_tensor = ark.mul(input_tensor, other_tensor)

    with ark.Runtime() as rt:
        rt.launch()
        input_tensor.from_numpy(input_tensor_host)
        other_tensor.from_numpy(other_tensor_host)

        input_view = input_tensor.to_torch()
        other_view = other_tensor.to_torch()
        output_view = output_tensor.to_torch()
        # make changes to the views
        input_view[1, 1] = 20
        other_view[0, 0] = 30
        rt.run()
        output_view[3, 0] = 40

        output_tensor_host = output_tensor.to_numpy()
        input_view_numpy = input_view.cpu().numpy()
        other_view_numpy = other_view.cpu().numpy()
        output_view_numpy = output_view.cpu().numpy()

    # Check if changes to the views are reflected in the original tensors
    assert check_diff(input_tensor_host, input_view_numpy, 20, (1, 1))
    assert check_diff(other_tensor_host, other_view_numpy, 30, (0, 0))
    assert check_diff(output_tensor_host, output_view_numpy, 40, (3, 0))


@pytest_ark(need_torch=True)
def test_conversion_torch():
    import torch

    dimensions = [4, 4]
    t = ark.constant(7, dimensions)

    with ark.Runtime() as rt:
        rt.launch()

        torch_tensor = t.to_torch()

        assert torch_tensor.shape == (4, 4)
        assert torch_tensor.dtype == torch.float32
        assert torch_tensor.device.type == "cuda"
        assert torch.all(torch_tensor == 0)

        rt.run()

        torch_tensor = t.to_torch()
        assert torch.all(torch_tensor == 7)


# Torch to ARK tests

ArkBinOp = Callable[[ark.Tensor, ark.Tensor], ark.Tensor]
TorchBinOp = Callable[..., "torch.Tensor"]
ArkUnOp = Callable[[ark.Tensor], ark.Tensor]
TorchUnOp = Callable[..., "torch.Tensor"]


# Verify the accuracy of binary operations involving ARK view tensors
@pytest_ark(need_torch=True)
def test_bin_op():
    import torch

    dtype = torch.float16
    tensor_dims = (2, 3)
    input_tensor = torch.randn(tensor_dims, dtype=dtype, device="cuda:0")
    other_tensor = torch.randn(tensor_dims, dtype=dtype, device="cuda:0")
    expected_output = torch.add(input_tensor, other_tensor).cpu().numpy()
    input_ark_view = ark.Tensor.from_torch(input_tensor)
    other_ark_view = ark.Tensor.from_torch(other_tensor)
    output = ark.add(input_ark_view, other_ark_view)

    with ark.Runtime() as rt:
        rt.launch()
        rt.run()
        output_host = output.to_numpy()

    assert np.allclose(output_host, expected_output)


# Verify the accuracy of unary operations involving ARK view tensors
@pytest_ark(need_torch=True)
def test_unary_op():
    import torch

    dtype = torch.float16
    tensor_dims = (3, 3)
    input_tensor = torch.randn(tensor_dims, dtype=dtype, device="cuda:0")
    expected_output = torch.exp(input_tensor).cpu().numpy()
    input_ark_view = ark.Tensor.from_torch(input_tensor)
    output = ark.exp(input_ark_view)

    with ark.Runtime() as rt:
        rt.launch()
        rt.run()
        output_host = output.to_numpy()

    assert np.allclose(output_host, expected_output)


# Test function to check if changes in torch tensors are reflected in ARK views
@pytest_ark(need_torch=True)
def test_torch_to_ark_aliasing():
    import torch

    dtype = torch.float16
    tensor_dims = (64, 64)
    # Initialize a PyTorch tensor
    input_tensor = torch.randn(tensor_dims, dtype=dtype, device="cuda:0")
    other_tensor = torch.randn(tensor_dims, dtype=dtype, device="cuda:0")

    input_ark_view = ark.Tensor.from_torch(input_tensor)
    other_ark_view = ark.Tensor.from_torch(other_tensor)

    output = ark.add(input_ark_view, other_ark_view)
    # Perform in place operations
    input_tensor += other_tensor
    other_tensor += input_tensor
    expected_output = (input_tensor + other_tensor).cpu().numpy()

    with ark.Runtime() as rt:
        rt.launch()
        rt.run()
        output_host = output.to_numpy()

    assert np.allclose(output_host, expected_output)
