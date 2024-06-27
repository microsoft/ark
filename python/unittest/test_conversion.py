import pytest
import numpy as np
import ark
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
@pytest.mark.parametrize("num_dims,size", [(1, 5), (1, 1024), (2, 5), (2, 32)])
@pytest.mark.parametrize("dtype", [ark.fp16, ark.fp32])
def test_values_fixed_dims(num_dims: int, size: int, dtype: ark.DataType):
    ark.init()
    dimensions = [size] * num_dims

    input_tensor, input_tensor_host = initialize_tensor(dimensions, dtype)
    other_tensor, other_tensor_host = initialize_tensor(dimensions, dtype)
    output_tensor = ark.add(input_tensor, other_tensor)

    runtime = ark.Runtime()
    runtime.launch()

    input_tensor.from_numpy(input_tensor_host)
    other_tensor.from_numpy(other_tensor_host)

    input_view = input_tensor.get_torch_view()
    other_view = other_tensor.get_torch_view()
    output_view = output_tensor.get_torch_view()

    runtime.run()

    input_view_numpy = input_view.cpu().numpy()
    other_view_numpy = other_view.cpu().numpy()
    output_view_numpy = output_view.cpu().numpy()

    output_tensor_host = output_tensor.to_numpy()

    runtime.stop()
    runtime.delete_all_runtimes()

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
@pytest.mark.parametrize("dtype", [ark.fp16, ark.fp32])
def test_ark_to_torch_aliasing(dtype: ark.DataType):
    ark.init()
    dimensions = [4, 4]
    input_tensor, input_tensor_host = initialize_tensor(dimensions, dtype)
    other_tensor, other_tensor_host = initialize_tensor(dimensions, dtype)
    output_tensor = ark.mul(input_tensor, other_tensor)
    runtime = ark.Runtime()
    runtime.launch()
    input_tensor.from_numpy(input_tensor_host)
    other_tensor.from_numpy(other_tensor_host)

    input_view = input_tensor.get_torch_view()
    other_view = other_tensor.get_torch_view()
    output_view = output_tensor.get_torch_view()
    # make changes to the views
    input_view[1, 1] = 20
    other_view[0, 0] = 30
    runtime.run()
    output_view[3, 0] = 40

    output_tensor_host = output_tensor.to_numpy()
    input_view_numpy = input_view.cpu().numpy()
    other_view_numpy = other_view.cpu().numpy()
    output_view_numpy = output_view.cpu().numpy()
    # Check if changes to the views are reflected in the original tensors
    print(input_view_numpy)
    assert check_diff(input_tensor_host, input_view_numpy, 20, (1, 1))
    assert check_diff(other_tensor_host, other_view_numpy, 30, (0, 0))
    assert check_diff(output_tensor_host, output_view_numpy, 40, (3, 0))

    runtime.stop()
    runtime.reset()


def test_conversion_torch():
    if _no_torch:
        pytest.skip("PyTorch not available")

    dimensions = [4, 4]

    ark.init()
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
TorchBinOp = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
ArkUnOp = Callable[[ark.Tensor], ark.Tensor]
TorchUnOp = Callable[[torch.Tensor], torch.Tensor]


# Verify the accuracy of binary operations involving ARK view tensors
@pytest.mark.parametrize(
    "dtype, ark_op, torch_op, tensor_dims",
    [(torch.float16, ark.add, torch.add, (2, 3))],
)
def test_bin_op(dtype, ark_op: ArkBinOp, torch_op: TorchBinOp, tensor_dims):
    ark.init()
    input_tensor = torch.randn(tensor_dims, dtype=dtype, device="cuda:0")
    other_tensor = torch.randn(tensor_dims, dtype=dtype, device="cuda:0")
    expected_output = torch_op(input_tensor, other_tensor).cpu().numpy()
    input_ark_view = ark.Tensor.get_ark_view(input_tensor)
    other_ark_view = ark.Tensor.get_ark_view(other_tensor)
    output = ark_op(input_ark_view, other_ark_view)
    runtime = ark.Runtime()
    runtime.launch()
    runtime.run()
    output_host = output.to_numpy()
    runtime.stop()
    runtime.reset()
    assert np.allclose(output_host, expected_output)


# Verify the accuracy of unary operations involving ARK view tensors
@pytest.mark.parametrize(
    "dtype, ark_op, torch_op, tensor_dims",
    [(torch.float16, ark.exp, torch.exp, (3, 3))],
)
def test_unary_op(dtype, ark_op: ArkUnOp, torch_op: TorchUnOp, tensor_dims):
    ark.init()
    input_tensor = torch.randn(tensor_dims, dtype=dtype, device="cuda:0")
    expected_output = torch_op(input_tensor).cpu().numpy()
    input_ark_view = ark.Tensor.get_ark_view(input_tensor)
    output = ark_op(input_ark_view)
    runtime = ark.Runtime()
    runtime.launch()
    runtime.run()
    output_host = output.to_numpy()
    runtime.stop()
    runtime.reset()
    assert np.allclose(output_host, expected_output)


# Test function to check if changes in torch tensors are reflected in ARK views
@pytest.mark.parametrize("dtype, tensor_dims", [(torch.float16, (64, 64))])
def test_torch_to_ark_aliasing(dtype, tensor_dims):
    ark.init()
    # Initialize a PyTorch tensor
    input_tensor = torch.randn(tensor_dims, dtype=dtype, device="cuda:0")
    other_tensor = torch.randn(tensor_dims, dtype=dtype, device="cuda:0")

    input_ark_view = ark.Tensor.get_ark_view(input_tensor)
    other_ark_view = ark.Tensor.get_ark_view(other_tensor)

    output = ark.add(input_ark_view, other_ark_view)
    # Perform in place operations
    input_tensor += other_tensor
    other_tensor += input_tensor
    expected_output = (input_tensor + other_tensor).cpu().numpy()

    runtime = ark.Runtime()
    runtime.launch()
    runtime.run()
    output_host = output.to_numpy()
    runtime.stop()
    runtime.reset()
    assert np.allclose(output_host, expected_output)
