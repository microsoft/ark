# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import ark
import unittest
import torch
import torch.nn.functional as F


def test_activation_internal(
    batch_size, m, n, data_type="float", activation_type="relu", iter=1
):
    if data_type == "float":
        ark_data_type = ark.FP32
        numpy_data_type = np.float32
    elif data_type == "half":
        ark_data_type = ark.FP16
        numpy_data_type = np.float16
    # Initialize the ARK runtime
    runtime = ark.Runtime()

    input_tensor = ark.tensor([batch_size, m, n], ark_data_type)

    if activation_type == "relu":
        output_tensor = ark.relu(input_tensor)
    elif activation_type == "gelu":
        output_tensor = ark.gelu(input_tensor)
    elif activation_type == "sigmoid":
        output_tensor = ark.sigmoid(input_tensor)

    # Launch the ARK runtime
    runtime.launch()

    # Initialize the input and other tensor with random values
    input_tensor_host = np.random.rand(batch_size, m, n).astype(numpy_data_type)
    input_tensor.from_numpy(input_tensor_host)

    # Run the ARK program
    runtime.run(iter=iter, async_run=True)
    elapsed = runtime.stop()

    output_tensor_host = output_tensor.to_numpy()
    if activation_type == "relu":
        gt = np.maximum(input_tensor_host, 0)
    elif activation_type == "gelu":
        torch_input = torch.from_numpy(input_tensor_host.astype(np.float32))
        gt = F.gelu(torch_input).detach().numpy().astype(np.float16)
    elif activation_type == "sigmoid":
        gt = 1 / (1 + np.exp(-input_tensor_host))
    # Check if the output tensor is equal to the sum of the input and other tensor
    # test if the result is correct
    max_abs_error = np.max(np.abs(output_tensor_host - gt))
    mean_abs_error = np.mean(np.abs(output_tensor_host - gt))
    # The numeric error of half precision of the machine
    numeric_epsilon_half = np.finfo(np.float16).eps
    np.testing.assert_allclose(
        output_tensor_host, gt, atol=numeric_epsilon_half * 2
    )
    print(
        f"{activation_type} test batch_size: {batch_size:6d} m: {m:6d} "
        f"n: {n:6d} data_type: {data_type} max_abs_error: "
        f"{max_abs_error:.5f} mean_abs_error: {mean_abs_error:.5f} "
        f"elapsed {elapsed:.5f} ms iter {iter} elapsed_per_iter {elapsed / iter:.5f} ms"
    )


def test_activation(activation_type="relu"):
    test_activation_internal(1, 64, 4, "half", activation_type)
    test_activation_internal(1, 128, 128, "half", activation_type)
    test_activation_internal(1, 256, 256, "half", activation_type)
    test_activation_internal(1, 512, 512, "half", activation_type)

    test_activation_internal(1, 64, 4, "float", activation_type)
    test_activation_internal(1, 128, 128, "float", activation_type)
    test_activation_internal(1, 256, 256, "float", activation_type)
    test_activation_internal(1, 512, 512, "float", activation_type)
    test_activation_internal(1, 1024, 1024, "float", activation_type)
    test_activation_internal(1, 4096, 1024, "float", activation_type)
    test_activation_internal(1, 1024, 4096, "float", activation_type)
    test_activation_internal(2, 64, 64, "float", activation_type)
    test_activation_internal(2, 128, 128, "float", activation_type)
    test_activation_internal(8, 4096, 1024, "float", activation_type)
    test_activation_internal(8, 1024, 4096, "float", activation_type)


class TestActivation(unittest.TestCase):
    def test_relu(self):
        test_activation("relu")

    def test_gelu(self):
        test_activation("gelu")

    def test_sigmoid(self):
        test_activation("sigmoid")


if __name__ == "__main__":
    unittest.main()
