# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark

import torch
import torch.nn.functional as F
import numpy as np
import unittest


def test_reduce_internal(batch_size, m, n, data_type="float", iter=1):
    runtime = ark.Runtime()
    if data_type == "float":
        ark_data_type = ark.TensorType.FP32
        numpy_data_type = np.float32
    elif data_type == "half":
        ark_data_type = ark.TensorType.FP16
        numpy_data_type = np.float16
    input_tensor = ark.tensor(ark.Dims(batch_size, m, n), ark_data_type)

    output_tensor = ark.reduce_sum(input_tensor, 2)
    # Test the mul method
    runtime.launch()
    input_tensor_host = np.random.rand(batch_size, m, n).astype(numpy_data_type)

    input_tensor.from_numpy(input_tensor_host)
    runtime.run(iter)

    elapsed = runtime.stop()

    output_tensor_host = output_tensor.to_numpy()

    input_tensor_host_float32 = input_tensor_host.astype(np.float32)

    torch_input = torch.from_numpy(input_tensor_host_float32)

    gt = torch.sum(torch_input, dim=2, keepdim=True).numpy()

    # test if the result is correct
    max_abs_error = np.max(np.abs(output_tensor_host - gt))
    mean_abs_error = np.mean(np.abs(output_tensor_host - gt))
    # The numeric error of half precision of the machine
    numeric_epsilon_half = np.finfo(np.float16).eps
    # reduce add n numbers, so we assume the atol to be n * numeric_epsilon_half
    # rtol should be atol/max(abs(gt) + epsilon). The epsilon is to avoid
    # divide by zero, here we set it to be numeric_epsilon_half

    atol = numeric_epsilon_half * n
    np.testing.assert_allclose(output_tensor_host, gt, atol=atol)

    print(
        f"reduce test: batch_size {batch_size:6d} m {m:6d} n {n:6d} data_type {data_type} "
        f"max_abs_error {max_abs_error:.5f} mean_abs_error {mean_abs_error:.5f} elapsed "
        f"{elapsed:.5f} ms iter {iter} elapsed_per_iter {elapsed / iter:.5f} ms"
    )


class TestReduce(unittest.TestCase):
    def test_reduce(self):
        test_reduce_internal(1, 64, 4, "half")
        test_reduce_internal(1, 128, 128, "half")
        test_reduce_internal(1, 256, 256, "half")
        test_reduce_internal(1, 512, 512, "half")

        test_reduce_internal(1, 64, 4)
        test_reduce_internal(1, 128, 128)
        test_reduce_internal(1, 256, 256)
        test_reduce_internal(1, 512, 512)
        test_reduce_internal(1, 1024, 1024)
        test_reduce_internal(1, 4096, 1024)
        test_reduce_internal(1, 1024, 4096)
        test_reduce_internal(2, 64, 64)
        test_reduce_internal(2, 128, 128)
        test_reduce_internal(8, 4096, 1024)
        test_reduce_internal(8, 1024, 4096)


if __name__ == "__main__":
    unittest.main()
