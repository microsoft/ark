# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark

import torch
import torch.nn.functional as F
import numpy as np
import unittest


def test_sharding_internal(batch_size, m, n, data_type="float", iter=1):
    runtime = ark.Runtime()
    if data_type == "float":
        ark_data_type = ark.TensorType.FP32
        numpy_data_type = np.float32
    elif data_type == "half":
        ark_data_type = ark.TensorType.FP16
        numpy_data_type = np.float16
    input_tensor0 = ark.tensor(ark.Dims(batch_size, m, n // 2), ark_data_type)
    input_tensor1 = ark.tensor(ark.Dims(batch_size, m, n // 2), ark_data_type)
    output_tensor = ark.tensor(ark.Dims(batch_size, m, n), ark_data_type)
    output_tensor_shards = ark.sharding(output_tensor, 2, n // 2)
    ark.scale(input_tensor0, 1.0, output_tensor_shards[0])
    ark.scale(input_tensor1, 1.0, output_tensor_shards[1])
    # Test the mul method
    runtime.launch()
    input_tensor0_host = np.random.rand(batch_size, m, n // 2).astype(
        numpy_data_type
    )
    input_tensor1_host = np.random.rand(batch_size, m, n // 2).astype(
        numpy_data_type
    )

    input_tensor0.from_numpy(input_tensor0_host)
    input_tensor1.from_numpy(input_tensor1_host)
    runtime.run(iter)

    elapsed = runtime.stop()

    output_tensor_host = output_tensor.to_numpy()
    print(input_tensor0_host)
    print(input_tensor1_host)
    print(output_tensor_host)
    # test if the result is correct
    gt = np.concatenate((input_tensor0_host, input_tensor1_host), axis=2)
    print(gt)
    max_abs_error = np.max(np.abs(output_tensor_host - gt))
    mean_abs_error = np.mean(np.abs(output_tensor_host - gt))
    # The numeric error of half precision of the machine
    numeric_epsilon_half = np.finfo(np.float16).eps
    # sharding add n numbers, so we assume the atol to be n * numeric_epsilon_half
    # rtol should be atol/max(abs(gt) + epsilon). The epsilon is to avoid
    # divide by zero, here we set it to be numeric_epsilon_half

    atol = numeric_epsilon_half * n
    # np.testing.assert_allclose(output_tensor_host, gt, atol=atol)

    print(
        f"sharding test: batch_size {batch_size:6d} m {m:6d} n {n:6d} data_type {data_type} "
        f"max_abs_error {max_abs_error:.5f} mean_abs_error {mean_abs_error:.5f} elapsed "
        f"{elapsed:.5f} ms iter {iter} elapsed_per_iter {elapsed / iter:.5f} ms"
    )


class TestSharding(unittest.TestCase):
    def test_sharding(self):
        test_sharding_internal(1, 32, 128, "float")


if __name__ == "__main__":
    unittest.main()
