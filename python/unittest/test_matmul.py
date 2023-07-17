# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark

import torch
import torch.nn.functional as F
import numpy as np
import unittest
from parameterized import parameterized


def test_matmul_internal(
    self,
    batch_size,
    m,
    n,
    k,
    bs_a,
    bs_b,
    split_k,
    is_relu=False,
    gran_lev=-1,
    iter=1,
    data_type="float",
):
    ark.init()

    # Create a Model instance
    model = ark.Model()
    if data_type == "float":
        ark_data_type = ark.TensorType.FP32
        numpy_data_type = np.float32
    elif data_type == "half":
        ark_data_type = ark.TensorType.FP16
        numpy_data_type = np.float16
    input_tensor = model.tensor(ark.Dims(bs_a, m, k), ark_data_type)
    other_tensor = model.tensor(ark.Dims(bs_b, k, n), ark_data_type)

    output_tensor = model.matmul(
        input_tensor,
        other_tensor,
        None,
        split_k,
        False,
        False,
        is_relu,
        "matmul",
        gran_lev,
    )
    # Test the mul method
    exe = ark.Executor(0, 0, 1, model, "ops_matmul_test")
    exe.compile()
    input_tensor_host = np.random.rand(batch_size, m, k).astype(numpy_data_type)
    other_tensor_host = np.random.rand(batch_size, k, n).astype(numpy_data_type)
    exe.launch()
    exe.tensor_memcpy_host_to_device(input_tensor, input_tensor_host)
    exe.tensor_memcpy_host_to_device(other_tensor, other_tensor_host)
    exe.run(1)

    elapsed = exe.stop()

    output_tensor_host = np.zeros((batch_size, m, n), dtype=numpy_data_type)

    exe.tensor_memcpy_device_to_host(output_tensor_host, output_tensor)

    gt = np.matmul(input_tensor_host, other_tensor_host)

    # test if the result is correct
    max_error = np.max(np.abs(output_tensor_host - gt))
    avg_error = np.mean(np.abs(output_tensor_host - gt))
    np.testing.assert_allclose(output_tensor_host, gt, rtol=1e-2, atol=1e-2)

    print(
        "matmul:",
        m,
        "x",
        n,
        "x",
        k,
        "(split_k=",
        split_k,
        ", relu=",
        is_relu,
        ", gran_lev=",
        gran_lev,
        ") ",
        " mse ",
        avg_error,
        " max_err ",
        max_error,
        " elapsed ",
        elapsed,
        " ms iter ",
        iter,
    )
    return True


class TestMatmul(unittest.TestCase):
    @parameterized.expand(
        [
            (1, 32, 32, 4, "half"),
            (1, 32, 32, 8, "half"),
            (1, 64, 64, 4, "half"),
            (1, 128, 128, 64, "half"),
            (1, 256, 256, 64, "half"),
            (1, 512, 512, 128, "half"),
        ]
    )
    def test_matmul(self, batch_size, m, n, k, data_type):
        ret = test_matmul_internal(
            self, batch_size, m, n, k, 1, 1, 1, False, -1, 1, data_type
        )
        self.assertTrue(ret)


if __name__ == "__main__":
    unittest.main()
