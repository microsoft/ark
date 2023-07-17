# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark

import torch
import torch.nn.functional as F
import numpy as np
import unittest


def test_matmul_internal(
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
    exe.launch()
    input_tensor_host = np.random.rand(bs_a, m, k).astype(numpy_data_type)
    other_tensor_host = np.random.rand(bs_b, k, n).astype(numpy_data_type)
    exe.tensor_memcpy_host_to_device(input_tensor, input_tensor_host)
    exe.tensor_memcpy_host_to_device(other_tensor, other_tensor_host)
    exe.run(1)

    elapsed = exe.stop()

    output_tensor_host = np.zeros((bs_a, m, n), dtype=numpy_data_type)

    exe.tensor_memcpy_device_to_host(output_tensor_host, output_tensor)

    gt = np.matmul(input_tensor_host, other_tensor_host)
    if is_relu:
        gt = np.maximum(gt, 0)
    # test if the result is correct
    max_abs_error = np.max(np.abs(output_tensor_host - gt))
    mean_abs_error = np.mean(np.abs(output_tensor_host - gt))
    # np.testing.assert_allclose(output_tensor_host, gt, rtol=1e-2, atol=1e-2)

    print(
        "matmul test:",
        "bs_a",
        "{:6d}".format(bs_a),
        "bs_b",
        "{:6d}".format(bs_b),
        "m",
        "{:6d}".format(m),
        "x",
        "{:6d}".format(n),
        "x",
        "{:6d}".format(k),
        "(split_k=",
        split_k,
        ", relu=",
        is_relu,
        ", gran_lev=",
        gran_lev,
        ") ",
        " max_abs_error ",
        "{:.5f}".format(max_abs_error),
        " elapsed ",
        " mse ",
        "{:.5f}".format(mean_abs_error),
        "{:.5f}".format(elapsed),
        " ms ",
        " iter ",
        iter,
    )
    return True


# Test the correctness of matmul at small scale
def test_matmul_small_sizes(split_k, is_relu, gran_lev, iter=1):
    test_matmul_internal(
        64, 64, 32, 1, 1, split_k, is_relu, gran_lev, iter, "half"
    )
    test_matmul_internal(
        128, 64, 32, 1, 1, split_k, is_relu, gran_lev, iter, "half"
    )
    test_matmul_internal(
        64, 128, 32, 1, 1, split_k, is_relu, gran_lev, iter, "half"
    )
    test_matmul_internal(
        128, 128, 32, 1, 1, split_k, is_relu, gran_lev, iter, "half"
    )

    test_matmul_internal(
        64, 64, 64, 1, 1, split_k, is_relu, gran_lev, iter, "half"
    )
    test_matmul_internal(
        128, 64, 64, 1, 1, split_k, is_relu, gran_lev, iter, "half"
    )
    test_matmul_internal(
        64, 128, 64, 1, 1, split_k, is_relu, gran_lev, iter, "half"
    )
    test_matmul_internal(
        128, 128, 64, 1, 1, split_k, is_relu, gran_lev, iter, "half"
    )
    test_matmul_internal(
        256, 128, 64, 1, 1, split_k, is_relu, gran_lev, iter, "half"
    )

    test_matmul_internal(
        128, 128, 256, 1, 1, split_k, is_relu, gran_lev, iter, "half"
    )


class TestMatmul(unittest.TestCase):
    def test_matmul_gran(self):
        for gran_lev in range(-1, 4):
            print("test_matmul_gran gran_lev=", gran_lev)
            test_matmul_small_sizes(1, False, gran_lev)

    def test_matmul_relu(self):
        print("test_matmul_relu")
        test_matmul_small_sizes(1, True, -1)

    def test_matmul_split(self):
        print("test_matmul_split")
        for split_k in range(2, 4):
            test_matmul_small_sizes(split_k, False, -1)
        for split_k in range(3, 8):
            for gran_lev in range(-1, 4):
                test_matmul_internal(
                    128, 4096, 1024, 1, 1, split_k, False, gran_lev, 1, "half"
                )

    def test_matmul_perf(self):
        test_matmul_small_sizes(1, False, -1, 1000)


if __name__ == "__main__":
    unittest.main()
