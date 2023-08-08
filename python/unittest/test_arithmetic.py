# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import ark
import unittest


def test_arithmetic_internal(
    batch_size, m, n, data_type="float", arithmetic_func="add", iter=1
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
    other_tensor = ark.tensor([batch_size, m, n], ark_data_type)

    if arithmetic_func == "add":
        output_tensor = ark.add(input_tensor, other_tensor)
    elif arithmetic_func == "sub":
        output_tensor = ark.sub(input_tensor, other_tensor)
    elif arithmetic_func == "mul":
        output_tensor = ark.mul(input_tensor, other_tensor)
    elif arithmetic_func == "div":
        output_tensor = ark.div(input_tensor, other_tensor)

    # Launch the ARK runtime
    runtime.launch()

    # Initialize the input and other tensor with random values between 1 and 2
    # (to avoid division by 0)
    input_tensor_host = (
        np.random.rand(batch_size, m, n).astype(numpy_data_type) + 1
    )
    input_tensor.from_numpy(input_tensor_host)
    other_tensor_host = (
        np.random.rand(batch_size, m, n).astype(numpy_data_type) + 1
    )
    other_tensor.from_numpy(other_tensor_host)

    # Run the ARK program
    runtime.run(iter=iter, async_run=True)
    elapsed = runtime.stop()

    output_tensor_host = output_tensor.to_numpy()
    if arithmetic_func == "add":
        gt = input_tensor_host + other_tensor_host
    elif arithmetic_func == "sub":
        gt = input_tensor_host - other_tensor_host
    elif arithmetic_func == "mul":
        gt = input_tensor_host * other_tensor_host
    elif arithmetic_func == "div":
        gt = input_tensor_host / other_tensor_host
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
        arithmetic_func,
        "test",
        "batch_size:",
        "{:6d}".format(batch_size),
        "m:",
        "{:6d}".format(m),
        "n:",
        "{:6d}".format(n),
        "data_type:",
        data_type,
        "max_abs_error:",
        "{:.5f}".format(max_abs_error),
        "mean_abs_error:",
        "{:.5f}".format(mean_abs_error),
        "elapsed",
        "{:.5f}".format(elapsed),
        " ms ",
        " iter ",
        iter,
        "elapsed_per_iter",
        "{:.5f}".format(elapsed / iter),
        " ms ",
    )


def test_arithmetic(arithmetic_func="add"):
    test_arithmetic_internal(1, 64, 4, "half", arithmetic_func)
    test_arithmetic_internal(1, 128, 128, "half", arithmetic_func)
    test_arithmetic_internal(1, 256, 256, "half", arithmetic_func)
    test_arithmetic_internal(1, 512, 512, "half", arithmetic_func)

    test_arithmetic_internal(1, 64, 4, "float", arithmetic_func)
    test_arithmetic_internal(1, 128, 128, "float", arithmetic_func)
    test_arithmetic_internal(1, 256, 256, "float", arithmetic_func)
    test_arithmetic_internal(1, 512, 512, "float", arithmetic_func)
    test_arithmetic_internal(1, 1024, 1024, "float", arithmetic_func)
    test_arithmetic_internal(1, 4096, 1024, "float", arithmetic_func)
    test_arithmetic_internal(1, 1024, 4096, "float", arithmetic_func)
    test_arithmetic_internal(2, 64, 64, "float", arithmetic_func)
    test_arithmetic_internal(2, 128, 128, "float", arithmetic_func)
    test_arithmetic_internal(8, 4096, 1024, "float", arithmetic_func)
    test_arithmetic_internal(8, 1024, 4096, "float", arithmetic_func)


class TestArithmetic(unittest.TestCase):
    def test_add(self):
        test_arithmetic("add")

    def test_sub(self):
        test_arithmetic("sub")

    def test_mul(self):
        test_arithmetic("mul")

    def test_div(self):
        test_arithmetic("div")


if __name__ == "__main__":
    unittest.main()
