# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import ark
import unittest


def test_math_functions_internal(
    batch_size, m, n, data_type="float", function_type="exp", iter=1
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

    if function_type == "exp":
        output_tensor = ark.exp(input_tensor)
    elif function_type == "sqrt":
        output_tensor = ark.sqrt(input_tensor)

    # Launch the ARK runtime
    runtime.launch()

    # Initialize the input and other tensor with random values
    input_tensor_host = np.random.rand(batch_size, m, n).astype(numpy_data_type)
    input_tensor.from_numpy(input_tensor_host)

    # Run the ARK program
    runtime.run(iter=iter, async_run=True)
    elapsed = runtime.stop()

    output_tensor_host = output_tensor.to_numpy()
    if function_type == "exp":
        gt = np.exp(input_tensor_host)
    elif function_type == "sqrt":
        gt = np.sqrt(input_tensor_host)
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
        f"{function_type} test: batch_size {batch_size:6d} m {m:6d} n {n:6d} data_type {data_type} "
        f"max_abs_error {max_abs_error:.5f} mean_abs_error {mean_abs_error:.5f} elapsed {elapsed:.5f} ms "
        f"iter {iter} elapsed_per_iter {elapsed / iter:.5f} ms"
    )


def test_math_functions(function_type="exp"):
    test_math_functions_internal(1, 64, 4, "half", function_type)
    test_math_functions_internal(1, 128, 128, "half", function_type)
    test_math_functions_internal(1, 256, 256, "half", function_type)
    test_math_functions_internal(1, 512, 512, "half", function_type)

    test_math_functions_internal(1, 64, 4, "float", function_type)
    test_math_functions_internal(1, 128, 128, "float", function_type)
    test_math_functions_internal(1, 256, 256, "float", function_type)
    test_math_functions_internal(1, 512, 512, "float", function_type)
    test_math_functions_internal(1, 1024, 1024, "float", function_type)
    test_math_functions_internal(1, 4096, 1024, "float", function_type)
    test_math_functions_internal(1, 1024, 4096, "float", function_type)
    test_math_functions_internal(2, 64, 64, "float", function_type)
    test_math_functions_internal(2, 128, 128, "float", function_type)
    test_math_functions_internal(8, 4096, 1024, "float", function_type)
    test_math_functions_internal(8, 1024, 4096, "float", function_type)


class TestSqrt(unittest.TestCase):
    def test_exp(self):
        test_math_functions("exp")

    def test_sqrt(self):
        test_math_functions("sqrt")


if __name__ == "__main__":
    unittest.main()
