# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import ark
import unittest


def test_sub_internal(batch_size, m, n, data_type="float", iter=1):
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

    output_tensor = ark.sub(input_tensor, other_tensor)

    # Launch the ARK runtime
    runtime.launch()

    # Initialize the input and other tensor with random values between 1 and 2
    # (to avoid subision by 0)
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
    gt = input_tensor_host - other_tensor_host
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
        "sub test",
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


class TestSub(unittest.TestCase):
    def test_sub(self):
        test_sub_internal(1, 64, 4, "half")
        test_sub_internal(1, 128, 128, "half")
        test_sub_internal(1, 256, 256, "half")
        test_sub_internal(1, 512, 512, "half")

        test_sub_internal(1, 64, 4)
        test_sub_internal(1, 128, 128)
        test_sub_internal(1, 256, 256)
        test_sub_internal(1, 512, 512)
        test_sub_internal(1, 1024, 1024)
        test_sub_internal(1, 4096, 1024)
        test_sub_internal(1, 1024, 4096)
        test_sub_internal(2, 64, 64)
        test_sub_internal(2, 128, 128)
        test_sub_internal(8, 4096, 1024)
        test_sub_internal(8, 1024, 4096)


if __name__ == "__main__":
    unittest.main()
