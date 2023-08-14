# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import ark
import unittest
import random


def test_scale_internal(batch_size, m, n, data_type="float", iter=1):
    if data_type == "float":
        ark_data_type = ark.FP32
        numpy_data_type = np.float32
    elif data_type == "half":
        ark_data_type = ark.FP16
        numpy_data_type = np.float16
    # Initialize the ARK runtime
    runtime = ark.Runtime()

    input_tensor = ark.tensor([batch_size, m, n], ark_data_type)
    scaled_value = random.uniform(-1, 1)
    output_tensor = ark.scale(input_tensor, scaled_value)

    # Launch the ARK runtime
    runtime.launch()

    # Initialize the input and other tensor with random values between 1 and 2
    # (to avoid sigmoidision by 0)
    input_tensor_host = (
        np.random.rand(batch_size, m, n).astype(numpy_data_type) + 1
    )
    input_tensor.from_numpy(input_tensor_host)

    # Run the ARK program
    runtime.run(iter=iter, async_run=True)
    elapsed = runtime.stop()

    output_tensor_host = output_tensor.to_numpy()
    gt = input_tensor_host * scaled_value
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
        f"scale test batch_size: {batch_size:6d} m: {m:6d} "
        f"n: {n:6d} data_type: {data_type} max_abs_error: "
        f"{max_abs_error:.5f} mean_abs_error: {mean_abs_error:.5f} "
        f"elapsed {elapsed:.5f} ms iter {iter} elapsed_per_iter {elapsed / iter:.5f} ms"
    )


class TestScale(unittest.TestCase):
    def test_scale(self):
        test_scale_internal(1, 64, 4, "half")
        test_scale_internal(1, 128, 128, "half")
        test_scale_internal(1, 256, 256, "half")
        test_scale_internal(1, 512, 512, "half")

        test_scale_internal(1, 64, 4, "float")
        test_scale_internal(1, 128, 128, "float")
        test_scale_internal(1, 256, 256, "float")
        test_scale_internal(1, 512, 512, "float")
        test_scale_internal(1, 1024, 1024, "float")
        test_scale_internal(1, 4096, 1024, "float")
        test_scale_internal(1, 1024, 4096, "float")
        test_scale_internal(2, 64, 64, "float")
        test_scale_internal(2, 128, 128, "float")
        test_scale_internal(8, 4096, 1024, "float")
        test_scale_internal(8, 1024, 4096, "float")


if __name__ == "__main__":
    unittest.main()
