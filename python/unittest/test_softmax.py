# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark

import torch
import torch.nn.functional as F
import numpy as np
import unittest


def test_softmax_internal(batch_size, m, n, data_type="float", iter=1):
    ark.init()

    # Create a Model instance
    model = ark.Model()
    if data_type == "float":
        ark_data_type = ark.TensorType.FP32
        numpy_data_type = np.float32
    elif data_type == "half":
        ark_data_type = ark.TensorType.FP16
        numpy_data_type = np.float16
    input_tensor = model.tensor(ark.Dims(batch_size, m, n), ark_data_type)

    output_tensor = model.softmax(input_tensor)
    # Test the mul method
    exe = ark.Executor(0, 0, 1, model, "ops_softmax_test")
    exe.compile()
    input_tensor_host = np.random.rand(batch_size, m, n).astype(numpy_data_type)

    exe.launch()
    exe.tensor_memcpy_host_to_device(input_tensor, input_tensor_host)

    exe.run(iter)

    elapsed = exe.stop()

    output_tensor_host = np.zeros((batch_size, m, n), dtype=numpy_data_type)

    exe.tensor_memcpy_device_to_host(output_tensor_host, output_tensor)

    input_tensor_host_float32 = input_tensor_host.astype(np.float32)

    torch_input = torch.from_numpy(input_tensor_host_float32)

    # get the ground truth
    gt = F.softmax(torch_input, dim=-1).numpy()
    # test if the result is correct
    max_abs_error = np.max(np.abs(output_tensor_host - gt))
    mean_abs_error = np.mean(np.abs(output_tensor_host - gt))
    np.testing.assert_allclose(output_tensor_host, gt, rtol=1e-2, atol=1e-2)

    print(
        "softmax test",
        "batch_size:",
        "{:6d}".format(batch_size),
        "m:",
        "{:6d}".format(m),
        "n:",
        "{:6d}".format(n),
        "data_type:",
        data_type,
        "max abs error:",
        "{:.5f}".format(max_abs_error),
        "mean abs error:",
        "{:.5f}".format(mean_abs_error),
        "elapsed",
        "{:.5f}".format(elapsed),
        " ms ",
        " iter ",
        iter,
    )


class TestSoftmax(unittest.TestCase):
    def test_softmax(self):
        test_softmax_internal(1, 32, 4, "half")
        test_softmax_internal(1, 32, 512, "half")
        test_softmax_internal(1, 64, 4, "half")
        test_softmax_internal(1, 128, 128, "half")
        test_softmax_internal(1, 256, 256, "half")
        test_softmax_internal(1, 512, 512, "half")

        test_softmax_internal(1, 8, 4)
        test_softmax_internal(1, 128, 128)
        test_softmax_internal(1, 256, 256)
        test_softmax_internal(1, 512, 512)
        test_softmax_internal(1, 1024, 1024)
        test_softmax_internal(1, 4096, 1024)
        test_softmax_internal(1, 1024, 4096)
        test_softmax_internal(2, 64, 64)
        test_softmax_internal(2, 128, 128)
        test_softmax_internal(8, 4096, 1024)
        test_softmax_internal(8, 1024, 4096)


if __name__ == "__main__":
    unittest.main()
