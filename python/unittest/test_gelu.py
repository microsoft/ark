# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark

import torch
import torch.nn.functional as F
import numpy as np
import unittest


def test_gelu_internal(batch_size, m, n, iter=1):
    ark.init()
    # Create a Model instance
    model = ark.Model()

    input_tensor = model.tensor(ark.Dims(batch_size, m, n), ark.TensorType.FP16)

    output_tensor = model.gelu(input_tensor)
    # Test the mul method
    exe = ark.Executor(0, 0, 1, model, "ops_gelu_test")
    exe.compile()
    input_tensor_host = np.random.rand(batch_size, m, n).astype(np.float16)

    exe.launch()
    exe.tensor_memcpy_host_to_device(input_tensor, input_tensor_host)

    exe.run(iter)

    elapsed = exe.stop()

    output_tensor_host = np.zeros((batch_size, m, n), dtype=np.float16)

    exe.tensor_memcpy_device_to_host(output_tensor_host, output_tensor)

    input_tensor_host_float32 = input_tensor_host.astype(np.float32)

    torch_input = torch.from_numpy(input_tensor_host_float32)

    gt = F.gelu(torch_input).detach().numpy().astype(np.float16)

    # test if the result is correct
    max_abs_error = np.max(np.abs(output_tensor_host - gt))
    mean_abs_error = np.mean(np.abs(output_tensor_host - gt))

    numeric_epsilon_half = np.finfo(np.float16).eps

    np.testing.assert_allclose(
        output_tensor_host, gt, atol=numeric_epsilon_half
    )

    print(
        "gelu test:",
        "batch_size",
        "{:6d}".format(batch_size),
        "m",
        "{:6d}".format(m),
        "n",
        "{:6d}".format(n),
        "max_abs_error",
        "{:.5f}".format(max_abs_error),
        "mean_abs_error",
        "{:.5f}".format(mean_abs_error),
        "elapsed",
        "{:.5f}".format(elapsed),
        "ms",
        "iter",
        iter,
        "elapsed per iter",
        "{:.5f}".format(elapsed / iter),
        " ms ",
    )


class TestGelu(unittest.TestCase):
    def test_gelu(self):
        test_gelu_internal(1, 1, 64)
        test_gelu_internal(1, 128, 128)
        test_gelu_internal(1, 4096, 1024)
        test_gelu_internal(1, 1024, 4096)
        test_gelu_internal(2, 1, 64)
        test_gelu_internal(2, 128, 128)
        test_gelu_internal(8, 4096, 1024)
        test_gelu_internal(8, 1024, 4096)


if __name__ == "__main__":
    unittest.main()
