# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark

import torch
import torch.nn.functional as F
import numpy as np
import unittest
from parameterized import parameterized


class TestGelu(unittest.TestCase):
    @parameterized.expand(
        [
            (1, 1, 64),
            (1, 128, 128),
            (1, 4096, 1024),
            (1, 1024, 4096),
            (2, 1, 64),
            (2, 128, 128),
            (8, 4096, 1024),
            (8, 1024, 4096),
        ]
    )
    def test_gelu_internal(self, batch_size, seq_len, d_model):
        ark.init()

        # Create a Model instance
        model = ark.Model()

        input_tensor = model.tensor(
            ark.Dims(batch_size, seq_len, d_model), ark.TensorType.FP16
        )

        output_tensor = model.gelu(input_tensor)
        # Test the mul method
        exe = ark.Executor(0, 0, 1, model, "ops_gelu_test")
        exe.compile()
        input_tensor_host = np.random.rand(batch_size, seq_len, d_model).astype(
            np.float16
        )

        exe.launch()
        exe.tensor_memcpy_host_to_device(input_tensor, input_tensor_host)

        exe.run(1)

        exe.stop()

        output_tensor_host = np.zeros(
            (batch_size, seq_len, d_model), dtype=np.float16
        )

        exe.tensor_memcpy_device_to_host(output_tensor_host, output_tensor)

        input_tensor_host_float32 = input_tensor_host.astype(np.float32)

        torch_input = torch.from_numpy(input_tensor_host_float32)

        gt = F.gelu(torch_input).detach().numpy().astype(np.float16)

        # test if the result is correct
        max_error = np.max(np.abs(output_tensor_host - gt))
        avg_error = np.mean(np.abs(output_tensor_host - gt))
        np.testing.assert_allclose(output_tensor_host, gt, rtol=1e-3, atol=1e-3)
        print("max error: ", max_error)
        print("avg error: ", avg_error)
        print("gelu test success")


if __name__ == "__main__":
    unittest.main()
