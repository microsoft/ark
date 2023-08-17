# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark

import torch
import numpy as np
import unittest


def test_norm_internal(
    batch_size, m, n, data_type="float", norm_type="layernorm", iter=1
):
    runtime = ark.Runtime()
    if data_type == "float":
        ark_data_type = ark.TensorType.FP32
        numpy_data_type = np.float32
    elif data_type == "half":
        ark_data_type = ark.TensorType.FP16
        numpy_data_type = np.float16
    input_tensor = ark.tensor(ark.Dims(batch_size, m, n), ark_data_type)
    if norm_type == "layernorm":
        output_tensor = ark.layernorm(input_tensor)
    elif norm_type == "rmsnorm":
        output_tensor = ark.rmsnorm(input_tensor)
    runtime.launch()

    input_tensor_host = np.random.rand(batch_size, m, n).astype(numpy_data_type)
    input_tensor.from_numpy(input_tensor_host)

    runtime.run(iter)

    elapsed = runtime.stop()

    output_tensor_host = output_tensor.to_numpy()

    input_tensor_host_float32 = input_tensor_host.astype(np.float32)

    torch_input = torch.from_numpy(input_tensor_host_float32)
    if norm_type == "layernorm":
        # get the ground truth
        gt = torch.nn.LayerNorm([n], eps=1e-5, elementwise_affine=False)(
            torch_input
        ).numpy()
    elif norm_type == "rmsnorm":
        gt = (
            torch_input
            * torch.rsqrt(torch_input.pow(2).mean(-1, keepdim=True) + 1e-5)
        ).numpy()

    # test if the result is correct
    max_abs_error = np.max(np.abs(output_tensor_host - gt))
    mean_abs_error = np.mean(np.abs(output_tensor_host - gt))
    numeric_epsilon_half = np.finfo(np.float16).eps
    # layernorm half precision error is too large now
    np.testing.assert_allclose(
        output_tensor_host, gt, atol=10 * numeric_epsilon_half
    )

    print(
        f"{norm_type} test batch_size: {batch_size:6d} m: {m:6d} "
        f"n: {n:6d} data_type: {data_type} max_abs_error: {max_abs_error:.5f} "
        f"mean_abs_error: {mean_abs_error:.5f} elapsed {elapsed:.5f} ms iter "
        f"{iter} elapsed_per_iter {elapsed / iter:.5f} ms"
    )


class TestNorm(unittest.TestCase):
    def test_layernorm(self):
        test_norm_internal(1, 32, 4, "half", "layernorm")
        test_norm_internal(1, 32, 512, "half", "layernorm")
        test_norm_internal(1, 64, 4, "half", "layernorm")
        test_norm_internal(1, 128, 128, "half", "layernorm")
        test_norm_internal(1, 256, 256, "half", "layernorm")
        test_norm_internal(1, 512, 512, "half", "layernorm")

        test_norm_internal(1, 8, 4, "float", "layernorm")
        test_norm_internal(1, 128, 128, "float", "layernorm")
        test_norm_internal(1, 256, 256, "float", "layernorm")
        test_norm_internal(1, 512, 512, "float", "layernorm")
        test_norm_internal(1, 1024, 1024, "float", "layernorm")
        test_norm_internal(1, 4096, 1024, "float", "layernorm")
        test_norm_internal(1, 1024, 4096, "float", "layernorm")
        test_norm_internal(2, 64, 64, "float", "layernorm")
        test_norm_internal(2, 128, 128, "float", "layernorm")
        test_norm_internal(8, 4096, 1024, "float", "layernorm")
        test_norm_internal(8, 1024, 4096, "float", "layernorm")

    def test_rmsnorm(self):
        test_norm_internal(1, 32, 4, "half", "rmsnorm")
        test_norm_internal(1, 32, 512, "half", "rmsnorm")
        test_norm_internal(1, 64, 4, "half", "rmsnorm")
        test_norm_internal(1, 128, 128, "half", "rmsnorm")
        test_norm_internal(1, 256, 256, "half", "rmsnorm")
        test_norm_internal(1, 512, 512, "half", "rmsnorm")

        test_norm_internal(1, 8, 4, "float", "rmsnorm")
        test_norm_internal(1, 128, 128, "float", "rmsnorm")
        test_norm_internal(1, 256, 256, "float", "rmsnorm")
        test_norm_internal(1, 512, 512, "float", "rmsnorm")
        test_norm_internal(1, 1024, 1024, "float", "rmsnorm")
        test_norm_internal(1, 4096, 1024, "float", "rmsnorm")
        test_norm_internal(1, 1024, 4096, "float", "rmsnorm")
        test_norm_internal(2, 64, 64, "float", "rmsnorm")
        test_norm_internal(2, 128, 128, "float", "rmsnorm")
        test_norm_internal(8, 4096, 1024, "float", "rmsnorm")
        test_norm_internal(8, 1024, 4096, "float", "rmsnorm")


if __name__ == "__main__":
    unittest.main()
