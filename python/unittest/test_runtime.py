# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from common import ark, pytest_ark
import numpy as np


@pytest_ark()
def test_runtime_empty():
    with ark.Runtime.get_runtime() as rt:
        rt.launch()
        rt.run()
        rt.stop()

@pytest_ark()
def test_runtime_init():
    M, N = 64, 64
    input_tensor = ark.tensor([M, N], ark.fp16)
    other_tensor = ark.tensor([M, N], ark.fp16)
    output_tensor = ark.add(input_tensor, other_tensor)
    runtime = ark.Runtime()
    runtime.launch()
    input_tensor_host = np.random.rand(M, N).astype(np.float16)
    input_tensor.from_numpy(input_tensor_host)
    other_tensor_host = np.random.rand(M, N).astype(np.float16)
    other_tensor.from_numpy(other_tensor_host)
    runtime.run()
    output_tensor_host = output_tensor.to_numpy()
    np.testing.assert_allclose(
        output_tensor_host, input_tensor_host + other_tensor_host
    )
    runtime.stop()
    ark.Model.reset()
    prev_output = output_tensor
    new_tensor = ark.tensor([M, N], ark.fp16)
    final_output = ark.add(prev_output, new_tensor)
    runtime.launch()
    new_tensor_host = np.random.rand(M, N).astype(np.float16)
    new_tensor.from_numpy(new_tensor_host)
    runtime.run()
    final_output_host = final_output.to_numpy()
    np.testing.assert_allclose(
        final_output_host, output_tensor_host + new_tensor_host
    )
    runtime.reset()


@pytest_ark()
def test_runtime_reuse_plans():
    M, N = 64, 64
    input_tensor = ark.tensor([M, N], ark.fp16)
    other_tensor = ark.tensor([M, N], ark.fp16)
    output_tensor = ark.add(input_tensor, other_tensor)
    runtime = ark.Runtime()
    runtime.launch()
    input_tensor_host = np.random.rand(M, N).astype(np.float16)
    input_tensor.from_numpy(input_tensor_host)
    other_tensor_host = np.random.rand(M, N).astype(np.float16)
    other_tensor.from_numpy(other_tensor_host)
    runtime.run()
    output_tensor_host = output_tensor.to_numpy()
    np.testing.assert_allclose(
        output_tensor_host, input_tensor_host + other_tensor_host
    )
    runtime.stop()
    ark.Model.reset()
    runtime.launch()
    runtime.run()
    output_tensor_host = output_tensor.to_numpy()
    np.testing.assert_allclose(
        output_tensor_host, input_tensor_host + other_tensor_host
    )
    runtime.reset()
