# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import numpy as np

ark.init()

# Create a Model instance
model = ark.Model()

# Create two tensors
input = model.tensor(ark.Dims(1, 1, 1, 32), ark.TensorType.FP16)
other = model.tensor(ark.Dims(1, 1, 1, 32), ark.TensorType.FP16)

# Add input and other to get output tensor
output = model.add(input, other)

# Create the executor instance, the scheduler will be created and start scheduling the model when the executor is created
exe = ark.Executor(0, 0, 1, model, "test_python_bindings")
exe.compile()
input_np = np.random.rand(1, 32).astype(np.float16)
other_np = np.random.rand(1, 32).astype(np.float16)

exe.tensor_memcpy_host_to_device(input, input_np)
exe.tensor_memcpy_host_to_device(other, other_np)

exe.launch()
exe.run(1)

exe.stop()
output_np = np.zeros((1, 32), dtype=np.float16)

exe.tensor_memcpy_device_to_host(output_np, output)

# test if the result is correct
assert np.allclose(output_np, input_np + other_np)
print("ark test success")
