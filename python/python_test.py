# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark

ark.init()

a = ark.Dims([1, 2, 3, 4])
print(ark.NO_DIM)
print(a[2])

ark.srand(42)

random_number = ark.rand()

print(random_number)

# Create a TensorBuf object
buf = ark.TensorBuf(1024, 1)

# Create a Dims object for shape, ldims, offs, and pads
shape = ark.Dims(1, 2, 3, 4)
ldims = ark.Dims(4, 4, 4, 4)
offs = ark.Dims(0, 0, 0, 0)
pads = ark.Dims(1, 1, 1, 1)

# # Create a Tensor object
tensor = ark.Tensor(
    shape,
    ark.TensorType.FP32,
    buf,
    ldims,
    offs,
    pads,
    False,
    False,
    0,
    "my_tensor",
)

# # Call Tensor methods
tensor_size = tensor.size()
tensor_offset = tensor.offset(1, 2, 3, 4)

print(tensor_size)
print(tensor_offset)

# Create a Model instance
model = ark.Model()

# Test the tensor method
t1 = model.tensor(ark.Dims(1, 1, 1, 32), ark.TensorType.FP16)
t2 = model.tensor(ark.Dims(1, 1, 1, 32), ark.TensorType.FP16)

# scaled_tensor = model.scale(t1, 2.0)
import numpy as np
# Test the add method
added_tensor = model.add(t1, t2)

# Test the mul method
# multiplied_tensor = model.mul(t1, t2)
exe = ark.Executor(0, 0, 1, model, "test_python_bindings")
exe.compile()
datasrc_np = np.random.rand(1, 32).astype(np.float16)

exe.tensor_memcpy_host_to_device(t1, datasrc_np)
exe.tensor_memcpy_host_to_device(t2, datasrc_np)
# data_test = np.zeros((32, 32), dtype=np.float16)
# exe.tensor_memcpy_device_to_host(data_test, t1)
# assert np.allclose(data_test, datasrc_np)

exe.launch()
exe.run(1)

exe.stop()
datadst_np = np.zeros((1, 32), dtype=np.float16)

exe.tensor_memcpy_device_to_host(datadst_np, added_tensor)

# test if the result is correct
assert np.allclose(datadst_np, datasrc_np * 2.0)
print("ark test success")
