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
tensor = ark.Tensor(shape, ark.TensorType.FP32, buf, ldims,
                    offs, pads, False, False, 0, "my_tensor")

# # Call Tensor methods
tensor_size = tensor.size()
tensor_offset = tensor.offset(1, 2, 3, 4)

print(tensor_size)
print(tensor_offset)

# Create a Model instance
model = ark.Model()

# Test the tensor method
t1 = model.tensor(ark.Dims(1, 1, 64, 64), ark.TensorType.FP32)
t2 = model.tensor(ark.Dims(1, 1, 64, 64), ark.TensorType.FP32)

# scaled_tensor = model.scale(t1, 2.0)

# Test the add method
added_tensor = model.add(t1, t2)

# Test the mul method
# multiplied_tensor = model.mul(t1, t2)
exe=ark.Executor(0, 0, 1, model, "test_python_bindings")
exe.compile()
exe.launch()
exe.run(1)
exe.stop()

print("ark test success")
