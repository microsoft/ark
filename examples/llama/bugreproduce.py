import ark
import numpy as np

runtime = ark.Runtime()

input = ark.tensor([1, 1, 64, 128], dtype=ark.FP32)

reduce = ark.reduce_mean(input, axis=input.ndims() - 1)

other = ark.tensor([1, 1, 64, 128], dtype=ark.FP32)

output = ark.add(reduce, other)

runtime.launch()

input_numpy = np.ones([64, 128], dtype=np.float32)
other_numpy = np.ones([64, 128], dtype=np.float32)

input.from_numpy(input_numpy)

other.from_numpy(other_numpy)

runtime.run()

# print(output.to_numpy())
print(reduce.to_numpy())
