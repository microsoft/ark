# A Quick Guide to Using ARK with Python for DNN Applications

Welcome to this tutorial on using ARK to run a simple deep neural network (DNN) application in Python. We will walk you through a basic Python example to illustrate the process.

After completing the [installation](./install.md) and setting the ARK_ROOT, you can run the tutorial example at [tutorial.py](./example/tutorial.py) to see how ARK works. 

```bash
python3 example/tutorial.py
```

ARK is an innovative GPU-driven code execution system, its architecture are depicted here: ![GPU-driven System Architecture](./imgs/GPU-driven_System_Architecture.png)

There are environment variables available to configure ARK. For more details about these variables, please refer to [Environment Variables](./env.md).

Before diving in, let's import the required modules and initialize ARK:

```python
import ark
import numpy as np
# clean up the shared memory directory. 
ark.init()
```
First, we need to create the operational graph for our DNN models. In this example, we define a simple model with two input tensors. The output tensor will be the sum of these input tensors.

```python
# Create a Model instance
model = ark.Model()

# Create two tensors
input = model.tensor(ark.Dims(1, 1, 1, 32), ark.TensorType.FP16)
other = model.tensor(ark.Dims(1, 1, 1, 32), ark.TensorType.FP16)

# Add input and other to get output tensor
output = model.add(input, other)
```

Next, we need to create the executor for the model. The scheduler will be created and start scheduling the model once the executor is created. After scheduling, the code generator of the scheduler will produce the CUDA kernel that can be used to run the model.

```python
exe = ark.Executor(0, 0, 1, model, "test_add")
```

Now, we can instruct the executor to compile the model:

```python
exe.compile()
```

At this step, users can see a compiling INFO such as:

```bash
Compiling /tmp/ark_\*\*\*\*\*\*\*\*\*\*.cu
```

You can open the compiled file and inspect the generated CUDA kernel code. Here is a snippet of the generated CUDA kernel code:

```cuda
#include "ark_kernels.h"
__device__ ark::sync::State _ARK_LOOP_SYNC_STATE;
__device__ char *_ARK_BUF;
DEVICE void uop0(ark::half *_0, ark::half *_1, ark::half *_2, int tx, int ty, int tz) {
  ark::add<ark::Vec<1, 1, 1, 64>, ark::Vec<1, 1, 1, 32>, ark::Vec<1, 1, 1, 64>, ark::Vec<1, 1, 1, 32>, ark::Vec<1, 1, 1, 64>, ark::Vec<1, 1, 1, 32>, ark::Vec<1, 1, 1, 64>, 32, 0>(_0, _1, _2, tx, ty, tz);
}
// tile dims: (1, 1, 1)
__noinline__ __device__ void op0(int _ti) {
  uop0((ark::half *)&_ARK_BUF[256], (ark::half *)&_ARK_BUF[0], (ark::half *)&_ARK_BUF[128], 0, 0, _ti);
}
DEVICE void depth0() {
  if (blockIdx.x < 1) {
    if (threadIdx.x < 32) {
      op0(blockIdx.x);
    }
  }
}
__device__ void ark_loop_body(int _iter) {
  depth0();
}
```
As seen in the code above, the generated CUDA kernel has one depth, which utilizes the first block and 32 threads to execute the unit operation uop0. The uop0 unit operation is the add operation we defined in the Python code. It employs the ark::add function from the ark_kernels.h to perform the operation.

Before launching the kernel, it is necessary to initialize the input tensors. You can randomly initialize the input tensors with numpy and then copy the data to the GPU using exe.tensor_memcpy_host_to_device.

```python
# Initialize the input tensors
input_np = np.random.rand(1, 32).astype(np.float16)
other_np = np.random.rand(1, 32).astype(np.float16)

exe.tensor_memcpy_host_to_device(input, input_np)
exe.tensor_memcpy_host_to_device(other, other_np)
```

Next, you can launch the kernel and run it for one iteration. To wait for the kernel to finish, use exe.stop().

```python
# Launch the kernel and run for 1 iteration
exe.launch()
exe.run(1)

# Wait for the kernel to finish
exe.stop()
```

Lastly, copy the output tensor back to the host and verify the result.

```python
# Copy the output tensor back to host
output_np = np.zeros((1, 32), dtype=np.float16)
exe.tensor_memcpy_device_to_host(output_np, output)

# test if the result is correct
assert np.allclose(output_np, input_np + other_np)
```

Congratulations! You have successfully learned how to run a DNN model over ARK. Happy coding!

