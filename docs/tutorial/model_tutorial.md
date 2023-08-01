# Using the Model and Executor APIs in ARK
ARK provides two levels of APIs for running deep neural network (DNN) applications: the ARK runtime API and the Model and Executor APIs. ARK runtime is built on top of the Model and Executor APIs.

While the ARK runtime API provides a simple interface for running DNN models, the Model and Executor APIs offer more advanced features for customizing and optimizing DNN applications. For historical reasons, the Model and Executor APIs are used in previous versions of ARK Applications.

If you're interested in using the Model and Executor APIs, this guide will take a step further from the [quickstart guide](./quickstart.md) to show a low-level example of using the Model and Executor APIs to run a simple DNN application in Python.

After completing the [installation](./install.md), you can run the tutorial example at [model_tutorial.py](../examples/tutorial/model_tutorial.py).

```bash
python examples/tutorial/tutorial.py
```

Before diving in, let's import the required modules and initialize ARK:

```python
import ark
import numpy as np

# Initialize the ARK runtime.
ark.init()
```
First, we need to create the operational graph for our DNN model. In this example, we define a simple model with two input tensors. The output tensor is the sum of these input tensors.

```python
# Create a Model instance
model = ark.Model()

# Create two tensors
input = model.tensor(ark.Dims(32,), ark.TensorType.FP16)
other = model.tensor(ark.Dims(32,), ark.TensorType.FP16)

# Add input and other to get output tensor
output = model.add(input, other)
```

Next, we need to create an executor for the model. Once it is created, the executor automatically runs a scheduler that schedules the model's GPU tasks. After the scheduling, a code generator will generate a CUDA kernel that runs the model.

```python
exe = ark.Executor(0, 0, 1, model, "tutorial_model")
```

Now, we instruct the executor to compile the model.

```python
exe.compile()
```

At this step, the system prints a compiling message such as:

```bash
INFO ark/gpu/gpu_compile.cc:236 Compiling /tmp/ark_xxxxxxxxxxx.cu
```

If you are interested, you can open the compiled file to inspect the generated CUDA kernel code. Here is a snippet of the generated CUDA kernel code from this tutorial.

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
As shown in the code above, the generated CUDA kernel has one depth (`depth0`), which utilizes the first thread block and 32 threads to run the unit operator `uop0`. `uop0` corresponds to the add operator we defined in the Python code. It employs `ark::add` function from `ark_kernels.h`.

Before launching the kernel, users may want to initialize input tensors. You can copy a numpy array into a tensor on GPU using `exe.tensor_memcpy_host_to_device()`.

```python
# Initialize the input tensors
input_np = np.random.rand(1, 32).astype(np.float16)
other_np = np.random.rand(1, 32).astype(np.float16)

exe.tensor_memcpy_host_to_device(input, input_np)
exe.tensor_memcpy_host_to_device(other, other_np)
```

Next, you can launch the kernel and run it for one iteration. We use `exe.stop()` here that waits for completion of all iterations and then exits the kernel.

```python
# Launch the kernel and run for 1 iteration
exe.launch()
exe.run(1)

# Wait for the 1 iteration to finish and exit the kernel
exe.stop()
```

Lastly, copy the output tensor back to the host and verify the result.

```python
# Copy the output tensor back to host
output_np = np.zeros((1, 32), dtype=np.float16)
exe.tensor_memcpy_device_to_host(output_np, output)

# test if the result is correct
assert np.allclose(output_np, input_np + other_np)

max_error = np.max(np.abs(output_np - (input_np + other_np)))
mean_error = np.mean(np.abs(output_np - (input_np + other_np)))

print("max error: ", max_error, "mean error: ", mean_error)
```