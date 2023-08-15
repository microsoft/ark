# A Quick Guide to Using ARK with Python

Welcome to this tutorial on using ARK in Python. We will walk you through a basic Python example to illustrate the process.

## Install & Configuration

Please refer to the [ARK Install Instructions](./install.md) to install ARK for Python. You may also want to check environment variables available to configure ARK. For more details about these variables, please refer to [Environment Variables](./env.md).

## Quick Start Tutorial

You can run a tutorial example at [tutorial.py](../examples/tutorial/quickstart_tutorial.py) to see how ARK works.

```bash
python examples/tutorial/quickstart_tutorial.py
```

Before diving in, let's import the required modules and initialize ARK runtime:

```python
import ark
import numpy as np

# Initialize the ARK runtime
runtime = ark.Runtime()

```
First, we need to create the operational graph for our DNN model. In this example, we define a simple model with two input tensors. The output tensor is the sum of these input tensors.

```python
M, N = 64, 64
# Create an input tensor
input_tensor = ark.tensor([M, N])
# Create another tensor
other_tensor = ark.tensor([M, N])

# Add the two tensors
output_tensor = ark.add(input_tensor, other_tensor)
```

Next, we need to launch the ARK runtime and initialize the input and output tensors. You can copy a numpy array into a tensor on GPU using `tensor.from_numpy(ndarray)`. By calling `runtime.launch()`, the ARK runtime will be launched. It will freeze the model and allocate GPU memory. Then it will schedule the model, generate and compile the CUDA kernel for the model. Therefore, it is necessary to call `runtime.launch()` before copying the tensor between the host and device. It is not allowed to modify the model after launching the ARK runtime.


```python
# Launch the ARK runtime
runtime.launch()

# Initialize the input and other tensor with random values
input_tensor_host = np.random.rand(M, N).astype(np.float32)
input_tensor.from_numpy(input_tensor_host)
other_tensor_host = np.random.rand(M, N).astype(np.float32)
other_tensor.from_numpy(other_tensor_host)
```

Next, you can run the ARK runtime using runtime.run(). This will launch the CUDA kernel and wait for the kernel to finish.

```python
# Run the ARK program
runtime.run()
```

Lastly, copy the output tensor back to the host and verify the result.

```python
# Copy the output tensor from device memory to host memory, if dst is 
# None, a new numpy array of the same shape as the src tensor will be returned
output_tensor_host = output_tensor.to_numpy()
# Check if the output tensor is equal to the sum of the input and other tensor
np.testing.assert_allclose(
    output_tensor_host, input_tensor_host + other_tensor_host
)
```

Congratulations! You have successfully learned how to run ARK. Happy coding!

For more tutorials, please refer to [tutorials](./tutorial/).
