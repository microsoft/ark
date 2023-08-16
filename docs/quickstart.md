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
First, we need to create the operational graph for our model. In this example, we define a simple model with two input tensors. The output tensor is the sum of these input tensors.

```python
M, N = 64, 64
# Create an input tensor
input_tensor = ark.tensor([M, N])
# Create another tensor
other_tensor = ark.tensor([M, N])

# Add the two tensors
output_tensor = ark.add(input_tensor, other_tensor)
```

Next, we need to launch the ARK runtime by calling `runtime.launch()`. This call will freeze the model, schedule GPU tasks, and allocate GPU memory. Then it will generate and compile the GPU kernel for the model. Finally, it will launch the GPU kernel that will be waiting for a `runtime.run()` call. Modifying the model after launching the runtime will take no effect.

> **NOTE:** Note the difference from other GPU frameworks such as PyTorch. In PyTorch, each GPU kernel represents a single GPU task and a kernel launch will immediately start computation. In ARK, the GPU kernel represents the entire GPU tasks needed to run the model, throughout the entire lifetime of the model. Therefore, ARK launches only a single kernel and the kernel will be always running until the runtime stops. Instead of immediately starting computation after launch, the ARK kernel will run computation upon a `runtime.run()` call to ensure that the host side is ready to provide input data & read results. This design allows ARK to achieve better performance by removing the overhead from the host side.

Next, we need to initialize the input and output tensors. You can copy a numpy array into a tensor on GPU using `tensor.from_numpy(ndarray)`. Since `runtime.launch()` allocates GPU memory, it is necessary to call `runtime.launch()` before copying the tensor between the host and device.

```python
# Launch the ARK runtime
runtime.launch()

# Initialize the input and other tensor with random values
input_tensor_host = np.random.rand(M, N).astype(np.float32)
input_tensor.from_numpy(input_tensor_host)
other_tensor_host = np.random.rand(M, N).astype(np.float32)
other_tensor.from_numpy(other_tensor_host)
```

Next, you can run the ARK runtime using `runtime.run()`. If you want to run multiple iterations, you can provide the number as an argument like `runtime.run(100)`.

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
