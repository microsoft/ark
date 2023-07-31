# A Quick Guide to Using ARK with Python for DNN Applications

Welcome to this tutorial on using ARK to run a simple deep neural network (DNN) application in Python. We will walk you through a basic Python example to illustrate the process.

After completing the [installation](./install.md), you can run the tutorial example at [tutorial.py](../examples/tutorial/quickstart_tutorial.py) to see how ARK works.

```bash
python examples/tutorial/quickstart_tutorial.py
```

There are environment variables available to configure ARK. For more details about these variables, please refer to [Environment Variables](./env.md).

Before diving in, let's import the required modules and initialize ARK:

```python
import ark
import numpy as np

# Initialize the ARK model
ark.init_model()

```
First, we need to create the operational graph for our DNN model. In this example, we define a simple model with two input tensors. The output tensor is the sum of these input tensors.

```python
M, N = 64, 64
# Create an input tensor
input_tensor = ark.tensor([M, N])
# Create another tensor
other_tensor = ark.tensor([M, N])

# Add the two tensors
output_tensor = ark.add(input_tensor, input_tensor)
```

Next, we need to launch the ARK runtime and initialize the input and output tensors. You can copy a numpy array into a tensor on GPU using `ark.tensor_memcpy_host_to_device()`. Therefore, you must call ark.launch() before copying the tensor between host and device. Changing the model after launching the ARK runtime is not allowed.


```python
# Launch the ARK runtime
ark.launch()

# Initialize the input and other tensor with random values
input_tensor_host = np.random.rand(M, N).astype(np.float32)
ark.tensor_memcpy_host_to_device(input_tensor, input_tensor_host)
other_tensor_host = np.random.rand(M, N).astype(np.float32)
ark.tensor_memcpy_host_to_device(other_tensor, other_tensor_host)
```

Next, you can run the ARK runtime using ark.run(). This will launch the CUDA kernel and wait for the kernel to finish.

```python
# Run the ARK program
ark.run()
```

Lastly, copy the output tensor back to the host and verify the result.

```python
# Copy the output tensor from device memory to host memory, if dst is 
# None, a new numpy array of the same shape as the src tensor will be returned
output_tensor_host = ark.tensor_memcpy_device_to_host(
    None, output_tensor
)
# Check if the output tensor is equal to the sum of the input and other tensor
np.testing.assert_allclose(
    output_tensor_host, input_tensor_host + other_tensor_host
)
```

Congratulations! You have successfully learned how to run a DNN model over ARK. Happy coding!

For more tutorials, please refer to [Tutorials](./tutorial/).
