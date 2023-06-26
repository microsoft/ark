# Quick Start for Using ARK with Python

In this tutorial, we will demonstrate how to use ARK to run a simple DNN appication in Python. We will be using a basic Python example to illustrate the process.

After we finish the [installation](./install.md) and set ARK_ROOT, we can build the example and run it.

ARK is a novel GPU-driven code execution system. It consists of several parts as shown in [GPU-driven System Architecture](./imgs/GPU-driven_System_Architecture.png). 

There is some environment variables that can be used to configure ARK. In this tutorial, please set ARK_LOG_LEVEL to DEBUG to see more details of the execution process.

```bash
export ARK_LOG_LEVEL=DEBUG
```

Please refer to [Environment Variables](./env.md) for more details about the environment variables.

Before we start, we need to import required modules and initialize ARK.

```python
import ark
import numpy as np
# clean up the shared memory directory. 
ark.init()
```

First , we need to create the operational graph of our DNN models. 
Here we define a simple model with two input tensors, the output tensor is the sum of the two input tensors.

```python
# Create a Model instance
model = ark.Model()

# Create two tensors
input = model.tensor(ark.Dims(1, 1, 1, 32), ark.TensorType.FP16)
other = model.tensor(ark.Dims(1, 1, 1, 32), ark.TensorType.FP16)

# Add input and other to get output tensor
output = model.add(input, other)
```
