# Module Tutorial
This tutorial provides an overview of how to use the module and its features. The module is similar to the pytorch module API and provides a set of functionalities such as model save and load.

## Usage
To use the module, you need to create a class that inherits from the `ark.Module` class. You can then define the `forward` and `backward` functions of the class. The parameters and submodules of the module is defined in the `__init__` function.

```python

# Define the parameters of the model
batch_size = 1
seq_len = 64
d_model = 512
d_ff = 2048

class SubModuleARK(ark.Module):
    def __init__(self):
        super(SubModuleARK, self).__init__()
        # Define the parameters of the submodule
        self.weight_2 = ark.Parameter(ark.tensor([d_ff, d_model], ark.FP16))

    def forward(self, inputs):
        # Perform the forward pass of the submodule
        middle_result1 = ark.matmul(inputs, self.weight_2)
        return middle_result1


class TestModelARK(ark.Module):
    def __init__(self):
        super(TestModelARK, self).__init__()
        # Define the parameters of the module
        self.weight_1 = ark.Parameter(ark.tensor([d_model, d_ff], ark.FP16))
        # Create a submodule of the module
        self.submodule = SubModuleARK()

    def forward(self, inputs):
        # Perform the forward pass of the model
        middle_result = ark.matmul(inputs, self.weight_1, is_relu=True)
        middle_result1 = self.submodule(middle_result)
        output = ark.add(middle_result1, inputs)
        output_layernorm = ark.layernorm(output)
        return output_layernorm
```

Here, we can create this model and then launch it.

```python
# Initialize the ARK runtime
runtime = ark.Runtime()
# Create an input tensor
input_tensor = ark.tensor([batch_size, seq_len, d_model], ark.FP16)

# Create an ARK module
ark_model = TestModelARK()

# Perform the forward pass
output_tensor = ark_model(input_tensor)

# Launch the ARK runtime
runtime.launch()
```

The initialization of the model can be done using a state_dict. Note that the parameters of this model in the state_dict must have the same name as the parameters defined in the module. Then, we can use `load_state_dict` to import the parameters of this model.

```python
# Initialize the input tensor
input_tensor_host = (
    (np.random.rand(batch_size, seq_len, d_model) - 0.5) * 0.1
).astype(np.float16)
input_tensor.from_numpy(input_tensor_host)
    
# Initialize the parameters of the ARK module using numpy state_dict
weight_1_host = ((np.random.rand(d_model, d_ff) - 0.5) * 0.1).astype(
    np.float16
)
weight_2_host = ((np.random.rand(d_ff, d_model) - 0.5) * 0.1).astype(
    np.float16
)
state_dict = {
    "weight_1": weight_1_host,
    "submodule.weight_2": weight_2_host,
}

# Load model parameters
ark_model.load_state_dict(state_dict)
```

If needed, we can save this state_dict using `save`. We provide a set of modules for saving and loading this model's parameters using Python's `pickle` library.

```python
ark.save(ark_model.state_dict(), "test_model.pt")
ark.load("test_model.pt")
```


Then we can run the model and get the output.

```python
# Run the ARK model
runtime.run()

# Copy the ARK module output tensor from device to host
output_tensor_host = output_tensor.to_numpy()
```

ARK's module is similar to PyTorch's module. Here we can use a similar pytorch module to compare their results.

```python
# Use pytorch to define the same model
class SubModulePytorch(nn.Module):
    def __init__(self):
        super(SubModulePytorch, self).__init__()
        self.weight_2 = nn.Parameter(torch.FloatTensor(d_ff, d_model))

    def forward(self, inputs):
        middle_result1 = torch.matmul(inputs, self.weight_2)
        return middle_result1


class TestModelPytorch(nn.Module):
    def __init__(self):
        super(TestModelPytorch, self).__init__()
        # Define the parameters of the module
        self.weight_1 = nn.Parameter(torch.FloatTensor(d_model, d_ff))
        # Create a submodule of the module
        self.submodule = SubModulePytorch()

    def forward(self, inputs):
        # Perform the forward pass of the model
        output = torch.matmul(inputs, self.weight_1)
        output = nn.ReLU()(output)
        output = self.submodule(output)
        output = nn.LayerNorm(d_model)(output + inputs)
        return output
```

We can then create this model and run it.

```python
 # For simplicity, we use float32 to compute the ground truth using pytorch
input_tensor_host_float32 = input_tensor_host.astype(np.float32)
torch_input = torch.from_numpy(input_tensor_host_float32)

torch_model = TestModelPytorch()
```

We can also convert ARK's state_dict into a PyTorch state_dict. This way, we can directly import the parameters of this model into the corresponding PyTorch model.

ARK state_dict's format is
```
{
    "weight_1": weight_1_numpy,
    "submodule.weight_2": weight_2_numpy,
}
```
`weight_1_numpy` and `weight_2_numpy` are `numpy.ndarray` type. PyTorch state_dict's format is
```
{
    "weight_1": weight_1_torch,
    "submodule.weight_2": weight_2_torch,
}
```
`weight_1_torch` and `weight_2_torch` are `torch.Tensor` type. We need to convert the `numpy.ndarray` type state_dict to `torch.Tensor` type state_dict using `ark.convert_state_dict`.


```python
    # Convert the numpy.ndarray type state_dict to torch.Tensor type state_dict using       
    #  ark.convert_state_dict
    torch_state_dict = ark.convert_state_dict(state_dict, "torch")
    # Load model parameters
    torch_model.load_state_dict(torch_state_dict)
```

Then we can run the model and compare the results.

```python
    # Run the pytorch model to compute the ground truth
    gt = torch_model(torch_input).detach().numpy().astype(np.float16)

    # Test if the result is correct
    max_error = np.max(np.abs(output_tensor_host - gt))
    avg_error = np.mean(np.abs(output_tensor_host - gt))

    # Use ark_model.state_dict() to get the state_dict of the ARK module
    # Note that the state_dict of the ARK module might be modified at the ARK kernel launch time
    ark_state_dict = ark_model.state_dict()

    # Test if the parameters are the same
    for k, v in state_dict.items():
        np.testing.assert_allclose(v, ark_state_dict[k])

    print("ARK module test")
    print(
        "batch_size:",
        batch_size,
        "seq_len:",
        seq_len,
        "d_model:",
        d_model,
        "d_ff:",
        d_ff,
    )
    print("max error: ", max_error, "avg error: ", avg_error)

```