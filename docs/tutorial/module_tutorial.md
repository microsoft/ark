# Module Tutorial
This tutorial provides an overview of how to use the module and its features. The module is designed to mimic the touch API and provides a set of functionalities based on the module.

## Usage
To use the module, you need to create a class that inherits from the `ark.Module` class. You can then define the `forward` and `backward` functions of the class.

```python
class TestModelARK(ark.Module):
    def __init__(self):
        super(TestModelARK, self).__init__()
        self.weight_1 = ark.tensor(ark.Dims(d_model, d_ff), ark.TensorType.FP16)
        self.weight_2 = ark.tensor(ark.Dims(d_ff, d_model), ark.TensorType.FP16)

    def forward(self, inputs):
        middle_result = ark.matmul(inputs, self.weight_1, is_relu=True)
        middle_result1 = ark.matmul(middle_result, self.weight_2)
        output = ark.add(middle_result1, inputs)
        output_layernorm = ark.layernorm(output)
        return output_layernorm
```

Here, we can create this model and then run it.

```python
ark.init_model()

input_tensor = ark.tensor(
    ark.Dims(batch_size, seq_len, d_model), ark.TensorType.FP16
)
ark_model = TestModelARK()
output_tensor = ark_model(input_tensor)

# Test the mul method
ark.launch()
```

The initialization part of the model can be done using a state detector. Note that the parameters of this model in the state detector must have the same name as the parameters defined in the module. Then, we can use `load_state_dict` to import the parameters of this model.

```python
input_tensor_host = (
    (np.random.rand(batch_size, seq_len, d_model) - 0.5) * 0.1
).astype(np.float16)

ark.tensor_memcpy_host_to_device(input_tensor, input_tensor_host)

weight_1_host = ((np.random.rand(d_model, d_ff) - 0.5) * 0.1).astype(
    np.float16
)
weight_2_host = ((np.random.rand(d_ff, d_model) - 0.5) * 0.1).astype(
    np.float16
)
state_dict = {"weight_1": weight_1_host, "weight_2": weight_2_host}

ark_model.load_state_dict(state_dict)
```

If needed, we can save this state detector using `save`. We provide a set of modules for saving and importing this model's parameters using Python's `pickle` library.

```python
ark.save()
```

We can also convert this state detector into a PyTorch state detector. This way, we can directly import the parameters of this model into the corresponding PyTorch model.

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

