# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import numpy as np
import torch.nn as nn
import ark

# Define the parameters of the model
batch_size = 1
seq_len = 64
d_model = 512
d_ff = 2048


def convert_state_dict(state_dict: dict, type="numpy"):
    """
    Convert the state_dict of a module to np.ndarray or torch.Tensor type
    """
    new_state_dict = {}
    for key in state_dict:
        if type == "torch":
            new_state_dict[key] = torch.from_numpy(state_dict[key])
        elif type == "numpy":
            new_state_dict[key] = state_dict[key].numpy()
    return new_state_dict


class SubModuleARK(ark.Module):
    def __init__(self):
        super(SubModuleARK, self).__init__()
        # Define the parameters of the submodule
        self.weight_2 = ark.parameter([d_ff, d_model], ark.FP16)

    def forward(self, inputs):
        # Perform the forward pass of the submodule
        middle_result1 = ark.matmul(inputs, self.weight_2)
        return middle_result1


class TestModelARK(ark.Module):
    def __init__(self):
        super(TestModelARK, self).__init__()
        # Define the parameters of the module
        self.weight_1 = ark.parameter([d_model, d_ff], ark.FP16)
        # Create a submodule of the module
        self.submodule = SubModuleARK()

    def forward(self, inputs):
        # Perform the forward pass of the model
        output = ark.matmul(inputs, self.weight_1)
        output = ark.relu(output)
        output = self.submodule(output)
        output = ark.add(output, inputs)
        output = ark.layernorm(output)
        return output


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


# An example of using the ARK module
def module_test():
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

    # Run the ARK model
    runtime.run()

    # Copy the ARK module output tensor from device to host
    output_tensor_host = output_tensor.to_numpy()

    # For simplicity, we use float32 to compute the ground truth using pytorch
    input_tensor_host_float32 = input_tensor_host.astype(np.float32)
    torch_input = torch.from_numpy(input_tensor_host_float32)

    torch_model = TestModelPytorch()

    # Convert the numpy.ndarray type state_dict to torch.Tensor type state_dict
    torch_state_dict = convert_state_dict(state_dict, "torch")
    # Load model parameters
    torch_model.load_state_dict(torch_state_dict)

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


if __name__ == "__main__":
    module_test()
