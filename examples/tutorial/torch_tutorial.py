# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import torch
from ark.module import ARKComponent, ARKFunction, ARKLayer
from ark import Tensor, Parameter



# Define a custom ARK function for a linear layer
class ARKLinear(ARKFunction):
    @staticmethod
    def build_backward(ctx, ark_grad_output, ark_input, ark_weight):
        grad_input = grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_input = ark.matmul(
                ark_grad_output, ark_weight, transpose_other=False
            )
        if ctx.needs_input_grad[1]:
            grad_weight = ark.matmul(
                ark_input, ark_grad_output, transpose_input=True
            )
        return grad_input, grad_weight


# Define a simple PyTorch model
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(512, 256, bias=False),  # Layer 1
            torch.nn.Linear(256, 256, bias=False),  # Layer 2
            torch.nn.Linear(256, 256, bias=False),  # Layer 3
            torch.nn.Linear(256, 256, bias=False),  # Layer 4
            torch.nn.Linear(256, 256, bias=False),  # Layer 5
            torch.nn.ReLU(),  # Activation function
        )

    def forward(self, x):
        return self.layers(x)


# Let's define the same model but, incorporate ARK
class ARKModel(SimpleModel):
    def __init__(self):
        super().__init__()
        # We want to run Linear layers 0 and 1 on ARK
        layers = list(self.layers.children())
        weight_0 = Parameter.from_tensor(
            Tensor.from_torch(layers[0].weight.to("cuda:0"))
        )
        weight_1 = Parameter.from_tensor(
            Tensor.from_torch(layers[1].weight.to("cuda:0"))
        )
        ark_layers = [
            ARKLayer(ARKLinear, weight_0),
            ARKLayer(ARKLinear, weight_1),
        ]
        # Create an ARK component consisting of our consecutive ARK layers
        ark_component = ARKComponent(ark_layers)
        # Replace the first two linear layers with our ARK component
        new_layers = [ark_component, layers[2], layers[3], layers[4], layers[5]]
        self.layers = torch.nn.Sequential(*new_layers)


# Instantiate the default PyTorch model
pytorch_model = SimpleModel()

# Instantiate the ARK model
ark_model = ARKModel()

# Let's print the layers of our PyTorch model and ARK model
print("PyTorch model:\n", pytorch_model)
print("\nARK model:\n", ark_model)

# Move models to GPU
pytorch_model.to("cuda:0")
ark_model.to("cuda:0")

# Now lets run the model on some random input
input_tensor = torch.randn(128, 512).to("cuda:0")
input_2 = input_tensor.clone()

pytorch_output = pytorch_model(input_tensor)
ark_output = ark_model(input_2)

# Print outputs of the models to verify
print("PyTorch Model Output:\n", pytorch_output)
print("ARK Model Output:\n", ark_output)
