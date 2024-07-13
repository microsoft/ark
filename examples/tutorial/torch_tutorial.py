# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import torch
from torch.nn.functional import mse_loss
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


# Define a PyTorch model
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(256, 256, bias=False),  # Layer 0
            torch.nn.Linear(256, 256, bias=False),  # Layer 1
            torch.nn.Linear(256, 256, bias=False),  # Layer 2
            torch.nn.Linear(256, 256, bias=False),  # Layer 3
            torch.nn.Linear(256, 256, bias=False),  # Layer 4
            torch.nn.ReLU(),  # Activation function
        )

    def forward(self, x):
        return self.layers(x)


# Move the PyTorch model to GPU
pytorch_model = SimpleModel()
pytorch_model.to("cuda:0")


# Let's define the same model but, incorporate ARK
class ARKModel(SimpleModel):
    def __init__(self, simple_model):
        super().__init__()
        # We want to run Linear layers 0 and 1 on ARK
        weight_0 = Parameter.from_tensor(
            Tensor.from_torch(simple_model.layers[1].weight.to("cuda:0"))
        )
        weight_1 = Parameter.from_tensor(
            Tensor.from_torch(simple_model.layers[2].weight.to("cuda:0"))
        )
        ark_layers = [
            ARKLayer(ARKLinear, weight_0),
            ARKLayer(ARKLinear, weight_1),
        ]
        # Create an ARK component consisting of our consecutive ARK layers
        ark_component = ARKComponent(ark_layers)
        # Replace the first two linear layers with our ARK component
        new_layers = [
            simple_model.layers[0],
            ark_component,
            simple_model.layers[3],
            simple_model.layers[4],
            simple_model.layers[5],
        ]
        self.layers = torch.nn.Sequential(*new_layers)


# Instantiate the hybrid PyTorch/ARK model
ark_model = ARKModel(pytorch_model)

# Let's print the layers of our PyTorch model and the hybrid model
print("PyTorch model:\n", pytorch_model)
print("\nARK model:\n", ark_model)

# Move the hybrid to GPU
ark_model.to("cuda:0")

# Now lets run the model on some random input
input_tensor = torch.randn(128, 256).to("cuda:0").requires_grad_(True)

pytorch_output = pytorch_model(input_tensor)
ark_output = ark_model(input_tensor)

# Compare the results of both models
assert(torch.allclose(pytorch_output, ark_output, atol=1e-4, rtol=1e-2))

# Define an arbitrary target
target = torch.randn(128, 256).to("cuda:0")

# Compute losses
torch_loss, ark_loss = mse_loss(pytorch_output, target), mse_loss(ark_output, target)

# Perform a backward pass
torch_loss.backward()
ark_loss.backward()

print("PyTorch loss:", torch_loss.item())
print("ARK loss:", ark_loss.item())

