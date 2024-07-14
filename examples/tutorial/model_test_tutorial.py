# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import torch
import torch.optim as optim

from ark.module import ARKComponent, ARKFunction, ARKLayer
from ark import Tensor, Parameter


# Define a custom ARK function for a linear layer
class ARKLinear(ARKFunction):
    @staticmethod
    def forward(ark_input, ark_weight):
        return ark.matmul(ark_input, ark_weight, transpose_other=True)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, weight = inputs
        ctx.save_for_backward(input, weight)

    @staticmethod
    def backward(ctx, ark_grad_output):
        inp, weight = ctx.saved_tensors
        grad_input = grad_weight = None
        if ctx.input_requires_grad[0]:
            grad_input = ark.matmul(
                ark_grad_output, weight, transpose_other=False
            )
        if ctx.input_requires_grad[1]:
            grad_weight = ark.matmul(inp, ark_grad_output, transpose_input=True)
        return grad_input, grad_weight


ark_linear_fn = ARKLinear


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
            torch.nn.ReLU(),    #Activation
        )

    def forward(self, x):
        return self.layers(x)


# For our ARK model we will replace the first two layers with ARK layers
def replace_layers_with_ark(model):
    weight_0 = model.layers[0].weight.clone().detach().to("cuda:0")
    ark_weight_0 = Parameter.from_tensor(
        Tensor.from_torch(weight_0.requires_grad_(True))
    )

    weight_1 = model.layers[1].weight.clone().detach().to("cuda:0")
    ark_weight_1 = Parameter.from_tensor(
        Tensor.from_torch(weight_1.requires_grad_(True))
    )

    model.layers[0] = ARKComponent(
        [
            ARKLayer(ark_linear_fn, ark_weight_0),
            ARKLayer(ark_linear_fn, ark_weight_1),
        ]
    )
    del model.layers[1]
    return model


# Instantiate our models
pytorch_model = SimpleModel()
ark_model = SimpleModel()

# Ensure both models have the same weights
ark_model.load_state_dict(pytorch_model.state_dict())
ark_model = replace_layers_with_ark(ark_model)

print("PyTorch model:\n", pytorch_model)
print("\nARK model:\n", ark_model)


# Move both models to GPU
pytorch_model.to("cuda:0")
ark_model.to("cuda:0")

# Now let's run the models on some random input
input_torch = torch.randn(128, 256).to("cuda:0").requires_grad_(True)
input_ark = input_torch.clone().detach().requires_grad_(True)


# Define an arbitrary target
target = torch.randn(128, 256).to("cuda:0")

loss_fn = torch.nn.MSELoss()
optim_torch = optim.SGD(pytorch_model.parameters(), lr=0.01)
optim_ark = optim.SGD(ark_model.parameters(), lr=0.01)
num_iters = 3
for iter in range(num_iters):
    print(f"Iteration {iter+1}/{num_iters}")

    optim_torch.zero_grad()
    optim_ark.zero_grad()

    pytorch_output = pytorch_model(input_torch)
    ark_output = ark_model(input_ark)

    # The outputs for both models should be roughly the same
    assert torch.allclose(pytorch_output, ark_output, atol=1e-4, rtol=1e-2)

    # Compute losses
    torch_loss = loss_fn(pytorch_output, target)
    ark_loss = loss_fn(ark_output, target)

    # Compare the results of both models
    print(f"PyTorch loss: {torch_loss.item()}")
    print(f"ARK loss: {ark_loss.item()}")

    # Perform a backward pass
    torch_loss.backward()
    ark_loss.backward()

    optim_torch.step()
    optim_ark.step()
