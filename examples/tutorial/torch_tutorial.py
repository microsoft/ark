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
        torch_weight_1 = simple_model.layers[1].weight.clone().detach().to("cuda:0")
        print("MODEL WEIGHT (PYTORCH):", simple_model.layers[1].weight)
        print("MODEL WEIGHT (ARK):", torch_weight_1)
        weight_0 = Parameter.from_tensor(
            Tensor.from_torch(torch_weight_1.requires_grad_(True))
        )
        torch_weight_2 = simple_model.layers[2].weight.clone().detach().to("cuda:0")
        print("MODEL WEIGHT (PYTORCH):", simple_model.layers[2].weight)
        print("MODEL WEIGHT (ARK):", torch_weight_2)
        weight_1 = Parameter.from_tensor(
            Tensor.from_torch(torch_weight_2.requires_grad_(True))
        )
        ark_funcs = [
            ARKLinear,
            ARKLinear
        ]
        args_list = [
            [weight_0],
            [weight_1]
        ]
        kwargs_list = [
            {},
            {}
        ]
        ark_layers = [ARKLayer(ark_func, *args, **kwargs) for ark_func, args, kwargs in zip(ark_funcs, args_list, kwargs_list)]
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
input2 = input_tensor.clone().detach().requires_grad_(True)

# Compare the results of both models

# Define an arbitrary target
target = torch.randn(128, 256).to("cuda:0")

loss_fn = torch.nn.MSELoss()
optim_torch = optim.SGD(pytorch_model.parameters(), lr=0.01)
optim_ark = optim.SGD(ark_model.parameters(), lr=0.01)
num_iters = 2
for iter in range(num_iters):
    print(f"Iteration {iter+1}/{num_iters}")

    optim_torch.zero_grad()
    optim_ark.zero_grad()

    pytorch_output = pytorch_model(input_tensor)
    ark_output = ark_model(input2)

    print("PyTorch output:", pytorch_output)
    print("ARK output:", ark_output)

    #Compute loss
    torch_loss = loss_fn(pytorch_output, target)
    ark_loss = loss_fn(ark_output, target)

    print(f"PyTorch loss: {torch_loss.item()}")
    print(f"ARK loss: {ark_loss.item()}")

    # Perform a backward pass
    torch_loss.backward()
    ark_loss.backward()

    optim_torch.step()
    optim_ark.step()

