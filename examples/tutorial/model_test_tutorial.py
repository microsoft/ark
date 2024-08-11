# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import torch
import torch.optim as optim


# Set random seed for reproducibility.
torch.manual_seed(42)

# Let's first define a linear layer using ARK.
class ARKLinear(ark.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, input):
        self.saved_input = input
        output = ark.matmul(input, self.weight, transpose_other=True)
        return output

    def backward(self, grad_output):
        grad_weight = ark.matmul(
            grad_output, self.saved_input, transpose_input=True
        )
        grad_input = ark.matmul(grad_output, self.weight, transpose_other=False)
        self.weight.update_gradient(grad_weight)
        return grad_input, grad_weight


# Let's use our previous module to define a double linear layer.
class MyARKModule(ark.Module):
    def __init__(self, weight0, weight1):
        super().__init__()
        self.linear1 = ARKLinear(weight0)
        self.linear2 = ARKLinear(weight1)

    def forward(self, x):
        x = self.linear1.forward(x)
        x = self.linear2.forward(x)
        return x

    def backward(self, grad_output):
        grad_x, grad_weight2 = self.linear2.backward(grad_output)
        grad_x, grad_weight1 = self.linear1.backward(grad_x)
        return grad_x, grad_weight1, grad_weight2


# Define a PyTorch model.
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(256, 256, bias=False),  # Layer 0
            torch.nn.Linear(256, 256, bias=False),  # Layer 1
            torch.nn.Linear(256, 256, bias=False),  # Layer 2
            torch.nn.Linear(256, 256, bias=False),  # Layer 3
            torch.nn.Linear(256, 256, bias=False),  # Layer 4
            torch.nn.ReLU(),  # Activation
        )

    def forward(self, x):
        return self.layers(x)


# Function to compare the gradients of two models of the same architecture and parameter order.
def compare_grad(ark_model, torch_model, atol=1e-4, rtol=1e-2):
    ark_params = list(ark_model.named_parameters())
    torch_params = list(torch_model.named_parameters())
    for (ark_name, ark_param), (torch_name, torch_param) in zip(
        ark_params, torch_params
    ):
        if (ark_param.grad is None) ^ (torch_param.grad is None):
            print("Exactly one of the gradients is None")
        else:
            grads_equal = torch.allclose(
                ark_param.grad, torch_param.grad, atol=atol, rtol=rtol
            )
            if not grads_equal:
                print(
                    f"Gradient for {ark_name} when compared to {torch_name} is different:"
                )
                print(f"ARK gradient: {ark_param.grad}")
                print(f"Torch gradient: {torch_param.grad}")


# For our ARK model we will replace the first two layers with ARK layers.
def replace_layers_with_ark(model):
    weight_0 = torch.nn.Parameter(
        model.layers[0].weight.to("cuda:0").requires_grad_(True)
    )
    weight_1 = torch.nn.Parameter(
        model.layers[1].weight.to("cuda:0").requires_grad_(True)
    )
    ark_module = ark.RuntimeModule(MyARKModule(weight_0, weight_1))
    model.layers[0] = ark_module
    del model.layers[1]

    # Since we replaced the PyTorch layer with an ARK layer, we need to register the PyTorch parameters
    # our ARK module utilizes with the original PyTorch model so ARK can leverage PyTorch's optimizers.
    model.register_parameter("weight_0", weight_0)
    model.register_parameter("weight_1", weight_1)

    return model


# Instantiate our models.
pytorch_model = SimpleModel()
ark_model = SimpleModel()


# Ensure both models have the same weights.
ark_model.load_state_dict(pytorch_model.state_dict())
ark_model = replace_layers_with_ark(ark_model)


# Move both models to GPU.
pytorch_model.to("cuda:0")
ark_model.to("cuda:0")

# Now let's run the models on some random input.
input_torch = torch.randn(128, 256).to("cuda:0").requires_grad_(True)
input_ark = input_torch.clone().detach().requires_grad_(True)


# Define an arbitrary target.
target = torch.randn(128, 256).to("cuda:0")

loss_fn = torch.nn.MSELoss()
optim_torch = optim.SGD(pytorch_model.parameters(), lr=0.01)
optim_ark = optim.SGD(ark_model.parameters(), lr=0.01)

num_iters = 5
for iter in range(num_iters):
    print(f"Iteration {iter+1}/{num_iters}")

    optim_torch.zero_grad()
    optim_ark.zero_grad()

    pytorch_output = pytorch_model(input_torch)
    ark_output = ark_model(input_ark)

    assert torch.allclose(pytorch_output, ark_output, atol=1e-4, rtol=1e-2)

    # Compute losses.
    torch_loss = loss_fn(pytorch_output, target)
    ark_loss = loss_fn(ark_output, target)

    # See how ARK's loss compares to PyTorch's loss.
    print(f"\nPyTorch loss: {torch_loss.item()}")
    print(f"\nARK loss: {ark_loss.item()}\n")
    assert torch.allclose(torch_loss, ark_loss, atol=1e-4, rtol=1e-2)

    # Perform a backward pass.
    torch_loss.backward()
    ark_loss.backward()

    optim_torch.step()
    optim_ark.step()

    # Ensure gradients of both models are updated accordingly.
    compare_grad(ark_model, pytorch_model)
