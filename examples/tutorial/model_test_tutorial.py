# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import torch
import torch.optim as optim

from ark.module import ARKFunction, ARKWrapper
from ark import Tensor, Parameter


# Define a custom ARK function for a linear layer
class ARKLinear(ARKFunction):
    def forward(self, ark_input, ark_weight):
        return ark.matmul(ark_input, ark_weight, transpose_other=True)

    def backward(self, ark_grad_output, saved_context):
        inp, weight = saved_context['input'], saved_context['weight']
        grad_input = grad_weight = None
        grad_input = ark.matmul(
                ark_grad_output, weight, transpose_other=False
            ) 
        grad_weight = ark.matmul(inp, ark_grad_output, transpose_input=True)
        return grad_input, grad_weight
    
    def save_context(self, ctx_dict, idx, ark_input, ark_weight):
        ctx_dict[idx] = {
            'input': ark_input,
            'weight': ark_weight
        }

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
    weight_0 = torch.nn.Parameter(model.layers[0].weight.clone().detach().to("cuda:0").requires_grad_(True))
    weight_1 = torch.nn.Parameter(model.layers[1].weight.clone().detach().to("cuda:0").requires_grad_(True))

    model.layers[0] = ARKWrapper([ARKLinear(), ARKLinear()], [[weight_0], [weight_1]])    
    del model.layers[1]
    model.register_parameter('weight_0', weight_0)
    model.register_parameter('weight_1', weight_1)
    return model




# Instantiate our models
pytorch_model = SimpleModel()
ark_model = SimpleModel()



            
print("TORCH MODEL PARAMS:")
for name, param in pytorch_model.named_parameters():
    if param.requires_grad:
        print(name, param, hex(param.data_ptr()), "REQUIRES GRAD")
    else:
        print(name, param, hex(param.data_ptr()), "NOGRAD")
print('\n')

# Ensure both models have the same weights
ark_model.load_state_dict(pytorch_model.state_dict())
ark_model = replace_layers_with_ark(ark_model)

print("ARK MODEL PARAMS:")
for name, param in ark_model.named_parameters():
    if param.requires_grad:
        print(name, param, hex(param.data_ptr()), "REQUIRES GRAD")
    else:
        print(name, param, hex(param.data_ptr()), "NO GRAD")
print('\n')

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
    #assert torch.allclose(pytorch_output, ark_output, atol=1e-4, rtol=1e-2)

    # Compute losses
    torch_loss = loss_fn(pytorch_output, target)
    ark_loss = loss_fn(ark_output, target)

    # Compare the results of both models
    print(f"PyTorch loss: {torch_loss.item()}")
    print(f"ARK loss: {ark_loss.item()}")

    # Perform a backward pass
    torch_loss.backward()
    ark_loss.backward()
    print("ARK MODEL GRADIENT CHECK:")
    for name, param in ark_model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                print(name, param.size(), hex(param.data_ptr()), "HAS STORED GRADIENTS")
            else:
                print(name, param.size(), hex(param.data_ptr()), "REQUIRES GRAD BUT NO GRADIENT STORED")
        else:
            print(name, param.size(), hex(param.data_ptr()), "NO GRAD")
    print('\n')
    print('\n')
    print("PYTORCH MODEL GRADIENT CHECK")
    for name, param in pytorch_model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                print(name, param.size(), hex(param.data_ptr()), "HAS STORED GRADIENTS")
            else:
                print(name, param.size(), hex(param.data_ptr()), "REQUIRES GRAD BUT NO GRADIENT STORED")
        else:
            print(name, param.size(), hex(param.data_ptr()), "NO GRAD")

    optim_torch.step()
    optim_ark.step()
