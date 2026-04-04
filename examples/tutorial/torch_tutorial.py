# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tutorial: Using ARK with PyTorch tensors.

Shows how to use eval() to run ARK computation on torch tensors
and get torch tensor results directly.
"""

import ark
import torch

ark.init()

# Create torch tensors on GPU
x = torch.ones(64, dtype=torch.float32, device="cuda:0") * 2
y = torch.ones(64, dtype=torch.float32, device="cuda:0") * 3

# Run ARK computation and get result as a torch tensor
result = ark.add(x, y).eval()
print(f"x + y = {result}")  # tensor([5., 5., ...])

# Run again with different values
x = torch.ones(64, dtype=torch.float32, device="cuda:0") * 10
y = torch.ones(64, dtype=torch.float32, device="cuda:0") * 20
result = ark.add(x, y).eval()
print(f"10 + 20 = {result}")  # tensor([30., 30., ...])
