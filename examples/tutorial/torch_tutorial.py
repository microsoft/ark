# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tutorial: Using ARK with PyTorch tensors.

Shows how to:
1. Create ARK placeholder tensors backed by PyTorch memory
2. Run ARK computation on torch-owned GPU memory
3. Read results back as PyTorch tensors
"""

import ark
import torch

ark.init()

# Create torch tensors on GPU
x = torch.ones(64, dtype=torch.float32, device="cuda:0") * 2
y = torch.ones(64, dtype=torch.float32, device="cuda:0") * 3

# Create ARK placeholders backed by torch memory
a = ark.placeholder([64], ark.fp32, data=x)
b = ark.placeholder([64], ark.fp32, data=y)

# Define ARK computation
z = ark.add(a, b)

# Launch and run
with ark.Runtime() as rt:
    rt.launch()
    rt.run()

    # Read result back as a torch tensor (zero-copy via DLPack)
    result = z.to_torch()
    print(f"x + y = {result}")  # tensor([5., 5., ...])

    # Modify torch inputs and re-run
    x.fill_(10)
    y.fill_(20)
    rt.run()

    result = z.to_torch()
    print(f"10 + 20 = {result}")  # tensor([30., 30., ...])
