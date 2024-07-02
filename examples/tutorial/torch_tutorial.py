# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import torch


class ArkAddModule(ark.RuntimeModule):
    def build_forward(self, x: ark.Tensor, y: ark.Tensor) -> ark.Tensor:
        return ark.add(x, y)


# ARK module for addition
module = ArkAddModule()

# Define two torch arrays
x = torch.ones(64) * 2
y = torch.ones(64) * 3

# Run the ARK module
z = module(x, y)

w = module(x, z)

# Print the result
print(z)  # 5
print(w)  # 7
