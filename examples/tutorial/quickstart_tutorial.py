# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import ark


def quickstart_tutorial():
    # Initialize the ARK environments
    ark.init()

    M, N = 64, 64
    # Create input tensors on GPU
    input_tensor = torch.randn(M, N, dtype=torch.float16, device="cuda:0")
    other_tensor = torch.randn(M, N, dtype=torch.float16, device="cuda:0")

    # Add the two tensors using ARK and evaluate
    output = ark.add(input_tensor, other_tensor).eval()

    # Check if the output tensor is equal to the sum of the input and other tensor
    torch.testing.assert_close(
        output, input_tensor + other_tensor, atol=0, rtol=0
    )

    print("Quickstart tutorial is successful!")


if __name__ == "__main__":
    quickstart_tutorial()
