# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import llama_pytorch
import llama_ark
import ark
import numpy as np
import torch

if __name__ == "__main__":
    # Initialize the ARK runtime
    runtime = ark.Runtime()
    rmsnorm_pytorch = llama_pytorch.RMSNorm(4096)
    rmsnorm_ark = llama_ark.RMSNorm(4096)
    input_numpy = np.ones((1, 4096))
    torch_input = torch.from_numpy(input_numpy)
    output_pytorch = rmsnorm_pytorch(torch_input)

    ark_input = ark.tensor([1, 4096], ark.FP16)
    output_ark = rmsnorm_ark(ark_input)
    # Launch the ARK runtime
    runtime.launch()
    ark_input.from_numpy(input_numpy.astype(np.float16))
    # Run the ARK program
    runtime.run()
    output_ark_host = output_ark.to_numpy()
    print(output_ark_host)
    # Check if the output tensor is equal to the sum of the input and other tensor
    # test if the result is correct
    gt = output_pytorch.detach().numpy()
    max_abs_error = np.max(np.abs(output_ark_host - gt))
    mean_abs_error = np.mean(np.abs(output_ark_host - gt))
    print(
        "rmsnorm test",
        "max_abs_error:",
        "{:.5f}".format(max_abs_error),
        "mean_abs_error:",
        "{:.5f}".format(mean_abs_error),
    )
