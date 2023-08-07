# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import llama_pytorch
import llama_ark
import ark

if __name__ == "__main__":
    # Initialize the ARK runtime
    runtime = ark.Runtime()
    rmsnorm_pytorch = llama_pytorch.RMSNorm(4096)
    rmsnorm_ark = llama_ark.RMSNorm(4096)
    output_pytorch = rmsnorm_pytorch(torch.ones(1, 4096))
    output_ark = rmsnorm_ark(ark.tensor(1, 4096))
    # Launch the ARK runtime
    runtime.launch()
