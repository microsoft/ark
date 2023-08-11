# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import llama_pytorch
import llama_ark
import ark
import numpy as np
import torch

from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

dim = 4096


def test_rmsnorm():
    # Initialize the ARK runtime
    runtime = ark.Runtime()
    rmsnorm_pytorch = llama_pytorch.RMSNorm(dim)
    rmsnorm_ark = llama_ark.RMSNorm(dim)
    input_numpy = np.random.uniform(
        low=-1, high=1, size=(batch_size, 1, dim)
    ).astype(np.float32)
    torch_input = torch.from_numpy(input_numpy)
    state_dict = {
        "weight": np.random.uniform(low=-1, high=1, size=(dim,)).astype(
            np.float32
        ),
    }
    state_dict_torch = ark.convert_state_dict(state_dict, "torch")
    rmsnorm_pytorch.load_state_dict(state_dict_torch)
    output_pytorch = rmsnorm_pytorch(torch_input)

    ark_input = ark.tensor([batch_size, 1, dim], ark.FP32)
    output_ark = rmsnorm_ark(ark_input)

    # Launch the ARK runtime
    runtime.launch()
    ark_input.from_numpy(input_numpy.astype(np.float32))
    rmsnorm_ark.load_state_dict(state_dict)

    # Run the ARK program
    runtime.run()
    output_ark_host = output_ark.to_numpy()

    # test if the result is correct

    gt = output_pytorch.detach().numpy().astype(np.float32)
    max_abs_error = np.max(np.abs(output_ark_host - gt))
    mean_abs_error = np.mean(np.abs(output_ark_host - gt))
    print(
        "rmsnorm test",
        "max_abs_error:",
        "{:.5f}".format(max_abs_error),
        "mean_abs_error:",
        "{:.5f}".format(mean_abs_error),
    )


batch_size = 1


def test_feedforward():
    # Initialize the ARK runtime
    runtime = ark.Runtime()
    rmsnorm_pytorch = llama_pytorch.FeedForward(dim, 16384, 256, None)
    rmsnorm_ark = llama_ark.FeedForward(dim, 16384, 256, None)
    input_numpy = np.random.uniform(
        low=-1, high=1, size=(batch_size, 1, dim)
    ).astype(np.float32)
    torch_input = torch.from_numpy(input_numpy)
    state_dict = {
        "w1.weight": np.random.uniform(
            low=-1, high=1, size=(11008, dim)
        ).astype(np.float32),
        "w2.weight": np.random.uniform(
            low=-1, high=1, size=(dim, 11008)
        ).astype(np.float32),
        "w3.weight": np.random.uniform(
            low=-1, high=1, size=(11008, dim)
        ).astype(np.float32),
    }
    state_dict_torch = ark.convert_state_dict(state_dict, "torch")
    rmsnorm_pytorch.load_state_dict(state_dict_torch)
    output_pytorch = rmsnorm_pytorch(torch_input)

    ark_input = ark.tensor([batch_size, 1, dim], ark.FP32)
    output_ark = rmsnorm_ark(ark_input)

    # Launch the ARK runtime
    runtime.launch()
    ark_input.from_numpy(input_numpy.astype(np.float32))
    rmsnorm_ark.load_state_dict(state_dict)

    # Run the ARK program
    runtime.run()
    output_ark_host = output_ark.to_numpy()

    # test if the result is correct

    gt = output_pytorch.detach().numpy().astype(np.float32)
    print("output_ark_host:", output_ark_host)
    print("gt", gt)

    max_abs_error = np.max(np.abs(output_ark_host - gt))
    mean_abs_error = np.mean(np.abs(output_ark_host - gt))
    print(
        "rmsnorm test",
        "max_abs_error:",
        "{:.5f}".format(max_abs_error),
        "mean_abs_error:",
        "{:.5f}".format(mean_abs_error),
    )


if __name__ == "__main__":
    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(1)
    # test_rmsnorm()
    test_feedforward()
