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
seq_len = 64


def test_rmsnorm():
    # Initialize the ARK runtime
    runtime = ark.Runtime()
    rmsnorm_pytorch = llama_pytorch.RMSNorm(dim)
    rmsnorm_ark = llama_ark.RMSNorm(dim)
    input_numpy = np.random.uniform(
        low=-1, high=1, size=(batch_size, seq_len, dim)
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

    ark_input = ark.tensor([batch_size, seq_len, dim], ark.FP32)
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


def test_attention():
    # Initialize the ARK runtime
    runtime = ark.Runtime()
    args = llama_ark.ModelArgs()
    attention_pytorch = llama_pytorch.Attention(args)
    attention_ark = llama_ark.Attention(args)
    dim = args.dim
    input_numpy = np.random.uniform(
        low=-1, high=1, size=(batch_size, seq_len, dim)
    ).astype(np.float32)
    torch_input = torch.from_numpy(input_numpy)

    head_dim = args.dim // args.n_heads
    n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
    state_dict = {
        "wq.weight": np.random.uniform(
            low=-1, high=1, size=(args.n_heads * head_dim, args.dim)
        ).astype(np.float32),
        "wk.weight": np.random.uniform(
            low=-1, high=1, size=(n_kv_heads * head_dim, args.dim)
        ).astype(np.float32),
        "wv.weight": np.random.uniform(
            low=-1, high=1, size=(n_kv_heads * head_dim, args.dim)
        ).astype(np.float32),
        "wo.weight": np.random.uniform(
            low=-1, high=1, size=(args.dim, args.n_heads * head_dim)
        ).astype(np.float32),
    }
    state_dict_torch = ark.convert_state_dict(state_dict, "torch")
    attention_pytorch.load_state_dict(state_dict_torch)
    xq_torch, xk_torch, xv_torch = attention_pytorch(torch_input, 0, None, None)

    ark_input = ark.tensor([batch_size, seq_len, dim], ark.FP32)
    xq, xk, xv = attention_ark(ark_input, 0, None, None)

    # Launch the ARK runtime
    runtime.launch()
    ark_input.from_numpy(input_numpy.astype(np.float32))
    attention_ark.load_state_dict(state_dict)

    # Run the ARK program
    runtime.run()
    xq_ark_host = xq.to_numpy()
    xk_ark_host = xk.to_numpy()
    xv_ark_host = xv.to_numpy()
    # test if the result is correct

    xq_gt = xq_torch.detach().numpy().astype(np.float32)
    xk_gt = xk_torch.detach().numpy().astype(np.float32)
    xv_gt = xv_torch.detach().numpy().astype(np.float32)

    xq_max_abs_error = np.max(np.abs(xq_ark_host - xq_gt))
    xq_mean_abs_error = np.mean(np.abs(xq_ark_host - xq_gt))
    print(xq_ark_host)
    print(xq_gt)
    print(
        "xq test",
        "max_abs_error:",
        "{:.5f}".format(xq_max_abs_error),
        "mean_abs_error:",
        "{:.5f}".format(xq_mean_abs_error),
    )

    xk_max_abs_error = np.max(np.abs(xk_ark_host - xk_gt))
    xk_mean_abs_error = np.mean(np.abs(xk_ark_host - xk_gt))
    print(
        "xk test",
        "max_abs_error:",
        "{:.5f}".format(xk_max_abs_error),
        "mean_abs_error:",
        "{:.5f}".format(xk_mean_abs_error),
    )

    xv_max_abs_error = np.max(np.abs(xv_ark_host - xv_gt))
    xv_mean_abs_error = np.mean(np.abs(xv_ark_host - xv_gt))
    print(
        "xv test",
        "max_abs_error:",
        "{:.5f}".format(xv_max_abs_error),
        "mean_abs_error:",
        "{:.5f}".format(xv_mean_abs_error),
    )


def test_feedforward():
    # Initialize the ARK runtime
    runtime = ark.Runtime()
    feedforward_pytorch = llama_pytorch.FeedForward(dim, 16384, 256, None)
    feedforward_ark = llama_ark.FeedForward(dim, 16384, 256, None)
    input_numpy = np.random.uniform(
        low=-1, high=1, size=(batch_size, seq_len, dim)
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
    feedforward_pytorch.load_state_dict(state_dict_torch)
    output_pytorch = feedforward_pytorch(torch_input)

    ark_input = ark.tensor([batch_size, seq_len, dim], ark.FP32)
    output_ark = feedforward_ark(ark_input)

    # Launch the ARK runtime
    runtime.launch()
    ark_input.from_numpy(input_numpy.astype(np.float32))
    feedforward_ark.load_state_dict(state_dict)

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
    test_attention()
    # test_feedforward()
