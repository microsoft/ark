# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import os
sys.path.append("./llama/llama")
import model as llama_pytorch
import llama_ark
import ark
import numpy as np
import torch
import torch.nn as nn

from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

dim = 4096
seq_len = 64


def convert_state_dict(state_dict: dict, type="numpy"):
    """
    Convert the state_dict of a module to np.ndarray or torch.Tensor type
    """
    new_state_dict = {}
    for key in state_dict:
        if type == "torch":
            new_state_dict[key] = torch.from_numpy(state_dict[key])
        elif type == "numpy":
            new_state_dict[key] = state_dict[key].numpy()
    return new_state_dict


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
    state_dict_torch = convert_state_dict(state_dict, "torch")
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
    state_dict_torch = convert_state_dict(state_dict, "torch")
    attention_pytorch.load_state_dict(state_dict_torch)
    freqs_cis_torch = llama_pytorch.precompute_freqs_cis(
        args.dim // args.n_heads, args.max_seq_len * 2
    )
    freqs_cis_torch = freqs_cis_torch[0:seq_len]
    output_torch = attention_pytorch(torch_input, 0, freqs_cis_torch, None)

    ark_input = ark.tensor([batch_size, seq_len, dim], ark.FP32)
    freqs_cis_ark = ark.tensor([1, seq_len, 1, head_dim], ark.FP32)
    output = attention_ark(ark_input, 0, freqs_cis_ark, None)

    # Launch the ARK runtime
    runtime.launch()
    ark_input.from_numpy(input_numpy.astype(np.float32))
    attention_ark.load_state_dict(state_dict)
    freqs_cis_complex = freqs_cis_torch.numpy().astype(np.complex64)
    # stack real and imag parts
    freqs_cis_stack = np.stack(
        [freqs_cis_complex.real, freqs_cis_complex.imag], axis=-1
    ).astype(np.float32)

    freqs_cis_ark.from_numpy(freqs_cis_stack)

    # Run the ARK program
    runtime.run()
    output_ark_host = output.to_numpy()

    # test if the result is correct
    output_gt = output_torch.detach().numpy().astype(np.float32)
    max_abs_error = np.max(np.abs(output_ark_host - output_gt))
    mean_abs_error = np.mean(np.abs(output_ark_host - output_gt))
    print(
        "attention test",
        "max_abs_error:",
        "{:.5f}".format(max_abs_error),
        "mean_abs_error:",
        "{:.5f}".format(mean_abs_error),
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
    state_dict_torch = convert_state_dict(state_dict, "torch")
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
        "feedforward test",
        "max_abs_error:",
        "{:.5f}".format(max_abs_error),
        "mean_abs_error:",
        "{:.5f}".format(mean_abs_error),
    )


def test_transformerblock():
    # Initialize the ARK runtime
    runtime = ark.Runtime()
    args = llama_ark.ModelArgs()
    transformer_block_pytorch = llama_pytorch.TransformerBlock(0, args)
    transformer_block_ark = llama_ark.TransformerBlock(0, args)
    dim = args.dim
    input_numpy = np.random.uniform(
        low=-1, high=1, size=(batch_size, seq_len, dim)
    ).astype(np.float32)
    torch_input = torch.from_numpy(input_numpy)

    # random init the torch model
    for param in transformer_block_pytorch.parameters():
        nn.init.uniform_(param, a=-0.1, b=0.1)
    state_dict = transformer_block_pytorch.state_dict()

    freqs_cis_torch = llama_pytorch.precompute_freqs_cis(
        args.dim // args.n_heads, args.max_seq_len * 2
    )
    freqs_cis_torch = freqs_cis_torch[0:seq_len]
    output_torch = transformer_block_pytorch(
        torch_input, 0, freqs_cis_torch, None
    )

    ark_input = ark.tensor([batch_size, seq_len, dim], ark.FP32)
    freqs_cis_ark = ark.tensor(
        [1, seq_len, 1, args.dim // args.n_heads], ark.FP32
    )
    output = transformer_block_ark(ark_input, 0, freqs_cis_ark, None)

    # Launch the ARK runtime
    runtime.launch()
    ark_input.from_numpy(input_numpy.astype(np.float32))

    ark_state_dict = convert_state_dict(state_dict, "numpy")
    transformer_block_ark.load_state_dict(ark_state_dict)
    freqs_cis_complex = freqs_cis_torch.numpy().astype(np.complex64)
    # stack real and imag parts
    freqs_cis_stack = np.stack(
        [freqs_cis_complex.real, freqs_cis_complex.imag], axis=-1
    ).astype(np.float32)

    freqs_cis_ark.from_numpy(freqs_cis_stack)
    # Run the ARK program
    runtime.run()
    output_ark_host = output.to_numpy()

    # test if the result is correct
    output_gt = output_torch.detach().numpy().astype(np.float32)
    max_abs_error = np.max(np.abs(output_ark_host - output_gt))
    mean_abs_error = np.mean(np.abs(output_ark_host - output_gt))
    print(
        "transformer_block test",
        "max_abs_error:",
        "{:.5f}".format(max_abs_error),
        "mean_abs_error:",
        "{:.5f}".format(mean_abs_error),
    )


def test_transformer():
    # Initialize the ARK runtime
    runtime = ark.Runtime()
    args = llama_ark.ModelArgs()
    args.vocab_size = 1024
    transformer_pytorch = llama_pytorch.Transformer(args)
    transformer_ark = llama_ark.Transformer(args)
    dim = args.dim
    input_numpy = np.random.uniform(
        low=-1, high=1, size=(batch_size, seq_len, dim)
    ).astype(np.float32)
    torch_input = torch.from_numpy(input_numpy)

    # random init the torch model
    for param in transformer_pytorch.parameters():
        nn.init.uniform_(param, a=-0.1, b=0.1)
    state_dict = transformer_pytorch.state_dict()

    # transformer_pytorch.load_state_dict(state_dict_torch)
    output_torch = transformer_pytorch(torch_input, 0, None, None)

    ark_input = ark.tensor([batch_size, seq_len, dim], ark.FP32)
    output = transformer_ark(ark_input, 0)

    # Launch the ARK runtime
    runtime.launch()
    ark_input.from_numpy(input_numpy.astype(np.float32))

    ark_state_dict = convert_state_dict(state_dict, "numpy")
    transformer_ark.load_state_dict(ark_state_dict)
    # Run the ARK program
    runtime.run()
    output_ark_host = output.to_numpy()

    # test if the result is correct
    output_gt = output_torch.detach().numpy().astype(np.float32)
    max_abs_error = np.max(np.abs(output_ark_host - output_gt))
    mean_abs_error = np.mean(np.abs(output_ark_host - output_gt))
    print(
        "transformer_block test",
        "max_abs_error:",
        "{:.5f}".format(max_abs_error),
        "mean_abs_error:",
        "{:.5f}".format(mean_abs_error),
    )


def test_rotary_embedding():
    # Initialize the ARK runtimes
    args = llama_pytorch.ModelArgs()

    freqs_cis = llama_pytorch.precompute_freqs_cis(
        args.dim // args.n_heads, args.max_seq_len * 2
    )
    batch_size = 1
    start_pos = 0
    seqlen = 64
    freqs_cis_torch = freqs_cis[start_pos : start_pos + seqlen]
    head_dim = args.dim // args.n_heads
    xq_torch = torch.randn(
        [batch_size, seq_len, args.n_heads, head_dim],
        dtype=torch.float32,
    )

    xk_torch = torch.randn(
        [batch_size, seq_len, args.n_heads, head_dim],
        dtype=torch.float32,
    )

    xq_out_torch, xk_out_torch = llama_pytorch.apply_rotary_emb(
        xq_torch, xk_torch, freqs_cis_torch
    )

    runtime = ark.Runtime()
    xq_ark = ark.tensor([batch_size, seq_len, args.n_heads, head_dim], ark.FP32)
    xk_ark = ark.tensor([batch_size, seq_len, args.n_heads, head_dim], ark.FP32)

    freqs_cis_ark = ark.tensor([1, seqlen, 1, head_dim], ark.FP32)

    xq_out_ark, xk_out_ark = llama_ark.apply_rotary_emb(
        xq_ark, xk_ark, freqs_cis_ark
    )

    runtime.launch()
    xq_ark.from_numpy(xq_torch.numpy().astype(np.float32))
    xk_ark.from_numpy(xk_torch.numpy().astype(np.float32))
    freqs_cis_complex = freqs_cis_torch.numpy().astype(np.complex64)

    # stack real and imag parts
    freqs_cis_stack = np.stack(
        [freqs_cis_complex.real, freqs_cis_complex.imag], axis=-1
    ).astype(np.float32)

    freqs_cis_ark.from_numpy(freqs_cis_stack)
    runtime.run()

    xq_out_ark_host = xq_out_ark.to_numpy()

    max_abs_error = np.max(
        np.abs(xq_out_ark_host - xq_out_torch.detach().numpy())
    )
    mean_abs_error = np.mean(
        np.abs(xq_out_ark_host - xq_out_torch.detach().numpy())
    )
    print(
        "rotary_embedding test",
        "max_abs_error:",
        "{:.5f}".format(max_abs_error),
        "mean_abs_error:",
        "{:.5f}".format(mean_abs_error),
    )

    xk_ark_host = xk_out_ark.to_numpy()

    max_abs_error = np.max(np.abs(xk_ark_host - xk_out_torch.detach().numpy()))
    mean_abs_error = np.mean(
        np.abs(xk_ark_host - xk_out_torch.detach().numpy())
    )
    print(
        "rotary_embedding test",
        "max_abs_error:",
        "{:.5f}".format(max_abs_error),
        "mean_abs_error:",
        "{:.5f}".format(mean_abs_error),
    )


if __name__ == "__main__":
    # Set up the environment variables for nccl
    # export RANK=0
    # export WORLD_SIZE=1
    # export MASTER_ADDR=localhost
    # export MASTER_PORT=29500
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(1)
    test_rmsnorm()
    test_attention()
    test_feedforward()
    test_transformerblock()
    # test_transformer()
    test_rotary_embedding()
