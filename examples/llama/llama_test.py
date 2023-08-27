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
import time
import fairscale

batch_size = 1
seq_len = 64
dim = llama_ark.ModelArgs().dim
start_pos = 0

ark_type = llama_ark.ark_type

# np_type = np.float16 if ark_type == ark.FP16 else np.float32
np_type = np.float32

performance_analysis = False
torch_device = None

total_execution_time = 1
warmup_iter = 50

local_rank = 0
world_size = 1


def performance_ark(runtime, iter=None):
    # Restart the ARK runtime
    runtime.launch()
    # Rough measure the execution time
    runtime.run(iter=warmup_iter)
    warmup_sec_per_iter = 1.0 * runtime.stop() / (warmup_iter * 1000)

    if iter is None:
        iter = max(int(total_execution_time / warmup_sec_per_iter), warmup_iter)
    runtime.launch()
    runtime.run(iter=iter, non_blocking=True)
    elapsed = runtime.stop()
    return 1.0 * elapsed / (1000 * iter), iter


def performance_torch(torch_func, iter=None):
    torch.cuda.synchronize()
    start_torch = time.time()
    for i in range(warmup_iter):
        torch_func()
    torch.cuda.synchronize()
    end_torch = time.time()
    elapsed_per_iter = 1.0 * (end_torch - start_torch) / warmup_iter

    # Rough measure the execution time
    if iter is None:
        iter = max(int(total_execution_time / elapsed_per_iter), warmup_iter)
    torch.cuda.synchronize()
    start_torch = time.time()
    for i in range(iter):
        torch_func()
    torch.cuda.synchronize()
    end_torch = time.time()
    elapsed = end_torch - start_torch
    return 1.0 * elapsed / iter, iter


def performance_comparison(runtime, torch_func):
    # The performance comparison is only enabled when performance_analysis is True
    if not performance_analysis:
        return
    ark_elapsed_per_iter, iter_ark = performance_ark(runtime)
    torch_elapsed_per_iter, iter_torch = performance_torch(torch_func)
    print(
        "performance comparison",
        "iter_ark:",
        iter_ark,
        "iter_torch:",
        iter_torch,
        "ark_elapsed_per_iter:",
        "{:.5f}".format(ark_elapsed_per_iter),
        "torch_elapsed_per_iter:",
        "{:.5f}".format(torch_elapsed_per_iter),
    )


def convert_state_dict(state_dict: dict, type="numpy"):
    """
    Convert the state_dict of a module to np.ndarray or torch.Tensor type
    """
    new_state_dict = {}
    for key in state_dict:
        if type == "torch":
            new_state_dict[key] = torch.from_numpy(state_dict[key])
        elif type == "numpy":
            new_state_dict[key] = state_dict[key].cpu().numpy()
    return new_state_dict


def test_rmsnorm():
    # Initialize the ARK runtime
    runtime = ark.Runtime(local_rank, world_size)
    rmsnorm_pytorch = llama_pytorch.RMSNorm(dim)
    rmsnorm_ark = llama_ark.RMSNorm(dim)
    input_numpy = np.random.uniform(
        low=-1, high=1, size=(batch_size, seq_len, dim)
    ).astype(np_type)
    torch_input = torch.from_numpy(input_numpy)
    for param in rmsnorm_pytorch.parameters():
        nn.init.uniform_(param, a=-0.1, b=0.1)
    state_dict_torch = rmsnorm_pytorch.state_dict()

    rmsnorm_pytorch = rmsnorm_pytorch.to(torch_device)
    torch_input = torch_input.to(torch_device)
    output_pytorch = rmsnorm_pytorch(torch_input)
    output_pytorch = output_pytorch.cpu()
    ark_input = ark.tensor([batch_size, seq_len, dim], ark_type)
    output_ark = rmsnorm_ark(ark_input)
    # Launch the ARK runtime
    runtime.launch()
    ark_input.from_numpy(input_numpy.astype(np_type))
    state_dict_ark = convert_state_dict(state_dict_torch, "numpy")

    rmsnorm_ark.load_state_dict(state_dict_ark)
    # Run the ARK program
    runtime.run()
    output_ark_host = output_ark.to_numpy()
    # test if the result is correct
    gt = output_pytorch.detach().numpy().astype(np_type)
    max_abs_error = np.max(np.abs(output_ark_host - gt))
    mean_abs_error = np.mean(np.abs(output_ark_host - gt))

    print(
        "rmsnorm test",
        "max_abs_error:",
        "{:.5f}".format(max_abs_error),
        "mean_abs_error:",
        "{:.5f}".format(mean_abs_error),
    )

    def pytorch_func():
        rmsnorm_pytorch(torch_input)

    performance_comparison(runtime, pytorch_func)


def test_linear():
    # Initialize the ARK runtime
    runtime = ark.Runtime(local_rank, world_size)
    parallel_linear_ark = llama_ark.RowParallelLinear(dim, dim)
    parallel_linear_pytorch = fairscale.nn.model_parallel.RowParallelLinear(
        dim,
        dim,
        bias=False,
        input_is_parallel=True,
        init_method=lambda x: x,
    )
    input_numpy = np.random.uniform(
        low=-1, high=1, size=(batch_size, seq_len, dim)
    ).astype(np_type)
    torch_input = torch.from_numpy(input_numpy)
    for param in parallel_linear_pytorch.parameters():
        nn.init.uniform_(param, a=-0.1, b=0.1)
    state_dict_torch = parallel_linear_pytorch.state_dict()

    parallel_linear_pytorch = parallel_linear_pytorch.to(torch_device)
    torch_input = torch_input.to(torch_device)
    output_pytorch = parallel_linear_pytorch(torch_input)
    output_pytorch = output_pytorch.cpu()
    ark_input = ark.tensor([batch_size, seq_len, dim], ark_type)
    output_ark = parallel_linear_ark(ark_input)
    # Launch the ARK runtime
    runtime.launch()
    ark_input.from_numpy(input_numpy.astype(np_type))
    state_dict_ark = convert_state_dict(state_dict_torch, "numpy")

    parallel_linear_ark.load_state_dict(state_dict_ark)
    # Run the ARK program
    runtime.run()
    output_ark_host = output_ark.to_numpy()
    # test if the result is correct
    gt = output_pytorch.detach().numpy().astype(np_type)
    max_abs_error = np.max(np.abs(output_ark_host - gt))
    mean_abs_error = np.mean(np.abs(output_ark_host - gt))

    print(
        "rmsnorm test",
        "max_abs_error:",
        "{:.5f}".format(max_abs_error),
        "mean_abs_error:",
        "{:.5f}".format(mean_abs_error),
    )

    def pytorch_func():
        parallel_linear_pytorch(torch_input)

    performance_comparison(runtime, pytorch_func)


def test_attention():
    # Initialize the ARK runtime
    runtime = ark.Runtime(local_rank, world_size)
    args = llama_ark.ModelArgs()
    attention_pytorch = llama_pytorch.Attention(args)
    attention_ark = llama_ark.Attention(args)
    dim = args.dim
    input_numpy = np.random.uniform(
        low=-1, high=1, size=(batch_size, seq_len, dim)
    ).astype(np_type)
    torch_input = torch.from_numpy(input_numpy)

    # random init the torch model
    for param in attention_pytorch.parameters():
        nn.init.uniform_(param, a=-0.1, b=0.1)
    state_dict_torch = attention_pytorch.state_dict()

    freqs_cis_torch = llama_pytorch.precompute_freqs_cis(
        args.dim // args.n_heads, args.max_seq_len * 2
    )
    freqs_cis_torch = freqs_cis_torch[0:seq_len]

    attention_pytorch = attention_pytorch.to(torch_device)
    torch_input = torch_input.to(torch_device)
    freqs_cis_torch = freqs_cis_torch.to(torch_device)
    output_torch = attention_pytorch(torch_input, 0, freqs_cis_torch, None)
    output_torch = output_torch.cpu()

    ark_input = ark.tensor([batch_size, seq_len, dim], ark_type)
    freqs_cis_ark = ark.tensor(
        [1, seq_len, 1, args.dim // args.n_heads], ark_type
    )
    output = attention_ark(ark_input, 0, freqs_cis_ark, None)

    # Launch the ARK runtime
    runtime.launch()
    ark_input.from_numpy(input_numpy.astype(np_type))
    state_dict_ark = convert_state_dict(state_dict_torch, "numpy")
    attention_ark.load_state_dict(state_dict_ark)
    freqs_cis_complex = freqs_cis_torch.cpu().numpy().astype(np.complex64)
    # stack real and imag parts
    freqs_cis_stack = np.stack(
        [freqs_cis_complex.real, freqs_cis_complex.imag], axis=-1
    ).astype(np_type)

    freqs_cis_ark.from_numpy(freqs_cis_stack)

    # Run the ARK program
    runtime.run()
    output_ark_host = output.to_numpy()

    # test if the result is correct
    output_gt = output_torch.detach().numpy().astype(np_type)
    max_abs_error = np.max(np.abs(output_ark_host - output_gt))
    mean_abs_error = np.mean(np.abs(output_ark_host - output_gt))
    print(
        "attention test",
        "max_abs_error:",
        "{:.5f}".format(max_abs_error),
        "mean_abs_error:",
        "{:.5f}".format(mean_abs_error),
    )

    def pytorch_func():
        attention_pytorch(torch_input, 0, freqs_cis_torch, None)

    performance_comparison(runtime, pytorch_func)


def test_feedforward():
    # Initialize the ARK runtime
    runtime = ark.Runtime(local_rank, world_size)
    feedforward_pytorch = llama_pytorch.FeedForward(dim, 16384, 256, None)
    feedforward_ark = llama_ark.FeedForward(dim, 16384, 256, None)
    input_numpy = np.random.uniform(
        low=-1, high=1, size=(batch_size, seq_len, dim)
    ).astype(np_type)
    torch_input = torch.from_numpy(input_numpy)
    for param in feedforward_pytorch.parameters():
        nn.init.uniform_(param, a=-0.1, b=0.1)
    state_dict_torch = feedforward_pytorch.state_dict()
    feedforward_pytorch.to(torch_device)
    torch_input = torch_input.to(torch_device)
    output_pytorch = feedforward_pytorch(torch_input)
    output_pytorch = output_pytorch.cpu()
    ark_input = ark.tensor([batch_size, seq_len, dim], ark_type)
    output_ark = feedforward_ark(ark_input)

    # Launch the ARK runtime
    runtime.launch()
    ark_input.from_numpy(input_numpy.astype(np_type))
    state_dict_ark = convert_state_dict(state_dict_torch, "numpy")
    feedforward_ark.load_state_dict(state_dict_ark)

    # Run the ARK program
    runtime.run()
    output_ark_host = output_ark.to_numpy()

    # test if the result is correct

    gt = output_pytorch.detach().numpy().astype(np_type)

    max_abs_error = np.max(np.abs(output_ark_host - gt))
    mean_abs_error = np.mean(np.abs(output_ark_host - gt))
    print(
        "feedforward test",
        "max_abs_error:",
        "{:.5f}".format(max_abs_error),
        "mean_abs_error:",
        "{:.5f}".format(mean_abs_error),
    )

    def pytorch_func():
        feedforward_pytorch(torch_input)

    performance_comparison(runtime, pytorch_func)


def test_transformerblock():
    # Initialize the ARK runtime
    runtime = ark.Runtime(local_rank, world_size)
    args = llama_ark.ModelArgs()
    transformer_block_pytorch = llama_pytorch.TransformerBlock(0, args)
    transformer_block_ark = llama_ark.TransformerBlock(0, args)
    dim = args.dim
    input_numpy = np.random.uniform(
        low=-1, high=1, size=(batch_size, seq_len, dim)
    ).astype(np_type)
    torch_input = torch.from_numpy(input_numpy)

    # random init the torch model
    for param in transformer_block_pytorch.parameters():
        nn.init.uniform_(param, a=-0.1, b=0.1)
    state_dict = transformer_block_pytorch.state_dict()

    freqs_cis_torch = llama_pytorch.precompute_freqs_cis(
        args.dim // args.n_heads, args.max_seq_len * 2
    )
    freqs_cis_torch = freqs_cis_torch[0:seq_len]
    transformer_block_pytorch.to(torch_device)
    torch_input = torch_input.to(torch_device)
    freqs_cis_torch = freqs_cis_torch.to(torch_device)
    output_torch = transformer_block_pytorch(
        torch_input, 0, freqs_cis_torch, None
    )
    output_torch = output_torch.cpu()
    ark_input = ark.tensor([batch_size, seq_len, dim], ark_type)
    freqs_cis_ark = ark.tensor(
        [1, seq_len, 1, args.dim // args.n_heads], ark_type
    )
    output = transformer_block_ark(ark_input, 0, freqs_cis_ark, None)

    # Launch the ARK runtime
    runtime.launch()
    ark_input.from_numpy(input_numpy.astype(np_type))

    ark_state_dict = convert_state_dict(state_dict, "numpy")
    transformer_block_ark.load_state_dict(ark_state_dict)
    freqs_cis_complex = freqs_cis_torch.cpu().numpy().astype(np.complex64)
    # stack real and imag parts
    freqs_cis_stack = np.stack(
        [freqs_cis_complex.real, freqs_cis_complex.imag], axis=-1
    ).astype(np_type)

    freqs_cis_ark.from_numpy(freqs_cis_stack)
    # Run the ARK program
    runtime.run()
    output_ark_host = output.to_numpy()

    # test if the result is correct
    output_gt = output_torch.detach().numpy().astype(np_type)
    max_abs_error = np.max(np.abs(output_ark_host - output_gt))
    mean_abs_error = np.mean(np.abs(output_ark_host - output_gt))
    print(
        "transformer_block test",
        "max_abs_error:",
        "{:.5f}".format(max_abs_error),
        "mean_abs_error:",
        "{:.5f}".format(mean_abs_error),
    )

    def pytorch_func():
        transformer_block_pytorch(torch_input, 0, freqs_cis_torch, None)

    performance_comparison(runtime, pytorch_func)


def test_transformer():
    # Initialize the ARK runtime
    runtime = ark.Runtime(local_rank, world_size)
    args = llama_ark.ModelArgs()
    # To make sure that we can run this test on a single GPU, we reduce the model layer number to 2
    args.n_layers = 2
    args.vocab_size = 1024
    transformer_pytorch = llama_pytorch.Transformer(args)
    transformer_ark = llama_ark.Transformer(args)
    dim = args.dim
    input_tokens = np.random.randint(
        low=0, high=args.vocab_size, size=(batch_size, seq_len)
    ).astype(np.int64)
    torch_input = torch.from_numpy(input_tokens)

    # random init the torch model
    for param in transformer_pytorch.parameters():
        nn.init.uniform_(param, a=-0.1, b=0.1)
    state_dict = transformer_pytorch.state_dict()
    # transformer_pytorch.load_state_dict(state_dict_torch)
    output_torch = transformer_pytorch(torch_input, 0)
    output_torch = output_torch.cpu()
    ark_input = ark.tensor([batch_size, seq_len, dim], ark_type)
    freqs_cis_ark = ark.tensor(
        [1, seq_len, 1, args.dim // args.n_heads], ark_type
    )
    mask_ark = ark.tensor([1, seq_len, seq_len], ark_type)
    output = transformer_ark(ark_input, 0, freqs_cis_ark, mask_ark)
    # Launch the ARK runtime
    runtime.launch()
    input_embedding = transformer_pytorch.tok_embeddings(torch_input)
    input_embedding_numpy = input_embedding.detach().numpy().astype(np_type)

    ark_input.from_numpy(input_embedding_numpy.astype(np_type))

    freqs_cis_torch = llama_pytorch.precompute_freqs_cis(
        args.dim // args.n_heads, args.max_seq_len * 2
    )
    freqs_cis_torch = freqs_cis_torch[0:seq_len]

    freqs_cis_complex = freqs_cis_torch.numpy().astype(np.complex64)
    # stack real and imag parts
    freqs_cis_stack = np.stack(
        [freqs_cis_complex.real, freqs_cis_complex.imag], axis=-1
    ).astype(np_type)

    freqs_cis_ark.from_numpy(freqs_cis_stack)
    ark_state_dict = convert_state_dict(state_dict, "numpy")
    transformer_ark.load_state_dict(ark_state_dict)

    mask_torch = None
    if seq_len > 1:
        mask_torch = torch.full((1, 1, seq_len, seq_len), float("-inf"))
        mask_torch = torch.triu(mask_torch, diagonal=start_pos + 1)

    mask_numpy = mask_torch.numpy().astype(np_type)
    mask_ark.from_numpy(mask_numpy)
    # Run the ARK program
    runtime.run()
    output_ark_host = output.to_numpy()

    # test if the result is correct
    output_gt = output_torch.detach().numpy().astype(np_type)
    max_abs_error = np.max(np.abs(output_ark_host - output_gt))
    mean_abs_error = np.mean(np.abs(output_ark_host - output_gt))
    print(
        "transformer test",
        "max_abs_error:",
        "{:.5f}".format(max_abs_error),
        "mean_abs_error:",
        "{:.5f}".format(mean_abs_error),
    )

    def pytorch_func():
        transformer_pytorch(torch_input, 0)

    performance_comparison(runtime, pytorch_func)


if __name__ == "__main__":
    # Usage: python -m torch.distributed.launch --nproc_per_node num_gpus llama_test.py
    # Set up the environment variables for nccl

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    llama_ark.local_rank = local_rank
    llama_ark.world_size = world_size
    torch_device = torch.device("cuda", local_rank)
    # Seed must be the same in all processes
    torch.manual_seed(1)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    # If you want to test the performance of ARK, set performance_analysis to True
    performance_analysis = False
    torch.distributed.init_process_group("nccl")
    fairscale.nn.model_parallel.initialize.initialize_model_parallel(world_size)
    # test_rmsnorm()
    test_linear()
    exit(0)

    # Make sure that all processes have finished the rmsnorm test
    # torch.distributed.barrier()
    test_attention()
    # torch.distributed.barrier()
    test_feedforward()
    # torch.distributed.barrier()
    test_transformerblock()
    # torch.distributed.barrier()
    test_transformer()
    # torch.distributed.barrier()
    torch.distributed.destroy_process_group()
