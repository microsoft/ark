# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import sys
import torch
import os
import time
import fairscale
import argparse
import multiprocessing as mp
from pathlib import Path


sys.path.append("llama")
import llama.model as model_pt
import model as model_ark
import numpy as np
from typing import Dict, List
from dataclasses import dataclass
from model import ModelArgs, ModelArgs7B
from generator import precompute_freqs_cis


ckpt_dir: str = ""
output_dir: str = ""

numpy_dtype_to_torch_dtype: dict = {
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.int32: torch.int32,
}


@dataclass
class RunResults:
    outputs: List[np.ndarray] = None
    runtime: float = 0.0  # in seconds

def run_ark(
    module: ark.Module,
    state_dict: Dict[str, np.ndarray],
    inputs: list = [],
    iterations: int = 1,
    rank: int = 0,
    world_size: int = 1,
) -> List[np.ndarray]:
    ark.set_rank(rank)
    ark.set_world_size(world_size)

    module_inputs = [
        ark.tensor(list(i.shape), ark.DataType.from_numpy(i.dtype))
        if isinstance(i, np.ndarray)
        else i
        for i in inputs
    ]
    output = module(*module_inputs)

    opgraph = ark._ark_core._OpGraph(ark.Model.get_model())

    json_str = opgraph.serialize(4)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    file_name = output_dir+f'/opgraph_worker_{rank}.json'

    with open(file_name, "w") as json_file:
        json_file.write(json_str)
        print(f'{file_name} written successfully')

    return None

@torch.inference_mode()
def run_pt(
    module: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
    inputs: list = [],
    iterations: int = 1,
) -> List[np.ndarray]:
    # Update the current state_dict with the given one
    if state_dict:
        cur_state_dict = module.state_dict()
        for k, v in state_dict.items():
            cur_state_dict[k] = v
        module.load_state_dict(cur_state_dict)

    # Load input data to GPU
    input_tensors = [
        torch.from_numpy(i).to("cuda") if isinstance(i, np.ndarray) else i
        for i in inputs
    ]

    # Load the module to GPU
    module = module.to("cuda")

    start_time = time.time()

    # Run the module
    with torch.no_grad():
        for _ in range(iterations):
            output = module(*input_tensors)

    end_time = time.time()

    if isinstance(output, list) or isinstance(output, tuple):
        outputs = [o.detach().to("cpu").numpy() for o in output]
    outputs = [output.detach().to("cpu").numpy()]

    return RunResults(outputs=outputs, runtime=end_time - start_time)


def test_module(
    module_class_ark: ark.Module,
    module_args_ark: list,
    inputs_ark: List[np.ndarray],
    module_class_pt: torch.nn.Module,
    module_args_pt: list,
    inputs_pt: List[np.ndarray],
    module_name_prefix: str = "",
    dtype: np.dtype = np.float16,
    test_thru: bool = False,
    test_thru_iterations: int = 100,
    test_thru_ark_only: bool = False,
    rank: int = 0,
    world_size: int = 1,
):
    if test_thru:
        print(f"Throughput test (iterations: {test_thru_iterations})")
    else:
        print(f"Correctness test")

    # ARK module
    module_ark: ark.Module = module_class_ark(*module_args_ark)


    params_dict = module_ark.params_dict()
    param_names = set(params_dict.keys())

    state_dict_ark =[]
    # Run the ARK module
    res_ark = run_ark(
        module_ark,
        state_dict_ark,
        inputs_ark,
        iterations=test_thru_iterations if test_thru else 1,
        rank=rank,
        world_size=world_size,
    )


def test_rmsnorm(
    args: ModelArgs, batch_size: int, seq_len: int, dtype: np.dtype
):
    # Create random input data
    inputs_ark = [
        np.random.uniform(
            low=-0.1, high=0.1, size=(batch_size, seq_len, args.dim)
        ).astype(dtype)
    ]
    inputs_pt = [i.astype(dtype) for i in inputs_ark]

    test_module(
        module_class_ark=model_ark.RMSNorm,
        module_args_ark=[
            args.dim,
            args.norm_eps,
            ark.DataType.from_numpy(dtype),
        ],
        inputs_ark=inputs_ark,
        module_class_pt=model_pt.RMSNorm,
        module_args_pt=[args.dim],
        inputs_pt=inputs_pt,
        module_name_prefix="norm",
    )


def test_row_parallel_linear(
    args: ModelArgs,
    batch_size: int,
    seq_len: int,
    dtype: np.dtype,
    rank: int = 0,
    world_size: int = 1,
):
    # Create random input data
    inputs_ark = [
        np.random.uniform(
            low=-0.1,
            high=0.1,
            size=(batch_size, seq_len, args.dim // args.n_heads * args.n_heads),
        ).astype(dtype)
    ]
    inputs_pt = [i.astype(dtype) for i in inputs_ark]

    test_module(
        module_class_ark=model_ark.RowParallelLinear,
        module_args_ark=[
            args.dim // args.n_heads * args.n_heads,
            args.dim,
            ark.DataType.from_numpy(dtype),
            False,
            rank,
            world_size,
        ],
        inputs_ark=inputs_ark,
        module_class_pt=fairscale.nn.model_parallel.RowParallelLinear,
        module_args_pt=[
            args.dim // args.n_heads * args.n_heads,
            args.dim,
            False,
            False,
            lambda x: x,
        ],
        inputs_pt=inputs_pt,
        module_name_prefix="layers.0.attention.wo",
        dtype=dtype,
        # test_thru = True,
        # test_thru_iterations = 200,
    )


def test_column_parallel_linear(
    args: ModelArgs,
    batch_size: int,
    seq_len: int,
    dtype: np.dtype,
    rank: int = 0,
    world_size: int = 1,
):
    seed = 1695878986  # int(time.time())
    print(f"seed: {seed}")
    np.random.seed(seed)
    # Create random input data
    inputs_ark = [
        np.random.uniform(
            low=-0.1, high=0.1, size=(batch_size, seq_len, args.dim)
        ).astype(dtype)
    ]
    inputs_pt = [i.astype(dtype) for i in inputs_ark]

    test_module(
        module_class_ark=model_ark.ColumnParallelLinear,
        module_args_ark=[
            args.dim,
            args.dim // args.n_heads * args.n_heads,
            ark.DataType.from_numpy(dtype),
            True,
            rank,
            world_size,
        ],
        inputs_ark=inputs_ark,
        module_class_pt=fairscale.nn.model_parallel.ColumnParallelLinear,
        module_args_pt=[
            args.dim,
            args.dim // args.n_heads * args.n_heads,
            False,
            True,
            lambda x: x,
        ],
        inputs_pt=inputs_pt,
        module_name_prefix="layers.0.attention.wq",
        dtype=dtype,
    )


def test_attention(
    args: ModelArgs,
    batch_size: int,
    seq_len: int,
    dtype: np.dtype,
    rank: int = 0,
    world_size: int = 1,
):
    #
    freqs_cis = precompute_freqs_cis(
        args.dim // args.n_heads, args.max_seq_len * 2
    )[0:seq_len]

    freqs_cis_ark = freqs_cis.astype(np.complex64)
    freqs_cis_ark = (
        np.stack([freqs_cis_ark.real, freqs_cis_ark.imag], axis=-1)
        .astype(dtype)
        .reshape(1, seq_len, 1, args.dim // args.n_heads)
    )

    seed = 1695878986  # int(time.time())
    print(f"seed: {seed}")
    np.random.seed(seed)
    feature = np.random.uniform(
        low=-0.1, high=0.1, size=(batch_size, seq_len, args.dim)
    ).astype(dtype)

    test_module(
        module_class_ark=model_ark.Attention,
        module_args_ark=[
            args,
            ark.DataType.from_numpy(dtype),
            rank,
            world_size,
        ],
        inputs_ark=[feature, 0, freqs_cis_ark, None],
        module_class_pt=model_pt.Attention,
        module_args_pt=[args],
        inputs_pt=[feature.astype(dtype), 0, freqs_cis, None],
        module_name_prefix="layers.0.attention",
    )


def test_transformer_block(
    args: ModelArgs,
    batch_size: int,
    seq_len: int,
    dtype: np.dtype,
    rank: int = 0,
    world_size: int = 1,
):
    #
    freqs_cis = precompute_freqs_cis(
        args.dim // args.n_heads, args.max_seq_len * 2
    )[0:seq_len]

    freqs_cis_ark = freqs_cis.astype(np.complex64)
    freqs_cis_ark = (
        np.stack([freqs_cis_ark.real, freqs_cis_ark.imag], axis=-1)
        .astype(dtype)
        .reshape(1, seq_len, 1, args.dim // args.n_heads)
    )

    feature = np.random.uniform(
        low=-1, high=1, size=(batch_size, seq_len, args.dim)
    ).astype(dtype)

    test_module(
        module_class_ark=model_ark.TransformerBlock,
        module_args_ark=[
            0,
            args,
            ark.DataType.from_numpy(dtype),
            rank,
            world_size,
        ],
        inputs_ark=[feature, 0, freqs_cis_ark, None],
        module_class_pt=model_pt.TransformerBlock,
        module_args_pt=[0, args],
        inputs_pt=[feature.astype(dtype), 0, freqs_cis, None],
        module_name_prefix="layers.0",
        rank=rank,
        world_size=world_size,
    )


def test_transformer(
    args: ModelArgs,
    batch_size: int,
    seq_len: int,
    dtype: np.dtype,
    rank: int = 0,
    world_size: int = 1,
):
    # Random input tokens

    seed = 1695878986  # int(time.time())
    print(f"seed: {seed}")
    np.random.seed(seed)

    tokens = np.random.randint(
        low=0, high=args.vocab_size, size=(batch_size, seq_len)
    ).astype(np.int32)

    start_pos = 0

    # Pre-calculated freqs_cis

    freqs_cis = precompute_freqs_cis(
        args.dim // args.n_heads, args.max_seq_len * 2
    )[0:seq_len]

    freqs_cis_ark = freqs_cis.astype(np.complex64)
    freqs_cis_ark = (
        np.stack([freqs_cis_ark.real, freqs_cis_ark.imag], axis=-1)
        .astype(dtype)
        .reshape(1, seq_len, 1, args.dim // args.n_heads)
    )

    # Pre-calculated mask

    if seq_len == 1:
        mask = None
    else:
        mask = np.full((1, 1, seq_len, seq_len), -np.inf, dtype=dtype)
        mask = np.triu(mask, k=start_pos + 1)

    test_module(
        module_class_ark=model_ark.Transformer,
        module_args_ark=[
            args,
            ark.DataType.from_numpy(dtype),
            rank,
            world_size,
        ],
        inputs_ark=[tokens, start_pos, freqs_cis_ark, mask],
        module_class_pt=model_pt.Transformer,
        module_args_pt=[args],
        inputs_pt=[tokens, start_pos],
        # test_thru=True,
        # test_thru_iterations=200,
    )


def test(args, batch_size, seq_len, dtype, rank, world_size):
    ark.init()
    ark.set_rank(rank)
    ark.set_world_size(world_size)

    # test_rmsnorm(args, batch_size, seq_len, dtype)
    # test_row_parallel_linear(args, batch_size, seq_len, dtype, rank, world_size)
    # test_column_parallel_linear(args, batch_size, seq_len, dtype, rank, world_size)
    # test_attention(args, batch_size, seq_len, dtype, rank, world_size)
    test_transformer_block(args, batch_size, seq_len, dtype, rank, world_size)
    # test_transformer(args, batch_size, seq_len, dtype, rank, world_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--ngpus", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="opgraph")

    # ckpt_dir = parser.parse_args().ckpt_dir
    ngpus = parser.parse_args().ngpus
    output_dir = parser.parse_args().output_dir

    # Configurations
    args = ModelArgs7B()
    batch_size = 1
    seq_len = 512
    dtype = np.float16
    world_size = ngpus

    # Default from HuggingFace
    args.vocab_size = 32000

    # Reduce max_seq_len due to OOM from the PyTorch model
    args.max_seq_len = 512

    # Verify the configurations
    assert batch_size <= args.max_batch_size
    assert seq_len <= args.max_seq_len


    if world_size == 1:
        test(args, batch_size, seq_len, dtype, 0, 1)
    else:
        procs = []
        for i in range(ngpus):
            p = mp.Process(
                target=test,
                args=(args, batch_size, seq_len, dtype, i, world_size),
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
