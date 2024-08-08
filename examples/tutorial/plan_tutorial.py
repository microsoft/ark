# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import ark
import time
import json
import numpy as np
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = (
        256  # make SwiGLU hidden layer size multiple of large power of 2
    )
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048


class ColumnParallelLinear(ark.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].
    Here the weight = A^T, so we need to partition the weight matrix along
    its first dimension.

    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dtype: np.dtype,
        gather_output: bool = True,
        local_rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dtype = dtype
        self.local_rank = local_rank
        self.world_size = world_size
        self.gather_output = gather_output

        self.weight = ark.parameter(
            [out_dim // world_size, in_dim], ark.DataType.from_numpy(dtype)
        )
        self.data = None

    def forward(self, x):
        if self.world_size == 1 or self.gather_output == False:
            return ark.matmul(x, self.weight, transpose_other=True)
        # We need to concat the output_tensor_shards along the last dimension
        output_tensor = ark.tensor(
            [x.shape()[0], x.shape()[1], self.out_dim],
            ark.DataType.from_numpy(self.dtype),
        )
        output_tensor_shards = ark.sharding(
            output_tensor,
            axis=2,
            dim_per_shard=self.out_dim // self.world_size,
        )
        local_result = ark.identity(
            output_tensor_shards[self.local_rank], deps=output_tensor_shards
        )
        # (batch_size, seq_len, out_dim // world_size)
        local_result = ark.matmul(
            x, self.weight, local_result, transpose_other=True
        )
        gather_input = ark.identity(output_tensor, deps=[local_result])
        # return gather_input
        gather_reshape = ark.reshape(
            gather_input, [x.shape()[0] * x.shape()[1], self.out_dim]
        )
        gather_out = ark.local_all_gather(
            gather_reshape, self.local_rank, self.world_size, 1
        )
        return ark.reshape(
            gather_out, [x.shape()[0], x.shape()[1], self.out_dim]
        )

    def initialize(self):
        if self.data is None:
            data = np.random.uniform(
                low=-0.1, high=0.1, size=self.weight.shape()
            ).astype(self.dtype)
            self.data = data
        self.weight.from_numpy(self.data)


class RowParallelLinear(ark.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -

    Here the weight = A^T, so we need to partition the weight matrix along
    its second dimension.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dtype: ark.DataType = ark.fp16,
        input_is_parallel: bool = False,
        local_rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dtype = dtype
        self.local_rank = local_rank
        self.world_size = world_size
        self.input_is_parallel = input_is_parallel

        self.weight = ark.parameter(
            [out_dim, in_dim // world_size], ark.DataType.from_numpy(self.dtype)
        )
        self.data = None

    def forward(self, x):
        if self.world_size == 1:
            return ark.matmul(x, self.weight, transpose_other=True)
        x_ndims = len(x.shape())
        if self.input_is_parallel:
            input_parallel = x
        else:
            x_shards = ark.sharding(
                x, x_ndims - 1, self.in_dim // self.world_size
            )
            input_parallel = x_shards[self.local_rank]
        local_result = ark.matmul(
            input_parallel, self.weight, transpose_other=True
        )
        reduced_result = ark.local_all_reduce(
            local_result, self.local_rank, self.world_size
        )
        return reduced_result

    def initialize(self):
        if self.data is None:
            data = np.random.uniform(
                low=-0.1, high=0.1, size=self.weight.shape()
            ).astype(self.dtype)
            self.data = data
        self.weight.from_numpy(self.data)


class Silu(ark.Module):
    """
    Silu activation function, silu(x) = x * sigmoid(x)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: ark.Tensor):
        x1 = ark.sigmoid(x)
        return ark.mul(x, x1)


class FeedForward(ark.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        dtype: np.dtype,
        local_rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * (
            (hidden_dim + multiple_of - 1) // multiple_of
        )

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, dtype, False, local_rank, world_size
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, dtype, True, local_rank, world_size
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, dtype, False, local_rank, world_size
        )

    def forward(self, x):
        # self.w2(F.silu(self.w1(x)) * self.w3(x))
        x1 = self.w1(x)
        x1 = Silu()(x1)
        x2 = self.w3(x)
        x3 = ark.mul(x1, x2)
        x4 = self.w2(x3)
        return x4

    def initialize(self):
        self.w1.initialize()
        self.w2.initialize()
        self.w3.initialize()


class Input(ark.Module):
    def __init__(
        self, batch_size: int, seq_len: int, dim: int, dtype: np.dtype
    ):
        super().__init__()
        self.tensor = ark.tensor(
            (batch_size, seq_len, dim), ark.DataType.from_numpy(dtype)
        )
        self.data = None

    def forward(self):
        return self.tensor

    def initialize(self):
        if self.data is None:
            self.data = np.random.uniform(
                low=-0.1, high=0.1, size=self.tensor.shape()
            ).astype(self.tensor.dtype().to_numpy())
        self.tensor.from_numpy(self.data)


def compare_results(result, ground_truth):
    eps = np.finfo(result.dtype).eps
    result = result.flatten()
    ground_truth = ground_truth.flatten()

    max_value_idx = np.argmax(ground_truth)
    min_value_idx = np.argmin(ground_truth)

    abs_diff = np.abs(result - ground_truth)
    max_abs_diff_idx = np.argmax(abs_diff)
    max_abs_diff = abs_diff[max_abs_diff_idx]

    abs_pt = np.abs(ground_truth)
    rel_diff = abs_diff / (abs_pt + eps)
    max_rel_diff_idx = np.argmax(rel_diff)
    max_rel_diff = rel_diff[max_rel_diff_idx]

    # max rel_diff where abs_pt is larger than 1e-3
    max_rel_diff_3_idx = np.argmax(rel_diff * (abs_pt > 1e-3))
    max_rel_diff_3 = rel_diff[max_rel_diff_3_idx]

    mean_square_error = np.mean(np.square(result - ground_truth))

    # Test info as string

    print(
        f"Comparing ground truth vs results\n"
        f"  max_value: {ground_truth[max_value_idx]} vs {result[max_value_idx]} at index {max_value_idx}\n"
        f"  min_value: {ground_truth[min_value_idx]} vs {result[min_value_idx]} at index {min_value_idx}\n"
        f"  max_abs_diff: {max_abs_diff:.4e} ({ground_truth[max_abs_diff_idx]} vs {result[max_abs_diff_idx]} at index {max_abs_diff_idx})\n"
        f"  max_rel_diff: {max_rel_diff:.4e} ({ground_truth[max_rel_diff_idx]} vs {result[max_rel_diff_idx]} at index {max_rel_diff_idx})\n"
        f"  max_rel_diff_3: {max_rel_diff_3:.4e} ({ground_truth[max_rel_diff_3_idx]} vs {result[max_rel_diff_3_idx]} at index {max_rel_diff_3_idx})\n"
        f"  mean_square_error: {mean_square_error:.4e}\n"
    )


def config_rule_larger_tile(op: str, arch: str) -> str:
    j = json.loads(op)
    op_type = j["Type"]
    if op_type == "Sigmoid" or op_type == "Mul":
        pshape = j["ResultTensors"][0]["PaddedShape"]
        if len(pshape) < 2 or pshape[-2] % 128 != 0 or pshape[-1] % 256 != 0:
            return ""
        num_tasks = pshape[-2] // 128 * pshape[-1] // 256
        cfg = {
            "NumWarps": 8,
            "SramBytes": 0,
            "Tile": [128, 256],
            "NumTasks": num_tasks,
        }
        return json.dumps(cfg)
    return ""


def main(plan_path: str):
    args = ModelArgs()
    batch_size = 1
    seq_len = 512
    dtype = np.float16
    seed = int(time.time())

    print(f"seed: {seed}")
    np.random.seed(seed)
    ark.srand(seed)

    InputModule = Input(batch_size, seq_len, args.dim, dtype)
    input_tensor = InputModule()

    # Declare model
    FeedForwardModule = FeedForward(
        dim=args.dim,
        hidden_dim=4 * args.dim,
        multiple_of=args.multiple_of,
        ffn_dim_multiplier=args.ffn_dim_multiplier,
        dtype=dtype,
    )
    output_tensor = FeedForwardModule(input_tensor)

    # Write model.json
    with open("model.json", "w") as f:
        f.write(ark.Model.get_model().compress().serialize())

    # Calculate default result
    ground_truth = None
    with ark.Runtime.get_runtime() as rt:
        planner = ark.Planner()

        # If this rule is installed, default planner will perform the same as
        # `plan_1_larger_tile.json` on A100.
        # planner.install_config_rule(config_rule_larger_tile)

        plan = planner.plan()
        with open("default_plan.json", "w") as f:
            f.write(str(plan))
        rt.launch(plan=plan)

        # Initialize
        InputModule.initialize()
        FeedForwardModule.initialize()

        # Calculate output
        rt.run()
        ground_truth = output_tensor.to_numpy()

        # Measure throughput
        iter = 100
        ts = time.time()
        rt.run(iter)
        elapsed_ms = (time.time() - ts) * 1e3
        print(
            f"DefaultPlan elapsed time: total {elapsed_ms:.6f} ms, {elapsed_ms/iter:.6f} ms/iter"
        )

    # Run `plan_path` file if exists
    if not Path(plan_path).is_file():
        print(f"File {plan_path} does not exist. Exiting...")
        return
    with ark.Runtime.get_runtime() as rt:
        rt.launch(plan=ark.Plan.from_file(plan_path))

        # Initialize
        InputModule.initialize()
        FeedForwardModule.initialize()

        # Calculate output
        rt.run()
        result = output_tensor.to_numpy()

        # Measure throughput
        iter = 100
        ts = time.time()
        rt.run(iter)
        elapsed_ms = (time.time() - ts) * 1e3
        print(
            f"Plan elapsed time: total {elapsed_ms:.6f} ms, {elapsed_ms/iter:.6f} ms/iter"
        )

    # Compare results
    compare_results(result, ground_truth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan_path", type=str, default="plan.json")

    args = parser.parse_args()
    main(args.plan_path)
