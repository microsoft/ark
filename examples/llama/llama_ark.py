# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import math
from dataclasses import dataclass
from typing import Optional

ark_type = ark.FP32
local_rank = 0
world_size = 1


def ModelArgs():
    return ModelArgs13B()


@dataclass
class ModelArgs7B:
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


@dataclass
class ModelArgs13B:
    dim: int = 5120
    n_layers: int = 40
    n_heads: int = 40
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = (
        256  # make SwiGLU hidden layer size multiple of large power of 2
    )
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


@dataclass
class ModelArgs70B:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = (
        256  # make SwiGLU hidden layer size multiple of large power of 2
    )
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(ark.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = ark.Parameter(ark.tensor([1, 1, dim], ark_type))

    def _norm(self, x):
        # x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return ark.rmsnorm(x)

    def forward(self, x):
        output = self._norm(x)
        return ark.mul(output, self.weight)


class ColumnParallelLinear(ark.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].
    Here the weight = A^T, so we need to partition the weight matrix along
    its first dimension.

    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = ark.Parameter(
            ark.tensor([out_dim // world_size, in_dim], ark_type)
        )

    def forward(self, x):
        if world_size == 1:
            return ark.matmul(x, self.weight, transpose_b=True)
        # (batch_size, seq_len, out_dim // world_size)
        output_tensor_shard = ark.matmul(x, self.weight, transpose_b=True)
        all_gather_tensor_shards = ark.all_gather(
            output_tensor_shard, local_rank, world_size
        )
        # We need to concat the output_tensor_shards along the last dimension
        assert len(all_gather_tensor_shards) == world_size
        output_tensor = ark.tensor(
            [x.shape[0], x.shape[1], self.out_dim], ark_type
        )
        output_tensor_shards = ark.sharding(
            output_tensor, 2, self.out_dim // world_size
        )
        output_dependency = []
        # Copy all the all_gather_tensor_shards to output_tensor_shards
        for i in range(world_size):
            output_tensor_shard = ark.scale(
                all_gather_tensor_shards[i], 1.0, output_tensor_shards[i]
            )
            output_dependency.append(output_tensor_shard)
        # The output_tensor should depend on the scale operators
        output_tensor = ark.identity(output_tensor, output_dependency)
        return output_tensor


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

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = ark.Parameter(
            ark.tensor([out_dim, in_dim // world_size], ark_type)
        )

    def forward(self, x):
        if world_size == 1:
            return ark.matmul(x, self.weight, transpose_b=True)
        x_ndims = len(x.shape)
        x_shards = ark.sharding(x, x_ndims - 1, self.in_dim // world_size)
        output_parallel = ark.matmul(
            x_shards[local_rank], self.weight, transpose_b=True
        )
        # allreduce the output_parallel, currently we only support allreduce on 1D tensor,
        # so we need to reshape the output_parallel to 1D
        output_shape = output_parallel.shape
        # multiply the output_shape list
        output_shape_bytes = 1
        for i in range(len(output_shape)):
            output_shape_bytes *= output_shape[i]
        output_parallel_reshape = ark.reshape(
            output_parallel,
            [output_shape_bytes],
        )
        output_reshape = ark.all_reduce(
            output_parallel_reshape, local_rank, world_size
        )
        output = ark.reshape(output_reshape, output_shape)
        return output


class Linear(ark.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = ark.Parameter(ark.tensor([out_dim, in_dim], ark_type))

    def forward(self, x):
        return ark.matmul(x, self.weight, transpose_b=True)


class Silu(ark.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # We need to specify output tensor so that the sigmoid op will not be an in-place operator
        output = ark.tensor(x.shape, ark_type)
        x1 = ark.sigmoid(x, output)
        return ark.mul(x, x1)


class FeedForward(ark.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * (
            (hidden_dim + multiple_of - 1) // multiple_of
        )

        self.w1 = ColumnParallelLinear(dim, hidden_dim)
        self.w2 = RowParallelLinear(hidden_dim, dim)
        self.w3 = ColumnParallelLinear(dim, hidden_dim)

    def forward(self, x):
        # self.w2(F.silu(self.w1(x)) * self.w3(x))
        x1 = self.w1(x)
        x1 = Silu()(x1)
        x2 = self.w3(x)
        x3 = ark.mul(x1, x2)
        x4 = self.w2(x3)
        return x4


def apply_rotary_emb(xq, xk, freqs_cis):
    xq_out = ark.rope(xq, freqs_cis)
    xk_out = ark.rope(xk, freqs_cis)
    return xq_out, xk_out


class Attention(ark.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = (
            args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        )
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = ColumnParallelLinear(args.dim, args.n_heads * self.head_dim)
        self.wk = ColumnParallelLinear(
            args.dim, self.n_kv_heads * self.head_dim
        )
        self.wv = ColumnParallelLinear(
            args.dim, self.n_kv_heads * self.head_dim
        )
        self.wo = RowParallelLinear(args.n_heads * self.head_dim, args.dim)

    def forward(
        self,
        x: ark.Tensor,
        start_pos: int,
        freqs_cis: ark.Tensor,
        mask: Optional[ark.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        # xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        # xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xq = ark.reshape(xq, [bsz, seqlen, self.n_local_heads, self.head_dim])
        xk = ark.reshape(
            xk, [bsz, seqlen, self.n_local_kv_heads, self.head_dim]
        )
        xv = ark.reshape(
            xv, [bsz, seqlen, self.n_local_kv_heads, self.head_dim]
        )
        if freqs_cis is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # TODO: enable kv cache and mask later
        keys = xk
        values = xv
        # (bs, n_local_heads, seqlen, head_dim)
        xq = ark.transpose(xq, [0, 2, 1, 3])
        keys = ark.transpose(keys, [0, 2, 1, 3])
        values = ark.transpose(values, [0, 2, 1, 3])

        # (bs, n_local_heads, head_dim, seqlen)
        keys_transpose = ark.transpose(keys, [0, 1, 3, 2])
        scores = ark.matmul(xq, keys_transpose)
        scores = ark.scale(scores, 1.0 / math.sqrt(self.head_dim))

        if mask is not None:
            scores = ark.add(scores, mask)
        scores = ark.softmax(scores)

        output = ark.matmul(
            scores, values
        )  # (bs, n_local_heads, seqlen, head_dim)
        output = ark.transpose(output, [0, 2, 1, 3])
        output = ark.reshape(
            output, [bsz, seqlen, self.head_dim * self.n_local_heads]
        )
        output = self.wo(output)
        return output


class TransformerBlock(ark.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: ark.Tensor,
        start_pos: int,
        freqs_cis: ark.Tensor,
        mask: Optional[ark.Tensor],
    ):
        attention_norm_x = self.attention_norm(x)
        h = self.attention.forward(attention_norm_x, start_pos, freqs_cis, mask)
        h = ark.add(x, h)
        out = ark.add(h, self.feed_forward(self.ffn_norm(h)))
        return out


class Transformer(ark.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        # TODO: implement the token embedding layer later
        # self.tok_embeddings = Linear(
        #     params.vocab_size, params.dim, init_method=lambda x: x
        # )

        self.layers = []
        for layer_id in range(self.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
            self.register_module(f"layers.{layer_id}", self.layers[layer_id])
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(params.dim, params.vocab_size)

    def forward(
        self,
        h: ark.Tensor,
        start_pos: int,
        freqs_cis: ark.Tensor,
        mask: Optional[ark.Tensor],
    ):
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h)
        return output


def unittest(test_func):
    def _test_func():
        test_func()

    proc = []
    global local_rank

    for rank in range(world_size):
        local_rank = rank
        proc.append(multiprocessing.Process(target=_test_func))
        proc[rank].start()
    for rank in range(world_size):
        proc[rank].join()
        assert proc[rank].exitcode == 0


import multiprocessing


def test_rmsnorm():
    batch_size = 1
    seq_len = 64
    dim = ModelArgs().dim
    # Initialize the ARK runtime
    runtime = ark.Runtime(local_rank, world_size)
    rmsnorm_ark = RMSNorm(dim)
    ark_input = ark.tensor([batch_size, seq_len, dim], ark_type)
    output = rmsnorm_ark(ark_input)
    # Launch the ARK runtime
    runtime.launch()

    # Run the ARK program
    runtime.run()
    print("ARK LLaMA RMSNorm test passed.")


def test_column_parallel_linear():
    batch_size = 1
    seq_len = 64
    dim = ModelArgs().dim
    # Initialize the ARK runtime
    runtime = ark.Runtime(local_rank, world_size)
    column_parallel_linear_ark = ColumnParallelLinear(dim, dim)
    ark_input = ark.tensor([batch_size, seq_len, dim], ark_type)
    output = column_parallel_linear_ark(ark_input)
    # Launch the ARK runtime
    runtime.launch()

    # Run the ARK program
    runtime.run()
    print("ARK LLaMA ColumnParallelLinear test passed.")


def test_row_parallel_linear():
    batch_size = 1
    seq_len = 64
    dim = ModelArgs().dim
    # Initialize the ARK runtime
    runtime = ark.Runtime(local_rank, world_size)
    row_parallel_linear_ark = RowParallelLinear(dim, dim)
    ark_input = ark.tensor([batch_size, seq_len, dim], ark_type)
    output = row_parallel_linear_ark(ark_input)
    # Launch the ARK runtime
    runtime.launch()

    # Run the ARK program
    runtime.run()
    print("ARK LLaMA RowParallelLinear test passed.")


def test_transformer():
    batch_size = 1
    seq_len = 64
    dim = ModelArgs().dim
    # Initialize the ARK runtime
    runtime = ark.Runtime(local_rank, world_size)
    args = ModelArgs()
    # To make sure that we can run this test on a single GPU, we reduce the model layer number to 2
    args.n_layers = 4
    args.vocab_size = 1024
    transformer_ark = Transformer(args)
    dim = args.dim

    ark_input = ark.tensor([batch_size, seq_len, dim], ark_type)
    freqs_cis_ark = ark.tensor(
        [1, seq_len, 1, args.dim // args.n_heads], ark_type
    )
    mask_ark = ark.tensor([1, seq_len, seq_len], ark_type)
    output = transformer_ark(ark_input, 0, freqs_cis_ark, mask_ark)
    # Launch the ARK runtime
    runtime.launch()

    # Run the ARK program
    runtime.run()
    print("ARK LLaMA Transformer test passed.")


if __name__ == "__main__":
    world_size = 8
    unittest(test_transformer)
