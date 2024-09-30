# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""LLaMA 2 Transformer model.
   Correspond to https://github.com/facebookresearch/llama/blob/main/llama/model.py
"""

import ark
import math
from dataclasses import dataclass
from typing import Optional
from ark import PlannerContext as Context


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


@dataclass
class ModelArgs7B(ModelArgs):
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
class ModelArgs13B(ModelArgs):
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
class ModelArgs70B(ModelArgs):
    dim: int = 8192
    n_layers: int = 80
    n_heads: int = 64
    n_kv_heads: Optional[int] = 8
    vocab_size: int = -1
    multiple_of: int = (
        4096  # make SwiGLU hidden layer size multiple of large power of 2
    )
    ffn_dim_multiplier: Optional[float] = 1.3
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 4096


class RMSNorm(ark.Module):
    """
    Root mean square layer normalization (RMSNorm).
    """

    def __init__(
        self, dim: int, eps: float = 1e-6, dtype: ark.DataType = ark.fp16
    ):
        super().__init__()
        self.eps = eps
        self.dtype = dtype
        self.weight = ark.parameter([1, 1, dim], ark.fp32)

    def forward(self, x):
        with Context(
            sync=False,
            config={
                "NumWarps": 1,
                "SramBytes": 0,
                "Granularity": 7,
            },
        ):
            with Context(config={"Tile": [1, 4096]}):
                x = ark.cast(x, ark.fp32)
                x2 = ark.mul(x, x)
            with Context(config={"Tile": [1], "ImplType": "WarpWise"}):
                mean = ark.reduce_mean(x2, axis=-1)
                mean = ark.add(mean, self.eps)
                rrms = ark.rsqrt(mean)
            with Context(config={"Tile": [1, 4096]}):
                x = ark.mul(x, rrms)
                x = ark.mul(x, self.weight, x)
                return ark.cast(x, self.dtype)


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
        dtype: ark.DataType = ark.fp16,
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

        self.weight = ark.parameter([out_dim // world_size, in_dim], dtype)

    def forward(self, x):
        if self.world_size == 1 or self.gather_output == False:
            return ark.matmul(x, self.weight, transpose_other=True)
        # We need to concat the output_tensor_shards along the last dimension
        output_tensor = ark.tensor(
            [x.shape()[0], x.shape()[1], self.out_dim], self.dtype
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

        self.weight = ark.parameter([out_dim, in_dim // world_size], dtype)

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
        reduced_result = ark.all_reduce(
            local_result, self.local_rank, self.world_size
        )
        return reduced_result


class ParallelEmbedding(ark.Module):
    """Embedding layer."""

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        dtype: ark.DataType,
        local_rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.weight = ark.parameter([vocab_size, dim // world_size], dtype)
        self.out_dim = dim
        self.dtype = dtype
        self.world_size = world_size
        self.local_rank = local_rank

    def forward(self, x):
        if self.world_size == 1:
            return ark.embedding(x, self.weight)

        output_tensor = ark.tensor(
            [x.shape()[0], x.shape()[1], self.out_dim], self.dtype
        )
        output_tensor_shards = ark.sharding(
            output_tensor, axis=2, dim_per_shard=self.out_dim // self.world_size
        )
        local_result = ark.identity(
            output_tensor_shards[self.local_rank], deps=output_tensor_shards
        )
        local_result = ark.embedding(x, self.weight, local_result)
        gather_input = ark.identity(output_tensor, deps=[local_result])
        gather_reshape = ark.reshape(
            gather_input, [x.shape()[0] * x.shape()[1], self.out_dim]
        )
        gather_out = ark.local_all_gather(
            gather_reshape, self.local_rank, self.world_size, 1
        )
        return ark.reshape(
            gather_out, [x.shape()[0], x.shape()[1], self.out_dim]
        )


class Linear(ark.Module):
    """
    Linear layer module with weights and no bias.
    """

    def __init__(
        self, in_dim: int, out_dim: int, dtype: ark.DataType = ark.fp16
    ):
        super().__init__()
        self.dtype = dtype
        self.weight = ark.parameter([out_dim, in_dim], dtype)

    def forward(self, x):
        return ark.matmul(x, self.weight, transpose_other=True)


class Silu(ark.Module):
    """
    Silu activation function, silu(x) = x * sigmoid(x)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: ark.Tensor):
        # We need to specify output tensor so that the sigmoid op will not be an in-place operator
        output = ark.tensor(x.shape(), x.dtype())
        x1 = ark.sigmoid(x, output)
        return ark.mul(x, x1)


class FeedForward(ark.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        dtype: ark.DataType = ark.fp16,
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

    def forward(self, x, ffn_norm):
        h = ffn_norm(x)
        with Context(
            processor_range=[0, 304],
            sram_range=[0, 49344],
            config={"NumWarps": 4},
        ):
            out_shape = h.shape()
            out_shape[-1] = self.w1.out_dim
            out = ark.tensor(out_shape, h.dtype())
            pos = 0
            for dim, tile, sram in [
                [1792, [256, 128], 24672],
                [256, [128, 128], 16480],
            ]:
                with Context(
                    processor_range=[0, 304], sync=False, config={"Tile": tile}
                ):
                    h_shard = h[:, pos : pos + dim, :]
                    out_shard = out[:, pos : pos + dim, :]
                    with Context(config={"SramBytes": sram}):
                        x1 = ark.matmul(
                            h_shard, self.w1.weight, transpose_other=True
                        )
                    with Context(config={"SramBytes": 0}):
                        x1 = Silu()(x1)
                # We don't need a barrier here but somehow the performance is better with it
                with Context(
                    processor_range=[0, 304], sync=False, config={"Tile": tile}
                ):
                    with Context(config={"SramBytes": sram}):
                        x2 = ark.matmul(
                            h_shard, self.w3.weight, transpose_other=True
                        )
                    with Context(config={"SramBytes": 0}):
                        x3 = ark.mul(x1, x2, out_shard)
                    out = ark.identity(out, deps=[x3])
                    pos += dim

        with Context(
            warp_range=[0, 4],
            config={
                "NumWarps": 4,
                "Tile": [256, 128],
                "SramBytes": 24672,
            },
            sync=False,
        ):
            ff = self.w2(out)
            return ark.add(x, ff)


def apply_rotary_emb(xq, xk, freqs_cis):
    """
    Apply rotary embeddings to xq and xk.
    """
    xq_out = ark.rope(xq, freqs_cis)
    xk_out = ark.rope(xk, freqs_cis)
    return xq_out, xk_out


class Attention(ark.Module):
    def __init__(
        self,
        args: ModelArgs,
        dtype: ark.DataType = ark.fp16,
        local_rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        self.n_kv_heads = (
            args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        )
        model_parallel_size = world_size
        self.dtype = dtype
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            dtype,
            False,
            local_rank,
            world_size,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            dtype,
            False,
            local_rank,
            world_size,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            dtype,
            False,
            local_rank,
            world_size,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            dtype,
            True,
            local_rank,
            world_size,
        )

    def forward(
        self,
        x: ark.Tensor,
        start_pos: int,
        freqs_cis: ark.Tensor,
        mask: Optional[ark.Tensor],
        attention_norm,
    ):
        bsz, seqlen, _ = x.shape()

        x_norm = attention_norm(x)

        xq_scratch = ark.tensor(
            [
                bsz,
                seqlen * self.n_local_heads,
                self.n_local_heads,
                self.head_dim,
            ],
            self.dtype,
        )
        xk_scratch = ark.tensor(
            [
                bsz,
                seqlen * self.n_local_kv_heads,
                self.n_local_kv_heads,
                self.head_dim,
            ],
            self.dtype,
        )

        with Context(
            warp_range=[0, 4],
            sram_range=[0, 24672],
            sync=False,
            config={"NumWarps": 4},
        ):
            with Context(config={"SramBytes": 24672, "Tile": [256, 128]}):
                xq = ark.matmul(x_norm, self.wq.weight, transpose_other=True)
            xq = ark.reshape(
                xq, [bsz, seqlen, self.n_local_heads, self.head_dim]
            )
            with Context(config={"SramBytes": 0, "Tile": [256, 1, 128]}):
                if freqs_cis is not None:
                    xq = ark.rope(xq, freqs_cis, xq_scratch[:, :seqlen, :, :])

            xq_scratch = ark.identity(xq_scratch, deps=[xq])

        with Context(
            warp_range=[0, 4],
            sram_range=[0, 24672],
            sync=False,
            config={"NumWarps": 4},
        ):
            with Context(config={"SramBytes": 24672, "Tile": [256, 128]}):
                xk = ark.matmul(x_norm, self.wk.weight, transpose_other=True)
            xk = ark.reshape(
                xk, [bsz, seqlen, self.n_local_kv_heads, self.head_dim]
            )
            with Context(config={"SramBytes": 0, "Tile": [256, 1, 128]}):
                if freqs_cis is not None:
                    xk = ark.rope(xk, freqs_cis, xk_scratch[:, :seqlen, :, :])

            xk_scratch = ark.identity(xk_scratch, deps=[xk])

        with Context(
            warp_range=[0, 4],
            sram_range=[0, 24672],
            sync=False,
            config={"NumWarps": 4},
        ):
            with Context(config={"SramBytes": 24672, "Tile": [256, 128]}):
                xv = ark.matmul(x_norm, self.wv.weight, transpose_other=True)
            xv = ark.reshape(
                xv, [bsz, seqlen, self.n_local_kv_heads, self.head_dim]
            )

        def calc_scores(xq_scratch, xk_scratch, mask):
            xq = xq_scratch[:, :, 0, :]
            xk = xk_scratch[:, :, 0, :]
            xq = ark.reshape(
                xq, [bsz, self.n_local_heads, seqlen, self.head_dim]
            )
            xk = ark.reshape(
                xk, [bsz, self.n_local_kv_heads, seqlen, self.head_dim]
            )
            with Context(
                sync=False,
                config={
                    "Tile": [256, 128],
                    "SramBytes": 24672,
                    "NumWarps": 4,
                    "BatchStrideCA": self.head_dim,
                    "BatchStrideNA": (
                        self.n_local_heads * seqlen * self.head_dim
                    ),
                    "BatchStrideCB": self.head_dim,
                    "BatchStrideNB": (
                        self.n_local_kv_heads * seqlen * self.head_dim
                    ),
                },
            ):
                scores = ark.matmul(xq, xk, transpose_other=True)
                scores = ark.mul(scores, 1.0 / math.sqrt(self.head_dim), scores)
                if mask is not None:
                    scores = ark.add(scores, mask, scores)
            return scores

        def softmax(scores):
            with Context(
                sram_range=[0, 0],
                sync=False,
                config={
                    "NumWarps": 1,
                    "SramBytes": 0,
                },
            ):
                with Context(config={"ImplType": "WarpWise", "Tile": [1]}):
                    max = ark.reduce_max(scores, axis=-1)
                with Context(config={"Tile": [1, 2048]}):
                    tmp = ark.sub(scores, max)
                    tmp = ark.exp(tmp)
                with Context(config={"ImplType": "WarpWise", "Tile": [1]}):
                    sum = ark.reduce_sum(tmp, axis=-1)
                with Context(config={"Tile": [1, 2048]}):
                    output = ark.div(tmp, sum)
            return output

        scores = calc_scores(xq_scratch, xk_scratch, mask)
        scores = softmax(scores)

        output_scratch = ark.tensor(
            [
                bsz,
                seqlen * self.n_local_heads,
                self.n_local_heads,
                self.head_dim,
            ],
            dtype=self.dtype,
        )
        with Context(
            sync=False,
            config={
                "Tile": [256, 128],
                "SramBytes": 24672,
                "NumWarps": 4,
                "BatchStrideCB": self.head_dim,
                "BatchStrideNB": self.n_local_kv_heads * seqlen * self.head_dim,
                "BatchStrideCC": self.head_dim,
                "BatchStrideNC": self.n_local_kv_heads * seqlen * self.head_dim,
            },
        ):
            xv = ark.reshape(xv[:, :, 0, :], [bsz, 1, seqlen, self.head_dim])
            output = ark.reshape(
                output_scratch[:, :, 0, :],
                [bsz, self.n_local_heads, seqlen, self.head_dim],
            )
            output = ark.matmul(scores, xv, output)
            output = ark.identity(
                output_scratch[:, :seqlen, :, :], deps=[output]
            )

        output = ark.reshape(
            output, [bsz, seqlen, self.head_dim * self.n_local_heads]
        )
        with Context(
            config={
                "NumWarps": 4,
                "Tile": [256, 128],
                "SramBytes": 24672,
            },
            sync=False,
        ):
            output = self.wo(output)
            return ark.add(x, output)


class TransformerBlock(ark.Module):
    def __init__(
        self,
        layer_id: int,
        args: ModelArgs,
        dtype: ark.DataType = ark.fp16,
        local_rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args, dtype, local_rank, world_size)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            dtype=dtype,
            local_rank=local_rank,
            world_size=world_size,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps, dtype=dtype)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps, dtype=dtype)

    def forward(
        self,
        x: ark.Tensor,
        start_pos: int,
        freqs_cis: ark.Tensor,
        mask: Optional[ark.Tensor],
    ):
        h = self.attention.forward(
            x, start_pos, freqs_cis, mask, self.attention_norm
        )
        return self.feed_forward(h, self.ffn_norm)


class Transformer(ark.Module):
    def __init__(
        self,
        params: ModelArgs,
        dtype: ark.DataType = ark.fp16,
        local_rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = ParallelEmbedding(
            params.vocab_size, params.dim, dtype, local_rank, world_size
        )

        self.layers = []
        for layer_id in range(self.n_layers):
            self.layers.append(
                TransformerBlock(
                    layer_id, params, dtype, local_rank, world_size
                )
            )
            self.register_module(f"layers.{layer_id}", self.layers[layer_id])
        self.norm = RMSNorm(params.dim, eps=params.norm_eps, dtype=dtype)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, dtype, True, local_rank, world_size
        )

    def forward(
        self,
        tokens: ark.Tensor,
        start_pos: int,
        freqs_cis: ark.Tensor,
        mask: Optional[ark.Tensor],
    ):
        with Context(warp_range=[0, 8], sram_range=[0, 49344]):
            h = self.tok_embeddings(tokens)

            for layer in self.layers:
                h = layer(h, start_pos, freqs_cis, mask)
            h = self.norm(h)
            with Context(
                config={"Tile": [256, 128], "SramBytes": 24672, "NumWarps": 4}
            ):
                output = self.output(h)
            return output
