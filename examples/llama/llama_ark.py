# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import math
from dataclasses import dataclass
from typing import Optional

ark_type = ark.FP16
local_rank = 0
world_size = 1


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
        input_shape = x.shape
        # ColumnParallelLinear only support 3D tensor
        assert len(input_shape) == 3
        batch_size = input_shape[0]
        output_trans_shape = [self.out_dim, input_shape[0] * input_shape[1]]

        # (out_dim, batch_size * seq_len)
        output_trans_tensor = ark.tensor(output_trans_shape, ark_type)

        # (out_dim // world_size, batch_size * seq_len) for each rank
        output_trans_tensor_shards = ark.sharding(
            output_trans_tensor, 0, self.out_dim // world_size
        )

        # (batch_size * seq_len, in_dim)
        x = ark.reshape(x, [input_shape[0] * input_shape[1], input_shape[2]])

        # (out_dim // world_size, in_dim) * (batch_size * seq_len, in_dim)^T =
        # (out_dim // world_size, batch_size * seq_len)
        ark.matmul(
            self.weight,
            x,
            output_trans_tensor_shards[local_rank],
            transpose_b=True,
        )
        output_trans_tensor_shards = ark.all_gather(
            output_trans_tensor_shards[local_rank],
            local_rank,
            world_size,
            output_trans_tensor_shards,
        )
        # Here the output_trans_tensor should have the dependence on the all_gather operator
        output_trans_tensor = ark.identity(
            output_trans_tensor, output_trans_tensor_shards
        )
        # Currently we only support transpose on 4D tensor
        # (1, 1, out_dim, batch_size * seq_len)
        output_trans_tensor = ark.reshape(
            output_trans_tensor,
            [1, 1, output_trans_tensor.shape[0], output_trans_tensor.shape[1]],
        )
        # (1, 1, batch_size * seq_len, out_dim)
        output_tensor = ark.transpose(output_trans_tensor, [0, 1, 3, 2])
        # (batch_size, seq_len, out_dim)
        output_tensor = ark.reshape(
            output_tensor, [input_shape[0], input_shape[1], self.out_dim]
        )
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
