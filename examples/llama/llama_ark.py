# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple


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
        self.weight = ark.Parameter(ark.tensor([1, 1, dim], ark.FP32))

    def _norm(self, x):
        # x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        square = ark.mul(x, x)
        square_mean = ark.reduce_mean(square, axis=square.ndims() - 1)
        return ark.div(x, ark.sqrt(square_mean))

    def forward(self, x):
        output = self._norm(x)
        return ark.mul(output, self.weight)


class Linear(ark.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = ark.Parameter(ark.tensor([out_dim, in_dim], ark.FP32))

    def forward(self, x):
        return ark.matmul(x, self.weight, transpose_b=True)


class Silu(ark.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # We need to specify output tensor so that the sigmoid op will not be an in-place operator
        output = ark.tensor(x.shape, ark.FP32)
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

        self.w1 = Linear(dim, hidden_dim)
        self.w2 = Linear(hidden_dim, dim)
        self.w3 = Linear(dim, hidden_dim)

    def forward(self, x):
        # self.w2(F.silu(self.w1(x)) * self.w3(x))
        x1 = self.w1(x)
        x1 = Silu()(x1)
        x2 = self.w3(x)
        x3 = ark.mul(x1, x2)
        x4 = self.w2(x3)
        return x4


def apply_rotary_emb(xq, xk, freqs_cis):
    xq_shape = xq.shape
    xq_ = ark.reshape(
        xq, [xq_shape[0] * xq_shape[1], xq_shape[2], xq_shape[3] // 2, 2]
    )
    xq_split = ark.sharding(xq_, axis=3, dim_per_shard=1)

    xq_real = xq_split[0]
    xq_real = ark.reshape(
        xq_real, [xq_real.shape[0], xq_real.shape[1], xq_real.shape[2]]
    )
    xq_imag = xq_split[1]
    xq_imag = ark.reshape(
        xq_imag, [xq_imag.shape[0], xq_imag.shape[1], xq_imag.shape[2]]
    )
    xq_out = ark.tensor(xq_.shape, ark.FP32)
    xq_out_ = ark.reshape(
        xq_out, [xq_shape[0] * xq_shape[1], xq_shape[2], xq_shape[3] // 2, 2]
    )
    xq_out_shards = ark.sharding(xq_out_, axis=3, dim_per_shard=1)

    xq_out_real = xq_out_shards[0]
    xq_out_real = ark.reshape(
        xq_out_real,
        [xq_out_real.shape[0], xq_out_real.shape[1], xq_out_real.shape[2]],
    )
    xq_out_imag = xq_out_shards[1]
    xq_out_imag = ark.reshape(
        xq_out_imag,
        [xq_out_imag.shape[0], xq_out_imag.shape[1], xq_out_imag.shape[2]],
    )

    freqs_cis_shard = ark.sharding(freqs_cis, axis=3, dim_per_shard=1)
    freqs_cis_real = freqs_cis_shard[0]
    freqs_cis_real = ark.reshape(
        freqs_cis_real,
        [
            freqs_cis_real.shape[0],
            freqs_cis_real.shape[1],
            freqs_cis_real.shape[2],
        ],
    )
    freqs_cis_imag = freqs_cis_shard[1]
    freqs_cis_imag = ark.reshape(
        freqs_cis_imag,
        [
            freqs_cis_imag.shape[0],
            freqs_cis_imag.shape[1],
            freqs_cis_imag.shape[2],
        ],
    )
    # (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    tmp1 = ark.mul(xq_real, freqs_cis_real)
    tmp2 = ark.mul(xq_imag, freqs_cis_imag)
    ark.sub(tmp1, tmp2, xq_out_real)
    # ark.mul(xq_real, freqs_cis_imag, xq_out_imag)
    # ark.sub(
    #     ark.mul(xq_real, freqs_cis_real),
    #     ark.mul(xq_imag, freqs_cis_imag),
    #     xq_out_real,
    # )
    # ark.add(
    #     ark.mul(xq_real, freqs_cis_imag),
    #     ark.mul(xq_imag, freqs_cis_real),
    #     xq_out_imag,
    # )
    # change back to original shape
    xq_out = ark.reshape(xq_out_, xq_shape)
    return xq_out


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
        self.wq = Linear(args.dim, args.n_heads * self.head_dim)
        self.wk = Linear(args.dim, self.n_kv_heads * self.head_dim)
        self.wv = Linear(args.dim, self.n_kv_heads * self.head_dim)
        self.wo = Linear(args.n_heads * self.head_dim, args.dim)

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

        # self.tok_embeddings = Linear(
        #     params.vocab_size, params.dim, init_method=lambda x: x
        # )

        self.layers = []
        for layer_id in range(6):
            self.tmp_layer = TransformerBlock(layer_id, params)
            self.layers.append(self.tmp_layer)

        # self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        # self.output = Linear(
        #     params.dim, params.vocab_size
        # )

        # self.freqs_cis = precompute_freqs_cis(
        #     self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        # )

    def forward(self, h: ark.Tensor, start_pos: int):
        freqs_cis = None

        mask = None

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        # h = self.norm(h)
        # output = self.output(h).float()
        return h
