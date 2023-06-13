# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import torch
import numpy as np
import torch.nn as nn

d_model = 512  # Dimension of word embeddings
d_ff = 2048  # Dimension of the hidden layer in the feed-forward network
d_k = d_v = 64  # Dimensions of K(=Q) and V in the attention mechanism
n_layers = 6  # Number of encoder and decoder layers
n_heads = 8  # Number of heads in Multi-Head Attention set to 8

batch_size = 1
seq_len = 32


class PoswiseFeedForwardNetPytorch(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNetPytorch, self).__init__()
        self.weight_1 = nn.Parameter(torch.FloatTensor(d_model, d_ff))
        self.weight_2 = nn.Parameter(torch.FloatTensor(d_ff, d_model))

    # inputs: [batch_size, seq_len, d_model]
    def forward(self, inputs):
        output = torch.matmul(
            inputs, self.weight_1
        )  # [batch_size, seq_len, d_ff]
        output = nn.ReLU()(output)
        output = torch.matmul(
            output, self.weight_2
        )  # [batch_size, seq_len, d_model]
        output = nn.LayerNorm(d_model)(
            output + inputs
        )  # [batch_size, seq_len, d_model]
        return output

    def init_model(self, param):
        self.weight_1.data.copy_(torch.from_numpy(param["weight_1"]))
        self.weight_2.data.copy_(torch.from_numpy(param["weight_2"]))


class PoswiseFeedForwardNetArk:
    def __init__(self, model):
        self.model = model
        self.weight_1 = model.tensor(
            ark.Dims(d_model, d_ff), ark.TensorType.FP16
        )
        self.weight_2 = model.tensor(
            ark.Dims(d_ff, d_model), ark.TensorType.FP16
        )

    def forward(self, inputs):
        middle_result = self.model.matmul(inputs, self.weight_1, is_relu=True)
        middle_result1 = self.model.matmul(middle_result, self.weight_2)
        output = self.model.add(middle_result1, inputs)
        output_layernorm = self.model.layer_norm(output)
        return output_layernorm

    def init_model(self, param, exe):
        exe.tensor_memcpy_host_to_device(self.weight_1, param["weight_1"])
        exe.tensor_memcpy_host_to_device(self.weight_2, param["weight_2"])


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(
        self, Q, K, V, attn_mask=None
    ):  # Q: [batch_size, n_heads, len_q, d_k]
        # K: [batch_size, n_heads, len_k, d_k]
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
            d_k
        )  # scores : [batch_size, n_heads, len_q, len_k]
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class ScaledDotProductAttentionArk:
    def __init__(self, model):
        self.model = model

    def forward(self, Q, K, V, attn_mask=None):
        K_transpose = self.model.transpose(K, ark.Dims(0,2, 1))
        print(K_transpose.shape())
        scores = self.model.matmul(Q, K_transpose)
        scores = self.model.scale(scores, 1 / np.sqrt(d_k))
        if attn_mask is not None:
            scores = self.model.add(scores, attn_mask, alpha=-1e9)
        attn = self.model.softmax(scores)
        context = self.model.matmul(scores, V)
        return context, attn
