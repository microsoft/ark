# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from transformer_utils import *

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
        output_layernorm = self.model.layernorm(output)
        return output_layernorm

    def init_model(self, param, exe):
        exe.tensor_memcpy_host_to_device(self.weight_1, param["weight_1"])
        exe.tensor_memcpy_host_to_device(self.weight_2, param["weight_2"])





class ScaledDotProductAttentionArk:
    def __init__(self, model):
        self.model = model

    def forward(self, Q, K, V, attn_mask=None):
        # Q: [batch_size, n_heads, len_q, d_k]
        # K: [batch_size, n_heads, len_k, d_k]
        # V: [batch_size, n_heads, len_v(=len_k), d_v]

        K_transpose = self.model.transpose(K, ark.Dims(0, 1, 3, 2))
        # reshape K_transpose to [batch_size * n_heads, d_k, len_k]
        K_transpose_shape = K_transpose.shape()
        K_transpose_reshape = self.model.reshape(K_transpose, ark.Dims(
            K_transpose_shape[0]*K_transpose_shape[1], K_transpose_shape[2], K_transpose_shape[3]))
        # reshape Q to [batch_size * n_heads, len_q, d_k]
        Q_shape = Q.shape()
        Q_reshape = self.model.reshape(Q, ark.Dims(
            Q_shape[0]*Q_shape[1], Q_shape[2], Q_shape[3]))
        # scores: [batch_size * n_heads, len_q, len_k]
        scores = self.model.matmul(Q_reshape, K_transpose_reshape)
        scores_scale = self.model.scale(scores, 1 / np.sqrt(d_k))
        if attn_mask is not None:
            scores_scale = self.model.add(scores_scale, attn_mask, alpha=-1e9)
        attn = self.model.softmax(scores_scale)

        # reshape V to [batch_size * n_heads, len_v, d_v]
        V_shape = V.shape()
        V_reshape = self.model.reshape(V, ark.Dims(
            V_shape[0]*V_shape[1], V_shape[2], V_shape[3]))

        context = self.model.matmul(attn, V_reshape)
        return context, attn


