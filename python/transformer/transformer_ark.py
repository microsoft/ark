# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from transformer_utils import *


def attn_pad_mask_init(attention_mask, exe, seq_k_len):
    # attention_mask: [batch_size * n_heads, len_q, len_k]
    attention_mask_shape = attention_mask.shape()
    attention_mask_host = np.zeros(
        (
            attention_mask_shape[0],
            attention_mask_shape[1],
            attention_mask_shape[2],
        ),
        dtype=np.float16,
    )
    for i in range(attention_mask_shape[0]):
        for j in range(attention_mask_shape[1]):
            for k in range(attention_mask_shape[2]):
                if k >= seq_k_len:
                    attention_mask_host[i][j][k] = -1000

    exe.tensor_memcpy_host_to_device(attention_mask, attention_mask_host)


class PoswiseFeedForwardNet:
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

    def init_model(self, param, exe, prefix=""):
        exe.tensor_memcpy_host_to_device(
            self.weight_1, param[prefix + "weight_1"]
        )
        exe.tensor_memcpy_host_to_device(
            self.weight_2, param[prefix + "weight_2"]
        )


class ScaledDotProductAttention:
    def __init__(self, model):
        self.model = model

    def forward(self, Q, K, V, attn_mask=None):
        # Q: [batch_size, n_heads, len_q, d_k]
        # K: [batch_size, n_heads, len_k, d_k]
        # V: [batch_size, n_heads, len_v(=len_k), d_v]

        K_transpose = self.model.transpose(K, ark.Dims(0, 1, 3, 2))
        # reshape K_transpose to [batch_size * n_heads, d_k, len_k]
        K_transpose_shape = K_transpose.shape()
        K_transpose_reshape = self.model.reshape(
            K_transpose,
            ark.Dims(
                K_transpose_shape[0] * K_transpose_shape[1],
                K_transpose_shape[2],
                K_transpose_shape[3],
            ),
        )
        # reshape Q to [batch_size * n_heads, len_q, d_k]
        Q_shape = Q.shape()
        Q_reshape = self.model.reshape(
            Q, ark.Dims(Q_shape[0] * Q_shape[1], Q_shape[2], Q_shape[3])
        )
        # scores: [batch_size * n_heads, len_q, len_k]
        scores = self.model.matmul(Q_reshape, K_transpose_reshape)
        scores_scale = self.model.scale(scores, 1 / np.sqrt(d_k))
        if attn_mask is not None:
            scores_scale = self.model.add(scores_scale, attn_mask)
        attn = self.model.softmax(scores_scale)

        # reshape V to [batch_size * n_heads, len_v, d_v]
        V_shape = V.shape()
        V_reshape = self.model.reshape(
            V, ark.Dims(V_shape[0] * V_shape[1], V_shape[2], V_shape[3])
        )

        context = self.model.matmul(attn, V_reshape)
        return context, attn


class MultiHeadAttention:
    def __init__(self, model):
        self.model = model
        self.W_Q = model.tensor(
            ark.Dims(d_model, d_k * n_heads), ark.TensorType.FP16
        )
        self.W_K = model.tensor(
            ark.Dims(d_model, d_k * n_heads), ark.TensorType.FP16
        )
        self.W_V = model.tensor(
            ark.Dims(d_model, d_v * n_heads), ark.TensorType.FP16
        )
        self.fc = model.tensor(
            ark.Dims(d_v * n_heads, d_model), ark.TensorType.FP16
        )
        self.scaled_dot_product_attention = ScaledDotProductAttention(model)

    def forward(self, input_Q, input_K, input_V, attn_mask=None):
        # input_Q: [batch_size, len_q, d_model]
        # input_K: [batch_size, len_k, d_model]
        # input_V: [batch_size, len_v(=len_k), d_model]
        # attn_mask: [batch_size, seq_len, seq_len]
        # residual: [batch_size, len_q, d_model]
        batch_size = input_Q.shape()[0]
        len_q = input_Q.shape()[1]
        # Q: [batch_size, len_q, n_heads * d_k]
        Q = self.model.matmul(input_Q, self.W_Q)
        # Q: [batch_size, len_q, n_heads, d_k]
        Q = self.model.reshape(Q, ark.Dims(batch_size, len_q, n_heads, d_k))
        # Q: [batch_size, n_heads, len_q, d_k]
        Q = self.model.transpose(Q, ark.Dims(0, 2, 1, 3))

        len_k = input_K.shape()[1]
        # K: [batch_size, len_k, n_heads * d_k]
        K = self.model.matmul(input_K, self.W_K)
        # K: [batch_size, len_k, n_heads, d_k]
        K = self.model.reshape(K, ark.Dims(batch_size, len_k, n_heads, d_k))
        # K: [batch_size, n_heads, len_k, d_k]
        K = self.model.transpose(K, ark.Dims(0, 2, 1, 3))

        len_v = input_V.shape()[1]
        # V: [batch_size, len_v(=len_k), n_heads * d_v]
        V = self.model.matmul(input_V, self.W_V)
        # V: [batch_size, len_v(=len_k), n_heads, d_v]
        V = self.model.reshape(V, ark.Dims(batch_size, len_v, n_heads, d_v))
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        V = self.model.transpose(V, ark.Dims(0, 2, 1, 3))

        context, attn = self.scaled_dot_product_attention.forward(
            Q, K, V, attn_mask
        )

        # context: [batch_size, n_heads, len_q, d_v]
        context1 = self.model.reshape(
            context, ark.Dims(batch_size, n_heads, len_q, d_v)
        )

        # context: [batch_size, len_q, n_heads, d_v]
        context2 = self.model.transpose(context1, ark.Dims(0, 2, 1, 3))

        context3 = self.model.reshape(
            context2, ark.Dims(batch_size, len_q, n_heads * d_v)
        )  # context: [batch_size, len_q, n_heads * d_v]

        # output: [batch_size, len_q, d_model]
        output = self.model.matmul(context3, self.fc)
        output_plus_residual = self.model.add(output, input_Q)
        output_layernorm = self.model.layernorm(output_plus_residual)
        return output_layernorm, attn

    def init_model(self, param, exe, prefix=""):
        exe.tensor_memcpy_host_to_device(self.W_Q, param[prefix + "W_Q"])
        exe.tensor_memcpy_host_to_device(self.W_K, param[prefix + "W_K"])
        exe.tensor_memcpy_host_to_device(self.W_V, param[prefix + "W_V"])
        exe.tensor_memcpy_host_to_device(self.fc, param[prefix + "fc"])


class EncoderLayer:
    def __init__(self, model):
        self.model = model
        self.enc_self_attn = MultiHeadAttention(
            model
        )  # Multi-Head Attention mechanism
        self.pos_ffn = PoswiseFeedForwardNet(model)

    def forward(self, enc_inputs, enc_self_attn_mask=None):
        enc_outputs, attn = self.enc_self_attn.forward(
            enc_inputs,
            enc_inputs,
            enc_inputs,
            # enc_outputs: [batch_size, src_len, d_model],
            enc_self_attn_mask,
        )  # attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs1 = self.pos_ffn.forward(
            enc_outputs
        )  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs1, attn

    def init_model(self, param, exe, prefix=""):
        self.enc_self_attn.init_model(param, exe, prefix + "enc_self_attn.")
        self.pos_ffn.init_model(param, exe, prefix + "pos_ffn.")
