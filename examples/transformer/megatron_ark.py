# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from transformer_utils import *
import transformer_ark


# PoswiseFeedForwardNet that uses tensor parallelism
class PoswiseFeedForwardNet:
    def __init__(self, model, rank):
        self.model = model
        self.rank = rank
        # The weight_1 and weight_2 are split into num_gpu parts, using tensor parallelism
        self.weight_1 = model.tensor(
            ark.Dims(d_model, d_ff // num_gpu), ark.TensorType.FP16
        )
        self.weight_2 = model.tensor(
            ark.Dims(d_ff // num_gpu, d_model), ark.TensorType.FP16
        )

    def forward(self, inputs):
        middle_result = self.model.matmul(inputs, self.weight_1, is_relu=True)
        middle_result1 = self.model.matmul(middle_result, self.weight_2)
        middle_result1 = self.model.reshape(
            middle_result1, ark.Dims(batch_size * seq_len * d_model)
        )
        middle_result_allreduced = self.model.all_reduce(
            middle_result1, self.rank, num_gpu
        )
        # all_reduce the middle_result to all the GPUs
        middle_result_allreduced = self.model.reshape(
            middle_result_allreduced, ark.Dims(batch_size, seq_len, d_model)
        )
        output = self.model.add(middle_result_allreduced, inputs)

        output_layernorm = self.model.layernorm(output)
        return output_layernorm

    def init_model(self, param, exe, prefix=""):
        weight_1 = param[prefix + "weight_1"]
        weight_1_shared = np.split(weight_1, num_gpu, axis=1)[self.rank]
        print(
            "weight_1_shared.flags['C_CONTIGUOUS']",
            weight_1_shared.flags["C_CONTIGUOUS"],
        )
        weight_1_shared_copy = weight_1_shared.copy()
        print(
            "weight_1_shared_copy.flags['C_CONTIGUOUS']",
            weight_1_shared_copy.flags["C_CONTIGUOUS"],
        )
        exe.tensor_memcpy_host_to_device(self.weight_1, weight_1_shared_copy)
        weight_2 = param[prefix + "weight_2"]
        weight_2_shared = np.split(weight_2, num_gpu, axis=0)[self.rank]
        weight_2_shared_copy = weight_2_shared.copy()
        exe.tensor_memcpy_host_to_device(self.weight_2, weight_2_shared_copy)


# MultiHeadAttention that uses tensor parallelism, different heads are splited on different GPUs
# The final fc layer is also splited on different GPUs
class MultiHeadAttention:
    def __init__(self, model, rank):
        self.model = model
        self.rank = rank
        self.W_Q = model.tensor(
            ark.Dims(d_model, d_k * n_heads_per_gpu), ark.TensorType.FP16
        )
        self.W_K = model.tensor(
            ark.Dims(d_model, d_k * n_heads_per_gpu), ark.TensorType.FP16
        )
        self.W_V = model.tensor(
            ark.Dims(d_model, d_v * n_heads_per_gpu), ark.TensorType.FP16
        )
        self.fc = model.tensor(
            ark.Dims(d_v * n_heads_per_gpu, d_model), ark.TensorType.FP16
        )
        self.scaled_dot_product_attention = (
            transformer_ark.ScaledDotProductAttention(model)
        )

    def forward(self, input_Q, input_K, input_V, attn_mask=None):
        # input_Q: [batch_size, len_q, d_model]
        # input_K: [batch_size, len_k, d_model]
        # input_V: [batch_size, len_v(=len_k), d_model]
        # attn_mask: [batch_size, seq_len, seq_len]
        # residual: [batch_size, len_q, d_model]
        batch_size = input_Q.shape()[0]
        len_q = input_Q.shape()[1]

        # Q: [batch_size, len_q, n_heads_per_gpu * d_k]
        Q = self.model.matmul(input_Q, self.W_Q)
        # Q: [batch_size, len_q, n_heads_per_gpu, d_k]
        Q = self.model.reshape(
            Q, ark.Dims(batch_size, len_q, n_heads_per_gpu, d_k)
        )
        # Q: [batch_size, n_heads_per_gpu, len_q, d_k]
        Q = self.model.transpose(Q, ark.Dims(0, 2, 1, 3))

        len_k = input_K.shape()[1]
        # K: [batch_size, len_k, n_heads_per_gpu * d_k]
        K = self.model.matmul(input_K, self.W_K)
        # K: [batch_size, len_k, n_heads_per_gpu, d_k]
        K = self.model.reshape(
            K, ark.Dims(batch_size, len_k, n_heads_per_gpu, d_k)
        )
        # K: [batch_size, n_heads_per_gpu, len_k, d_k]
        K = self.model.transpose(K, ark.Dims(0, 2, 1, 3))

        len_v = input_V.shape()[1]
        # V: [batch_size, len_v(=len_k), n_heads_per_gpu * d_v]
        V = self.model.matmul(input_V, self.W_V)
        # V: [batch_size, len_v(=len_k), n_heads_per_gpu, d_v]
        V = self.model.reshape(
            V, ark.Dims(batch_size, len_v, n_heads_per_gpu, d_v)
        )
        # V: [batch_size, n_heads_per_gpu, len_v(=len_k), d_v]
        V = self.model.transpose(V, ark.Dims(0, 2, 1, 3))

        context, attn = self.scaled_dot_product_attention.forward(
            Q, K, V, attn_mask
        )

        # context: [batch_size, n_heads_per_gpu, len_q, d_v]
        context1 = self.model.reshape(
            context, ark.Dims(batch_size, n_heads_per_gpu, len_q, d_v)
        )

        # context: [batch_size, len_q, n_heads_per_gpu, d_v]
        context2 = self.model.transpose(context1, ark.Dims(0, 2, 1, 3))

        context3 = self.model.reshape(
            context2, ark.Dims(batch_size, len_q, n_heads_per_gpu * d_v)
        )  # context: [batch_size, len_q, n_heads_per_gpu * d_v]

        # output: [batch_size, len_q, d_model]
        output = self.model.matmul(context3, self.fc)
        output_reshape = self.model.reshape(
            output, ark.Dims(batch_size * len_q * d_model)
        )
        output_allreduce = self.model.all_reduce(
            output_reshape, self.rank, num_gpu
        )
        output_allreduce_reshape = self.model.reshape(
            output_allreduce, ark.Dims(batch_size, len_q, d_model)
        )
        output_plus_residual = self.model.add(output_allreduce_reshape, input_Q)
        output_layernorm = self.model.layernorm(output_plus_residual)
        return output_layernorm, attn
