# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from megatron_utils import *


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


class PoswiseFeedForwardNetTensorParallel:
    def __init__(self, model):
        self.model = model
        self.weight_1_shard = model.tensor(
            ark.Dims(d_model, d_ff // num_gpu), ark.TensorType.FP16
        )
        self.weight_2_shard = model.tensor(
            ark.Dims(d_ff // num_gpu, d_model), ark.TensorType.FP16
        )

    def forward(self, inputs, rank):
        middle_result = self.model.matmul(
            inputs, self.weight_1_shard, is_relu=True
        )
        middle_result1 = self.model.matmul(middle_result, self.weight_2_shard)
        middle_result1 = self.model.reshape(
            middle_result1, ark.Dims(batch_size * seq_len * d_model)
        )
        middle_result_allreduced = self.model.all_reduce(
            middle_result1, rank, num_gpu
        )
        middle_result_allreduced = self.model.reshape(
            middle_result_allreduced, ark.Dims(batch_size, seq_len, d_model)
        )
        output = self.model.add(middle_result_allreduced, inputs)

        output_layernorm = self.model.layernorm(output)
        return output_layernorm
