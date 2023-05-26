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
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False))

    # inputs: [batch_size, seq_len, d_model]
    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return (output + residual)  # [batch_size, seq_len, d_model]


class PoswiseFeedForwardNetArk():
    def __init__(self, model):
        self.model = model
        self.weight_1 = model.tensor(
            ark.Dims(d_model, d_ff), ark.TensorType.FP16)
        self.weight_2 = model.tensor(
            ark.Dims(d_ff, d_model), ark.TensorType.FP16)

    def forward(self, inputs):
        middle_result = model.matmul(inputs, self.weight_1, is_relu=True)
        middle_result1 = model.matmul(middle_result, self.weight_2)
        output = model.add(middle_result1, inputs)
        # TODO: add layer norm
        return output


ark.init()

# Create a Model instance
model = ark.Model()

input_tensor = model.tensor(
    ark.Dims(batch_size, seq_len, d_model), ark.TensorType.FP16)

output_tensor = PoswiseFeedForwardNetArk(model).forward(input_tensor)

# Test the mul method
# multiplied_tensor = model.mul(t1, t2)
exe = ark.Executor(0, 0, 1, model, "test_python_bindings")
exe.compile()
input_tensor_host = np.random.rand(
    batch_size, seq_len, d_model).astype(np.float16)



exe.launch()
exe.tensor_memcpy_host_to_device(input_tensor, input_tensor_host)

exe.run(1)

exe.stop()

output_tensor_host = np.zeros((batch_size, seq_len, d_model), dtype=np.float16)

exe.tensor_memcpy_device_to_host(output_tensor_host, output_tensor)

input_tensor_host_float32 = input_tensor_host.astype(np.float32)

torch_input =  torch.from_numpy(input_tensor_host_float32)

torch_model = PoswiseFeedForwardNetPytorch()

gt = torch_model(torch_input).detach().numpy().astype(np.float16)

# test if the result is correct
assert np.allclose(output_tensor_host, gt)
print("ark test success")
