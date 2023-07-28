# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import numpy as np
import torch.nn as nn
import ark

d_model = 512  # Dimension of word embeddings
d_ff = 2048  # Dimension of the hidden layer in the feed-forward network

batch_size = 1
seq_len = 64

import ark


class TestModelARK(ark.Module):
    def __init__(self):
        super(TestModelARK, self).__init__()
        self.weight_1 = ark.tensor(ark.Dims(d_model, d_ff), ark.TensorType.FP16)
        self.weight_2 = ark.tensor(ark.Dims(d_ff, d_model), ark.TensorType.FP16)

    def forward(self, inputs):
        middle_result = ark.matmul(inputs, self.weight_1, is_relu=True)
        middle_result1 = ark.matmul(middle_result, self.weight_2)
        output = ark.add(middle_result1, inputs)
        output_layernorm = ark.layernorm(output)
        return output_layernorm


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
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


def test_TestModel(state_dict_file):
    ark.init_model()

    input_tensor = ark.tensor(
        ark.Dims(batch_size, seq_len, d_model), ark.TensorType.FP16
    )
    ark_model = TestModelARK()
    output_tensor = ark_model(input_tensor)
    # Test the mul method

    ark.launch()

    input_tensor_host = (
        (np.random.rand(batch_size, seq_len, d_model) - 0.5) * 0.1
    ).astype(np.float16)

    ark.tensor_memcpy_host_to_device(input_tensor, input_tensor_host)

    weight_1_host = ((np.random.rand(d_model, d_ff) - 0.5) * 0.1).astype(
        np.float16
    )
    weight_2_host = ((np.random.rand(d_ff, d_model) - 0.5) * 0.1).astype(
        np.float16
    )
    # if os.path.exists(state_dict_file):
    #     print("load state dict from file", state_dict_file)
    #     # state_dict = ark.load(state_dict_file)
    # else:
    state_dict = {"weight_1": weight_1_host, "weight_2": weight_2_host}
    print("state_dict: ", state_dict)
    ark_model.load_state_dict(state_dict)
    ark.run()
    print(ark_model.state_dict())
    # ark.save(ark_model.state_dict(), state_dict_file)

    output_tensor_host = ark.tensor_memcpy_device_to_host(None, output_tensor)

    input_tensor_host_float32 = input_tensor_host.astype(np.float32)

    torch_input = torch.from_numpy(input_tensor_host_float32)

    torch_model = TestModel()

    torch_model.load_state_dict(ark.convert_state_dict(state_dict, "torch"))

    gt = torch_model(torch_input).detach().numpy().astype(np.float16)

    # test if the result is correct
    max_error = np.max(np.abs(output_tensor_host - gt))
    avg_error = np.mean(np.abs(output_tensor_host - gt))
    print(input_tensor_host)
    print(output_tensor_host)
    print(gt)
    print("ARK module test")
    print(
        "batch_size:",
        batch_size,
        "seq_len:",
        seq_len,
        "d_model:",
        d_model,
        "d_ff:",
        d_ff,
    )
    print("max error: ", max_error, "avg error: ", avg_error)


if __name__ == "__main__":
    # state_dict_file =
    # print("state_dict_file: ", state_dict_file)
    # if state_dict_file == None:
    #     print("Usage: python module_test.py state_dict_file")
    #     exit(-1)
    test_TestModel(None)
