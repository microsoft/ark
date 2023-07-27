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

for name in dir(ark.Model):
    if not name.startswith("__") and callable(getattr(ark.Model, name)):
        print(name)


class TestModelARK(ark.Module):
    def __init__(self, model):
        super(TestModelARK, self).__init__(model)
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
    ark.init()

    # Create a Model instance
    model = ark.Model()

    input_tensor = model.tensor(
        ark.Dims(batch_size, seq_len, d_model), ark.TensorType.FP16
    )
    print("input_tensor type: ", type(input_tensor))
    print("input_tensor: ", input_tensor.shape())
    ark_model = TestModelARK(model)
    output_tensor = ark_model.forward(input_tensor)
    # Test the mul method
    exe = ark.Executor(0, 0, 1, model, "test_TestModel")

    exe.compile()
    input_tensor_host = (
        (np.random.rand(batch_size, seq_len, d_model) - 0.5) * 0.1
    ).astype(np.float16)

    exe.launch()
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
    exe.run(1)
    exe.stop()
    print(ark_model.state_dict())
    # ark.save(ark_model.state_dict(), state_dict_file)

    output_tensor_host = ark.tensor_memcpy_device_to_host(None, output_tensor)

    input_tensor_host_float32 = input_tensor_host.astype(np.float32)

    torch_input = torch.from_numpy(input_tensor_host_float32)

    torch_model = TestModel()

    torch_model.load_state_dict(ark.convert_state_dict_to_pytorch(state_dict))

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
