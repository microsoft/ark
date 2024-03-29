# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import numpy as np
import torch.nn as nn
import ark
import unittest

d_model = 512
d_ff = 2048

batch_size = 1
seq_len = 64

import ark


def convert_state_dict(state_dict: dict, type="numpy"):
    """
    Convert the state_dict of a module to np.ndarray or torch.Tensor type
    """
    new_state_dict = {}
    for key in state_dict:
        if type == "torch":
            new_state_dict[key] = torch.from_numpy(state_dict[key])
        elif type == "numpy":
            new_state_dict[key] = state_dict[key].numpy()
    return new_state_dict


class TestModelARK(ark.Module):
    def __init__(self):
        super(TestModelARK, self).__init__()
        self.weight_1 = ark.parameter([d_model, d_ff], ark.fp16)
        self.weight_2 = ark.parameter([d_ff, d_model], ark.fp16)

    def forward(self, inputs):
        output = ark.matmul(inputs, self.weight_1)
        output = ark.relu(output)
        output = ark.matmul(output, self.weight_2)
        output = ark.add(output, inputs)
        output = ark.layernorm(output)
        return output


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


def test_TestModel():
    runtime = ark.Runtime()

    input_tensor = ark.tensor(
        ark.Dims(batch_size, seq_len, d_model), ark.TensorType.FP16
    )
    ark_model = TestModelARK()
    output_tensor = ark_model(input_tensor)
    # Test the mul method

    runtime.launch()

    input_tensor_host = (
        (np.random.rand(batch_size, seq_len, d_model) - 0.5) * 0.1
    ).astype(np.float16)

    input_tensor.from_numpy(input_tensor_host)

    weight_1_host = ((np.random.rand(d_model, d_ff) - 0.5) * 0.1).astype(
        np.float16
    )
    weight_2_host = ((np.random.rand(d_ff, d_model) - 0.5) * 0.1).astype(
        np.float16
    )
    state_dict = {"weight_1": weight_1_host, "weight_2": weight_2_host}

    ark_model.load_state_dict(state_dict)
    runtime.run()

    output_tensor_host = output_tensor.to_numpy()

    input_tensor_host_float32 = input_tensor_host.astype(np.float32)

    torch_input = torch.from_numpy(input_tensor_host_float32)

    torch_model = TestModel()

    torch_model.load_state_dict(convert_state_dict(state_dict, "torch"))

    gt = torch_model(torch_input).detach().numpy().astype(np.float16)

    # test if the result is correct
    max_error = np.max(np.abs(output_tensor_host - gt))
    avg_error = np.mean(np.abs(output_tensor_host - gt))
    ark_state_dict = ark_model.state_dict()
    for k, v in state_dict.items():
        np.testing.assert_allclose(v, ark_state_dict[k])
    # print(input_tensor_host)
    # print(output_tensor_host)
    # print(gt)
    print(
        f"ARK module test batch_size: {batch_size} seq_len: {seq_len} "
        f"d_model: {d_model} d_ff: {d_ff} max error: {max_error} "
        f"avg error: {avg_error}"
    )


class TestAPI(unittest.TestCase):
    def test_api(self):
        test_TestModel()


if __name__ == "__main__":
    unittest.main()
