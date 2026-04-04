# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import ark

# Define the parameters of the model
batch_size = 1
seq_len = 128
d_model = 512
d_ff = 2048


class SubModuleARK(ark.Module):
    def __init__(self, weight_2):
        super(SubModuleARK, self).__init__()
        self.weight_2 = ark.Tensor.from_torch(weight_2)

    def forward(self, inputs):
        return ark.matmul(inputs, self.weight_2)


class TestModelARK(ark.Module):
    def __init__(self, weight_1, weight_2):
        super(TestModelARK, self).__init__()
        self.weight_1 = ark.Tensor.from_torch(weight_1)
        self.submodule = SubModuleARK(weight_2)

    def forward(self, inputs):
        output = ark.matmul(inputs, self.weight_1)
        output = ark.relu(output)
        output = self.submodule(output)
        output = ark.add(output, inputs)
        output = ark.layernorm(output)
        return output


class TestModelPytorch(nn.Module):
    def __init__(self):
        super(TestModelPytorch, self).__init__()
        self.weight_1 = nn.Parameter(torch.ones(d_model, d_ff, device="cuda:0"))
        self.submodule_weight_2 = nn.Parameter(
            torch.ones(d_ff, d_model, device="cuda:0")
        )
        self.layernorm = nn.LayerNorm(d_model, device="cuda:0")

    def forward(self, inputs):
        output = torch.matmul(inputs, self.weight_1)
        output = nn.ReLU()(output)
        output = torch.matmul(output, self.submodule_weight_2)
        output = self.layernorm(output + inputs)
        return output


def module_test():
    # Create torch tensors for input and weights
    input_tensor = (
        torch.randn(
            batch_size, seq_len, d_model, dtype=torch.float32, device="cuda:0"
        )
        * 0.1
    )
    weight_1 = (
        torch.randn(d_model, d_ff, dtype=torch.float32, device="cuda:0") * 0.1
    )
    weight_2 = (
        torch.randn(d_ff, d_model, dtype=torch.float32, device="cuda:0") * 0.1
    )

    # Build and evaluate the ARK model
    ark_model = TestModelARK(weight_1, weight_2)
    output = ark_model(input_tensor).eval()

    # Compute PyTorch ground truth
    torch_model = TestModelPytorch()
    torch_model.load_state_dict(
        {"weight_1": weight_1, "submodule_weight_2": weight_2},
        strict=False,
    )
    gt = torch_model(input_tensor)

    # Compare results
    max_error = (output - gt).abs().max().item()
    avg_error = (output - gt).abs().mean().item()

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
    print("max error:", max_error, "avg error:", avg_error)


if __name__ == "__main__":
    ark.init()
    module_test()
