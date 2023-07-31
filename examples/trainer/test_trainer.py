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

from ark import Executor, Model, Module, Tensor


class Optimizer:
    def __init__(self, module: Module, lr: float):
        self.module = module
        self.model = module.model
        self.lr = lr

    def step(self, module=None):
        for param in module.parameters:
            grads = module.grads[param]
            grads_scale = self.model.scale(grads, -1.0 * self.lr)
            param_identity = self.model.identity(param)
            self.model.add(param, grads_scale, param_identity)
        for module in module._sub_modules:
            self.step(module)


class loss_fn:
    def forward(self, outputs: Tensor, labels: Tensor):
        return self.model.cross_entropy(outputs, labels)


class Trainer:
    def __init__(
        self,
        module: Module,
        loss_fn: loss_fn,
        optimizer: Optimizer,
        executor: Executor,
    ):
        self.module = module
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.executor = executor

    def trainer_init(self, inputs, labels):
        outputs = self.module(inputs)
        loss = self.loss_fn(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss

    def train(self, iter):
        self.executor.launch()
        self.executor.run(iter)
        elapsed_msec = self.executor.stop()
        print("Training time: ", elapsed_msec, "ms")


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


def test_TestModel():
    ark.init_model()

    input_tensor = ark.tensor(
        ark.Dims(batch_size, seq_len, d_model), ark.TensorType.FP16
    )
    ark_model = TestModelARK()
    output_tensor = ark_model(input_tensor)

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
    state_dict = {"weight_1": weight_1_host, "weight_2": weight_2_host}

    ark_model.load_state_dict(state_dict)
    ark.run()

    output_tensor_host = ark.tensor_memcpy_device_to_host(None, output_tensor)

    input_tensor_host_float32 = input_tensor_host.astype(np.float32)

    torch_input = torch.from_numpy(input_tensor_host_float32)
    
    ark.destroy()
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

class TestAPI(unittest.TestCase):
    def test_api(self):
        test_TestModel()


if __name__ == "__main__":
    unittest.main()
