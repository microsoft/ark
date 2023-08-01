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


class Optimizer:
    def __init__(self, module: ark.Module, lr: float):
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


class loss_fn(ark.Module):
    def __init__(self):
        pass
        # self.loss = ark.tensor(ark.Dims(1), ark.TensorType.FP16)

    def forward(self, outputs: ark.Tensor, labels: ark.Tensor):
        neg_ground_truth = self.model.scale(labels, -1.0)
        diff = self.model.add(outputs, neg_ground_truth)
        diff1 = self.model.scale(diff, 1.0)
        mse = self.model.mul(diff, diff1)
        return mse

    def backward(self, loss):
        grad_diff = self.model.scale(loss, 2.0)
        loss.backward()


class fully_connected_layer(ark.Module):
    def __init__(self, input_size, output_size):
        super(fully_connected_layer, self).__init__()
        self.weight = ark.tensor(
            ark.Dims(input_size, output_size), ark.TensorType.FP16
        )
        self.bias = ark.tensor(ark.Dims(output_size), ark.TensorType.FP16)

    def forward(self, inputs):
        output = ark.matmul(inputs, self.weight)
        output = ark.add(output, self.bias)
        return output

    def backward(self, loss):
        grad_bias = ark.scale(loss, 1.0)
        grad_weight = ark.matmul(loss, self.weight, transpose_a=True)
        grad_input = ark.matmul(loss, self.weight, transpose_b=True)
        self.grads[self.weight] = grad_weight
        self.grads[self.bias] = grad_bias
        return grad_input


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

    def backward(self, loss):
        loss.backward()


class Trainer:
    def __init__(
        self,
        module: ark.Module,
        loss_fn: loss_fn,
        optimizer: Optimizer,
    ):
        self.module = module
        self.loss_fn = loss_fn
        self.optimizer = optimizer

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

    def get_loss(self):
        loss = ark.tensor_mempcy_device_to_host(None, self.loss_fn.loss)


def test_TestModel():
    ark.init_model()

    input_tensor = ark.tensor(
        ark.Dims(batch_size, seq_len, d_model), ark.TensorType.FP16
    )
    ark_model = TestModelARK()
    output_tensor = ark_model(input_tensor)
    trainer = Trainer(ark_model, loss_fn(), Optimizer(ark_model, 0.001))
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
