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
        self.lr = lr

    def step(self):
        for param in self.module.parameters:
            grads = module.grads[param]
            grads_scale = ark.scale(grads, -1.0 * self.lr)
            param_identity = ark.identity(param)
            ark.add(param, grads_scale, param_identity)
        for module in module._sub_modules:
            self.step(module)


class loss_fn(ark.Module):
    def __init__(self):
        pass
        # self.loss = ark.tensor(ark.Dims(1), ark.TensorType.FP16)

    def forward(self, outputs: ark.Tensor, labels: ark.Tensor):
        neg_ground_truth = ark.scale(labels, -1.0)
        diff = ark.add(outputs, neg_ground_truth)
        diff1 = ark.scale(diff, 1.0)
        mse = ark.mul(diff, diff1)
        return mse

    def backward(self, loss):
        grad_diff = ark.scale(loss, 2.0)
        return grad_diff


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
        self.module1 = fully_connected_layer(d_model, d_ff)
        self.module2 = fully_connected_layer(d_ff, d_model)

    def forward(self, inputs):
        output = self.module1(inputs)
        output = self.module2(output)
        return output

    def backward(self, grads):
        grad_module2 = self.module2.backward(grads)
        grad_module1 = self.module1.backward(grad_module2)
        return grad_module1


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
        self.input = ark.tensor(
            [batch_size, seq_len, d_model], ark.TensorType.FP16
        )
        self.label = ark.tensor(
            [batch_size, seq_len, d_model], ark.TensorType.FP16
        )
        output = self.module(self.input)
        loss = self.loss_fn(output, self.label)
        self.loss_fn.backward(loss)
        self.optimizer.step()

    def trainer_init(self, inputs, labels):
        # Initialize the input and label tensors
        
        return

    def train(self, iter):
        for i in range(iter):
            ark.run()
            loss = self.get_loss()
            print("loss:", loss)

    def get_loss(self):
        loss = ark.tensor_mempcy_device_to_host(None, self.loss_fn.loss)


def test_TestModel():
    ark.init_model()

    input_tensor = ark.tensor(
        ark.Dims(batch_size, seq_len, d_model), ark.TensorType.FP16
    )
    ark_model = TestModelARK()
    trainer = Trainer(ark_model, loss_fn(), Optimizer(ark_model, 0.001))
    ark.launch()

    trainer.train(10)

    ark.destroy()
    print("ARK module test")


if __name__ == "__main__":
    test_TestModel()
