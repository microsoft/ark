# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import numpy as np
import torch.nn as nn
import ark
import unittest

d_model = 64
d_ff = 128

batch_size = 1

import ark


class Optimizer:
    def __init__(self, module: ark.Module, lr: float):
        self.module = module
        self.lr = lr

    def step(self, module: ark.Module = None):
        if module == None:
            module = self.module
        print("step module: ", module)
        for param_name in module.parameters:
            print("param_name: ", param_name, "grads: ", module.grads)
            param = module.parameters[param_name]
            grads = module.grads[param_name]
            print("grads shape: ", grads.shape)
            print("param shape: ", param.shape)
            grads = ark.reshape(grads, param.shape)
            grads_scale = ark.scale(grads, -1.0 * self.lr)
            param_identity = ark.identity(param)
            print(
                param.shape,
                grads.shape,
                grads_scale.shape,
                param_identity.shape,
            )
            ark.add(param, grads_scale, param_identity)
        for sub_module_name in module.sub_modules:
            print("sub_module_name: ", sub_module_name)
            self.step(module.sub_modules[sub_module_name])


class loss_fn(ark.Module):
    def __init__(self):
        self.other_parameter = {}

    def forward(self, outputs: ark.Tensor, labels: ark.Tensor):
        neg_ground_truth = ark.scale(labels, -1.0)
        diff = ark.add(outputs, neg_ground_truth)
        self.other_parameter["diff"] = diff
        diff1 = ark.scale(diff, 1.0)
        mse = ark.mul(diff, diff1)
        # If batch_size is larger than 1, we need to sum the loss over the batch dimension.
        if batch_size > 1:
            mse = ark.reduce_sum(mse, axis=0)
        mse = ark.reshape(mse, outputs.shape[1:])
        return mse

    def backward(self, loss):
        diff = self.other_parameter["diff"]
        grad_diff = ark.reshape(diff, diff.shape[1:])
        return grad_diff


class fully_connected_layer(ark.Module):
    def __init__(self, input_size, output_size):
        super(fully_connected_layer, self).__init__()
        self.weight = ark.tensor([input_size, output_size], ark.TensorType.FP16)
        self.bias = ark.tensor([1, output_size], ark.TensorType.FP16)
        self.other_parameter = {}

    def forward(self, inputs):
        self.other_parameter["inputs"] = inputs
        output = ark.matmul(inputs, self.weight)
        output = ark.add(output, self.bias)
        print("output: ", output.shape)
        return output

    def backward(self, grads_output):
        grad_bias = grads_output
        inputs = self.other_parameter["inputs"]
        print("inputs: ", inputs.shape, "grads_output: ", grads_output.shape)
        grad_weight = ark.matmul(inputs, grads_output, transpose_a=True)
        grad_input = ark.matmul(grads_output, self.weight, transpose_b=True)
        print("grad_weight: ", grad_weight.shape)
        print("self.weight: ", self.weight.shape)
        grad_weight = ark.reshape(grad_weight, self.weight.shape)
        grad_bias = ark.reshape(grad_bias, self.bias.shape)
        self.grads["weight"] = grad_weight
        self.grads["bias"] = grad_bias
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
        self.input = ark.tensor([batch_size, d_model], ark.TensorType.FP16)
        self.label = ark.tensor([batch_size, d_model], ark.TensorType.FP16)
        output = self.module(self.input)
        loss = self.loss_fn(output, self.label)
        grad_loss = self.loss_fn.backward(loss)
        print("grad_loss: ", grad_loss.shape)
        self.module.backward(grad_loss)
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
        [batch_size, d_model], ark.TensorType.FP16
    )
    ark_model = TestModelARK()
    trainer = Trainer(ark_model, loss_fn(), Optimizer(ark_model, 0.001))
    ark.launch()

    trainer.train(10)

    ark.destroy()
    print("ARK module test")


if __name__ == "__main__":
    test_TestModel()
