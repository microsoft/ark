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
        for param_name in module.parameters:
            param = module.parameters[param_name]
            grads = module.grads[param_name]

            grads = ark.reshape(grads, param.shape)
            grads_scale = ark.scale(grads, -1.0 * self.lr)
            param_identity = ark.identity(param)

            ark.add(param, grads_scale, param_identity)
        for sub_module_name in module.sub_modules:
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
        self.other_parameter["loss"] = mse
        return mse

    def backward(self, loss):
        diff = self.other_parameter["diff"]
        return diff


class matmul_layer(ark.Module):
    def __init__(self, input_size, output_size):
        super(matmul_layer, self).__init__()
        self.other_parameter = {}
        self.weight = ark.Parameter(
            ark.tensor([input_size, output_size], ark.TensorType.FP16)
        )

    def forward(self, inputs):
        self.other_parameter["inputs"] = inputs
        output = ark.matmul(inputs, self.weight)
        return output

    def backward(self, grads_output):
        inputs = self.other_parameter["inputs"]
        grad_weight = ark.matmul(inputs, grads_output, transpose_a=True)
        grad_input = ark.matmul(grads_output, self.weight, transpose_b=True)
        self.grads["weight"] = grad_weight
        return grad_input


class TestModelARK(ark.Module):
    def __init__(self):
        super(TestModelARK, self).__init__()
        self.module1 = matmul_layer(d_model, d_ff)
        self.module2 = matmul_layer(d_ff, d_model)

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
        self.module.backward(grad_loss)
        self.optimizer.step()

    def trainer_init(self, inputs, labels):
        # Initialize the input and label tensors
        ark.tensor_memcpy_host_to_device(self.input, inputs)
        ark.tensor_memcpy_host_to_device(self.label, labels)
        # Randomly initialize the weights
        state_dict = {
            "module1.weight": np.random.rand(d_model, d_ff).astype(np.float16),
            "module2.weight": np.random.rand(d_ff, d_model).astype(np.float16),
        }
        self.module.load_state_dict(state_dict)

    def train(self, iter):
        for i in range(iter):
            ark.run()
            loss = self.get_loss()
            print("loss: ", loss)

    def get_loss(self):
        loss_tensor = self.loss_fn.other_parameter["loss"]
        loss = ark.tensor_memcpy_device_to_host(None, loss_tensor)
        loss = np.sum(loss)
        return loss


def test_TestModel():
    ark.init_model()

    ark_model = TestModelARK()
    trainer = Trainer(ark_model, loss_fn(), Optimizer(ark_model, 0.001))
    ark.launch()
    # Initialize the input and label tensors with all 1
    input_tensor_host = np.ones([batch_size, d_model], dtype=np.float16)
    label_tensor_host = np.ones([batch_size, d_model], dtype=np.float16)
    trainer.trainer_init(input_tensor_host, label_tensor_host)
    trainer.train(1)

    ark.destroy()
    print("ARK module test")


if __name__ == "__main__":
    test_TestModel()
