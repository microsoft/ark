# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .ark_core import Model, Tensor, TensorType, Dims
from .executor import Executor

class optimizer:
    def __init__(self, module, lr):
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
    def forward(self, outputs, labels):
        return self.model.cross_entropy(outputs, labels)

class trainer:
    def __init__(self, module, loss_fn, optimizer: optimizer, executor: Executor):
        self.model = module.model
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

