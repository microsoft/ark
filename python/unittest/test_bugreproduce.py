# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import numpy as np


def model_tutorial():
    runtime = ark.Runtime()
    dim = 2048
    hidden_dim = 4096
    # Create two tensors
    input = ark.tensor(ark.Dims(4, dim), ark.TensorType.FP32)
    weight1 = ark.tensor(ark.Dims(hidden_dim, dim), ark.TensorType.FP32)

    weight2 =ark.tensor(ark.Dims(dim, hidden_dim), ark.TensorType.FP32)

    output1 = ark.matmul(input, weight1, transpose_b=True)
    print("output1.shape: ", output1.shape)
    output2 = ark.matmul(output1, weight2, transpose_b=True)
    print("output2.shape: ", output2.shape)
    runtime.launch()
    runtime.run()



if __name__ == "__main__":
    model_tutorial()
