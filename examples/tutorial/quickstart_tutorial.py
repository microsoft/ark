# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import numpy as np
import torch.nn as nn
import ark


if __name__ == "__main__":
    # Initialize the ARK model
    ark.init_model()

    M, N = 1024, 1024
    # Create an input tensor
    input_tensor = ark.tensor(ark.Dims(M, N))
    # Create another tensor
    other_tensor = ark.tensor(ark.Dims(M, N))

    # Add the two tensors
    output_tensor = ark.add(input_tensor, input_tensor)

    # Launch the ARK runtime
    ark.launch()

    # Initialize the input and other tensor with random values
    input_tensor_host = np.random.rand(M, N)
    ark.tensor_memcpy_host_to_device(input_tensor, input_tensor_host)
    other_tensor_host = np.random.rand(M, N)
    ark.tensor_memcpy_host_to_device(other_tensor, other_tensor_host)

    # Run the ARK program
    ark.run()

    # Copy the output tensor from device memory to host memory
    output_tensor_host = ark.tensor_memcpy_device_to_host(None, output_tensor)

    # Check if the output tensor is equal to the sum of the input and other tensor
    np.testing.assert_allclose(
        output_tensor_host, input_tensor_host + other_tensor_host
    )

    # Destroy the ARK runtime
    ark.destroy()
