# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import ark

def quickstart_tutorial():
    # Initialize the ARK model
    ark.init_model()

    M, N = 64, 64
    # Create an input tensor
    input_tensor = ark.tensor([M, N])
    # Create another tensor
    other_tensor = ark.tensor([M, N])

    # Add the two tensors
    output_tensor = ark.add(input_tensor, other_tensor)

    # Launch the ARK runtime
    ark.launch()

    # Initialize the input and other tensor with random values
    input_tensor_host = np.random.rand(M, N).astype(np.float32)
    ark.tensor_memcpy_host_to_device(input_tensor, input_tensor_host)
    other_tensor_host = np.random.rand(M, N).astype(np.float32)
    ark.tensor_memcpy_host_to_device(other_tensor, other_tensor_host)

    # Run the ARK program
    ark.run()

    # Copy the output tensor from device memory to host memory, if dst is
    # None, a new numpy array of the same shape as the src tensor will be returned
    output_tensor_host = ark.tensor_memcpy_device_to_host(None, output_tensor)
    # Check if the output tensor is equal to the sum of the input and other tensor
    np.testing.assert_allclose(
        output_tensor_host, input_tensor_host + other_tensor_host
    )


if __name__ == "__main__":
    quickstart_tutorial()
