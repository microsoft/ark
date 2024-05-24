# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import ark


def quickstart_tutorial():
    # Initialize the ARK environments
    ark.init()

    M, N = 64, 64
    # Create an input tensor
    input_tensor = ark.tensor([M, N], ark.fp16)
    # Create another tensor
    other_tensor = ark.tensor([M, N], ark.fp16)

    # Add the two tensors
    output_tensor = ark.add(input_tensor, other_tensor)

    # Initialize the ARK runtime
    runtime = ark.Runtime()

    # Launch the ARK runtime
    runtime.launch()

    # Initialize the input and other tensor with random values
    input_tensor_host = np.random.rand(M, N).astype(np.float16)
    input_tensor.from_numpy(input_tensor_host)
    other_tensor_host = np.random.rand(M, N).astype(np.float16)
    other_tensor.from_numpy(other_tensor_host)

    # Run the ARK program
    runtime.run()

    # Copy the output tensor from device memory to host memory, if dst is
    # None, a new numpy array of the same shape as the src tensor will be returned
    output_tensor_host = output_tensor.to_numpy()
    # Check if the output tensor is equal to the sum of the input and other tensor
    np.testing.assert_allclose(
        output_tensor_host, input_tensor_host + other_tensor_host
    )

    # Stop the ARK runtime (undo Runtime.launch())
    runtime.stop()

    # Reset the ARK runtime (free all resources)
    runtime.reset()

    print("Quickstart tutorial is successful!")


if __name__ == "__main__":
    quickstart_tutorial()
