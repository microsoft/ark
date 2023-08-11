# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import numpy as np


def model_tutorial():
    # Create a Model instance
    model = ark.Model()

    # Create two tensors
    input = model.tensor(ark.Dims(32), ark.TensorType.FP16)
    other = model.tensor(ark.Dims(32), ark.TensorType.FP16)

    # Add input and other to get output tensor
    output = model.add(input, other)

    # Create the executor instance, the scheduler will be created and
    # start scheduling the model when the executor is created
    exe = ark.Executor(0, 0, 1, model, "tutorial_model")

    # Compile the generated code from the code generator
    exe.compile()

    # Initialize the input tensors
    input_np = np.random.rand(1, 32).astype(np.float16)
    other_np = np.random.rand(1, 32).astype(np.float16)

    input.from_numpy(input_np)
    other.from_numpy(other_np)

    print("input: ", input_np)
    print("other: ", other_np)

    # Launch the kernel and run for 1 iteration
    exe.launch()
    exe.run(1)

    # Wait for the kernel to finish
    exe.stop()

    # Copy the output tensor back to host
    output_np = np.zeros((1, 32), dtype=np.float16)
    output.to_numpy(output_np)

    print("output: ", output_np)

    # test if the result is correct
    assert np.allclose(output_np, input_np + other_np)

    max_error = np.max(np.abs(output_np - (input_np + other_np)))
    mean_error = np.mean(np.abs(output_np - (input_np + other_np)))

    print("max error: ", max_error, "mean error: ", mean_error)
    print("test_add passed")


if __name__ == "__main__":
    model_tutorial()
