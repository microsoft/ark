# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import numpy as np
import multiprocessing


def all_reduce_test(rank, np_inputs, world_size, tensor_len):
    tensor_size = tensor_len * 2
    print("rank:", rank)

    # Create a Model instance
    model = ark.Model(rank)

    input_tensor = model.tensor(ark.Dims(tensor_len), ark.TensorType.FP16)

    allreduce_result = model.all_reduce(input_tensor, rank, world_size)

    exe = ark.Executor(rank, rank, world_size, model, "test_python_bindings")
    exe.compile()

    exe.launch()
    print("rank:", rank, " input_tensor:", np_inputs[rank])
    exe.tensor_memcpy_host_to_device(input_tensor, np_inputs[rank])
    exe.run(1)
    exe.stop()

    host_output = np.zeros(tensor_len, dtype=np.float16)
    exe.tensor_memcpy_device_to_host(host_output, allreduce_result)
    print("host_output:", host_output)
    gt = np.zeros(tensor_len, dtype=np.float16)
    for np_input in np_inputs:
        gt += np_input

    print("gt:", gt)
    max_error = np.max(np.abs(host_output - gt))
    mean_error = np.mean(np.abs(host_output - gt))
    print("max error:", max_error)
    print("mean error:", mean_error)
    print("rank:", rank, "done")


def all_reduce_test_main(world_size, tensor_len):
    ark.init()
    num_processes = world_size  # number of processes
    processes = []
    np_inputs = []
    for i in range(num_processes):
        np_inputs.append(np.random.rand(tensor_len).astype(np.float16))
    for i in range(num_processes):
        process = multiprocessing.Process(
            target=all_reduce_test, args=(i, np_inputs, world_size, tensor_len)
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()


if __name__ == "__main__":
    all_reduce_test_main(2, 2048)
    all_reduce_test_main(4, 2048)
