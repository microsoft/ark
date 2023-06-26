# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# import megatron_pytorch
# import megatron_ark
# from megatron_utils import *
import ark
import numpy as np
import multiprocessing

world_size = 2

tensor_len = 2048
tensor_size = tensor_len * 2


def my_function(rank, np_inputs):
    print("rank:", rank)

    # Create a Model instance
    model = ark.Model()

    input_tensor = model.tensor(ark.Dims(tensor_len), ark.TensorType.FP16)
    if rank == 0:
        model.send(input_tensor, 0, 1, tensor_size)
        model.send_done(input_tensor, 0)
    if rank == 1:
        model.recv(input_tensor, 0, 0)
    # model.all_reduce(input_tensor, rank, world_size)

    exe = ark.Executor(rank, rank, world_size, model, "test_python_bindings")
    exe.compile()

    exe.launch()
    if rank == 0:
        exe.tensor_memcpy_host_to_device(input_tensor, np_inputs)
    exe.run(1)
    exe.stop()
    if rank == 1:
        host_output = np.zeros(tensor_len, dtype=np.float16)
        exe.tensor_memcpy_device_to_host(host_output, input_tensor)
        print("host_output:", host_output)
        print("np_inputs:", np_inputs)
        max_error = np.max(np.abs(host_output - np_inputs))
        mean_error = np.mean(np.abs(host_output - np_inputs))
        print("max error:", max_error)
        print("mean error:", mean_error)
    print("rank:", rank, "done")


if __name__ == "__main__":
    ark.init()
    num_processes = world_size  # number of processes
    processes = []
    np_inputs = np.random.rand(tensor_len).astype(np.float16)
    for i in range(num_processes):
        process = multiprocessing.Process(
            target=my_function, args=(i, np_inputs)
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
