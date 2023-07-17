# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import numpy as np
import multiprocessing

world_size = 2

tensor_len = 2048
tensor_size = tensor_len * 2


def sendrecv_test_one_dir_function(rank, np_inputs):
    print("rank:", rank)

    # Create a Model instance
    model = ark.Model(rank)

    input_tensor = model.tensor(ark.Dims(tensor_len), ark.TensorType.FP16)
    if rank == 0:
        model.send(input_tensor, 0, 1, tensor_size)
        model.send_done(input_tensor, 0, 1)
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


def sendrecv_test_one_dir():
    ark.init()
    num_processes = world_size  # number of processes
    processes = []
    np_inputs = np.random.rand(tensor_len).astype(np.float16)
    for i in range(num_processes):
        process = multiprocessing.Process(
            target=sendrecv_test_one_dir_function, args=(i, np_inputs)
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()


def sendrecv_test_bi_dir_function(rank, np_inputs):
    print("rank:", rank)
    other_rank = 1 - rank
    # Create a Model instance
    model = ark.Model(rank)

    send_tensor = model.tensor(ark.Dims(tensor_len), ark.TensorType.FP16)
    recv_tensor = model.tensor(ark.Dims(tensor_len), ark.TensorType.FP16)
    model.send(send_tensor, rank, other_rank, tensor_size)
    model.send_done(send_tensor, rank, other_rank)
    model.recv(recv_tensor, other_rank, other_rank)
    # model.all_reduce(input_tensor, rank, world_size)

    exe = ark.Executor(rank, rank, world_size, model, "test_python_bindings")
    exe.compile()

    exe.launch()
    exe.tensor_memcpy_host_to_device(send_tensor, np_inputs[rank])

    exe.run(1)
    exe.stop()

    host_output = np.zeros(tensor_len, dtype=np.float16)
    exe.tensor_memcpy_device_to_host(host_output, recv_tensor)
    print("host_output:", host_output)
    print("np_inputs:", np_inputs[0])
    max_error = np.max(np.abs(host_output - np_inputs[other_rank]))
    mean_error = np.mean(np.abs(host_output - np_inputs[other_rank]))
    print("max error:", max_error)
    print("mean error:", mean_error)
    print("rank:", rank, "done")


def sendrecv_test_bi_dir():
    ark.init()
    num_processes = world_size  # number of processes
    processes = []
    np_inputs = []
    np_inputs.append(np.random.rand(tensor_len).astype(np.float16))
    np_inputs.append(np.random.rand(tensor_len).astype(np.float16))
    for i in range(num_processes):
        process = multiprocessing.Process(
            target=sendrecv_test_bi_dir_function, args=(i, np_inputs)
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()


if __name__ == "__main__":
    sendrecv_test_one_dir()
    sendrecv_test_bi_dir()
