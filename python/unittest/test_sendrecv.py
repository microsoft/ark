# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import numpy as np
import multiprocessing
import unittest

world_size = 2

tensor_len = 2048
tensor_size = tensor_len * 2


def calculate_errors(arr1, arr2, epsilon=2e-10):
    abs_errors = np.abs(arr1 - arr2)
    relative_errors = np.abs(arr1 - arr2) / (
        np.maximum(np.abs(arr1), np.abs(arr2)) + epsilon
    )
    return abs_errors, relative_errors


def sendrecv_test_one_dir_function(rank, np_inputs):
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

    exe = ark.Executor(rank, rank, world_size, model, "sendrecv_test_one_dir")
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
        max_abs_error = np.max(np.abs(host_output - np_inputs))
        mean_error = np.mean(np.abs(host_output - np_inputs))
        print("max_abs_error:", max_abs_error)
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
    model = ark.Model()

    send_tensor = model.tensor(ark.Dims(tensor_len), ark.TensorType.FP16)
    recv_tensor = model.tensor(ark.Dims(tensor_len), ark.TensorType.FP16)
    model.send(send_tensor, rank, other_rank, tensor_size)
    model.send_done(send_tensor, rank)
    model.recv(recv_tensor, other_rank, other_rank)

    exe = ark.Executor(rank, rank, world_size, model, "sendrecv_test_bi_dir")
    exe.compile()

    exe.launch()
    exe.tensor_memcpy_host_to_device(send_tensor, np_inputs[rank])

    exe.run(1)
    exe.stop()

    host_output = np.zeros(tensor_len, dtype=np.float16)
    exe.tensor_memcpy_device_to_host(host_output, recv_tensor)
    print("host_output:", host_output)
    print("np_inputs:", np_inputs[0])
    abs_errors, relative_errors = calculate_errors(
        host_output, np_inputs[other_rank]
    )

    max_abs_error = np.max(np.abs(host_output - np_inputs[other_rank]))
    mean_error = np.mean(np.abs(host_output - np_inputs[other_rank]))

    print("max_abs_error:", max_abs_error)
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


class SendRecvTest(unittest.TestCase):
    def test_sendrecv_one_dir(self):
        sendrecv_test_one_dir()

    def test_sendrecv_bi_dir(self):
        sendrecv_test_bi_dir()


if __name__ == "__main__":
    unittest.main()
