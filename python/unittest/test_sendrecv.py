# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import numpy as np
import multiprocessing
import unittest

world_size = 2

tensor_len = 2048
tensor_size = tensor_len * 2


def sendrecv_test_one_dir_function(rank, np_inputs, iter=1):
    # Create a Model instance
    model = ark.Model(rank)

    input_tensor = model.tensor(ark.Dims(tensor_len), ark.TensorType.FP16)
    if rank == 0:
        model.send(input_tensor, 0, 1, tensor_size)
        model.send_done(input_tensor, 0, 1)
    if rank == 1:
        model.recv(input_tensor, 0, 0)
    # model.all_reduce(input_tensor, rank, world_size)

    exe = ark.Executor(rank, rank, world_size, model, "sendrecv_test_one_dir")
    exe.compile()

    exe.launch()
    if rank == 0:
        exe.tensor_memcpy_host_to_device(input_tensor, np_inputs)
    exe.run(iter)
    elapsed = exe.stop()
    if rank == 1:
        host_output = np.zeros(tensor_len, dtype=np.float16)
        exe.tensor_memcpy_device_to_host(host_output, input_tensor)

        max_abs_error = np.max(np.abs(host_output - np_inputs))
        mean_abs_error = np.mean(np.abs(host_output - np_inputs))
        # The numeric error of half precision of the machine
        numeric_epsilon_half = np.finfo(np.float16).eps
        atol = numeric_epsilon_half
        np.testing.assert_allclose(host_output, np_inputs, atol=atol)
        print(
            "sendrecv_test_one_dir:",
            "rank",
            rank,
            "tensor_len",
            "{:6d}".format(tensor_len),
            "max_abs_error:",
            "{:.5f}".format(max_abs_error),
            "mean_abs_error:",
            "{:.5f}".format(mean_abs_error),
            "elapsed",
            "{:.5f}".format(elapsed),
            " ms ",
            " iter ",
            iter,
            "elapsed_per_iter",
            "{:.5f}".format(elapsed / iter),
            " ms ",
        )


def sendrecv_test_one_dir():
    ark.cleanup()
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


def sendrecv_test_bi_dir_function(rank, np_inputs, iter=1):
    other_rank = 1 - rank
    # Create a Model instance
    model = ark.Model(rank)

    send_tensor = model.tensor(ark.Dims(tensor_len), ark.TensorType.FP16)
    recv_tensor = model.tensor(ark.Dims(tensor_len), ark.TensorType.FP16)
    model.send(send_tensor, rank, other_rank, tensor_size)
    model.send_done(send_tensor, rank, other_rank)
    model.recv(recv_tensor, other_rank, other_rank)

    exe = ark.Executor(rank, rank, world_size, model, "sendrecv_test_bi_dir")
    exe.compile()

    exe.launch()
    exe.tensor_memcpy_host_to_device(send_tensor, np_inputs[rank])

    exe.run(iter)
    elapsed = exe.stop()

    host_output = np.zeros(tensor_len, dtype=np.float16)
    exe.tensor_memcpy_device_to_host(host_output, recv_tensor)

    gt = np_inputs[other_rank]
    max_abs_error = np.max(np.abs(host_output - gt))
    mean_abs_error = np.mean(np.abs(host_output - gt))
    # The numeric error of half precision of the machine
    numeric_epsilon_half = np.finfo(np.float16).eps
    np.testing.assert_allclose(host_output, gt, atol=numeric_epsilon_half)
    print(
        "sendrecv_test_bi_dir:",
        "rank",
        rank,
        "tensor_len",
        "{:6d}".format(tensor_len),
        "max_abs_error:",
        "{:.5f}".format(max_abs_error),
        "mean_abs_error:",
        "{:.5f}".format(mean_abs_error),
        "elapsed",
        "{:.5f}".format(elapsed),
        " ms ",
        " iter ",
        iter,
    )


def sendrecv_test_bi_dir():
    ark.cleanup()
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
