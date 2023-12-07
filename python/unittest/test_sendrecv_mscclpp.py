# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import numpy as np
import multiprocessing
import os
import unittest

world_size = 2

tensor_len = 2048
tensor_size = tensor_len * 2


def sendrecv_test_one_dir_function(rank, np_inputs, iter=1):
    # Create a Model instance
    runtime = ark.Runtime()
    ark.set_rank(rank)
    ark.set_world_size(world_size)

    input_tensor = ark.tensor([tensor_len], ark.fp16)
    output_tensor = ark.tensor([tensor_len], ark.fp16)
    if rank == 0:
        ark.send_mscclpp(input_tensor, 0, 1, tensor_size)
        ark.send_done_mscclpp(input_tensor, 1)
    if rank == 1:
        output_tensor = ark.recv_mscclpp(0, 0, 0, output_tensor)

    runtime.launch()
    if rank == 0:
        input_tensor.from_numpy(np_inputs)
    runtime.run(iter)
    elapsed = runtime.stop()
    if rank == 1:
        host_output = output_tensor.to_numpy()
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


def sendrecv_test_bi_dir_function(rank, np_inputs, iter=1):
    # Create a Model instance
    runtime = ark.Runtime()
    ark.set_rank(rank)
    ark.set_world_size(world_size)
    other_rank = 1 - rank

    send_tensor = ark.tensor([tensor_len], ark.fp16)
    recv_tensor = ark.tensor([tensor_len], ark.fp16)
    ark.send_mscclpp(send_tensor, rank, other_rank, tensor_size)
    ark.send_done_mscclpp(send_tensor, other_rank)
    ark.recv_mscclpp(other_rank, other_rank, 0, recv_tensor)

    runtime.launch()
    send_tensor.from_numpy(np_inputs[rank])

    runtime.run(iter)
    elapsed = runtime.stop()

    host_output = np.zeros(tensor_len, dtype=np.float16)
    host_output = recv_tensor.to_numpy()

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


class SendRecvMscclppTest(unittest.TestCase):
    def test_sendrecv_one_dir(self):
        sendrecv_test_one_dir()

    def test_sendrecv_bi_dir(self):
        sendrecv_test_bi_dir()


if __name__ == "__main__":
    unittest.main()
