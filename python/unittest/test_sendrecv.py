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
    runtime = ark.Runtime(rank, world_size)

    input_tensor = ark.tensor(ark.Dims(tensor_len), ark.TensorType.FP16)
    if rank == 0:
        ark.send(input_tensor, 0, 1, tensor_size)
        ark.send_done(input_tensor, 0, 1)
    if rank == 1:
        ark.recv(input_tensor, 0, 0)
    # ark.all_reduce(input_tensor, rank, world_size)

    runtime.launch()
    if rank == 0:
        input_tensor.from_numpy(np_inputs)
    runtime.run(iter)
    elapsed = runtime.stop()
    if rank == 1:
        host_output = input_tensor.to_numpy()

        max_abs_error = np.max(np.abs(host_output - np_inputs))
        mean_abs_error = np.mean(np.abs(host_output - np_inputs))
        # The numeric error of half precision of the machine
        numeric_epsilon_half = np.finfo(np.float16).eps
        atol = numeric_epsilon_half
        np.testing.assert_allclose(host_output, np_inputs, atol=atol)
        print(
            f"sendrecv_test_one_dir: rank {rank} tensor_len {tensor_len:6d} "
            f"max_abs_error {max_abs_error:.5f} mean_abs_error {mean_abs_error:.5f} "
            f"elapsed {elapsed:.5f} ms iter {iter} elapsed_per_iter {elapsed / iter:.5f} ms"
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
    runtime = ark.Runtime(rank, world_size)
    other_rank = 1 - rank

    send_tensor = ark.tensor(ark.Dims(tensor_len), ark.TensorType.FP16)
    recv_tensor = ark.tensor(ark.Dims(tensor_len), ark.TensorType.FP16)
    ark.send(send_tensor, rank, other_rank, tensor_size)
    ark.send_done(send_tensor, rank, other_rank)
    ark.recv(recv_tensor, other_rank, other_rank)

    runtime.launch()
    send_tensor.from_numpy(np_inputs[rank])

    runtime.run(iter)
    elapsed = runtime.stop()

    host_output = recv_tensor.to_numpy()

    gt = np_inputs[other_rank]
    max_abs_error = np.max(np.abs(host_output - gt))
    mean_abs_error = np.mean(np.abs(host_output - gt))
    # The numeric error of half precision of the machine
    numeric_epsilon_half = np.finfo(np.float16).eps
    np.testing.assert_allclose(host_output, gt, atol=numeric_epsilon_half)
    print(
        f"sendrecv_test_bi_dir: rank {rank} tensor_len {tensor_len:6d} "
        f"max_abs_error {max_abs_error:.5f} mean_abs_error {mean_abs_error:.5f} "
        f"elapsed {elapsed:.5f} ms iter {iter} elapsed_per_iter {elapsed / iter:.5f} ms"
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


class SendRecvTest(unittest.TestCase):
    def test_sendrecv_one_dir(self):
        sendrecv_test_one_dir()

    def test_sendrecv_bi_dir(self):
        sendrecv_test_bi_dir()


if __name__ == "__main__":
    unittest.main()
