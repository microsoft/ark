# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import numpy as np
import multiprocessing
import os
import unittest


def all_reduce_packet_test(rank, np_inputs, world_size, tensor_len, iter=1):
    # Create a Model instance
    runtime = ark.Runtime()
    ark.set_rank(rank)
    ark.set_world_size(world_size)

    input_tensor = ark.tensor([tensor_len], ark.fp16)

    allreduce_result = ark.local_all_reduce_packet_msll(input_tensor, rank, world_size)

    runtime.launch()
    input_tensor.from_numpy(np_inputs[rank])
    runtime.run(iter)
    elapsed = runtime.stop()

    host_output = allreduce_result.to_numpy()
    gt = np.zeros(tensor_len, dtype=np.float16)
    for np_input in np_inputs:
        gt += np_input

    max_abs_error = np.max(np.abs(host_output - gt))
    mean_abs_error = np.mean(np.abs(host_output - gt))
    # The numeric error of half precision of the machine
    numeric_epsilon_half = np.finfo(np.float16).eps
    np.testing.assert_allclose(
        host_output, gt, atol=2 * world_size * numeric_epsilon_half
    )
    print(
        f"allreduce all_reduce_packet_test: world_size {world_size} rank {rank} tensor_len "
        f"{tensor_len:6d} max_abs_error {max_abs_error:.5f} mean_abs_error "
        f"{mean_abs_error:.5f} elapsed {elapsed:.5f} ms iter {iter} "
        f"elapsed_per_iter {elapsed / iter:.5f} ms"
    )


def all_reduce_test(rank, np_inputs, world_size, tensor_len, iter=1):
    # Create a Model instance
    runtime = ark.Runtime()
    ark.set_rank(rank)
    ark.set_world_size(world_size)

    input_tensor = ark.tensor([tensor_len], ark.fp16)

    allreduce_result = ark.local_all_reduce_msll(input_tensor, rank, world_size)

    runtime.launch()
    input_tensor.from_numpy(np_inputs[rank])
    runtime.run(iter)
    elapsed = runtime.stop()

    host_output = allreduce_result.to_numpy()
    gt = np.zeros(tensor_len, dtype=np.float16)
    for np_input in np_inputs:
        gt += np_input

    max_abs_error = np.max(np.abs(host_output - gt))
    mean_abs_error = np.mean(np.abs(host_output - gt))
    # The numeric error of half precision of the machine
    numeric_epsilon_half = np.finfo(np.float16).eps
    np.testing.assert_allclose(
        host_output, gt, atol=2 * world_size * numeric_epsilon_half
    )
    print(
        f"allreduce all_reduce_test: world_size {world_size} rank {rank} tensor_len "
        f"{tensor_len:6d} max_abs_error {max_abs_error:.5f} mean_abs_error "
        f"{mean_abs_error:.5f} elapsed {elapsed:.5f} ms iter {iter} "
        f"elapsed_per_iter {elapsed / iter:.5f} ms"
    )


def test_allreduce_msll_packet_internal(world_size, tensor_len):
    ark.init()
    num_processes = world_size  # number of processes
    processes = []
    np_inputs = []
    for i in range(num_processes):
        np_inputs.append(np.random.rand(tensor_len).astype(np.float16))
    for i in range(num_processes):
        process = multiprocessing.Process(
            target=all_reduce_packet_test,
            args=(i, np_inputs, world_size, tensor_len),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()


def test_allreduce_msll_internal(world_size, tensor_len):
    ark.init()
    num_processes = world_size  # number of processes
    processes = []
    np_inputs = []
    for i in range(num_processes):
        np_inputs.append(np.random.rand(tensor_len).astype(np.float16))
    for i in range(num_processes):
        process = multiprocessing.Process(
            target=all_reduce_test,
            args=(i, np_inputs, world_size, tensor_len),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()


class TestAllreduce(unittest.TestCase):
    def test_allreduce_msll(self):
        test_allreduce_msll_packet_internal(8, 4096)
        test_allreduce_msll_internal(8, 32*1024*1024)


if __name__ == "__main__":
    os.environ["ARK_USE_MSLL"] = "1"
    unittest.main()
