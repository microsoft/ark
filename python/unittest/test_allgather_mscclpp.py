# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import numpy as np
import multiprocessing
import os
import unittest


def all_gather_test(rank, input, expected, world_size, height, width, iter=1):
    # Create a Model instance
    runtime = ark.Runtime()
    ark.set_rank(rank)
    ark.set_world_size(world_size)

    input_tensor = ark.tensor([height, width], ark.fp16)

    allgather_result = ark.local_all_gather_mscclpp(
        input_tensor, rank, world_size, 1
    )

    runtime.launch()
    input_tensor.from_numpy(input)
    runtime.run(iter)
    elapsed = runtime.stop()

    host_output = allgather_result.to_numpy()

    max_abs_error = np.max(np.abs(host_output - expected))
    mean_abs_error = np.mean(np.abs(host_output - expected))
    # The numeric error of half precision of the machine
    numeric_epsilon_half = np.finfo(np.float16).eps
    np.testing.assert_allclose(
        host_output, expected, atol=2 * world_size * numeric_epsilon_half
    )
    print(
        f"allgather all_gather_test: world_size {world_size} rank {rank} tensor_len "
        f"<{height},{width}> max_abs_error {max_abs_error:.5f} mean_abs_error "
        f"{mean_abs_error:.5f} elapsed {elapsed:.5f} ms iter {iter} "
        f"elapsed_per_iter {elapsed / iter:.5f} ms"
    )


def test_allgather2D_mscclpp_internal(world_size, height, width):
    ark.init()
    num_processes = world_size  # number of processes
    processes = []
    expected = np.random.rand(height, width).astype(np.float16)
    sharded_input = np.split(expected, world_size, axis=1)
    shard_width = width // world_size
    np_inputs = []
    for i in range(num_processes):
        input = np.zeros((height, width), dtype=np.float16)
        input[:, shard_width * i : shard_width * (i + 1)] = sharded_input[i]
        np_inputs.append(input)
    for i in range(num_processes):
        process = multiprocessing.Process(
            target=all_gather_test,
            args=(i, np_inputs[i], expected, world_size, height, width),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()


class TestAllgather(unittest.TestCase):
    def test_allgather_mscclpp(self):
        test_allgather2D_mscclpp_internal(8, 4096, 8192)


if __name__ == "__main__":
    os.environ["ARK_USE_MSCCLPP"] = "1"
    unittest.main()
