# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import numpy as np
import multiprocessing
import os
import unittest


def reduce_scatter_test(rank, inputs, world_size, nelems, iter=1):
    # Create a Model instance
    runtime = ark.Runtime()
    ark.set_rank(rank)
    ark.set_world_size(world_size)

    input_tensor = ark.tensor([nelems], ark.fp16)

    reducesactter_result = ark.local_reduce_scatter(
        input_tensor, rank, world_size
    )

    runtime.launch()
    input_tensor.from_numpy(inputs[rank])
    runtime.run(iter)
    elapsed = runtime.stop()

    host_output = reducesactter_result.to_numpy()
    expected = np.copy(inputs[rank])
    nelems_per_rank = nelems // world_size
    for i, input in enumerate(inputs):
        if i != rank:
            tmp = np.zeros(nelems, dtype=np.float16)
            tmp[nelems_per_rank * rank : nelems_per_rank * (rank + 1)] = input[nelems_per_rank * rank : nelems_per_rank * (rank + 1)]
            expected += tmp

    max_abs_error = np.max(np.abs(host_output - expected))
    mean_abs_error = np.mean(np.abs(host_output - expected))
    # The numeric error of half precision of the machine
    numeric_epsilon_half = np.finfo(np.float16).eps
    np.testing.assert_allclose(
        host_output, expected, atol=2 * world_size * numeric_epsilon_half
    )
    print(
        f"reducescatter reduce_scatter_test: world_size {world_size} rank {rank} tensor_len "
        f"<{nelems}> max_abs_error {max_abs_error:.5f} mean_abs_error "
        f"{mean_abs_error:.5f} elapsed {elapsed:.5f} ms iter {iter} "
        f"elapsed_per_iter {elapsed / iter:.5f} ms"
    )


def test_reducescatter_mscclpp_internal(world_size, nelems):
    ark.init()
    num_processes = world_size  # number of processes
    processes = []
    np_inputs = []
    for i in range(num_processes):
        np_inputs.append(np.random.rand(nelems).astype(np.float16))
    for i in range(num_processes):
        process = multiprocessing.Process(
            target=reduce_scatter_test,
            args=(i, np_inputs, world_size, nelems),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()


class TestReducescatter(unittest.TestCase):
    def test_reducescatter_mscclpp(self):
        test_reducescatter_mscclpp_internal(8, 1024 * 1024 * 32)


if __name__ == "__main__":
    unittest.main()
