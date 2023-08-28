# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import numpy as np
import multiprocessing
import unittest

m, n, k = 64, 256, 8


# Use all-gather operation to perform tensor parallel matmul
def all_reduce_tensor_parallel(rank, np_inputs, world_size, iter=1):
    print("rank:", rank)
    runtime = ark.Runtime(rank, world_size)
    # The number of columns per GPU

    # The input and other tensor of the matmul, note that the other is only a shard of the whole other tensor, we split the other tensor to perform tensor parallel matmul
    input_tensor = ark.tensor([m, k], ark.TensorType.FP16)
    other_tensor = ark.tensor([k, n], ark.TensorType.FP16)

    output = ark.matmul(input_tensor, other_tensor)
    # In-place all-gather operation to get the whole output tensor
    allreduce_result = ark.all_reduce(output, rank, world_size)

    runtime.launch()

    # Copy input and other tensors to device
    input_tensor.from_numpy(np_inputs["input"])
    other_tensor.from_numpy(np_inputs["other"])

    runtime.run(non_blocking=True)
    elapsed = runtime.stop()
    output_host = allreduce_result.to_numpy()

    # Calculate ground truth
    gt = np.matmul(np_inputs["input"], np_inputs["other"]) * world_size

    max_abs_error = np.max(np.abs(output_host - gt))
    mean_abs_error = np.mean(np.abs(output_host - gt))
    print(
        f"allreduce_parallel_test world_size: {world_size} rank: "
        f"{rank} m: {m} n: {n} k: {k} max_abs_error: {max_abs_error:.5f} "
        f"mean_abs_error: {mean_abs_error:.5f} elapsed: {elapsed:.5f} ms "
        f"iter: {iter} elapsed_per_iter: {elapsed / iter:.5f} ms"
    )


def all_reduce_test_main(
    world_size, allreduce_test_func=all_reduce_tensor_parallel
):
    ark.init()
    num_processes = world_size  # number of processes
    processes = []
    np_inputs = {
        "input": np.random.randn(m, k).astype(np.float16),
        "other": np.random.randn(k, n).astype(np.float16),
    }

    for i in range(num_processes):
        process = multiprocessing.Process(
            target=allreduce_test_func,
            args=(i, np_inputs, world_size),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()


class TestAllreduce(unittest.TestCase):
    def test_all_reduce(self):
        all_reduce_test_main(2, all_reduce_tensor_parallel)
        all_reduce_test_main(4, all_reduce_tensor_parallel)


if __name__ == "__main__":
    unittest.main()
