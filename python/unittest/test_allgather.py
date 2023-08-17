# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import numpy as np
import multiprocessing
import unittest


def all_gather_test_not_inplace(
    rank, np_inputs, world_size, tensor_len, iter=1
):
    runtime = ark.Runtime(rank, world_size)

    input_tensor = ark.tensor(ark.Dims(tensor_len), ark.TensorType.FP16)
    # The all_gather operation will create the recv tensor shards and return them as a list. The allgather_result[rank] is the same as input_tensor
    allgather_result = ark.all_gather(input_tensor, rank, world_size)

    runtime.launch()
    input_tensor.from_numpy(np_inputs[rank])
    runtime.run(iter)
    elapsed = runtime.stop()
    max_abs_error = 0
    for tensor_shard in range(world_size):
        # if tensor_shard == rank, then this is a local shard. The allgather_result[tensor_shard] is the same as input_tensor
        host_output = allgather_result[tensor_shard].to_numpy()
        gt = np_inputs[tensor_shard]

        max_abs_error = max(max_abs_error, np.max(np.abs(host_output - gt)))
        numeric_epsilon_half = np.finfo(np.float16).eps
        np.testing.assert_allclose(host_output, gt, rtol=numeric_epsilon_half)
    print(
        f"allgather not-inplace test world_size: {world_size} rank: "
        f"{rank} tensor_len: {tensor_len:6d} max_abs_error: {max_abs_error:.5f} "
        f"elapsed {elapsed:.5f} ms iter {iter} elapsed_per_iter {elapsed / iter:.5f} ms"
    )


def all_gather_test_inplace(rank, np_inputs, world_size, tensor_len, iter=1):
    runtime = ark.Runtime(rank, world_size)

    # input_tensor = ark.tensor(ark.Dims(tensor_len), ark.TensorType.FP16)

    output_tensor = ark.tensor(
        ark.Dims(tensor_len * world_size), ark.TensorType.FP16
    )
    # Shard the output tensor into world_size shards
    output_shard = ark.sharding(output_tensor, 0, tensor_len)
    # The input tensor is the rank'th shard of the output tensor
    input_tensor = output_shard[rank]
    allgather_result = ark.all_gather(
        input_tensor, rank, world_size, output_shard
    )

    runtime.launch()
    input_tensor.from_numpy(np_inputs[rank])
    runtime.run(iter)
    elapsed = runtime.stop()
    host_output = output_tensor.to_numpy()

    gt = np.concatenate(np_inputs, axis=0)

    max_abs_error = np.max(np.abs(host_output - gt))
    mean_abs_error = np.mean(np.abs(host_output - gt))
    print(
        "allgather-inplace test",
        "world_size:",
        world_size,
        "rank:",
        rank,
        "tensor_len:",
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


def all_gather_test_main(
    world_size, tensor_len, allgather_test_func=all_gather_test_not_inplace
):
    ark.init()
    num_processes = world_size  # number of processes
    processes = []
    np_inputs = []
    for i in range(num_processes):
        np_inputs.append(np.random.rand(tensor_len).astype(np.float16))
    for i in range(num_processes):
        process = multiprocessing.Process(
            target=allgather_test_func,
            args=(i, np_inputs, world_size, tensor_len),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()


class TestAllgather(unittest.TestCase):
    def test_allgather_not_inplace(self):
        all_gather_test_main(2, 2048, all_gather_test_not_inplace)
        all_gather_test_main(4, 2048, all_gather_test_not_inplace)

    def test_allgather_inplace(self):
        all_gather_test_main(2, 2048, all_gather_test_inplace)
        all_gather_test_main(4, 2048, all_gather_test_inplace)
        all_gather_test_main(6, 2048, all_gather_test_inplace)
        all_gather_test_main(8, 2048, all_gather_test_inplace)


if __name__ == "__main__":
    unittest.main()
