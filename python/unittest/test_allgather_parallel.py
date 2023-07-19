# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import numpy as np
import multiprocessing

m, n, k = 64, 256, 8


# Use all-gather operation to perform tensor parallel matmul
def all_gather_tensor_parallel(rank, np_inputs, world_size, iter=1):
    print("rank:", rank)

    # The number of columns per GPU
    N_pergpu = n // world_size
    # Create a Model instance
    model = ark.Model(rank)

    # The input and other tensor of the matmul, note that the other is only a shard of the whole other tensor, we split the other tensor to perform tensor parallel matmul
    input_tensor = model.tensor(ark.Dims(m, k), ark.TensorType.FP16)
    other_tensor_shard = model.tensor(
        ark.Dims(k, N_pergpu), ark.TensorType.FP16
    )
    # The whole output tensor of the matmul
    whole_output_trans = model.tensor(ark.Dims(n, m), ark.TensorType.FP16)

    # shard the other at the first dim, the rank'th shard is the output of the matmul, and we will perform an inplace all-gather operation to get the whole output tensor
    whole_output_shard = model.sharding(whole_output_trans, 0, N_pergpu)

    output_tensor_shard_trans = whole_output_shard[rank]

    # output_tensor_shard = matmul(input, other) => output_tensor_shard.transpose = matmul(other.transpose, input.transpose)
    model.matmul(
        other_tensor_shard,
        input_tensor,
        output_tensor_shard_trans,
        1,
        trans_input=True,
        trans_other=True,
    )
    # In-place all-gather operation to get the whole output tensor
    allgather_result = model.all_gather(
        output_tensor_shard_trans, rank, world_size, whole_output_shard
    )

    # Create an executor instance
    exe = ark.Executor(rank, rank, world_size, model, "test_all_gather")
    exe.compile()
    exe.launch()

    # Copy input and other tensors to device
    exe.tensor_memcpy_host_to_device(input_tensor, np_inputs["input"])
    np_other = np_inputs["other"]
    np_other_shard = np.split(np_other, world_size, axis=1)[rank]
    np_other_shard_copy = np_other_shard.copy()
    exe.tensor_memcpy_host_to_device(other_tensor_shard, np_other_shard_copy)

    # Run the executor
    exe.run(iter)
    elapsed = exe.stop()

    # Copy output tensor to host
    output_host_trans = np.zeros((n, m), dtype=np.float16)
    exe.tensor_memcpy_device_to_host(output_host_trans, whole_output_trans)
    output_host = output_host_trans.transpose()

    # Calculate ground truth
    gt = np.matmul(np_inputs["input"], np_other)

    max_abs_error = np.max(np.abs(output_host - gt))
    mean_abs_error = np.mean(np.abs(output_host - gt))
    print(
        "allgather_parallel_test",
        "world_size:",
        world_size,
        "rank:",
        rank,
        "m",
        m,
        "n",
        n,
        "k",
        k,
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
    world_size, allgather_test_func=all_gather_tensor_parallel
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
            target=allgather_test_func,
            args=(i, np_inputs, world_size),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()


if __name__ == "__main__":
    all_gather_test_main(2, all_gather_tensor_parallel)
    all_gather_test_main(4, all_gather_tensor_parallel)
