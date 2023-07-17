# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import numpy as np
import multiprocessing

M, N, K = 64, 256, 8


# Use all-gather operation to perform tensor parallel matmul
def all_gather_tensor_parallel(rank, np_inputs, world_size):
    print("rank:", rank)

    # The number of columns per GPU
    N_pergpu = N // world_size
    # Create a Model instance
    model = ark.Model()

    # The input and other tensor of the matmul, note that the other is only a shard of the whole other tensor, we split the other tensor to perform tensor parallel matmul
    input_tensor = model.tensor(ark.Dims(M, K), ark.TensorType.FP16)
    other_tensor_shard = model.tensor(
        ark.Dims(K, N_pergpu), ark.TensorType.FP16
    )
    # The whole output tensor of the matmul
    whole_output_trans = model.tensor(ark.Dims(N, M), ark.TensorType.FP16)

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
    exe.run(1)
    exe.stop()

    # Copy output tensor to host
    output_host_trans = np.zeros((N, M), dtype=np.float16)
    exe.tensor_memcpy_device_to_host(output_host_trans, whole_output_trans)
    output_host = output_host_trans.transpose()

    # Calculate ground truth
    gt = np.matmul(np_inputs["input"], np_other)

    max_error = np.max(np.abs(output_host - gt))
    avg_error = np.mean(np.abs(output_host - gt))
    print("max error:", max_error)
    print("avg error:", avg_error)
    print("output_host:", output_host)
    print("gt:", gt)
    gt_shard = np.split(gt, world_size, axis=1)[rank]
    print("gt_shard", gt_shard)


def all_gather_test_main(
    world_size, allgather_test_func=all_gather_tensor_parallel
):
    ark.init()
    num_processes = world_size  # number of processes
    processes = []
    np_inputs = {
        "input": np.random.randn(M, K).astype(np.float16),
        "other": np.random.randn(K, N).astype(np.float16),
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
