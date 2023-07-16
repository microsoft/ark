# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import numpy as np
import multiprocessing

M, N, K = 64, 64, 8


def all_gather_test_not_inplace(rank, np_inputs, world_size):
    print("rank:", rank)
    N_pergpu = N // world_size
    # Create a Model instance
    model = ark.Model()

    input_tensor = model.tensor(ark.Dims(M, K), ark.TensorType.FP16)
    other_tensor_shard = model.tensor(
        ark.Dims(K, N_pergpu), ark.TensorType.FP16
    )

    whole_output_trans = model.tensor(ark.Dims(M * N), ark.TensorType.FP16)

    whole_output_shard = model.sharding(whole_output_trans, 0, M * N_pergpu)

    output_tensor_shard_trans = model.reshape(
        whole_output_shard[rank], ark.Dims(N_pergpu, M)
    )
    # output_tensor_shard = matmul(input, other) => output_tensor_shard.transpose = matmul(other.transpose, input.transpose)
    model.matmul(
        other_tensor_shard,
        input_tensor,
        output_tensor_shard_trans,
        1,
        trans_input=True,
        trans_other=True,
    )
    output_tensor_shard_trans = model.reshape(
        output_tensor_shard_trans, ark.Dims(M * N_pergpu)
    )

    # allgather_result = model.all_gather(
    #     output_tensor_shard_trans, rank, world_size, whole_output_shard
    # )
    exe = ark.Executor(rank, rank, world_size, model, "test_all_gather")
    exe.compile()

    exe.launch()
    # init the input tensor and other tensor
    exe.tensor_memcpy_host_to_device(input_tensor, np_inputs["input"])
    np_other = np_inputs["other"]
    np_other_shard = np.split(np_other, world_size, axis=1)[rank]
    np_other_shard_copy = np_other_shard.copy()
    exe.tensor_memcpy_host_to_device(other_tensor_shard, np_other_shard_copy)

    exe.run(1)
    exe.stop()

    output_host_trans = np.zeros((N, M), dtype=np.float16)
    exe.tensor_memcpy_device_to_host(output_host_trans, whole_output_trans)
    output_host = output_host_trans.transpose()
    gt = np.matmul(np_inputs["input"], np_other)

    max_error = np.max(np.abs(output_host - gt))
    avg_error = np.mean(np.abs(output_host - gt))
    print("max error:", max_error)
    print("avg error:", avg_error)
    print("output_host:", output_host)
    print("gt:", gt)


def all_gather_test_main(
    world_size, allgather_test_func=all_gather_test_not_inplace
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
    all_gather_test_main(2, all_gather_test_not_inplace)
    # all_gather_test_main(3, all_gather_test_not_inplace)