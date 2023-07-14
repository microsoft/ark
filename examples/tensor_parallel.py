# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import numpy as np
import multiprocessing


def all_gather_test_not_inplace(rank, np_inputs, world_size):
    print("rank:", rank)
    M, N, K = 64, 64, 8
    N_pergpu = N // world_size
    # Create a Model instance
    model = ark.Model()

    input_tensor = model.tensor(ark.Dims(M, K), ark.TensorType.FP16)
    other_tensor = model.tensor(ark.Dims(K, N_pergpu), ark.TensorType.FP16)

    output_tensor_shard = model.matmul(input_tensor, other_tensor)

    whole_output = model.tensor(ark.Dims(M * N), ark.TensorType.FP16)

    whole_output_shard = model.sharding(whole_output, 0, M * N_pergpu)
    whole_output_shard[rank] = model.reshape(
        whole_output_shard[rank], ark.Dims(N_pergpu, M)
    )

    model.transpose(
        output_tensor_shard, ark.Dims(1, 0), whole_output_shard[rank]
    )

    whole_output_shard = model.reshape(whole_output_shard, ark.Dims(M * N))
    allgather_result = model.all_gather(whole_output_shard, rank, world_size)
    exe = ark.Executor(rank, rank, world_size, model, "test_python_bindings")
    exe.compile()

    exe.launch()

    exe.run(1)
    exe.stop()


def all_gather_test_main(
    world_size, allgather_test_func=all_gather_test_not_inplace
):
    ark.init()
    num_processes = world_size  # number of processes
    processes = []
    np_inputs = []

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
