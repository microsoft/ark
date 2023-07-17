# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import numpy as np
import multiprocessing


def all_gather_test_not_inplace(rank, np_inputs, world_size, tensor_len):
    tensor_size = tensor_len * 2
    print("rank:", rank)

    # Create a Model instance
    model = ark.Model()

    input_tensor = model.tensor(ark.Dims(tensor_len), ark.TensorType.FP16)
    # The all_gather operation will create the recv tensor shards and return them as a list. The allgather_result[rank] is the same as input_tensor
    allgather_result = model.all_gather(input_tensor, rank, world_size)

    exe = ark.Executor(rank, rank, world_size, model, "test_python_bindings")
    exe.compile()

    exe.launch()

    exe.tensor_memcpy_host_to_device(input_tensor, np_inputs[rank])
    exe.run(1)
    exe.stop()
    for tensor_shard in range(world_size):
        # if tensor_shard == rank, then this is a local shard. The allgather_result[tensor_shard] is the same as input_tensor
        host_output = np.zeros(tensor_len, dtype=np.float16)
        exe.tensor_memcpy_device_to_host(
            host_output, allgather_result[tensor_shard]
        )
        gt = np_inputs[tensor_shard]

        max_error = np.max(np.abs(host_output - gt))
        mean_error = np.mean(np.abs(host_output - gt))
        print("max error:", max_error)
        print("mean error:", mean_error)
    print("rank:", rank, "done")


def all_gather_test_inplace(rank, np_inputs, world_size, tensor_len):
    print("rank:", rank)

    # Create a Model instance
    model = ark.Model()

    # input_tensor = model.tensor(ark.Dims(tensor_len), ark.TensorType.FP16)

    output_tensor = model.tensor(
        ark.Dims(tensor_len * world_size), ark.TensorType.FP16
    )
    # Shard the output tensor into world_size shards
    output_shard = model.sharding(output_tensor, 0, tensor_len)
    # The input tensor is the rank'th shard of the output tensor
    input_tensor = output_shard[rank]
    allgather_result = model.all_gather(
        input_tensor, rank, world_size, output_shard
    )

    exe = ark.Executor(rank, rank, world_size, model, "test_python_bindings")
    exe.compile()

    exe.launch()
    print("rank:", rank, " input_tensor:", np_inputs[rank])
    exe.tensor_memcpy_host_to_device(input_tensor, np_inputs[rank])
    exe.run(1)
    exe.stop()
    host_output = np.zeros(tensor_len * world_size, dtype=np.float16)
    # for tensor_shard in range(world_size):
    # if tensor_shard == rank:
    # print("local rank: ")
    print(allgather_result)

    exe.tensor_memcpy_device_to_host(host_output, output_tensor)
    # return
    print("host_output:", host_output)
    # gt = np_inputs[tensor_shard]

    max_error = np.max(np.abs(host_output - gt))
    mean_error = np.mean(np.abs(host_output - gt))
    print("max error:", max_error)
    print("mean error:", mean_error)
    print("rank:", rank, "done")


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


if __name__ == "__main__":
    all_gather_test_main(2, 2048, all_gather_test_not_inplace)
    all_gather_test_main(4, 2048, all_gather_test_not_inplace)
    all_gather_test_main(2, 2048, all_gather_test_inplace)
    all_gather_test_main(4, 2048, all_gather_test_inplace)
    all_gather_test_main(6, 2048, all_gather_test_inplace)
    all_gather_test_main(8, 2048, all_gather_test_inplace)
