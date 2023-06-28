# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import numpy as np
import multiprocessing

world_size = 2

tensor_len = 2048
tensor_size = tensor_len * 2


def sendrecv_test_ping_pong_function(rank, np_inputs):
    print("rank:", rank)
    other_rank = 1 - rank
    # Create a Model instance
    model = ark.Model()

    if rank == 0:
        send_tensor = model.tensor(ark.Dims(tensor_len), ark.TensorType.FP16)
        recv_tensor = model.tensor(ark.Dims(tensor_len), ark.TensorType.FP16)
        model.send(send_tensor, 0, 1, tensor_size)
        model.send_done(send_tensor, 0)
        # model.recv(recv_tensor, 1, 1)
    if rank == 1:
        recv_tensor = model.tensor(ark.Dims(tensor_len), ark.TensorType.FP16)
        recv_dep = model.recv(recv_tensor, 0, 0)
        send_tensor = model.identity(recv_tensor, [recv_dep])
        # model.send(send_tensor, 1, 0, tensor_size)
        # model.send_done(send_tensor, 1)

    exe = ark.Executor(rank, rank, world_size, model, "test_python_bindings")
    exe.compile()

    exe.launch()
    if rank == 0:
        exe.tensor_memcpy_host_to_device(send_tensor, np_inputs)

    exe.run(1)
    exe.stop()
    host_output = np.zeros(tensor_len, dtype=np.float16)

    exe.tensor_memcpy_device_to_host(host_output, recv_tensor)
    print("host_output:", host_output)
    print("np_inputs:", np_inputs)
    max_error = np.max(np.abs(host_output - np_inputs))
    mean_error = np.mean(np.abs(host_output - np_inputs))
    print("max error:", max_error)
    print("mean error:", mean_error)
    print("rank:", rank, "done")


def sendrecv_test_ping_pong():
    ark.init()
    num_processes = world_size  # number of processes
    processes = []
    np_inputs = np.random.rand(tensor_len).astype(np.float16)
    for i in range(num_processes):
        process = multiprocessing.Process(
            target=sendrecv_test_ping_pong_function, args=(i, np_inputs)
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()


if __name__ == "__main__":
    sendrecv_test_ping_pong()
