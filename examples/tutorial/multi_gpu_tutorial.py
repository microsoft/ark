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
    # Create a Model instance
    model = ark.Model(rank)

    # Define the behavior for rank 0
    if rank == 0:
        send_tensor = model.tensor(ark.Dims(tensor_len), ark.TensorType.FP16)
        recv_tensor = model.tensor(ark.Dims(tensor_len), ark.TensorType.FP16)

        # send the tensor to rank 1
        send_id, dst_rank = 0, 1
        model.send(send_tensor, send_id, dst_rank, tensor_size)
        model.send_done(send_tensor, send_id, dst_rank)

        # recv the tensor from rank 1
        recv_id, recv_rank = 1, 1
        model.recv(recv_tensor, recv_id, recv_rank)

    # Define the behavior for rank 1
    if rank == 1:
        # recv the tensor from rank 0
        recv_tensor = model.tensor(ark.Dims(tensor_len), ark.TensorType.FP16)
        recv_id, recv_rank = 0, 0
        recv_dep = model.recv(recv_tensor, recv_id, recv_rank)

        # The send must be executed after the recv, in the current scheduler,
        # in one depth their will be send operation, compute operation and
        # recv operation
        send_tensor = model.identity(recv_tensor, [recv_dep])

        # Send the received tensor back to rank 0 after an identity operation
        send_id, dst_rank = 1, 0
        model.send(send_tensor, send_id, dst_rank, tensor_size)
        model.send_done(send_tensor, send_id, dst_rank)

    # Create an executor for the model
    exe = ark.Executor(rank, rank, world_size, model, "test_sendrecv_ping_pong")
    exe.compile()

    # Launch the execution and perform data transfers
    exe.launch()
    # Copy send data to GPU0
    if rank == 0:
        exe.tensor_memcpy_host_to_device(send_tensor, np_inputs)

    exe.run(1)
    exe.stop()

    # Copy data back to host and calculate errors
    host_output = np.zeros(tensor_len, dtype=np.float16)
    exe.tensor_memcpy_device_to_host(host_output, recv_tensor)

    # Print output and error information
    print("host_output:", host_output)
    print("np_inputs:", np_inputs)
    max_error = np.max(np.abs(host_output - np_inputs))
    mean_error = np.mean(np.abs(host_output - np_inputs))
    print("max error:", max_error, "mean error:", mean_error)
    print("rank:", rank, "done")


def sendrecv_test_ping_pong():
    ark.init()
    num_processes = world_size  # number of processes
    processes = []
    np_inputs = np.random.rand(tensor_len).astype(np.float16)

    # Create a process for each GPU
    for i in range(num_processes):
        process = multiprocessing.Process(
            target=sendrecv_test_ping_pong_function, args=(i, np_inputs)
        )
        process.start()
        processes.append(process)

    # Join the processes after completion
    for process in processes:
        process.join()


if __name__ == "__main__":
    sendrecv_test_ping_pong()
