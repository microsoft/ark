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
    # Initialize the ARK model
    ark.init_model(rank, world_size)

    # Define the behavior for rank 0
    if rank == 0:
        send_tensor = ark.tensor(ark.Dims(tensor_len), ark.FP16)
        recv_tensor = ark.tensor(ark.Dims(tensor_len), ark.FP16)

        # send the tensor to rank 1
        send_id, dst_rank = 0, 1
        send_dep_tensor = ark.send(send_tensor, send_id, dst_rank, tensor_size)
        # A identity operation is used to add an exectuion dependency and
        # make sure execution order correct
        ark.send_done(
            ark.identity(send_tensor, [send_dep_tensor]), send_id, dst_rank
        )
        # recv the tensor from rank 1
        recv_id, recv_rank = 1, 1
        ark.recv(recv_tensor, recv_id, recv_rank)

    # Define the behavior for rank 1
    if rank == 1:
        # recv the tensor from rank 0
        recv_tensor = ark.tensor(ark.Dims(tensor_len), ark.FP16)
        recv_id, recv_rank = 0, 0
        recv_dep = ark.recv(recv_tensor, recv_id, recv_rank)

        # The send must be executed after the recv, identity is used to
        # add an exectuion dependency between the two operations
        send_tensor = ark.identity(recv_tensor, [recv_dep])

        # Send the received tensor back to rank 0
        send_id, dst_rank = 1, 0
        send_dep_tensor = ark.send(send_tensor, send_id, dst_rank, tensor_size)
        # A identity operation is used to add an exectuion dependency and
        # make sure execution order correct
        ark.send_done(
            ark.identity(send_tensor, [send_dep_tensor]), send_id, dst_rank
        )

    # Launch the ARK runtime
    ark.launch()

    # Copy send data to GPU0
    if rank == 0:
        ark.tensor_memcpy_host_to_device(send_tensor, np_inputs)

    # Run the ARK program
    ark.run()

    # Copy data back to host and calculate errors
    host_output = np.zeros(tensor_len, dtype=np.float16)
    ark.tensor_memcpy_device_to_host(host_output, recv_tensor)

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
