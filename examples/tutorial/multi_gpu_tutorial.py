# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ark
import numpy as np
import multiprocessing

world_size = 2

tensor_len = 2048
tensor_size = tensor_len * ark.fp16.element_size()


def sendrecv_test_ping_pong_function(rank, np_inputs):
    print("rank:", rank)
    ark.set_rank(rank)
    ark.set_world_size(world_size)

    # Define the behavior for rank 0
    if rank == 0:
        # send a tensor to rank 1
        send_tensor = ark.tensor([tensor_len], ark.fp16)
        send_id, dst_rank = 0, 1
        send_tensor = ark.send(send_tensor, send_id, dst_rank, tensor_size)
        # wait until the send is done
        ark.send_done(send_tensor, send_id, dst_rank)
        # recv the tensor from rank 1
        recv_id, recv_rank = 1, 1
        recv_tensor = ark.recv(recv_id, recv_rank, bytes=tensor_size)
        # cast received bytes to fp16
        recv_tensor = ark.cast(recv_tensor, ark.fp16)

    # Define the behavior for rank 1
    if rank == 1:
        # recv the tensor from rank 0
        recv_id, recv_rank = 0, 0
        recv_tensor = ark.recv(recv_id, recv_rank, bytes=tensor_size)
        # cast received bytes to fp16
        recv_tensor = ark.cast(recv_tensor, ark.fp16)
        # send the received tensor back to rank 0
        send_tensor = recv_tensor
        send_id, dst_rank = 1, 0
        send_tensor = ark.send(send_tensor, send_id, dst_rank, tensor_size)
        # wait until the send is done
        ark.send_done(send_tensor, send_id, dst_rank)

    # Initialize the ARK runtime
    runtime = ark.Runtime()

    # Launch the ARK runtime
    runtime.launch()

    # Copy send data to GPU0
    if rank == 0:
        send_tensor.from_numpy(np_inputs)

    # Run the ARK program
    runtime.run()

    # Copy data back to host and calculate errors
    host_output = recv_tensor.to_numpy()

    # Print output and error information
    print("host_output:", host_output)
    print("np_inputs:", np_inputs)
    max_error = np.max(np.abs(host_output - np_inputs))
    mean_error = np.mean(np.abs(host_output - np_inputs))
    print("max error:", max_error, "mean error:", mean_error)
    print("rank:", rank, "done")


def sendrecv_test_ping_pong():
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
    ark.init()
    sendrecv_test_ping_pong()
