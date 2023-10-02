# Multi-GPU Parallel Training

This tutorial will guide you through implementing parallel training across multiple GPUs using multi-process programming. Parallel training is beneficial for harnessing the full potential of multiple GPUs and accelerating the training process.

## Prerequisites

Before you begin, make sure you have completed the [installation](./install.md) process. Next, you can run the tutorial example at [multi_gpu_tutorial.py](../examples/tutorial/multi_gpu_tutorial.py) to learn how to use ARK to communicate between different GPUs.

## Ping-Pong Transfer Example

In this example, we will demonstrate communication between two GPUs (GPU 0 and GPU 1) using a simple ping-pong transfer. We will send a tensor from GPU 0 to GPU 1, and then send back the received tensor from GPU 1 to GPU 0.

To begin, we need to import the necessary modules. We will use the `multiprocessing` module to create multiple processes. Alternatively, we can use MPI to create multiple processes. We will also use the `numpy` module.

```python
import ark
import numpy as np
import multiprocessing

world_size = 2

tensor_len = 2048
tensor_size = tensor_len * 2
```

Since we have 2 GPUs, we need to run 2 processes. Each process will be assigned to a GPU. 

```python
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

# Use process.join() to wait for the completion of all processes
for process in processes:
    process.join()
```

The following is the main function for the two processes. We first set the `rank` and `world_size` of the current process. In ARK, we assume that one process corresponds to one GPU. 



```python
def sendrecv_test_ping_pong_function(rank, np_inputs):
    print("rank:", rank)
    ark.set_rank(rank)
    ark.set_world_size(world_size)
```


The first Model on process 0 will send send_tensor from GPU 0 to GPU 1. Since multiple tensors can be sent to the same GPU, an identifier `id` is required to distinguish the tensor. Here, we set the first `id` to 0. Then, the first process will receive recv_tensor from GPU 1. The received `id` will be 1.

```python
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
```

The following is the model definition for GPU1. Here, GPU1 receives the tensor from GPU0 and sends it back to GPU0.

```python
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
```

Note that there is a line that describes the dependency between the send and recv operation:

```python
send_tensor = model.identity(recv_tensor, [recv_dep])
```

This is because the send operation must be executed after the recv operation. In the current scheduler, if this dependency is not specified, the send operation may be executed before the recv operation, causing an error. We will improve the scheduler in the future to automatically handle this situation.
    
Finally, we can launch the runtime to compile the kernel code and create contexts for each GPU. The connection between the two GPUs will be established automatically.

After we lauch the ARK model, we need to copy the send tensor to GPU0 to initialize the send tensor. Then we can run the ARK program.

```python
    # Construct the ARK runtime
    runtime = ark.Runtime()

    # Launch the ARK runtime
    runtime.launch()

    # Copy send data to GPU0
    if rank == 0:
        send_tensor.from_numpy(np_inputs)

    # Run the ARK program
    runtime.run()
```

Finally, we can copy the recv_tensor to the host to check the result. The recv_tensor on both GPUs should be the same as the original send tensor.

```python
    # Copy data back to host and calculate errors
    host_output = recv_tensor.to_numpy()

    # Print output and error information
    print("host_output:", host_output)
    print("np_inputs:", np_inputs)
    max_error = np.max(np.abs(host_output - np_inputs))
    mean_error = np.mean(np.abs(host_output - np_inputs))
    print("max error:", max_error, "mean error:", mean_error)
    print("rank:", rank, "done")
```
