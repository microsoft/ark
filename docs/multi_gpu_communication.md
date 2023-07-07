# Multi-GPU Parallel Training

This tutorial will guide you through implementing parallel training across multiple GPUs using multi-process programming. Parallel training is beneficial for harnessing the full potential of multiple GPUs and accelerating the training process.

## Prerequisites

Before you begin, make sure you have completed the [installation](./install.md) process. Next, you can run the tutorial example at [tutorial.py](../examples/tutorial/tutorial.py) to learn how to use ARK to communicate between different GPUs.

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

Since we have 2 GPUs, we need to run 2 processes. Each process will be assigned to a GPU. Note that `ark.init()` must be called before creating the processes. Otherwise, one process might delete the shared memory created by another process, resulting in an error.

```python
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

# Use process.join() to wait for the completion of all processes
for process in processes:
    process.join()
```

The following is the main function for the two processes. We first create a model instance on each GPU.

The first Model on process 0 will send send_tensor from GPU 0 to GPU 1. Since multiple tensors can be sent to the same GPU, an identifier `id` is required to distinguish the tensor. Here, we set the first `id` to 0. Then, the first process will receive recv_tensor from GPU 1. The received `id` will be 1.

For more information about the `send` and `recv` operator, please refer to the [API documentation](../docs/api.md).

```python
def sendrecv_test_ping_pong_function(rank, np_inputs):
    print("rank:", rank)
    # Create a Model instance
    model = ark.Model()

    # Define the behavior for rank 0
    if rank == 0:
        send_tensor = model.tensor(ark.Dims(tensor_len), ark.TensorType.FP16)
        recv_tensor = model.tensor(ark.Dims(tensor_len), ark.TensorType.FP16)

        # send the tensor to rank 1
        send_id, dst_rank = 0, 1
        model.send(send_tensor, send_id, dst_rank, tensor_size)
        model.send_done(send_tensor, send_id)

        # recv the tensor from rank 1
        recv_id, recv_rank = 1, 1
        model.recv(recv_tensor, recv_id, recv_rank)
```

The following is the model definition for GPU1. Here, GPU1 receives the tensor from GPU0 and sends it back to GPU0.

```python
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
        model.send_done(send_tensor, 1)
```

Note that there is a line that describes the dependency between the send and recv operation:

```python
send_tensor = model.identity(recv_tensor, [recv_dep])
```

This is because the send operation must be executed after the recv operation. In the current scheduler, if this dependency is not specified, the send operation may be executed before the recv operation, causing an error. We will improve the scheduler in the future to automatically handle this situation.
    
Finally, we can create the executor and run the model. We need to specify the rank and world_size of the executor. The rank of the executor is the rank of the process, and the world_size is the number of processes, which is 2 here.
After we lauch the executor, we need to copy the send tensor to GPU0 to initialize the send tensor. Then we can run the model for one step and stop the executor.

```python
    # Create an executor for the model
    exe = ark.Executor(rank, rank, world_size, model, "test_python_bindings")
    exe.compile()

    # Launch the execution and perform data transfers
    exe.launch()
    # Copy send data to GPU0
    if rank == 0:
        exe.tensor_memcpy_host_to_device(send_tensor, np_inputs)

    exe.run(1)
    exe.stop()
```

Finally, we can copy the recv_tensor to the host to check the result. In theory, the recv_tensor on both GPUs should be the same as the original send tensor.

```python
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
```