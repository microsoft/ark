# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import megatron_pytorch
import megatron_ark
from megatron_utils import *
import multiprocessing 

world_size = 2
  
def my_function(rank): 
    print("rank:", rank) 

    # Create a Model instance
    model = ark.Model()

    input_tensor = model.tensor(
        ark.Dims(2048), ark.TensorType.FP16
    )
    if(rank == 0):
        model.send(input_tensor, 0, 1, 2048)
        # model.send_done(input_tensor, 0)
    if(rank == 1):
        model.recv(input_tensor, 0, 0, 2048)
    # model.all_reduce(input_tensor, rank, world_size)

    exe = ark.Executor(rank, rank, world_size, model, "test_python_bindings")
    exe.compile()

    exe.launch()

    exe.run(1)
    exe.stop()
    print("rank:", rank, "done")
  
if __name__ == "__main__":  
    ark.init()

    num_processes = world_size  # number of processes
    processes = []  
  
    for i in range(num_processes):  
        process = multiprocessing.Process(target=my_function, args=(i,))  
        process.start()  
        processes.append(process)  
  
    for process in processes:  
        process.join()  


