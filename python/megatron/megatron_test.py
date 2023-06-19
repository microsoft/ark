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
        ark.Dims(1024), ark.TensorType.FP16
    )

    model.all_reduce(input_tensor, rank, world_size)

    exe = ark.Executor(rank, rank, world_size, model, "test_python_bindings")
    exe.compile()

    exe.launch()

    exe.run(1)
    exe.stop()

  
if __name__ == "__main__":  
    ark.init()

    num_processes = world_size  # 设置进程数量  
    processes = []  
  
    for i in range(num_processes):  
        process = multiprocessing.Process(target=my_function, args=(i,))  
        process.start()  
        processes.append(process)  
  
    for process in processes:  
        process.join()  
        process.wait()


