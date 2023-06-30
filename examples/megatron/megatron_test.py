# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import megatron_pytorch
import megatron_ark
from megatron_utils import *


def test_PoswiseFeedForwardNetTensorParallel(rank, np_inputs):
    # Create a Model instance
    model = ark.Model()

    input_tensor = model.tensor(
        ark.Dims(batch_size, seq_len, d_model), ark.TensorType.FP16
    )

    PoswiseFeedForwardNetTensorParallel = (
        megatron_ark.PoswiseFeedForwardNetTensorParallel(model)
    )

    PoswiseFeedForwardNetTensorParallel.forward(input_tensor, rank)

    exe = ark.Executor(rank, rank, num_gpu, model, "test_python_bindings")
    exe.compile()

    exe.launch()

    exe.run(1)
    exe.stop()


def multi_process_test_main(func, np_inputs):
    ark.init()
    num_processes = num_gpu  # number of processes
    processes = []
    for i in range(num_processes):
        process = multiprocessing.Process(target=func, args=(i, np_inputs))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()


if __name__ == "__main__":
    multi_process_test_main(test_PoswiseFeedForwardNetTensorParallel, None)
