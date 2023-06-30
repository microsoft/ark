# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import transformer_pytorch
import megatron_ark
from transformer_utils import *
import multiprocessing


def test_PoswiseFeedForwardNet(rank, param):
    # Create a Model instance
    model = ark.Model()

    input_tensor = model.tensor(
        ark.Dims(batch_size, seq_len, d_model), ark.TensorType.FP16
    )

    ark_model = megatron_ark.PoswiseFeedForwardNet(model, rank)

    output_tensor = ark_model.forward(input_tensor)

    exe = ark.Executor(rank, rank, num_gpu, model, "test_python_bindings")
    exe.compile()
    input_tensor_host = param["input_tensor"]
    exe.tensor_memcpy_host_to_device(input_tensor, input_tensor_host)

    exe.launch()
    ark_model.init_model(param, exe)
    exe.run(1)
    exe.stop()

    output_tensor_host = np.zeros(
        (batch_size, seq_len, d_model), dtype=np.float16
    )

    exe.tensor_memcpy_device_to_host(output_tensor_host, output_tensor)

    input_tensor_host_float32 = input_tensor_host.astype(np.float32)

    torch_input = torch.from_numpy(input_tensor_host_float32)

    torch_model = transformer_pytorch.PoswiseFeedForwardNet()

    torch_model.init_model(param)

    gt = torch_model(torch_input).detach().numpy().astype(np.float16)

    # test if the result is correct
    max_error = np.max(np.abs(output_tensor_host - gt))
    avg_error = np.mean(np.abs(output_tensor_host - gt))
    print("input_tensor_host", input_tensor_host)
    print("output_tensor_host", output_tensor_host)
    print("gt", gt)
    print("poswise feed forward net test")
    print(
        "batch_size:",
        batch_size,
        "seq_len:",
        seq_len,
        "d_model:",
        d_model,
        "d_ff:",
        d_ff,
    )
    print("max error: ", max_error, "avg error: ", avg_error)


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
    input_tensor_host = (
        (np.random.rand(batch_size, seq_len, d_model) - 0.5) * 0.1
    ).astype(np.float16)
    weight_1_host = ((np.random.rand(d_model, d_ff) - 0.5) * 0.1).astype(
        np.float16
    )
    weight_2_host = ((np.random.rand(d_ff, d_model) - 0.5) * 0.1).astype(
        np.float16
    )
    param = {
        "input_tensor": input_tensor_host,
        "weight_1": weight_1_host,
        "weight_2": weight_2_host,
    }
    multi_process_test_main(test_PoswiseFeedForwardNet, param)
