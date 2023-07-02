# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import transformer_pytorch
import megatron_ark
from transformer_utils import *
import multiprocessing


def test_PoswiseFeedForwardNet_process(rank, param):
    # Create a Model instance
    model = ark.Model()

    input_tensor = model.tensor(
        ark.Dims(batch_size, seq_len, d_model), ark.TensorType.FP16
    )

    ark_model = megatron_ark.PoswiseFeedForwardNet(model, rank)

    output_tensor = ark_model.forward(input_tensor)

    exe = ark.Executor(rank, rank, num_gpu, model, "test_python_bindings")
    exe.compile()

    exe.launch()

    input_tensor_host = param["input_tensor"]
    exe.tensor_memcpy_host_to_device(input_tensor, input_tensor_host)
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
    # print("input_tensor_host", input_tensor_host)
    # print("output_tensor_host", output_tensor_host)
    # print("gt", gt)
    print("Megatron poswise feed forward net test")
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


def test_PoswiseFeedForwardNet():
    # set random seed
    np.random.seed(1234)
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
    multi_process_test_main(test_PoswiseFeedForwardNet_process, param)


def test_MultiHeadAttention_process(rank, param):
    # Create a Model instance
    model = ark.Model()

    Q = model.tensor(
        ark.Dims(batch_size, seq_len, d_model), ark.TensorType.FP16
    )
    K = model.tensor(
        ark.Dims(batch_size, seq_len, d_model), ark.TensorType.FP16
    )
    V = model.tensor(
        ark.Dims(batch_size, seq_len, d_model), ark.TensorType.FP16
    )

    ark_model = megatron_ark.MultiHeadAttention(model, rank)

    attn_mask = model.tensor(
        ark.Dims(batch_size * n_heads_per_gpu, seq_len, seq_len),
        ark.TensorType.FP16,
    )

    context, attn = ark_model.forward(Q, K, V, attn_mask)

    exe = ark.Executor(rank, rank, num_gpu, model, "test_python_bindings")
    exe.compile()

    exe.launch()
    Q_host = param["Q"]
    K_host = param["K"]
    V_host = param["V"]
    # exe.tensor_memcpy_host_to_device(Q, Q_host)
    # exe.tensor_memcpy_host_to_device(K, K_host)
    # exe.tensor_memcpy_host_to_device(V, V_host)
    # ark_model.init_model(param, exe)
    exe.run(1)
    exe.stop()

    context_host = np.zeros(
        (batch_size, seq_len, n_heads * d_v), dtype=np.float16
    )
    attn_host = np.zeros(
        (batch_size, n_heads, seq_len, seq_len), dtype=np.float16
    )

    exe.tensor_memcpy_device_to_host(context_host, context)
    exe.tensor_memcpy_device_to_host(attn_host, attn)

    torch_Q = torch.from_numpy(Q_host.astype(np.float32))
    torch_K = torch.from_numpy(K_host.astype(np.float32))
    torch_V = torch.from_numpy(V_host.astype(np.float32))

    torch_model = transformer_pytorch.MultiHeadAttention()
    torch_model.init_model(param)
    input_seq = np.zeros((batch_size, seq_len), dtype=np.int32)
    for i in range(batch_size):
        for j in range(seq_len):
            if j < input_seq_len:
                input_seq[i][j] = 1
    input_seq_torch = torch.from_numpy(input_seq)
    attn_mask_torch = transformer_pytorch.get_attn_pad_mask(
        input_seq_torch, input_seq_torch
    )
    context_torch, attn_torch = torch_model(
        torch_Q, torch_K, torch_V, attn_mask_torch
    )

    gt_context = context_torch.detach().numpy().astype(np.float16)
    # gt_context = gt_context.reshape(batch_size*n_heads* seq_len* d_v)
    gt_attn = attn_torch.detach().numpy().astype(np.float16)

    context_max_error = np.max(np.abs(context_host - gt_context))
    context_avg_error = np.mean(np.abs(context_host - gt_context))
    attn_max_error = np.max(np.abs(attn_host - gt_attn))
    attn_avg_error = np.mean(np.abs(attn_host - gt_attn))
    print("multi head attention test")
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
    print(
        "max context error: ",
        context_max_error,
        "avg context error: ",
        context_avg_error,
        "max attn error: ",
        attn_max_error,
        "avg attn error: ",
        attn_avg_error,
    )
    # print(context_host)
    # print(gt_context)


def test_MultiHeadAttention():
    # set random seed
    np.random.seed(1234)
    Q_host = ((np.random.rand(batch_size, seq_len, d_model) - 0.5)).astype(
        np.float16
    )
    K_host = ((np.random.rand(batch_size, seq_len, d_model) - 0.5)).astype(
        np.float16
    )
    V_host = ((np.random.rand(batch_size, seq_len, d_model) - 0.5)).astype(
        np.float16
    )

    W_Q_host = ((np.random.rand(d_model, d_k * n_heads) - 0.5)).astype(
        np.float16
    )
    W_K_host = ((np.random.rand(d_model, d_k * n_heads) - 0.5)).astype(
        np.float16
    )
    W_V_host = ((np.random.rand(d_model, d_v * n_heads) - 0.5)).astype(
        np.float16
    )
    fc_host = ((np.random.rand(d_v * n_heads, d_model) - 0.5)).astype(
        np.float16
    )
    param = {
        "Q": Q_host,
        "K": K_host,
        "V": V_host,
        "W_Q": W_Q_host,
        "W_K": W_K_host,
        "W_V": W_V_host,
        "fc": fc_host,
    }
    multi_process_test_main(test_MultiHeadAttention_process, param)


if __name__ == "__main__":
    # test_PoswiseFeedForwardNet()
    test_MultiHeadAttention()
