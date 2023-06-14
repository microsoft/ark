# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import transformer_pytorch
import transformer_ark
from transformer_utils import *


def test_poswise_feed_forward_net():
    ark.init()

    # Create a Model instance
    model = ark.Model()

    input_tensor = model.tensor(
        ark.Dims(batch_size, seq_len, d_model), ark.TensorType.FP16
    )

    ark_model = transformer_ark.PoswiseFeedForwardNet(model)
    output_tensor = ark_model.forward(input_tensor)
    # Test the mul method
    exe = ark.Executor(0, 0, 1, model, "test_python_bindings")
    exe.compile()
    input_tensor_host = (
        (np.random.rand(batch_size, seq_len, d_model) - 0.5) * 0.1
    ).astype(np.float16)

    exe.launch()
    exe.tensor_memcpy_host_to_device(input_tensor, input_tensor_host)

    weight_1_host = ((np.random.rand(d_model, d_ff) - 0.5) * 0.1).astype(
        np.float16
    )
    weight_2_host = ((np.random.rand(d_ff, d_model) - 0.5) * 0.1).astype(
        np.float16
    )

    param = {"weight_1": weight_1_host, "weight_2": weight_2_host}

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
    # print(input_tensor_host)
    # print(output_tensor_host)
    # print(gt)
    print("poswise feed forward net test")
    print("batch_size:", batch_size, "seq_len:", seq_len, "d_model:", d_model, "d_ff:", d_ff)
    print("max error: ", max_error , "avg error: ", avg_error)

def test_ScaledDotProductAttention():
    ark.init()

    # Create a Model instance
    model = ark.Model()

    Q = model.tensor(ark.Dims(batch_size, n_heads,
                     seq_len, d_k), ark.TensorType.FP16)
    K = model.tensor(ark.Dims(batch_size, n_heads,
                     seq_len, d_k), ark.TensorType.FP16)
    V = model.tensor(ark.Dims(batch_size, n_heads,
                     seq_len, d_v), ark.TensorType.FP16)

    ark_model = transformer_ark.ScaledDotProductAttention(model)
    context, attn = ark_model.forward(Q, K, V)
    # Test the mul method
    exe = ark.Executor(0, 0, 1, model, "test_python_bindings")
    exe.compile()
    Q_host = ((np.random.rand(batch_size, n_heads, seq_len, d_k) - 0.5)).astype(
        np.float16
    )
    K_host = ((np.random.rand(batch_size, n_heads, seq_len, d_k) - 0.5)).astype(
        np.float16
    )
    V_host = ((np.random.rand(batch_size, n_heads, seq_len, d_v) - 0.5)).astype(
        np.float16
    )

    exe.launch()
    exe.tensor_memcpy_host_to_device(Q, Q_host)
    exe.tensor_memcpy_host_to_device(K, K_host)
    exe.tensor_memcpy_host_to_device(V, V_host)

    exe.run(1)
    exe.stop()

    context_host = np.zeros((batch_size,n_heads, seq_len, d_v), dtype=np.float16)
    attn_host = np.zeros((batch_size,n_heads, seq_len, seq_len), dtype=np.float16)

    exe.tensor_memcpy_device_to_host(context_host, context)
    exe.tensor_memcpy_device_to_host(attn_host, attn)

    torch_Q = torch.from_numpy(Q_host.astype(np.float32))
    torch_K = torch.from_numpy(K_host.astype(np.float32))
    torch_V = torch.from_numpy(V_host.astype(np.float32))

    torch_model = transformer_pytorch.ScaledDotProductAttention()

    context_torch, attn_torch = torch_model(torch_Q, torch_K, torch_V)

    gt_context = context_torch.detach().numpy().astype(np.float16)
    gt_attn = attn_torch.detach().numpy().astype(np.float16)
    
    context_max_error = np.max(np.abs(context_host - gt_context))
    context_avg_error = np.mean(np.abs(context_host - gt_context))
    attn_max_error = np.max(np.abs(attn_host - gt_attn))
    attn_avg_error = np.mean(np.abs(attn_host - gt_attn))
    print("scaled dot product attention test") 
    print("batch_size:", batch_size, "seq_len:", seq_len, "d_model:", d_model, "d_ff:", d_ff)
    print("max context error: ", context_max_error, "avg context error: ", context_avg_error, "max attn error: ", attn_max_error, "avg attn error: ", attn_avg_error)

if __name__ == "__main__":
    test_poswise_feed_forward_net()
    test_ScaledDotProductAttention()
