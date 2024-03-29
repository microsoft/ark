# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import transformer_pytorch
import transformer_ark
from transformer_utils import *


def test_PoswiseFeedForwardNet():
    ark.init()

    # Create a Model instance
    model = ark.Model()

    input_tensor = model.tensor(
        ark.Dims(batch_size, seq_len, d_model), ark.TensorType.FP16
    )

    ark_model = transformer_ark.PoswiseFeedForwardNet(model)
    output_tensor = ark_model.forward(input_tensor)
    # Test the mul method
    exe = ark.Executor(0, 0, 1, model, "test_PoswiseFeedForwardNet")
    exe.compile()
    input_tensor_host = (
        (np.random.rand(batch_size, seq_len, d_model) - 0.5) * 0.1
    ).astype(np.float16)

    exe.launch()
    input_tensor.from_numpy(input_tensor_host)

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

    output_tensor.to_numpy(output_tensor_host)

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


def test_ScaledDotProductAttention():
    ark.init()

    # Create a Model instance
    model = ark.Model()

    Q = model.tensor(
        ark.Dims(batch_size, n_heads, seq_len, d_k), ark.TensorType.FP16
    )
    K = model.tensor(
        ark.Dims(batch_size, n_heads, seq_len, d_k), ark.TensorType.FP16
    )
    V = model.tensor(
        ark.Dims(batch_size, n_heads, seq_len, d_v), ark.TensorType.FP16
    )

    ark_model = transformer_ark.ScaledDotProductAttention(model)

    attn_mask = model.tensor(
        ark.Dims(batch_size * n_heads, seq_len, seq_len), ark.TensorType.FP16
    )

    context, attn = ark_model.forward(Q, K, V, attn_mask)

    # Test the mul method
    exe = ark.Executor(0, 0, 1, model, "test_ScaledDotProductAttention")
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
    Q.from_numpy(Q_host)
    K.from_numpy(K_host)
    V.from_numpy(V_host)
    transformer_ark.attn_pad_mask_init(attn_mask, exe, input_seq_len)

    exe.run(1)
    exe.stop()

    context_host = np.zeros(
        (batch_size, n_heads, seq_len, d_v), dtype=np.float16
    )
    attn_host = np.zeros(
        (batch_size, n_heads, seq_len, seq_len), dtype=np.float16
    )

    context.to_numpy(context_host)
    attn.to_numpy(attn_host)

    input_seq = np.zeros((batch_size, seq_len), dtype=np.int32)
    for i in range(batch_size):
        for j in range(seq_len):
            if j < input_seq_len:
                input_seq[i][j] = 1
    input_seq_torch = torch.from_numpy(input_seq)
    attn_mask_torch = transformer_pytorch.get_attn_pad_mask(
        input_seq_torch, input_seq_torch
    )
    attn_mask_torch = attn_mask_torch.unsqueeze(1).repeat(1, n_heads, 1, 1)
    torch_Q = torch.from_numpy(Q_host.astype(np.float32))
    torch_K = torch.from_numpy(K_host.astype(np.float32))
    torch_V = torch.from_numpy(V_host.astype(np.float32))

    torch_model = transformer_pytorch.ScaledDotProductAttention()

    context_torch, attn_torch = torch_model(
        torch_Q, torch_K, torch_V, attn_mask_torch
    )

    gt_context = context_torch.detach().numpy().astype(np.float16)
    gt_attn = attn_torch.detach().numpy().astype(np.float16)

    context_max_error = np.max(np.abs(context_host - gt_context))
    context_avg_error = np.mean(np.abs(context_host - gt_context))
    attn_max_error = np.max(np.abs(attn_host - gt_attn))
    attn_avg_error = np.mean(np.abs(attn_host - gt_attn))
    print("scaled dot product attention test")
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


def test_MultiHeadAttention():
    ark.init()

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

    ark_model = transformer_ark.MultiHeadAttention(model)

    attn_mask = model.tensor(
        ark.Dims(batch_size * n_heads, seq_len, seq_len), ark.TensorType.FP16
    )

    context, attn = ark_model.forward(Q, K, V, attn_mask)
    # Test the mul method
    exe = ark.Executor(0, 0, 1, model, "test_MultiHeadAttention")
    exe.compile()
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

    param = {"W_Q": W_Q_host, "W_K": W_K_host, "W_V": W_V_host, "fc": fc_host}
    ark_model.init_model(param, exe)

    exe.launch()
    Q.from_numpy(Q_host)
    K.from_numpy(K_host)
    V.from_numpy(V_host)
    transformer_ark.attn_pad_mask_init(attn_mask, exe, input_seq_len)
    exe.run(1)
    exe.stop()

    context_host = np.zeros(
        (batch_size, seq_len, n_heads * d_v), dtype=np.float16
    )
    attn_host = np.zeros(
        (batch_size, n_heads, seq_len, seq_len), dtype=np.float16
    )

    context.to_numpy(context_host)
    attn.to_numpy(attn_host)

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


def test_EncoderLayer():
    ark.init()

    # Create a Model instance
    model = ark.Model()

    enc_inputs = model.tensor(
        ark.Dims(batch_size, seq_len, d_model), ark.TensorType.FP16
    )

    ark_model = transformer_ark.EncoderLayer(model)

    attn_mask = model.tensor(
        ark.Dims(batch_size * n_heads, seq_len, seq_len), ark.TensorType.FP16
    )

    context, attn = ark_model.forward(enc_inputs, attn_mask)
    # Test the mul method
    exe = ark.Executor(0, 0, 1, model, "test_EncoderLayer")
    exe.compile()
    enc_inputs_host = (
        (np.random.rand(batch_size, seq_len, d_model) - 0.5)
    ).astype(np.float16)

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
    pos_ffn_weight_1_host = (
        (np.random.rand(d_model, d_ff) - 0.5) * 0.1
    ).astype(np.float16)
    pos_ffn_weight_2_host = (
        (np.random.rand(d_ff, d_model) - 0.5) * 0.1
    ).astype(np.float16)

    param = {
        "enc_self_attn.W_Q": W_Q_host,
        "enc_self_attn.W_K": W_K_host,
        "enc_self_attn.W_V": W_V_host,
        "enc_self_attn.fc": fc_host,
        "pos_ffn.weight_1": pos_ffn_weight_1_host,
        "pos_ffn.weight_2": pos_ffn_weight_2_host,
    }
    ark_model.init_model(param, exe)

    exe.launch()
    enc_inputs.from_numpy(enc_inputs_host)
    transformer_ark.attn_pad_mask_init(attn_mask, exe, input_seq_len)
    exe.run(1)
    exe.stop()

    context_host = np.zeros(
        (batch_size, seq_len, n_heads * d_v), dtype=np.float16
    )
    attn_host = np.zeros(
        (batch_size, n_heads, seq_len, seq_len), dtype=np.float16
    )

    context.to_numpy(context_host)
    attn.to_numpy(attn_host)

    torch_enc_inputs = torch.from_numpy(enc_inputs_host.astype(np.float32))

    torch_model = transformer_pytorch.EncoderLayer()
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
    context_torch, attn_torch = torch_model(torch_enc_inputs, attn_mask_torch)

    gt_context = context_torch.detach().numpy().astype(np.float16)
    # gt_context = gt_context.reshape(batch_size*n_heads* seq_len* d_v)
    gt_attn = attn_torch.detach().numpy().astype(np.float16)

    context_max_error = np.max(np.abs(context_host - gt_context))
    context_avg_error = np.mean(np.abs(context_host - gt_context))
    attn_max_error = np.max(np.abs(attn_host - gt_attn))
    attn_avg_error = np.mean(np.abs(attn_host - gt_attn))
    print("EncoderLayer test")
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


def test_Encoder():
    ark.init()

    # Create a Model instance
    model = ark.Model()

    enc_inputs = model.tensor(
        ark.Dims(batch_size, seq_len, d_model), ark.TensorType.FP16
    )

    ark_model = transformer_ark.Encoder(model)

    attn_mask = model.tensor(
        ark.Dims(batch_size * n_heads, seq_len, seq_len), ark.TensorType.FP16
    )

    context, attns = ark_model.forward(enc_inputs, attn_mask)
    # Test the mul method
    exe = ark.Executor(0, 0, 1, model, "test_Encoder")
    exe.compile()
    enc_inputs_host = (
        (np.random.rand(batch_size, seq_len, d_model) - 0.5)
    ).astype(np.float16)

    param = {}

    for i in range(n_layers):
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
        pos_ffn_weight_1_host = (
            (np.random.rand(d_model, d_ff) - 0.5) * 0.1
        ).astype(np.float16)
        pos_ffn_weight_2_host = (
            (np.random.rand(d_ff, d_model) - 0.5) * 0.1
        ).astype(np.float16)

        prefix = "layers." + str(i) + "."
        param[prefix + "enc_self_attn.W_Q"] = W_Q_host
        param[prefix + "enc_self_attn.W_K"] = W_K_host
        param[prefix + "enc_self_attn.W_V"] = W_V_host
        param[prefix + "enc_self_attn.fc"] = fc_host
        param[prefix + "pos_ffn.weight_1"] = pos_ffn_weight_1_host
        param[prefix + "pos_ffn.weight_2"] = pos_ffn_weight_2_host

    ark_model.init_model(param, exe)

    exe.launch()
    enc_inputs.from_numpy(enc_inputs_host)
    transformer_ark.attn_pad_mask_init(attn_mask, exe, input_seq_len)
    exe.run(1)
    exe.stop()

    context_host = np.zeros(
        (batch_size, seq_len, n_heads * d_v), dtype=np.float16
    )

    context.to_numpy(context_host)
    attns_host = []
    for i in range(n_layers):
        attn_host = np.zeros(
            (batch_size, n_heads, seq_len, seq_len), dtype=np.float16
        )
        attns[i].to_numpy(attn_host)
        attns_host.append(attn_host)

    torch_enc_inputs = torch.from_numpy(enc_inputs_host.astype(np.float32))

    torch_model = transformer_pytorch.Encoder()
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
    context_torch, attns_torch = torch_model(torch_enc_inputs, attn_mask_torch)

    gt_context = context_torch.detach().numpy().astype(np.float16)
    # gt_context = gt_context.reshape(batch_size*n_heads* seq_len* d_v)
    # gt_attn = attn_torch.detach().numpy().astype(np.float16)

    context_max_error = np.max(np.abs(context_host - gt_context))
    context_avg_error = np.mean(np.abs(context_host - gt_context))
    attns_max_error = []
    attns_avg_error = []
    for i in range(n_layers):
        attn_host = attns_host[i]
        attn_torch = attns_torch[i]
        gt_attn = attn_torch.detach().numpy().astype(np.float16)
        attn_max_error = np.max(np.abs(attn_host - gt_attn))
        attn_avg_error = np.mean(np.abs(attn_host - gt_attn))
        attns_max_error.append(attn_max_error)
        attns_avg_error.append(attn_avg_error)
    # attn_max_error = np.max(np.abs(attn_host - gt_attn))
    # attn_avg_error = np.mean(np.abs(attn_host - gt_attn))
    print("Encoder test")
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
        attns_max_error,
        "avg attn error: ",
        attns_avg_error,
    )
    # print(context_host)
    # print(gt_context)


if __name__ == "__main__":
    test_PoswiseFeedForwardNet()
    test_ScaledDotProductAttention()
    test_MultiHeadAttention()
    test_EncoderLayer()
    # TODO: test_Encoder() and test_Decoder()
    # test_Encoder()
