
from transformer_ark import *

if __name__ == "__main__":
    ark.init()

    # Create a Model instance
    model = ark.Model()

    input_tensor = model.tensor(
        ark.Dims(batch_size, seq_len, d_model), ark.TensorType.FP16)

    ark_model = PoswiseFeedForwardNetArk(model)
    output_tensor = ark_model.forward(input_tensor)
    # Test the mul method
    exe = ark.Executor(0, 0, 1, model, "test_python_bindings")
    exe.compile()
    input_tensor_host = ((np.random.rand(
        batch_size, seq_len, d_model)-0.5)*0.1).astype(np.float16)


    exe.launch()
    exe.tensor_memcpy_host_to_device(input_tensor, input_tensor_host)

    weight_1_host = ((np.random.rand(d_model, d_ff)-0.5)*0.1).astype(np.float16)
    weight_2_host = ((np.random.rand(d_ff, d_model)-0.5)*0.1).astype(np.float16)

    param = {"weight_1": weight_1_host, "weight_2": weight_2_host}

    ark_model.init_model(param, exe)

    exe.run(1)
    exe.stop()

    output_tensor_host = np.zeros((batch_size, seq_len, d_model), dtype=np.float16)

    exe.tensor_memcpy_device_to_host(output_tensor_host, output_tensor)

    input_tensor_host_float32 = input_tensor_host.astype(np.float32)

    torch_input = torch.from_numpy(input_tensor_host_float32)

    torch_model = PoswiseFeedForwardNetPytorch()

    torch_model.init_model(param)

    gt = torch_model(torch_input).detach().numpy().astype(np.float16)

    # test if the result is correct
    max_error = np.max(np.abs(output_tensor_host - gt))
    avg_error = np.mean(np.abs(output_tensor_host - gt))
    print(input_tensor_host)
    print(output_tensor_host)
    print(gt)
    print("transformer test ", "batch_size:", batch_size, "seq_len:", seq_len, "d_model:", d_model, "d_ff:", d_ff, "max error: ", max_error, "avg error: ", avg_error)

    print("ark test success")

