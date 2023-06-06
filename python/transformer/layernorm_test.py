import ark

import torch
import torch.nn.functional as F
import numpy as np

class layer_norm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super(layer_norm, self).__init__()
        self.gamma = torch.nn.Parameter(torch.ones(hidden_size))
        self.beta = torch.nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class layer_norm_ark():
    def __init__(self, model):
        self.model = model
    def forward(self, inputs):
        mean = self.model.reduce(inputs, 2)
        x_minus_mean = self.model.add(inputs, self.model.scale(mean,-1))
        std = self.model.sqrt(self.model.reduce(self.model.square(x_minus_mean), 2))
        return self.model.add(self.model.mul(self.model.scale(self.model.div(x_minus_mean, self.model.add(std, 1e-5)), self.model.const(1.0)), self.model.const(1.0)), self.model.const(0.0))


def test_layernorm_internal(batch_size, m, n, data_type="float"):
    ark.init()

    # Create a Model instance
    model = ark.Model()
    if data_type == "float":
        ark_data_type = ark.TensorType.FP32
        numpy_data_type = np.float32
    elif data_type == "half":
        ark_data_type = ark.TensorType.FP16
        numpy_data_type = np.float16
    input_tensor = model.tensor(
        ark.Dims(batch_size, m, n), ark_data_type
    )
    ark_layer_norm = layer_norm_ark(model)
    output_tensor = ark_layer_norm.forward(input_tensor)
    # Test the mul method
    exe = ark.Executor(0, 0, 1, model, "ops_layernorm_test")
    exe.compile()
    input_tensor_host = np.random.rand(batch_size, m, n).astype(
        numpy_data_type
    )

    exe.launch()
    exe.tensor_memcpy_host_to_device(input_tensor, input_tensor_host)

    exe.run(1)

    exe.stop()

    output_tensor_host = np.zeros(
        (batch_size, m, 1), dtype=numpy_data_type,
    )

    exe.tensor_memcpy_device_to_host(output_tensor_host, output_tensor)

    input_tensor_host_float32 = input_tensor_host.astype(np.float32)

    torch_input = torch.from_numpy(input_tensor_host_float32)

    gt = torch.sum(torch_input, dim=2, keepdim=True).numpy().astype(numpy_data_type)

    # test if the result is correct
    max_error = np.max(np.abs(output_tensor_host - gt))
    avg_error = np.mean(np.abs(output_tensor_host - gt))
    np.testing.assert_allclose(output_tensor_host, gt, rtol=1e-3, atol=1e-3)
    # print(input_tensor_host)
    # print(output_tensor_host)
    # print(gt)
    print("layernorm test ", "batch_size:", batch_size, "m:", m, "n:", n, "data_type:", data_type, "max error: ", max_error, "avg error: ", avg_error)


if __name__ == "__main__":
    batch_size = 1
    m = 32
    n = 512
    test_layernorm_internal(batch_size, m, n, "float")
