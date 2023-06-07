import ark

import torch
import torch.nn.functional as F
import numpy as np


class layer_norm(torch.nn.Module):
    def __init__(self, eps=1e-5):
        super(layer_norm, self).__init__()

    def forward(self, x):
        x=x.reshape(x.shape[0],-1)
        mean = x.mean(-1, keepdim=True)
        return mean
        # std = x.std(-1, keepdim=True)
        # return self.gamma * (x - mean) / (std + self.eps) + self.beta


class layer_norm_ark():
    def __init__(self, model):
        self.model = model

    def forward(self, inputs):
        input_shape = inputs.padded_shape()
        inputs = self.model.reshape(inputs, ark.Dims(
            input_shape[0], 1, input_shape[1] * input_shape[2]))
        sum = self.model.reduce(inputs, 2)
        scale_val = 1.0 / (input_shape[1] * input_shape[2])
        mean = self.model.scale(sum, scale_val)
        x_minus_mean = self.model.add(inputs, self.model.scale(mean,-1))
        std_square = self.model.reduce(self.model.mul(x_minus_mean,x_minus_mean))
        std = self.model.sqrt(std_square, 2)
        return self.model.div(x_minus_mean, self.model.add(std, 1e-5))


def test_layernorm_internal(batch_size, m, n, data_type="float"):
    ark.init()

    # Create a Model instance
    model = ark.Model()
    if data_type == "float":
        ark_data_type = ark.TensorType.FP32
        numpy_data_type = np.float32
        elesize = 4
    elif data_type == "half":
        ark_data_type = ark.TensorType.FP16
        numpy_data_type = np.float16
        elesize = 2
    input_tensor = model.tensor(
        ark.Dims(batch_size, m, n), ark_data_type
    )
    ark_layer_norm = layer_norm_ark(model)
    output_tensor = ark_layer_norm.forward(input_tensor)
    output_tensor_size = output_tensor.ldims_bytes()
    output_tensor_dim = output_tensor_size // elesize
    print("output_tensor_shape", output_tensor_size)
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

    output_tensor_host = np.zeros(output_tensor_dim, dtype=numpy_data_type)

    exe.tensor_memcpy_device_to_host(output_tensor_host, output_tensor)

    input_tensor_host_float32 = input_tensor_host.astype(np.float32)

    torch_input = torch.from_numpy(input_tensor_host_float32)

    torch_layernorm = layer_norm()

    gt =  torch_layernorm(torch_input).numpy()

    # test if the result is correct
    max_error = np.max(np.abs(output_tensor_host - gt))
    avg_error = np.mean(np.abs(output_tensor_host - gt))
    # np.testing.assert_allclose(output_tensor_host, gt, rtol=1e-3, atol=1e-3)
    print(input_tensor_host)
    print(output_tensor_host)
    print(gt)
    print("layernorm test ", "batch_size:", batch_size, "m:", m, "n:", n,
          "data_type:", data_type, "max error: ", max_error, "avg error: ", avg_error)


if __name__ == "__main__":
    batch_size = 1
    m = 32
    n = 512
    test_layernorm_internal(batch_size, m, n, "float")
