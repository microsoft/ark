// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/executor.hpp"
#include "gpu/gpu.hpp"
#include "logging.hpp"
#include "model/model_op.hpp"
#include "ops_test_common.hpp"

ark::unittest::State test_ops_placeholder() {
    ark::Model model;
    ark::Dims shape{10, 1};

    // Allocate GPU memory for the external buffer
    float *d_ext_buffer = nullptr;
    ark::gpuMalloc(&d_ext_buffer, shape.nelems() * sizeof(float));

    // Initialize GPU Memory
    std::vector<float> h_ext_buffer(shape.nelems());
    std::iota(h_ext_buffer.begin(), h_ext_buffer.end(), 1.0f);
    ark::gpuMemcpy(d_ext_buffer, h_ext_buffer.data(),
                   shape.nelems() * sizeof(float), ark::gpuMemcpyHostToDevice);

    // Associate the initialized device buffer with a tensor produced from a
    // placeholder operation
    ark::Tensor tns =
        model.placeholder(shape, ark::FP32, {}, {}, {}, -1, "", d_ext_buffer);

    ark::Tensor res = model.add(tns, 1.0);

    ark::DefaultExecutor exe(model);

    exe.launch();
    exe.run(1);
    exe.stop();

    // Copy tensor data from GPU to CPU
    std::vector<float> h_res(shape.nelems(), 0.0f);
    exe.tensor_read(res, h_res);

    for (auto i = 0; i < shape.nelems(); ++i) {
        UNITTEST_EQ(h_res[i], i + 2);
    }

    ark::gpuFree(d_ext_buffer);

    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_ops_placeholder_value_contiguous);
    return ark::unittest::SUCCESS;
}