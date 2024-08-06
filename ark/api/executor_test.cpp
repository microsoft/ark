// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/executor.hpp"

#include "gpu/gpu.hpp"
#include "model/model_json.hpp"
#include "unittest/unittest_utils.h"

template <bool LoopMode>
ark::unittest::State test_executor() {
    ark::gpuStream stream;
    UNITTEST_EQ(
        ark::gpuStreamCreateWithFlags(&stream, ark::gpuStreamNonBlocking),
        ark::gpuSuccess);

    ark::Model empty;
    {
        ark::DefaultExecutor executor(empty, 0, stream, {}, "test", LoopMode);
        UNITTEST_EQ(executor.device_id(), 0);
        UNITTEST_EQ(executor.stream(), stream);

        executor.compile();
        executor.launch();
        executor.run(1);
        executor.wait();
        executor.stop();
        executor.destroy();
    }
    {
        ark::DefaultExecutor executor(empty, 0, stream, {}, "test", LoopMode);
        executor.compile();
        executor.launch();
        executor.run(1);
        executor.wait();
        executor.stop();

        executor.launch();
        executor.run(1);
        executor.wait();
        executor.stop();

        executor.destroy();
    }
    {
        ark::DefaultExecutor executor(empty, 0, stream, {}, "test", LoopMode);
        UNITTEST_THROW(executor.launch(), ark::InvalidUsageError);

        executor.compile();
        executor.launch();
        executor.launch();  // Will be ignored with a warning.
        executor.run(1);
        executor.wait();
        executor.wait();  // nothing to do

        // Stop & destroy automatically.
    }

    UNITTEST_EQ(ark::gpuStreamDestroy(stream), ark::gpuSuccess);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_executor_loop() { return test_executor<true>(); }

ark::unittest::State test_executor_no_loop() { return test_executor<false>(); }

ark::unittest::State test_executor_tensor_read_write() {
    // Alloc CPU array
    std::vector<float> host_data(1024);
    void *host_ptr = host_data.data();
    for (size_t i = 0; i < host_data.size(); ++i) {
        host_data[i] = static_cast<float>(i);
    }

    // Alloc GPU array
    void *dev_ptr;
    UNITTEST_EQ(ark::gpuMalloc(&dev_ptr, 1024 * sizeof(float)),
                ark::gpuSuccess);

    // Create an ARK tensor
    ark::Model m;
    auto tensor = m.tensor({1024}, ark::FP32);
    m.noop(tensor);

    ark::DefaultExecutor executor(m, 0);
    executor.compile();
    executor.launch();

    // Copy data from CPU array to ARK tensor
    executor.tensor_write(tensor, host_ptr, 1024 * sizeof(float));

    // Copy data from ARK tensor to GPU array
    executor.tensor_read(tensor, dev_ptr, 1024 * sizeof(float), nullptr, true);

    // Check the data
    std::vector<float> dev_data(1024);
    executor.tensor_read(tensor, dev_data.data(), 1024 * sizeof(float));
    for (size_t i = 0; i < dev_data.size(); ++i) {
        UNITTEST_EQ(dev_data[i], static_cast<float>(i));
        dev_data[i] = -1;
    }

    UNITTEST_EQ(ark::gpuMemcpy(dev_data.data(), dev_ptr, 1024 * sizeof(float),
                               ark::gpuMemcpyDeviceToHost),
                ark::gpuSuccess);
    for (size_t i = 0; i < dev_data.size(); ++i) {
        UNITTEST_EQ(dev_data[i], static_cast<float>(i));
        dev_data[i] = -1;
    }

    // Copy -1s back to GPU array
    UNITTEST_EQ(ark::gpuMemcpy(dev_ptr, dev_data.data(), 1024 * sizeof(float),
                               ark::gpuMemcpyHostToDevice),
                ark::gpuSuccess);

    // Copy data from GPU array to ARK tensor
    executor.tensor_write(tensor, dev_ptr, 1024 * sizeof(float), nullptr, true);

    // Copy data from ARK tensor to CPU array
    executor.tensor_read(tensor, host_ptr, 1024 * sizeof(float));

    // Check the data
    for (size_t i = 0; i < host_data.size(); ++i) {
        UNITTEST_EQ(host_data[i], -1);
    }

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_executor_invalid() {
    // Invalid device ID.
    UNITTEST_THROW(ark::Executor(-1, nullptr, "test", ""),
                   ark::InvalidUsageError);

    // Invalid rank.
    ark::PlanJson plan;
    plan["Rank"] = 1;
    UNITTEST_THROW(ark::Executor(0, nullptr, "test", plan.dump(), true),
                   ark::InvalidUsageError);

    return ark::unittest::SUCCESS;
}

int main() {
    UNITTEST(test_executor_loop);
    UNITTEST(test_executor_no_loop);
    UNITTEST(test_executor_tensor_read_write);
    UNITTEST(test_executor_invalid);
    return 0;
}
