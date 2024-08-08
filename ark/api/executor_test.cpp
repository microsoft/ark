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

        UNITTEST_TRUE(executor.destroyed());
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

ark::unittest::State test_executor_tensor_read_write(ark::Dims shape,
                                                     ark::Dims stride,
                                                     ark::Dims offset) {
    // Alloc CPU array
    std::vector<float> host_data(shape.nelems());
    for (size_t i = 0; i < host_data.size(); ++i) {
        host_data[i] = static_cast<float>(i);
    }

    // Alloc GPU array
    void *dev_ptr;
    UNITTEST_EQ(ark::gpuMalloc(&dev_ptr, shape.nelems() * sizeof(float)),
                ark::gpuSuccess);

    // Create an ARK tensor
    ark::Model m;
    auto tensor = m.tensor(shape, ark::FP32, stride, offset);
    m.noop(tensor);

    ark::DefaultExecutor executor(m, 0);
    executor.compile();
    executor.launch();
    UNITTEST_GT(executor.tensor_address(tensor), 0);

    // Copy data from CPU array to ARK tensor
    executor.tensor_write(tensor, host_data.data(),
                          shape.nelems() * sizeof(float));

    // Copy data from ARK tensor to GPU array
    executor.tensor_read(tensor, dev_ptr, shape.nelems() * sizeof(float),
                         nullptr, true);

    // Check the data
    std::vector<float> dev_data(shape.nelems());
    executor.tensor_read(tensor, dev_data.data(),
                         shape.nelems() * sizeof(float));
    for (size_t i = 0; i < dev_data.size(); ++i) {
        UNITTEST_EQ(dev_data[i], static_cast<float>(i));
        dev_data[i] = -1;
    }

    UNITTEST_EQ(
        ark::gpuMemcpy(dev_data.data(), dev_ptr, shape.nelems() * sizeof(float),
                       ark::gpuMemcpyDeviceToHost),
        ark::gpuSuccess);
    for (size_t i = 0; i < dev_data.size(); ++i) {
        UNITTEST_EQ(dev_data[i], static_cast<float>(i));
        dev_data[i] = -1;
    }

    // Copy -1s back to GPU array
    UNITTEST_EQ(
        ark::gpuMemcpy(dev_ptr, dev_data.data(), shape.nelems() * sizeof(float),
                       ark::gpuMemcpyHostToDevice),
        ark::gpuSuccess);

    // Copy data from GPU array to ARK tensor
    executor.tensor_write(tensor, dev_ptr, shape.nelems() * sizeof(float),
                          nullptr, true);

    // Copy data from ARK tensor to CPU array
    executor.tensor_read(tensor, host_data.data(),
                         shape.nelems() * sizeof(float));

    // Check the data
    for (size_t i = 0; i < host_data.size(); ++i) {
        UNITTEST_EQ(host_data[i], -1);
    }

    // Provide a stream
    ark::gpuStream stream;
    UNITTEST_EQ(
        ark::gpuStreamCreateWithFlags(&stream, ark::gpuStreamNonBlocking),
        ark::gpuSuccess);
    executor.tensor_read(tensor, host_data.data(),
                         shape.nelems() * sizeof(float), stream);
    executor.tensor_write(tensor, host_data.data(),
                          shape.nelems() * sizeof(float), stream);
    UNITTEST_EQ(ark::gpuStreamDestroy(stream), ark::gpuSuccess);

    // Invalid copy size
    UNITTEST_THROW(executor.tensor_read(tensor, host_data.data(),
                                        shape.nelems() * sizeof(float) + 1),
                   ark::InvalidUsageError);
    UNITTEST_THROW(executor.tensor_write(tensor, host_data.data(),
                                         shape.nelems() * sizeof(float) + 1),
                   ark::InvalidUsageError);

    executor.stop();

    UNITTEST_EQ(ark::gpuFree(dev_ptr), ark::gpuSuccess);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_executor_tensor_read_write_no_stride() {
    return test_executor_tensor_read_write({1024}, {}, {});
}

ark::unittest::State test_executor_tensor_read_write_stride_offset() {
    return test_executor_tensor_read_write({4, 512}, {4, 1024}, {0, 512});
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
    UNITTEST(test_executor_tensor_read_write_no_stride);
    UNITTEST(test_executor_tensor_read_write_stride_offset);
    UNITTEST(test_executor_invalid);
    return 0;
}
