// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_EXECUTOR_HPP
#define ARK_EXECUTOR_HPP

#include <ark/model_ref.hpp>
#include <ark/planner.hpp>
#include <ark/tensor.hpp>
#include <memory>
#include <string>
#include <vector>

namespace ark {

using Stream = void *;

class GpuMemory;

/// Convenience class for executing a model.
class Executor {
   public:
    /// Constructor.
    Executor();

    /// Destructor.
    ~Executor();

    /// Return the device ID.
    int device_id() const;

    /// Return the stream of the executor.
    Stream stream() const;

    /// Return the buffer of the executor.
    std::shared_ptr<GpuMemory> buffer() const;

    /// Return the plan string.
    std::string plan() const;

    const std::string &name() const;

    /// Compile the model. This must be called before `launch()`.
    void compile(const std::string &plan, int device_id,
                 const std::string &name = "executor");

    /// Launch the executor. This must be called after `compile()`.
    void launch(Stream stream = nullptr, bool loop_mode = true);

    /// Run the executor for `iter` iterations.
    void run(int iter);

    /// Wait for the previous run to finish.
    void wait(int64_t max_spin_count = -1);

    /// Stop the executor and return the elapsed time in milliseconds.
    /// Once this is called, we need to call `launch()` again to run the model
    /// again.
    float stop(int64_t max_spin_count = -1);

    /// Barrier for all rank executors.
    void barrier();

    /// Destroy the executor.
    void destroy();

    /// Return whether the executor is destroyed.
    bool destroyed() const;

    /// Return the raw virtual address of the tensor.
    void *tensor_address(const Tensor &tensor) const;

    template <typename T>
    void tensor_read(const Tensor &tensor, std::vector<T> &data,
                     Stream stream = nullptr) const {
        tensor_read(tensor, reinterpret_cast<void *>(data.data()),
                    data.size() * sizeof(T), stream);
    }

    template <typename T>
    void tensor_write(const Tensor &tensor, const std::vector<T> &data,
                      Stream stream = nullptr) const {
        tensor_write(tensor, reinterpret_cast<const void *>(data.data()),
                     data.size() * sizeof(T), stream);
    }

    void tensor_read(const Tensor &tensor, void *data, size_t bytes,
                     Stream stream = nullptr, bool is_d2d = false) const;

    void tensor_write(const Tensor &tensor, const void *data, size_t bytes,
                      Stream stream = nullptr, bool is_d2d = false) const;

   protected:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class Model;

class DefaultExecutor : public Executor {
   public:
    DefaultExecutor(
        const Model &model, int device_id = -1, Stream stream = nullptr,
        const std::vector<Planner::ConfigRule> &config_rules = {},
        const std::string &name = "DefaultExecutor", bool loop_mode = true);

    /// Launch the default executor.
    void launch();
};

}  // namespace ark

#endif  // ARK_EXECUTOR_HPP
