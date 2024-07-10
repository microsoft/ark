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

/// Convenience class for executing a model.
class Executor {
   public:
    /// Constructor.
    Executor(int device_id, Stream stream, const std::string &name,
             const std::string &plan);

    /// Destructor.
    ~Executor();

    /// Return the device ID.
    int device_id() const;

    /// Return the stream of the executor.
    Stream stream() const;

    /// Return the plan string.
    std::string plan() const;

    /// Compile the model. This must be called before `launch()`.
    void compile();

    /// Launch the model (not running yet). This must be called after
    /// `compile()`.
    void launch(int64_t max_spin_count = -1);

    /// Run the model for `iter` iterations.
    void run(int iter);

    /// Wait for the previous run to finish.
    void wait(int64_t max_spin_count = -1);

    /// Stop the model and return the elapsed time in milliseconds.
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
    uintptr_t tensor_address(const Tensor tensor) const;

    template <typename T>
    void tensor_read(const Tensor tensor, std::vector<T> &data,
                     Stream stream = nullptr) const {
        tensor_read(tensor, reinterpret_cast<void *>(data.data()),
                    data.size() * sizeof(T), stream);
    }

    template <typename T>
    void tensor_write(const Tensor tensor, const std::vector<T> &data,
                      Stream stream = nullptr) const {
        tensor_write(tensor, reinterpret_cast<const void *>(data.data()),
                     data.size() * sizeof(T), stream);
    }

    void tensor_read(const Tensor tensor, void *data, size_t bytes,
                     Stream stream = nullptr, bool is_d2d = false) const;

    void tensor_write(const Tensor tensor, const void *data, size_t bytes,
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
        const std::vector<DefaultPlanner::ConfigRule> &config_rules = {},
        const std::string &name = "DefaultExecutor");
};

}  // namespace ark

#endif  // ARK_EXECUTOR_HPP
