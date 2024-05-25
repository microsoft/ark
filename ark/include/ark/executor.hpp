// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_EXECUTOR_HPP
#define ARK_EXECUTOR_HPP

#include <ark/model_ref.hpp>
#include <ark/tensor.hpp>
#include <memory>
#include <string>
#include <vector>

namespace ark {

/// Convenience class for executing a model.
class Executor {
   public:
    /// Constructor.
    Executor(int rank, int world_size, int gpu_id, const std::string &name,
             const std::string &plan);

    ~Executor();

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

    void barrier();

    void destroy();

    bool destroyed() const;

    template <typename T>
    void tensor_read(const Tensor tensor, std::vector<T> &data) const {
        tensor_read(tensor, reinterpret_cast<void *>(data.data()),
                    data.size() * sizeof(T));
    }

    template <typename T>
    void tensor_write(const Tensor tensor, const std::vector<T> &data) const {
        tensor_write(tensor, reinterpret_cast<const void *>(data.data()),
                     data.size() * sizeof(T));
    }

    void tensor_read(const Tensor tensor, void *data, size_t bytes) const;

    void tensor_write(const Tensor tensor, const void *data,
                      size_t bytes) const;

   private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class Model;

class DefaultExecutor : public Executor {
   public:
    DefaultExecutor(const Model &model, int gpu_id = -1,
                    const std::string &name = "DefaultExecutor");
};

}  // namespace ark

#endif  // ARK_EXECUTOR_HPP
