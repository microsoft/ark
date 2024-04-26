// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_EXECUTOR_HPP
#define ARK_EXECUTOR_HPP

#include <memory>
#include <string>
#include <vector>

#include "ark/model_ref.hpp"
#include "bfloat16.h"
#include "half.h"

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
    void launch();

    /// Run the model for `iter` iterations.
    void run(int iter);

    /// Wait for the previous run to finish.
    void wait();

    /// Stop the model and return the elapsed time in milliseconds.
    /// Once this is called, we need to call `launch()` again to run the model
    /// again.
    float stop();

    template <typename T>
    void tensor_read(const ModelTensorRef tensor, std::vector<T> &data) const {
        tensor_read(tensor, reinterpret_cast<void *>(data.data()),
                    data.size() * sizeof(T));
    }

    template <typename T>
    void tensor_write(const ModelTensorRef tensor,
                      const std::vector<T> &data) const {
        tensor_write(tensor, reinterpret_cast<const void *>(data.data()),
                     data.size() * sizeof(T));
    }

   private:
    void tensor_read(const ModelTensorRef tensor, void *data,
                     size_t bytes) const;

    void tensor_write(const ModelTensorRef tensor, const void *data,
                      size_t bytes) const;

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
