// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_EXECUTOR_H
#define ARK_EXECUTOR_H

#include "model.h"

namespace ark {

/// Convenience class for executing a model.
class Executor {
   public:
    /// Constructor.
    Executor(int rank, int world_size, Model &model, const std::string &name,
             int num_warps_per_sm = 16);
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

   private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ark

#endif  // ARK_EXECUTOR_H
