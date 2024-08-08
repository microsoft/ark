// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_GPU_STREAM_HPP_
#define ARK_GPU_STREAM_HPP_

#include <memory>

#include "gpu/gpu.hpp"

namespace ark {

class GpuManager;

class GpuStream {
   public:
    ~GpuStream() = default;
    void sync() const;
    gpuError query() const;
    gpuStream get() const;

   protected:
    friend class GpuManager;

    GpuStream();

   private:
    class Impl;
    std::shared_ptr<Impl> pimpl_;
};
}  // namespace ark

#endif  // ARK_GPU_STREAM_HPP_
