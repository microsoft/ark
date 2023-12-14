// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_GPU_STREAM_H_
#define ARK_GPU_STREAM_H_

#include <memory>

#include "gpu/gpu.h"

namespace ark {
class GpuStreamV2 {
   public:
    GpuStreamV2();
    ~GpuStreamV2() = default;
    void sync() const;
    gpuStream get() const;

   private:
    class Impl;
    std::shared_ptr<Impl> pimpl_;
};
}  // namespace ark

#endif  // ARK_GPU_STREAM_H_
