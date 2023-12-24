// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_GPU_STREAM_H_
#define ARK_GPU_STREAM_H_

#include <memory>

#include "gpu/gpu.h"

namespace ark {

class GpuManager;
class GpuStream {
   public:
    ~GpuStream() = default;
    void sync() const;
    gpuError query() const;
    gpuStream get() const;

   private:
    friend class GpuManager;
    GpuStream(const GpuManager& manager);

    class Impl;
    std::shared_ptr<Impl> pimpl_;
};
}  // namespace ark

#endif  // ARK_GPU_STREAM_H_
