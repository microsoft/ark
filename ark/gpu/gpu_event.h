// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_GPU_EVENT_V2_H_
#define ARK_GPU_EVENT_V2_H_

#include <memory>

#include "gpu/gpu_stream.h"

namespace ark {
class GpuManager;
class GpuEventV2 {
   public:
    GpuEventV2(std::shared_ptr<GpuManager> manager,
               bool disable_timing = false);
    ~GpuEventV2() = default;
    GpuEventV2(const GpuEventV2 &) = delete;
    GpuEventV2 &operator=(const GpuEventV2 &) = delete;

    void record(std::shared_ptr<GpuStreamV2> stream);
    float elapsed_msec(const GpuEventV2 &other) const;

   private:
    class Impl;
    std::shared_ptr<Impl> pimpl_;
};
}  // namespace ark

#endif  // ARK_GPU_EVENT_V2_H_
