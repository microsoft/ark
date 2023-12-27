// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_GPU_EVENT_H_
#define ARK_GPU_EVENT_H_

#include <memory>

#include "gpu/gpu_stream.h"

namespace ark {
class GpuManager;
class GpuEvent {
   public:
    ~GpuEvent() = default;
    GpuEvent(const GpuEvent &) = delete;
    GpuEvent &operator=(const GpuEvent &) = delete;

    void record(std::shared_ptr<GpuStream> stream);
    float elapsed_msec(const GpuEvent &other) const;

   private:
    friend class GpuManager;
    GpuEvent(const GpuManager &manager, bool disable_timing = false);

    class Impl;
    std::shared_ptr<Impl> pimpl_;
};
}  // namespace ark

#endif  // ARK_GPU_EVENT_H_
