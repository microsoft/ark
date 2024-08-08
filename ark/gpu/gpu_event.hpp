// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_GPU_EVENT_HPP_
#define ARK_GPU_EVENT_HPP_

#include <memory>

#include "gpu/gpu.hpp"

namespace ark {

class GpuStream;
class GpuManager;

class GpuEvent {
   public:
    ~GpuEvent() = default;
    GpuEvent(const GpuEvent &) = delete;
    GpuEvent &operator=(const GpuEvent &) = delete;

    void record(gpuStream stream);
    float elapsed_msec(const GpuEvent &other) const;

   protected:
    friend class GpuManager;

    GpuEvent(bool disable_timing = false);

   private:
    class Impl;
    std::shared_ptr<Impl> pimpl_;
};
}  // namespace ark

#endif  // ARK_GPU_EVENT_HPP_
