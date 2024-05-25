// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_GPU_EVENT_H_
#define ARK_GPU_EVENT_H_

#include <memory>

namespace ark {

class GpuStream;
class GpuManager;

class GpuEvent {
   public:
    ~GpuEvent() = default;
    GpuEvent(const GpuEvent &) = delete;
    GpuEvent &operator=(const GpuEvent &) = delete;

    void record(std::shared_ptr<GpuStream> stream);
    float elapsed_msec(const GpuEvent &other) const;

   protected:
    friend class GpuManager;

    GpuEvent(bool disable_timing = false);

   private:
    class Impl;
    std::shared_ptr<Impl> pimpl_;
};
}  // namespace ark

#endif  // ARK_GPU_EVENT_H_
