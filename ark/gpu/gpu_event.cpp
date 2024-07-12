// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_event.hpp"

#include "gpu/gpu_logging.hpp"
#include "gpu/gpu_manager.hpp"

namespace ark {
class GpuEvent::Impl {
   public:
    Impl(bool disable_timing);
    ~Impl();
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    void record(gpuStream stream);
    float elapsed_msec(const GpuEvent& other) const;

   private:
    gpuEvent event_;
};

GpuEvent::Impl::Impl(bool disable_timing) {
    unsigned int flags = 0;
    if (disable_timing) {
        flags |= gpuEventDisableTiming;
    }
    GLOG(gpuEventCreateWithFlags(&event_, flags));
}

GpuEvent::Impl::~Impl() { GLOG(gpuEventDestroy(event_)); }

void GpuEvent::Impl::record(gpuStream stream) {
    GLOG(gpuEventRecord(event_, stream));
}

float GpuEvent::Impl::elapsed_msec(const GpuEvent& other) const {
    float elapsed;
    GLOG(gpuEventElapsedTime(&elapsed, other.pimpl_->event_, event_));
    return elapsed;
}

GpuEvent::GpuEvent(bool disable_timing)
    : pimpl_(std::make_shared<Impl>(disable_timing)) {}

void GpuEvent::record(gpuStream stream) { pimpl_->record(stream); }

float GpuEvent::elapsed_msec(const GpuEvent& other) const {
    return pimpl_->elapsed_msec(other);
}

}  // namespace ark
