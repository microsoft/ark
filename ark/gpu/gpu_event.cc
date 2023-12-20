// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_event.h"

#include "gpu/gpu.h"
#include "gpu/gpu_logging.h"
#include "gpu/gpu_manager.h"

namespace ark {
class GpuEventV2::Impl {
   public:
    Impl(std::shared_ptr<GpuManager> manager, bool disable_timing);
    ~Impl();
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    void record(std::shared_ptr<GpuStreamV2> stream);
    float elapsed_msec(const GpuEventV2& other) const;

   private:
    std::shared_ptr<GpuManager> manager_;
    gpuEvent event_;
};

GpuEventV2::Impl::Impl(std::shared_ptr<GpuManager> manager, bool disable_timing)
    : manager_(manager) {
    unsigned int flags = 0;
    if (disable_timing) {
        flags |= gpuEventDisableTiming;
    }
    GLOG(gpuEventCreate(&event_, flags));
}

GpuEventV2::Impl::~Impl() { GLOG(gpuEventDestroy(event_)); }

void GpuEventV2::Impl::record(std::shared_ptr<GpuStreamV2> stream) {
    manager_->set_current();
    GLOG(gpuEventRecord(event_, stream->get()));
}

float GpuEventV2::Impl::elapsed_msec(const GpuEventV2& other) const {
    float elapsed;
    manager_->set_current();
    GLOG(gpuEventElapsedTime(&elapsed, other.pimpl_->event_, event_));
    return elapsed;
}

GpuEventV2::GpuEventV2(std::shared_ptr<GpuManager> manager, bool disable_timing)
    : pimpl_(std::make_shared<Impl>(manager, disable_timing)) {}

void GpuEventV2::record(std::shared_ptr<GpuStreamV2> stream) {
    pimpl_->record(stream);
}

float GpuEventV2::elapsed_msec(const GpuEventV2& other) const {
    return pimpl_->elapsed_msec(other);
}

}  // namespace ark
