// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_stream.h"

#include "gpu/gpu_logging.h"
#include "gpu/gpu_manager.h"

namespace ark {
class GpuStreamV2::Impl {
   public:
    Impl(GpuManager &manager);
    ~Impl();
    Impl(const Impl &) = delete;
    Impl &operator=(const Impl &) = delete;

    gpuStream get() const { return gpu_stream_; }
    gpuError query() const {
        manager_.set_current();
        return gpuStreamQuery(gpu_stream_);
    }
    void sync() const {
        manager_.set_current();
        GLOG(gpuStreamSynchronize(gpu_stream_));
    }

   private:
    gpuStream gpu_stream_;
    GpuManager &manager_;
};

GpuStreamV2::GpuStreamV2(GpuManager &manager)
    : pimpl_(std::make_shared<Impl>(manager)) {}

void GpuStreamV2::sync() const { pimpl_->sync(); }

gpuError GpuStreamV2::query() const { return pimpl_->query(); }

gpuStream GpuStreamV2::get() const { return pimpl_->get(); }

GpuStreamV2::Impl::Impl(GpuManager &manager) : manager_(manager) {
    GLOG(gpuStreamCreate(&gpu_stream_, gpuStreamNonBlocking));
}

GpuStreamV2::Impl::~Impl() { GLOG(gpuStreamDestroy(gpu_stream_)); }

}  // namespace ark
