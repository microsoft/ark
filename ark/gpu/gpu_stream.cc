// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_stream.h"

#include "gpu/gpu_logging.h"
#include "gpu/gpu_manager.h"

namespace ark {
class GpuStream::Impl {
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

GpuStream::GpuStream(GpuManager &manager)
    : pimpl_(std::make_shared<Impl>(manager)) {}

void GpuStream::sync() const { pimpl_->sync(); }

gpuError GpuStream::query() const { return pimpl_->query(); }

gpuStream GpuStream::get() const { return pimpl_->get(); }

GpuStream::Impl::Impl(GpuManager &manager) : manager_(manager) {
    GLOG(gpuStreamCreate(&gpu_stream_, gpuStreamNonBlocking));
}

GpuStream::Impl::~Impl() { GLOG(gpuStreamDestroy(gpu_stream_)); }

}  // namespace ark
