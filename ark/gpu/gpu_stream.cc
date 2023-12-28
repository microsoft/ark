// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_stream.h"

#include "gpu/gpu_logging.h"
#include "gpu/gpu_manager.h"

namespace ark {
class GpuStream::Impl {
   public:
    Impl(const GpuManager &manager);
    ~Impl();
    Impl(const Impl &) = delete;
    Impl &operator=(const Impl &) = delete;

    gpuStream get() const { return gpu_stream_; }
    gpuError query() const { return gpuStreamQuery(gpu_stream_); }
    void sync() const { GLOG(gpuStreamSynchronize(gpu_stream_)); }

   private:
    gpuStream gpu_stream_;
    const GpuManager &manager_;
};

GpuStream::GpuStream(const GpuManager &manager)
    : pimpl_(std::make_shared<Impl>(manager)) {}

void GpuStream::sync() const { pimpl_->sync(); }

gpuError GpuStream::query() const { return pimpl_->query(); }

gpuStream GpuStream::get() const { return pimpl_->get(); }

GpuStream::Impl::Impl(const GpuManager &manager) : manager_(manager) {
    GLOG(gpuStreamCreateWithFlags(&gpu_stream_, gpuStreamNonBlocking));
}

GpuStream::Impl::~Impl() { GLOG(gpuStreamDestroy(gpu_stream_)); }

}  // namespace ark
