// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_stream.h"

#include "gpu/gpu_logging.h"

namespace ark {
class GpuStreamV2::Impl {
   public:
    Impl();
    ~Impl();
    Impl(const Impl &) = delete;
    Impl &operator=(const Impl &) = delete;

    gpuStream get() const { return gpu_stream_; }

   private:
    gpuStream gpu_stream_;
};

GpuStreamV2::GpuStreamV2() : pimpl_(std::make_shared<Impl>()) {}

void GpuStreamV2::sync() const { GLOG(gpuStreamSynchronize(pimpl_->get())); }

gpuStream GpuStreamV2::get() const { return pimpl_->get(); }

GpuStreamV2::Impl::Impl() {
    GLOG(gpuStreamCreate(&gpu_stream_, gpuStreamNonBlocking));
}

GpuStreamV2::Impl::~Impl() { GLOG(gpuStreamDestroy(gpu_stream_)); }

}  // namespace ark
