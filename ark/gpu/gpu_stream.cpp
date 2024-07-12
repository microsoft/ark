// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_stream.hpp"

#include "gpu/gpu_logging.hpp"
#include "gpu/gpu_manager.hpp"

namespace ark {
class GpuStream::Impl {
   public:
    Impl();
    ~Impl();
    Impl(const Impl &) = delete;
    Impl &operator=(const Impl &) = delete;

    gpuStream get() const { return gpu_stream_; }
    gpuError query() const { return gpuStreamQuery(gpu_stream_); }
    void sync() const { GLOG(gpuStreamSynchronize(gpu_stream_)); }

   private:
    gpuStream gpu_stream_;
};

GpuStream::GpuStream() : pimpl_(std::make_shared<Impl>()) {}

void GpuStream::sync() const { pimpl_->sync(); }

gpuError GpuStream::query() const { return pimpl_->query(); }

gpuStream GpuStream::get() const { return pimpl_->get(); }

GpuStream::Impl::Impl() {
    GLOG(gpuStreamCreateWithFlags(&gpu_stream_, gpuStreamNonBlocking));
}

GpuStream::Impl::~Impl() { GLOG(gpuStreamDestroy(gpu_stream_)); }

}  // namespace ark
