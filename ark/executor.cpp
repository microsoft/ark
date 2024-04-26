// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/executor.hpp"

#include <cmath>
#include <memory>

#include "ark/model.hpp"
#include "ark/planner.hpp"
#include "codegen.hpp"
#include "env.h"
#include "file_io.h"
#include "gpu/gpu.h"
#include "gpu/gpu_event.h"
#include "gpu/gpu_kernel.h"
#include "gpu/gpu_logging.h"
#include "logging.h"
#include "model/model_tensor.hpp"

#define MAX_LOOP_COUNTER 10000000

#if defined(ARK_CUDA)
#include <cuda/atomic>
static int atomicLoadRelaxed(int *ptr) {
    return cuda::atomic_ref<int, cuda::thread_scope_system>{*ptr}.load(
        cuda::memory_order_relaxed);
}
static void atomicStoreRelaxed(int *ptr, int val) {
    cuda::atomic_ref<int, cuda::thread_scope_system>{*ptr}.store(
        val, cuda::memory_order_relaxed);
}
#elif defined(ARK_ROCM)
static int atomicLoadRelaxed(int *ptr) {
    return __atomic_load_n(ptr, __ATOMIC_RELAXED);
}
static void atomicStoreRelaxed(int *ptr, int val) {
    __atomic_store_n(ptr, val, __ATOMIC_RELAXED);
}
#endif  // defined(ARK_ROCM)

namespace ark {

///
void tensor_to_data(const int8_t *tensor, int8_t *data, const Dims &shape,
                    const Dims &strides, const Dims &offsets,
                    size_t elem_bytes) {
    auto sh = shape;
    auto st = strides;
    auto of = offsets;
    sh[-1] *= elem_bytes;
    st[-1] *= elem_bytes;
    of[-1] *= elem_bytes;
    if (sh.dims4() == st.dims4()) {
        ::memcpy(data, tensor, sh.size());
        return;
    }
    if (sh.ndims() == 1) {
        ::memcpy(data, tensor + of[0], sh[0]);
        return;
    }
    for (auto i = 0; i < sh[0]; ++i) {
        if (sh.ndims() == 2) {
            ::memcpy(data + i * sh[1], tensor + ((i + of[0]) * st[1] + of[1]),
                     sh[1]);
            continue;
        }
        for (auto j = 0; j < sh[1]; ++j) {
            if (sh.ndims() == 3) {
                ::memcpy(data + ((i * sh[1] + j) * sh[2]),
                         tensor + (((i + of[0]) * st[1] + j + of[1]) * st[2] +
                                   of[2]),
                         sh[2]);
                continue;
            }
            for (auto k = 0; k < sh[2]; ++k) {
                ::memcpy(data + (((i * sh[1] + j) * sh[2] + k) * sh[3]),
                         tensor + ((((i + of[0]) * st[1] + j + of[1]) * st[2] +
                                    k + of[2]) *
                                       st[3] +
                                   of[3]),
                         sh[3]);
            }
        }
    }
}

///
void data_to_tensor(int8_t *tensor, const int8_t *data, const Dims &shape,
                    const Dims &strides, const Dims &offsets,
                    size_t elem_bytes) {
    auto sh = shape;
    auto st = strides;
    auto of = offsets;
    sh[-1] *= elem_bytes;
    st[-1] *= elem_bytes;
    of[-1] *= elem_bytes;
    if (sh.dims4() == st.dims4()) {
        ::memcpy(tensor, data, sh.size());
        return;
    }
    if (sh.ndims() == 1) {
        ::memcpy(tensor + of[0], data, sh[0]);
        return;
    }
    for (auto i = 0; i < sh[0]; ++i) {
        if (sh.ndims() == 2) {
            ::memcpy(tensor + ((i + of[0]) * st[1] + of[1]), data + i * sh[1],
                     sh[1]);
            continue;
        }
        for (auto j = 0; j < sh[1]; ++j) {
            if (sh.ndims() == 3) {
                ::memcpy(tensor + (((i + of[0]) * st[1] + j + of[1]) * st[2] +
                                   of[2]),
                         data + ((i * sh[1] + j) * sh[2]), sh[2]);
                continue;
            }
            for (auto k = 0; k < sh[2]; ++k) {
                ::memcpy(tensor + ((((i + of[0]) * st[1] + j + of[1]) * st[2] +
                                    k + of[2]) *
                                       st[3] +
                                   of[3]),
                         data + (((i * sh[1] + j) * sh[2] + k) * sh[3]), sh[3]);
            }
        }
    }
}

class Executor::Impl {
   public:
    Impl(int rank, int world_size, int gpu_id, const std::string &name,
         const std::string &plan);
    ~Impl() = default;

    void compile();
    void launch();
    void run(int iter);
    void wait();
    float stop();

    void tensor_read(const ModelTensorRef tensor, void *data,
                     size_t bytes) const;
    void tensor_write(const ModelTensorRef tensor, const void *data,
                      size_t bytes) const;

   protected:
    const int rank_;
    const int world_size_;
    int gpu_id_;

    bool is_launched_ = false;
    bool is_recording_ = false;
    float elapsed_msec_ = -1;

    std::shared_ptr<CodeGenerator> codegen_;
    std::shared_ptr<GpuEvent> timer_begin_;
    std::shared_ptr<GpuEvent> timer_end_;
    std::shared_ptr<GpuMemory> buffer_;
    std::shared_ptr<GpuHostMemory> flag_;
    std::shared_ptr<GpuStream> main_stream_;
    std::shared_ptr<GpuStream> copy_stream_;
    std::shared_ptr<GpuKernel> kernel_;
};

Executor::Impl::Impl(int rank, int world_size, int gpu_id,
                     const std::string &name, const std::string &plan)
    : rank_(rank), world_size_(world_size), gpu_id_(gpu_id) {
    auto &plan_path = get_env().enforce_plan_path;
    if (!plan_path.empty()) {
        LOG(INFO, "Enforce executor plan path: ", plan_path);
        codegen_ = std::make_shared<CodeGenerator>(read_file(plan_path), name);
    } else {
        codegen_ = std::make_shared<CodeGenerator>(plan, name);
    }
    auto gpu_manager = GpuManager::get_instance(gpu_id_);
    timer_begin_ = gpu_manager->create_event();
    timer_end_ = gpu_manager->create_event();
    buffer_ = gpu_manager->malloc(codegen_->total_memory_bytes(), 65536);
    flag_ = gpu_manager->malloc_host(
        sizeof(int), gpuHostAllocMapped | gpuHostAllocWriteCombined);
    main_stream_ = gpu_manager->create_stream();
    copy_stream_ = gpu_manager->create_stream();

    int threads_per_block = static_cast<int>(
        codegen_->num_warps_per_proc() * gpu_manager->info().threads_per_warp);
    int num_sm = static_cast<int>(codegen_->num_procs());
    int *flag = flag_->ref<int>();
    size_t smem_block_total =
        static_cast<size_t>(gpu_manager->info().smem_block_total);

    kernel_ = std::shared_ptr<GpuKernel>(
        new GpuKernel(gpu_id_, codegen_->code(), {threads_per_block, 1, 1},
                      {num_sm, 1, 1}, std::max(smem_block_total, size_t(4)),
                      name, {std::pair<void *, size_t>{flag, sizeof(flag)}}));
}

void Executor::Impl::compile() { kernel_->compile(); }

void Executor::Impl::launch() {
    if (!kernel_->is_compiled()) {
        ERR(InvalidUsageError, "Need to compile first before initialization.");
    }
    if (is_launched_) {
        // Wait until previous works finish.
        this->wait();
        return;
    }
    // Initialize global variables in the loop kernel.
    auto gpu_manager = GpuManager::get_instance(gpu_id_);
    void *buf_ptr_val = buffer_->ref();
    GpuPtr lss_ptr_addr = kernel_->get_global("ARK_LOOP_SYNC_STATE");
    GpuPtr buf_ptr_addr = kernel_->get_global("ARK_BUF");
    std::array<int, 4> data = {0, 0, 0, 0};
    gpu_manager->memcpy_htod((void *)lss_ptr_addr, 0, data.data(), 0,
                             sizeof(int) * data.size());
    gpu_manager->memcpy_htod((void *)buf_ptr_addr, 0, &buf_ptr_val, 0,
                             sizeof(GpuPtr));
    // TODO: remove this hack
    // GpuPtr lss_0_ptr_addr = kernel_->get_global("ARK_LOOP_SYNC_STATE_0",
    // true); GpuPtr lss_1_ptr_addr =
    // kernel_->get_global("ARK_LOOP_SYNC_STATE_1", true);

    // set the data buffer pointers of remote gpus
    // int nrph = get_env().num_ranks_per_host;
    // int nodes_id = gpu_manager->get_gpu_id() / nrph;
    // // only set the GPU remote data buf pointers of the GPUs on the same node
    // for (int i = nodes_id * nrph;
    //      i < (nodes_id + 1) * nrph && i < ctx_->world_size(); i++) {
    //     void* data_buf_value = ctx_->get_data_memory(i)->ref();
    //     if (data_buf_value == 0) {
    //         continue;
    //     }
    //     GpuPtr data_buf_ptr;
    //     std::string data_buf_name = ARK_BUF_NAME + std::to_string(i);
    //     gpuDrvError _e = gpuModuleGetGlobal(&data_buf_ptr, &tmp, module_,
    //                                         data_buf_name.c_str());
    //     if (_e == gpuErrorNotFound) {
    //         LOG(DEBUG, "global variable ", data_buf_name, " not found");
    //         continue;
    //     }
    //     LOG(DEBUG, data_buf_name, " data_buf_ptr=", std::hex, data_buf_ptr,
    //         " data_buf_value=", data_buf_value);
    //     gpu_manager->memcpy_htod((void*)data_buf_ptr, 0, &data_buf_value, 0,
    //                          sizeof(GpuPtr));
    // }

    // std::shared_ptr<GpuCommSw> comm = ctx_->get_comm_sw();
    // if (comm->get_proxy_channels_num() > 0) {
    //     GpuPtr channel_addr;
    //     GLOG_DRV(gpuModuleGetGlobal(&channel_addr, &tmp, module_,
    //                                 "_ARK_PROXY_CHANS"));
    //     const void* chans_ref = comm->get_proxy_channels_ref();
    //     size_t chans_bytes = comm->get_proxy_channels_bytes();
    //     gpu_manager->memcpy_htod((void*)channel_addr, 0,
    //                          const_cast<void*>(chans_ref), 0, chans_bytes);
    // }
    // if (comm->get_sm_channels_num() > 0) {
    //     GpuPtr channel_addr;
    //     GLOG_DRV(
    //         gpuModuleGetGlobal(&channel_addr, &tmp, module_,
    //         "_ARK_SM_CHANS"));
    //     const void* chans_ref = comm->get_sm_channels_ref();
    //     size_t chans_bytes = comm->get_sm_channels_bytes();
    //     gpu_manager->memcpy_htod((void*)channel_addr, 0,
    //                          const_cast<void*>(chans_ref), 0, chans_bytes);
    // }

    elapsed_msec_ = -1;
    if (!kernel_->is_compiled()) {
        ERR(InvalidUsageError, "Need to compile first before initialization.");
    } else if (is_launched_) {
        LOG(WARN, "Ignore launching twice.");
        return;
    }
    timer_begin_->record(main_stream_);

    // ctx_->get_comm_sw()->launch_request_loop();

    // Initialize loop flags.
    atomicStoreRelaxed(flag_->ref<int>(), 0);
    kernel_->launch(main_stream_);
    timer_end_->record(main_stream_);
    is_recording_ = true;
    is_launched_ = true;
}

void Executor::Impl::run(int iter) {
    if (iter > 0) {
        while (atomicLoadRelaxed(flag_->ref<int>()) > 0) {
        }
        atomicStoreRelaxed(flag_->ref<int>(), iter);
    }
}

void Executor::Impl::wait() {
    int cnt = MAX_LOOP_COUNTER;
    while (atomicLoadRelaxed(flag_->ref<int>()) > 0) {
        if (--cnt > 0) {
            continue;
        }
        // Check if the kernel encountered an error.
        gpuError res = main_stream_->query();
        if (res == gpuSuccess) {
            if (atomicLoadRelaxed(flag_->ref<int>()) > 0) {
                LOG(WARN, "Stream is finished but the loop flag is still set.");
                break;
            } else {
                LOG(WARN,
                    "wait() is delayed by a stream query. Regarding "
                    "timing measurements may be inaccurate.");
                break;
            }
        } else if (res == gpuErrorNotReady) {
            cnt = MAX_LOOP_COUNTER;
        } else {
            GLOG(res);
        }
    }
}

float Executor::Impl::stop() {
    this->wait();
    atomicStoreRelaxed(flag_->ref<int>(), -1);
    main_stream_->sync();
    if (is_recording_) {
        elapsed_msec_ = timer_end_->elapsed_msec(*timer_begin_);
        is_recording_ = false;
    }
    is_launched_ = false;
    // ctx_->get_comm_sw()->stop_request_loop();
    return elapsed_msec_;
}

void Executor::Impl::tensor_read(const ModelTensorRef tensor, void *data,
                                 size_t bytes) const {
    const auto &tensor_info = codegen_->tensor_info(tensor->id());
    if (bytes < tensor->shape_bytes()) {
        ERR(InvalidUsageError, "Data buffer is smaller than the tensor data.");
    }
    void *src = buffer_->ref(tensor_info.offset);
    if (tensor->strides() == tensor->shape()) {
        GLOG(gpuMemcpyAsync(data, src, bytes, gpuMemcpyDeviceToHost,
                            copy_stream_->get()));
        copy_stream_->sync();
    } else {
        std::vector<int8_t> tensor_host(tensor_info.bytes);
        GLOG(gpuMemcpyAsync(tensor_host.data(), src, tensor_info.bytes,
                            gpuMemcpyDeviceToHost, copy_stream_->get()));
        copy_stream_->sync();
        // std::cout << "tensor_host read (offset " << tensor_info.offset << "):
        // "; for (auto i : tensor_host) {
        //     std::cout << i << " ";
        // }
        // std::cout << std::endl;
        tensor_to_data(tensor_host.data(), static_cast<int8_t *>(data),
                       tensor->shape(), tensor->strides(), tensor->offsets(),
                       tensor->data_type()->bytes());
    }
}

void Executor::Impl::tensor_write(const ModelTensorRef tensor, const void *data,
                                  size_t bytes) const {
    const auto &tensor_info = codegen_->tensor_info(tensor->id());
    if (bytes < tensor->shape_bytes()) {
        ERR(InvalidUsageError, "Data buffer is smaller than the tensor data.");
    }
    void *dst = buffer_->ref(tensor_info.offset);
    if (tensor->strides() == tensor->shape()) {
        GLOG(gpuMemcpyAsync(dst, data, tensor_info.bytes, gpuMemcpyHostToDevice,
                            copy_stream_->get()));
    } else {
        std::vector<int8_t> tensor_host(tensor_info.bytes);
        GLOG(gpuMemcpyAsync(tensor_host.data(), dst, tensor_info.bytes,
                            gpuMemcpyDeviceToHost, copy_stream_->get()));
        copy_stream_->sync();
        // std::cout << "tensor_host write (offset " << tensor_info.offset
        //             << "): ";
        // for (auto i = 0; i < tensor_host.size() * sizeof(T) / sizeof(float);
        //         ++i) {
        //     std::cout << *((float *)tensor_host.data() + i) << " ";
        // }
        // std::cout << std::endl;
        // std::cout << "data write (offset " << tensor_info.offset << "): ";
        // for (auto i = 0; i < data.size() * sizeof(T) / sizeof(float); ++i) {
        //     std::cout << *((float *)data.data() + i) << " ";
        // }
        // std::cout << std::endl;
        // std::cout << "tensor shape: " << tensor->shape() << std::endl;
        // std::cout << "tensor strides: " << tensor->strides() << std::endl;
        // std::cout << "tensor offsets: " << tensor->offsets() << std::endl;
        data_to_tensor(tensor_host.data(), static_cast<const int8_t *>(data),
                       tensor->shape(), tensor->strides(), tensor->offsets(),
                       tensor->data_type()->bytes());
        // std::cout << "tensor_host write2 (offset " << tensor_info.offset
        //             << "): ";
        // for (auto i = 0; i < tensor_host.size() * sizeof(T) / sizeof(float);
        //         ++i) {
        //     std::cout << *((float *)tensor_host.data() + i) << " ";
        // }
        // std::cout << std::endl;
        GLOG(gpuMemcpyAsync(dst, tensor_host.data(), tensor_info.bytes,
                            gpuMemcpyHostToDevice, copy_stream_->get()));
    }
    copy_stream_->sync();
}

Executor::Executor(int rank, int world_size, int gpu_id,
                   const std::string &name, const std::string &plan)
    : impl_(std::make_unique<Executor::Impl>(rank, world_size, gpu_id, name,
                                             plan)) {}

Executor::~Executor() = default;

void Executor::compile() { impl_->compile(); }

void Executor::launch() { impl_->launch(); }

void Executor::run(int iter) { impl_->run(iter); }

void Executor::wait() { impl_->wait(); }

float Executor::stop() { return impl_->stop(); }

// ///
// std::shared_ptr<std::vector<char>> Executor::tensor_read(
//     const ModelTensorRef tensor) const {
//     return impl_->tensor_read<char>(tensor);
// }

// ///
// void Executor::tensor_write(const ModelTensorRef tensor,
//                             const std::vector<char> &data) const {
//     impl_->tensor_write<char>(tensor, data);
// }

///
void Executor::tensor_read(const ModelTensorRef tensor, void *data,
                           size_t bytes) const {
    impl_->tensor_read(tensor, data, bytes);
}

///
void Executor::tensor_write(const ModelTensorRef tensor, const void *data,
                            size_t bytes) const {
    impl_->tensor_write(tensor, data, bytes);
}

DefaultExecutor::DefaultExecutor(const Model &model, int gpu_id,
                                 const std::string &name)
    : Executor(
          model.rank(), model.world_size(),
          (gpu_id < 0) ? (model.rank() % get_env().num_ranks_per_host) : gpu_id,
          name,
          Planner(model, (gpu_id < 0)
                             ? (model.rank() % get_env().num_ranks_per_host)
                             : gpu_id)
              .plan()) {}

}  // namespace ark
