// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/executor.hpp"

#include <dlpack/dlpack.h>

#include <cmath>
#include <memory>
#include <mscclpp/core.hpp>
#include <mscclpp/proxy_channel.hpp>
#include <mscclpp/sm_channel.hpp>

#include "ark/data_type.hpp"
#include "ark/model.hpp"
#include "ark/planner.hpp"
#include "codegen.hpp"
#include "env.h"
#include "file_io.h"
#include "gpu/gpu.h"
#include "gpu/gpu_event.h"
#include "gpu/gpu_kernel.h"
#include "gpu/gpu_logging.h"
#include "gpu/gpu_manager.h"
#include "logging.h"
#include "model/model_buffer.hpp"
#include "model/model_data_type.hpp"
#include "model/model_tensor.hpp"
#include "utils/utils_net.hpp"

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
static void tensor_to_data(const int8_t *tensor, int8_t *data,
                           const Dims &shape, const Dims &strides,
                           const Dims &offsets, size_t elem_bytes) {
    auto sh = shape;
    auto st = strides;
    auto of = offsets;
    sh[-1] *= elem_bytes;
    st[-1] *= elem_bytes;
    of[-1] *= elem_bytes;
    if (sh.dims4() == st.dims4()) {
        ::memcpy(data, tensor, sh.nelems());
        return;
    }
    if (sh.ndims() == 1) {
        ::memcpy(data, tensor + of[0], sh[0]);
        return;
    }
    for (DimType i = 0; i < sh[0]; ++i) {
        if (sh.ndims() == 2) {
            ::memcpy(data + i * sh[1], tensor + ((i + of[0]) * st[1] + of[1]),
                     sh[1]);
            continue;
        }
        for (DimType j = 0; j < sh[1]; ++j) {
            if (sh.ndims() == 3) {
                ::memcpy(data + ((i * sh[1] + j) * sh[2]),
                         tensor + (((i + of[0]) * st[1] + j + of[1]) * st[2] +
                                   of[2]),
                         sh[2]);
                continue;
            }
            for (DimType k = 0; k < sh[2]; ++k) {
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
static void data_to_tensor(int8_t *tensor, const int8_t *data,
                           const Dims &shape, const Dims &strides,
                           const Dims &offsets, size_t elem_bytes) {
    auto sh = shape;
    auto st = strides;
    auto of = offsets;
    sh[-1] *= elem_bytes;
    st[-1] *= elem_bytes;
    of[-1] *= elem_bytes;
    if (sh.dims4() == st.dims4()) {
        ::memcpy(tensor, data, sh.nelems());
        return;
    }
    if (sh.ndims() == 1) {
        ::memcpy(tensor + of[0], data, sh[0]);
        return;
    }
    for (DimType i = 0; i < sh[0]; ++i) {
        if (sh.ndims() == 2) {
            ::memcpy(tensor + ((i + of[0]) * st[1] + of[1]), data + i * sh[1],
                     sh[1]);
            continue;
        }
        for (DimType j = 0; j < sh[1]; ++j) {
            if (sh.ndims() == 3) {
                ::memcpy(tensor + (((i + of[0]) * st[1] + j + of[1]) * st[2] +
                                   of[2]),
                         data + ((i * sh[1] + j) * sh[2]), sh[2]);
                continue;
            }
            for (DimType k = 0; k < sh[2]; ++k) {
                ::memcpy(tensor + ((((i + of[0]) * st[1] + j + of[1]) * st[2] +
                                    k + of[2]) *
                                       st[3] +
                                   of[3]),
                         data + (((i * sh[1] + j) * sh[2] + k) * sh[3]), sh[3]);
            }
        }
    }
}

static size_t tensor_stride_bytes(const Json &tensor) {
    Dims strides(tensor["Strides"].get<std::vector<DimType>>());
    size_t nelems = strides.nelems();
    return nelems * DataType::from_name(tensor["DataType"]).bytes();
}

class Executor::Impl {
   public:
    Impl(int rank, int world_size, int gpu_id, const std::string &name,
         const std::string &plan);
    ~Impl() = default;

    int gpu_id() const { return gpu_id_; }

    void compile();
    void launch(int64_t max_spin_count);
    void run(int iter);
    void wait(int64_t max_spin_count);
    float stop(int64_t max_spin_count);
    void barrier();

    uintptr_t tensor_address(const Tensor tensor) const;

    void tensor_read(const Tensor tensor, void *data, size_t bytes,
                     bool is_d2d) const;
    void tensor_write(const Tensor tensor, const void *data, size_t bytes,
                      bool is_d2d) const;

   private:
    void init_communicator();
    std::map<size_t, size_t> init_buffers(const Json &plan_json);
    std::set<int> init_remote_ranks(const Json &plan_json) const;
    void init_channels(const std::set<int> &remote_ranks);

   protected:
    const int rank_;
    const int world_size_;
    int gpu_id_;

    bool is_launched_ = false;
    bool is_recording_ = false;
    float elapsed_msec_ = -1;

    std::map<size_t, size_t> buffer_id_to_offset_;
    size_t total_bytes_;
    std::shared_ptr<CodeGenerator> codegen_;
    std::shared_ptr<GpuEvent> timer_begin_;
    std::shared_ptr<GpuEvent> timer_end_;
    std::shared_ptr<GpuMemory> buffer_;
    std::shared_ptr<GpuHostMemory> flag_;
    std::shared_ptr<GpuStream> main_stream_;
    std::shared_ptr<GpuStream> copy_stream_;
    std::shared_ptr<GpuKernel> kernel_;

    // For communication
    std::shared_ptr<mscclpp::Communicator> comm_;
    std::shared_ptr<mscclpp::ProxyService> proxy_service_;
    std::map<int, std::vector<std::shared_ptr<mscclpp::SimpleProxyChannel>>>
        rank_to_proxy_channels_;
    std::map<int, std::vector<std::shared_ptr<mscclpp::SmChannel>>>
        rank_to_sm_channels_;
};

Executor::Impl::Impl(int rank, int world_size, int gpu_id,
                     const std::string &name, const std::string &plan)
    : rank_(rank), world_size_(world_size), gpu_id_(gpu_id) {
    if (rank < 0 || rank >= world_size) {
        ERR(InvalidUsageError, "Invalid rank ", rank, " with world size ",
            world_size);
    }
    if (gpu_id < 0) {
        ERR(InvalidUsageError, "Invalid GPU ID ", gpu_id);
    }
    if (world_size_ > 1) {
        init_communicator();
    }

    Json plan_json;
    auto &plan_path = get_env().enforce_plan_path;
    if (!plan_path.empty()) {
        LOG(INFO, "Enforce executor plan path: ", plan_path);
        plan_json = Json::parse(read_file(plan_path));
    } else {
        plan_json = Json::parse(plan);
    }

    buffer_id_to_offset_ = init_buffers(plan_json);

    std::string buffer_id_to_offset_str;
    for (const auto &kv : buffer_id_to_offset_) {
        buffer_id_to_offset_str +=
            std::to_string(kv.first) + ": " + std::to_string(kv.second) + ", ";
    }

    codegen_ =
        std::make_shared<CodeGenerator>(plan_json, buffer_id_to_offset_, name);

    auto gpu_manager = GpuManager::get_instance(gpu_id_);
    timer_begin_ = gpu_manager->create_event();
    timer_end_ = gpu_manager->create_event();
    buffer_ = gpu_manager->malloc(total_bytes_, 65536);
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

    if (world_size_ > 1) {
        auto remote_ranks = init_remote_ranks(plan_json);
        init_channels(remote_ranks);
    }

    kernel_ = std::shared_ptr<GpuKernel>(
        new GpuKernel(gpu_id_, codegen_->code(), {threads_per_block, 1, 1},
                      {num_sm, 1, 1}, std::max(smem_block_total, size_t(4)),
                      name, {std::pair<void *, size_t>{flag, sizeof(flag)}}));
}

void Executor::Impl::init_communicator() {
    auto bootstrap =
        std::make_shared<mscclpp::TcpBootstrap>(rank_, world_size_);
    std::stringstream ip_port;
    ip_port << get_host(0) << ":" << get_env().mscclpp_port;
    bootstrap->initialize(ip_port.str());
    comm_ = std::make_shared<mscclpp::Communicator>(bootstrap);
}

std::map<size_t, size_t> Executor::Impl::init_buffers(const Json &plan_json) {
    class BufferInfo {
       public:
        BufferInfo(const std::shared_ptr<ModelBuffer> buffer)
            : buffer(buffer), bytes(0), is_input(true), is_output(true) {}

        // ID of this buffer
        const std::shared_ptr<ModelBuffer> buffer;

        // Total bytes of this buffer
        size_t bytes;

        // True if none of tensors in this buffer is a result tensor or a write
        // tensor of a non-virtual Op, i.e., this buffer is an input buffer
        bool is_input;

        // True if none of tensors in this buffer is a read tensor of a
        // non-virtual Op, i.e., this buffer is an output buffer
        bool is_output;

        // IDs of tensors in this buffer
        std::set<size_t> tensor_ids;

        // IDs of tasks that read/write from/to this buffer
        std::set<size_t> task_ids;
    };

    std::map<size_t, size_t> buffer_id_to_offset;
    std::map<size_t, std::shared_ptr<BufferInfo>> buffer_id_to_info;

    auto get_or_create_buffer_info = [&](const Json &buffer_json) {
        auto buffer = ModelBuffer::deserialize(buffer_json);
        if (buffer_id_to_info.find(buffer->id()) == buffer_id_to_info.end()) {
            auto buf_info = std::make_shared<BufferInfo>(buffer);
            buffer_id_to_info[buffer->id()] = buf_info;
            return buf_info;
        }
        return buffer_id_to_info[buffer->id()];
    };

    auto retrieve_buffer_info = [&](const Json &tensor, size_t task_id,
                                    bool is_input, bool is_output) {
        size_t tensor_id = tensor["Id"].get<size_t>();
        auto buf_info = get_or_create_buffer_info(tensor["Buffer"]);
        buf_info->bytes =
            std::max(buf_info->bytes, tensor_stride_bytes(tensor));
        buf_info->is_input = is_input;
        buf_info->is_output = is_output;
        buf_info->tensor_ids.insert(tensor_id);
        buf_info->task_ids.insert(task_id);
    };

    for (auto &task_info : plan_json["TaskInfos"]) {
        for (auto &op : task_info["Ops"]) {
            size_t task_id = task_info["Id"].get<size_t>();
            for (auto &tns : op["ReadTensors"]) {
                retrieve_buffer_info(tns, task_id, true, false);
            }
            for (auto &tns : op["WriteTensors"]) {
                retrieve_buffer_info(tns, task_id, false, true);
            }
            for (auto &tns : op["ResultTensors"]) {
                retrieve_buffer_info(tns, task_id, false, true);
            }
        }
    }

    std::map<int, std::pair<std::vector<int>, std::vector<size_t>>>
        remote_rank_to_send_tags_and_offsets;
    std::map<int, std::pair<std::vector<int>, std::vector<size_t>>>
        remote_rank_to_recv_tags_and_offsets;
    std::map<int, std::map<int, size_t>> remote_rank_to_send_tag_to_buffer_id;
    std::map<int, std::map<int, size_t>> remote_rank_to_recv_tag_to_buffer_id;

    // TODO: improve memory planning
    size_t offset = 0;
    for (auto &kv : buffer_id_to_info) {
        auto &buf_info = kv.second;
        int r = buf_info->buffer->rank();
        if (r != rank_ && r != -1) {
            // this is a remote buffer
            for (const auto &tag_info : buf_info->buffer->send_tags()) {
                remote_rank_to_send_tag_to_buffer_id[buf_info->buffer->rank()]
                                                    [tag_info.second] =
                                                        buf_info->buffer->id();
            }
            for (const auto &tag_info : buf_info->buffer->recv_tags()) {
                remote_rank_to_recv_tag_to_buffer_id[buf_info->buffer->rank()]
                                                    [tag_info.second] =
                                                        buf_info->buffer->id();
            }
            continue;
        }
        buffer_id_to_offset[buf_info->buffer->id()] = offset;
        for (const auto &tag_info : buf_info->buffer->send_tags()) {
            remote_rank_to_send_tags_and_offsets[tag_info.first]
                .first.push_back(tag_info.second);
            remote_rank_to_send_tags_and_offsets[tag_info.first]
                .second.push_back(offset);
        }
        for (const auto &tag_info : buf_info->buffer->recv_tags()) {
            remote_rank_to_recv_tags_and_offsets[tag_info.first]
                .first.push_back(tag_info.second);
            remote_rank_to_recv_tags_and_offsets[tag_info.first]
                .second.push_back(offset);
        }
        offset += buf_info->bytes;
    }
    total_bytes_ = offset;

    //
    // Send each tag (SendTag or RecvTag) and the corresponding offset to
    // remote ranks.
    //
    // If Rank 0 sends a local `Buffer X` data to `Buffer Y` in Rank 1 with
    // tag `t`, Rank 0 will declare another `Buffer Z` that represents
    // `Buffer Y` locally. Likewise, Rank 1 will declare `Buffer W` that
    // represents `Buffer X` locally. See the following example:
    //
    //         Rank 0 (Sender)               Rank 1 (Receiver)
    //    +----------------------+       +----------------------+
    //    | Buffer X             |       | Buffer Y             |
    //    | Rank: 0              |       | Rank: 1              |
    //    | Offset: 0x1000       |       | Offset: 0x2000       |
    //    | SendTag: [[1,t],...] |       | RecvTag: [[0,t],...] |
    //    +----------------------+       +----------------------+
    //    +----------------------+       +----------------------+
    //    | Buffer Z             |       | Buffer W             |
    //    | Rank: 1              |       | Rank: 0              |
    //    | Offset: ???          |       | Offset: ???          |
    //    | RecvTag: [[0,t],...] |       | SendTag: [[1,t],...] |
    //    +----------------------+       +----------------------+
    //
    // Offsets of Buffer Z and Buffer W are unknown at this point, because
    // they are determined by Rank 1 and Rank 0, respectively. To retrieve
    // the offsets, Rank 0 will go through SendTag of Buffer X and will send
    // the tag `t` and the offset `0x1000` to Rank 1. Rank 1 can then
    // determine the offset of Buffer W as `0x1000`, because Buffer W's rank
    // is 0 and it has a SendTag `t`. Likewise, Rank 1 will send the RecvTag `t`
    // and the offset `0x2000` to Rank 0, so that Rank 0 can determine the
    // offset of Buffer Z as `0x2000`.
    //

    for (auto &kv : remote_rank_to_send_tags_and_offsets) {
        auto remote_rank = kv.first;
        if (remote_rank == -1) continue;
        auto &tags_and_offsets = kv.second;
        auto &tags = tags_and_offsets.first;
        auto &offsets = tags_and_offsets.second;
        int len = tags.size();
        auto bootstrap = comm_->bootstrap();
        bootstrap->send(&len, sizeof(int), remote_rank, 0);
        bootstrap->send(tags.data(), tags.size() * sizeof(int), remote_rank, 1);
        bootstrap->send(offsets.data(), offsets.size() * sizeof(size_t),
                        remote_rank, 2);
    }
    for (auto &kv : remote_rank_to_recv_tags_and_offsets) {
        auto remote_rank = kv.first;
        if (remote_rank == -1) continue;
        auto &tags_and_offsets = kv.second;
        auto &tags = tags_and_offsets.first;
        auto &offsets = tags_and_offsets.second;
        int len = tags.size();
        auto bootstrap = comm_->bootstrap();
        bootstrap->send(&len, sizeof(int), remote_rank, 3);
        bootstrap->send(tags.data(), tags.size() * sizeof(int), remote_rank, 4);
        bootstrap->send(offsets.data(), offsets.size() * sizeof(size_t),
                        remote_rank, 5);
    }
    for (auto &kv : remote_rank_to_send_tag_to_buffer_id) {
        auto remote_rank = kv.first;
        auto &send_tag_to_buffer_id = kv.second;
        std::vector<int> tags;
        std::vector<size_t> offsets;
        int len;
        auto bootstrap = comm_->bootstrap();
        bootstrap->recv(&len, sizeof(int), remote_rank, 0);
        tags.resize(len);
        offsets.resize(len);
        bootstrap->recv(tags.data(), len * sizeof(int), remote_rank, 1);
        bootstrap->recv(offsets.data(), len * sizeof(size_t), remote_rank, 2);
        for (int i = 0; i < len; ++i) {
            buffer_id_to_offset[send_tag_to_buffer_id[tags[i]]] = offsets[i];
        }
    }
    for (auto &kv : remote_rank_to_recv_tag_to_buffer_id) {
        auto remote_rank = kv.first;
        auto &recv_tag_to_buffer_id = kv.second;
        std::vector<int> tags;
        std::vector<size_t> offsets;
        int len;
        auto bootstrap = comm_->bootstrap();
        bootstrap->recv(&len, sizeof(int), remote_rank, 3);
        tags.resize(len);
        offsets.resize(len);
        bootstrap->recv(tags.data(), len * sizeof(int), remote_rank, 4);
        bootstrap->recv(offsets.data(), len * sizeof(size_t), remote_rank, 5);
        for (int i = 0; i < len; ++i) {
            buffer_id_to_offset[recv_tag_to_buffer_id[tags[i]]] = offsets[i];
        }
    }

    return buffer_id_to_offset;
}

std::set<int> Executor::Impl::init_remote_ranks(const Json &plan_json) const {
    std::set<int> remote_ranks;
    for (auto &task_info : plan_json["TaskInfos"]) {
        for (auto &op : task_info["Ops"]) {
            for (auto &tns : op["ReadTensors"]) {
                auto buffer = ModelBuffer::deserialize(tns["Buffer"]);
                if (buffer->rank() != rank_ && buffer->rank() != -1) {
                    remote_ranks.insert(buffer->rank());
                }
            }
            for (auto &tns : op["WriteTensors"]) {
                auto buffer = ModelBuffer::deserialize(tns["Buffer"]);
                if (buffer->rank() != rank_ && buffer->rank() != -1) {
                    remote_ranks.insert(buffer->rank());
                }
            }
            for (auto &tns : op["ResultTensors"]) {
                auto buffer = ModelBuffer::deserialize(tns["Buffer"]);
                if (buffer->rank() != rank_ && buffer->rank() != -1) {
                    remote_ranks.insert(buffer->rank());
                }
            }
        }
    }
    return remote_ranks;
}

void Executor::Impl::init_channels(const std::set<int> &remote_ranks) {
    proxy_service_ = std::make_shared<mscclpp::ProxyService>();

    int num_ranks_per_node = get_env().num_ranks_per_host;
    auto rank_to_node = [&](int rank) { return rank / num_ranks_per_node; };
    int this_node = rank_to_node(rank_);

    const mscclpp::Transport IBs[] = {
        mscclpp::Transport::IB0, mscclpp::Transport::IB1,
        mscclpp::Transport::IB2, mscclpp::Transport::IB3,
        mscclpp::Transport::IB4, mscclpp::Transport::IB5,
        mscclpp::Transport::IB6, mscclpp::Transport::IB7};

    mscclpp::TransportFlags all_transports =
        mscclpp::Transport::CudaIpc | mscclpp::Transport::Ethernet;
    if (!get_env().disable_ib) {
        all_transports |= IBs[gpu_id_];
    }
    mscclpp::RegisteredMemory regmem =
        comm_->registerMemory(buffer_->ref(), buffer_->bytes(), all_transports);

    std::map<int, std::vector<mscclpp::NonblockingFuture<
                      std::shared_ptr<mscclpp::Connection>>>>
        rank_to_connections_future;
    std::map<int, mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>>
        rank_to_remote_regmem_future;

    for (auto remote_rank : remote_ranks) {
        int remote_node = rank_to_node(remote_rank);
        auto add_connection = [&](int remote_rank,
                                  mscclpp::Transport transport) {
            rank_to_connections_future[remote_rank].push_back(
                comm_->connectOnSetup(remote_rank, 0, transport));
        };
        if (remote_node == this_node) {
            add_connection(remote_rank, mscclpp::Transport::CudaIpc);
            if (!get_env().disable_ib) {
                add_connection(remote_rank, IBs[gpu_id_]);
            }
        } else {
            add_connection(remote_rank, get_env().disable_ib
                                            ? mscclpp::Transport::Ethernet
                                            : IBs[gpu_id_]);
        }
        comm_->sendMemoryOnSetup(regmem, remote_rank, 0);
        rank_to_remote_regmem_future[remote_rank] =
            comm_->recvMemoryOnSetup(remote_rank, 0);
    }
    comm_->setup();

    std::map<int, std::vector<std::shared_ptr<mscclpp::Connection>>>
        rank_to_connections;
    for (auto &kv : rank_to_connections_future) {
        for (auto &future : kv.second) {
            rank_to_connections[kv.first].push_back(future.get());
        }
    }
    for (auto &kv : rank_to_connections) {
        for (auto &conn : kv.second) {
            rank_to_proxy_channels_[kv.first].push_back(
                std::make_shared<mscclpp::SimpleProxyChannel>(
                    proxy_service_->proxyChannel(
                        proxy_service_->buildAndAddSemaphore(*comm_, conn)),
                    proxy_service_->addMemory(
                        rank_to_remote_regmem_future[kv.first].get()),
                    proxy_service_->addMemory(regmem)));
        }
    }
    comm_->setup();

    std::map<int,
             std::vector<std::shared_ptr<mscclpp::SmDevice2DeviceSemaphore>>>
        sm_semaphores;
    for (auto &kv : rank_to_connections) {
        for (auto &conn : kv.second) {
            if (conn->transport() != mscclpp::Transport::CudaIpc) continue;
            sm_semaphores[kv.first].push_back(
                std::make_shared<mscclpp::SmDevice2DeviceSemaphore>(*comm_,
                                                                    conn));
        }
    }
    comm_->setup();

    for (auto &kv : sm_semaphores) {
        for (auto &sem : kv.second) {
            rank_to_sm_channels_[kv.first].push_back(
                std::make_shared<mscclpp::SmChannel>(
                    sem, rank_to_remote_regmem_future[kv.first].get(),
                    regmem.data(), nullptr));
        }
    }
}

void Executor::Impl::compile() { kernel_->compile(); }

void Executor::Impl::launch(int64_t max_spin_count) {
    if (!kernel_->is_compiled()) {
        ERR(InvalidUsageError, "Need to compile first before initialization.");
    }
    if (is_launched_) {
        // Wait until previous works finish.
        this->wait(max_spin_count);
        return;
    }
    auto get_global_rt = [&](const std::string &symbol) {
        return reinterpret_cast<void *>(kernel_->get_global(symbol));
    };
    // Initialize global variables in the loop kernel.
    GLOG(gpuSetDevice(gpu_id_));
    void *buf_ptr_val = buffer_->ref();
    void *lss_ptr_addr = get_global_rt("ARK_LOOP_SYNC_STATE");
    void *buf_ptr_addr = get_global_rt("ARK_BUF");
    std::array<int, 4> data = {0, 0, 0, 0};
    GLOG(gpuMemcpyAsync(lss_ptr_addr, data.data(), sizeof(int) * data.size(),
                        gpuMemcpyHostToDevice, copy_stream_->get()));
    GLOG(gpuMemcpyAsync(buf_ptr_addr, &buf_ptr_val, sizeof(gpuDeviceptr),
                        gpuMemcpyHostToDevice, copy_stream_->get()));
    if (world_size_ > 1) {
        void *proxy_chan_addr = get_global_rt("ARK_PROXY_CHANS");
        void *proxy_secondary_chan_addr =
            get_global_rt("ARK_PROXY_SECONDARY_CHANS");
        void *sm_chan_addr = get_global_rt("ARK_SM_CHANS");
        std::vector<mscclpp::SimpleProxyChannel::DeviceHandle> proxy_handles(
            world_size_);
        std::vector<mscclpp::SimpleProxyChannel::DeviceHandle>
            proxy_secondary_handles(world_size_);
        std::vector<mscclpp::SmChannel::DeviceHandle> sm_handles(world_size_);
        for (int i = 0; i < world_size_; i++) {
            auto it = rank_to_proxy_channels_.find(i);
            if (it != rank_to_proxy_channels_.end() && it->second.size() > 0) {
                proxy_handles[i] = it->second[0]->deviceHandle();
                if (it->second.size() > 1) {
                    proxy_secondary_handles[i] = it->second[1]->deviceHandle();
                }
            }
            auto it2 = rank_to_sm_channels_.find(i);
            if (it2 != rank_to_sm_channels_.end() && it2->second.size() > 0) {
                sm_handles[i] = it2->second[0]->deviceHandle();
            }
        }
        GLOG(gpuMemcpyAsync(
            proxy_chan_addr, proxy_handles.data(),
            proxy_handles.size() *
                sizeof(mscclpp::SimpleProxyChannel::DeviceHandle),
            gpuMemcpyHostToDevice, copy_stream_->get()));
        GLOG(gpuMemcpyAsync(
            proxy_secondary_chan_addr, proxy_secondary_handles.data(),
            proxy_secondary_handles.size() *
                sizeof(mscclpp::SimpleProxyChannel::DeviceHandle),
            gpuMemcpyHostToDevice, copy_stream_->get()));
        GLOG(gpuMemcpyAsync(
            sm_chan_addr, sm_handles.data(),
            sm_handles.size() * sizeof(mscclpp::SmChannel::DeviceHandle),
            gpuMemcpyHostToDevice, copy_stream_->get()));
    }
    copy_stream_->sync();

    elapsed_msec_ = -1;
    if (!kernel_->is_compiled()) {
        ERR(InvalidUsageError, "Need to compile first before initialization.");
    } else if (is_launched_) {
        LOG(WARN, "Ignore launching twice.");
        return;
    }
    timer_begin_->record(main_stream_);

    if (world_size_ > 1) {
        proxy_service_->startProxy();
    }

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

void Executor::Impl::wait(int64_t max_spin_count) {
    int64_t cnt = max_spin_count;
    while (atomicLoadRelaxed(flag_->ref<int>()) > 0) {
        if (cnt-- > 0) {
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
            cnt = max_spin_count;
        } else {
            GLOG(res);
        }
    }
}

float Executor::Impl::stop(int64_t max_spin_count) {
    this->wait(max_spin_count);
    atomicStoreRelaxed(flag_->ref<int>(), -1);
    main_stream_->sync();
    if (is_recording_) {
        elapsed_msec_ = timer_end_->elapsed_msec(*timer_begin_);
        is_recording_ = false;
    }
    is_launched_ = false;
    if (world_size_ > 1) {
        proxy_service_->stopProxy();
    }
    return elapsed_msec_;
}

void Executor::Impl::barrier() {
    if (world_size_ > 1) {
        comm_->bootstrap()->barrier();
    }
}

uintptr_t Executor::Impl::tensor_address(const Tensor tensor) const {
    size_t buffer_id = tensor.ref()->buffer()->id();
    if (buffer_id_to_offset_.find(buffer_id) == buffer_id_to_offset_.end()) {
        ERR(NotFoundError, "Invalid buffer ID: ", buffer_id);
    }
    size_t offset = buffer_id_to_offset_.at(buffer_id);
    return reinterpret_cast<uintptr_t>(buffer_->ref(offset));
}

void Executor::Impl::tensor_read(const Tensor tensor, void *data, size_t bytes,
                                 bool is_d2d) const {
    GLOG(gpuSetDevice(gpu_id_));
    size_t tensor_data_bytes =
        tensor.shape().nelems() * tensor.data_type().bytes();
    if (bytes != tensor_data_bytes) {
        ERR(InvalidUsageError, "Destination bytes (", bytes,
            ") mismatches the tensor data bytes (", tensor_data_bytes, ").");
    }
    auto kind = (is_d2d) ? gpuMemcpyDeviceToDevice : gpuMemcpyDeviceToHost;
    void *src = reinterpret_cast<void *>(tensor_address(tensor));
    if (tensor.strides() == tensor.shape()) {
        GLOG(gpuMemcpyAsync(data, src, bytes, kind, copy_stream_->get()));
    } else {
        size_t tensor_bytes =
            tensor.strides().nelems() * tensor.data_type().bytes();
        std::vector<int8_t> tensor_host(tensor_bytes);
        GLOG(gpuMemcpyAsync(tensor_host.data(), src, tensor_bytes,
                            gpuMemcpyDeviceToHost, copy_stream_->get()));
        copy_stream_->sync();
        if (!is_d2d) {
            tensor_to_data(tensor_host.data(), static_cast<int8_t *>(data),
                           tensor.shape(), tensor.strides(), tensor.offsets(),
                           tensor.data_type().bytes());
            return;
        }
        // TODO: convert data layout on the device directly
        std::vector<int8_t> data_host(bytes);
        tensor_to_data(tensor_host.data(), data_host.data(), tensor.shape(),
                       tensor.strides(), tensor.offsets(),
                       tensor.data_type().bytes());
        GLOG(gpuMemcpyAsync(data, data_host.data(), bytes,
                            gpuMemcpyHostToDevice, copy_stream_->get()));
    }
    copy_stream_->sync();
}

void Executor::Impl::tensor_write(const Tensor tensor, const void *data,
                                  size_t bytes, bool is_d2d) const {
    GLOG(gpuSetDevice(gpu_id_));
    size_t tensor_data_bytes =
        tensor.shape().nelems() * tensor.data_type().bytes();
    if (bytes != tensor_data_bytes) {
        ERR(InvalidUsageError, "Source bytes (", bytes,
            ") mismatches the tensor data bytes (", tensor_data_bytes, ").");
    }
    size_t tensor_bytes =
        tensor.strides().nelems() * tensor.data_type().bytes();
    auto kind = (is_d2d) ? gpuMemcpyDeviceToDevice : gpuMemcpyHostToDevice;
    void *dst = reinterpret_cast<void *>(tensor_address(tensor));
    if (tensor.strides() == tensor.shape()) {
        GLOG(
            gpuMemcpyAsync(dst, data, tensor_bytes, kind, copy_stream_->get()));
    } else {
        std::vector<int8_t> tensor_host(tensor_bytes);
        if (!is_d2d) {
            data_to_tensor(tensor_host.data(),
                           static_cast<const int8_t *>(data), tensor.shape(),
                           tensor.strides(), tensor.offsets(),
                           tensor.data_type().bytes());
        } else {
            // TODO: convert data layout on the device directly
            std::vector<int8_t> tmp(bytes);
            GLOG(gpuMemcpyAsync(tmp.data(), data, bytes, gpuMemcpyDeviceToHost,
                                copy_stream_->get()));
            copy_stream_->sync();
            data_to_tensor(tensor_host.data(), tmp.data(), tensor.shape(),
                           tensor.strides(), tensor.offsets(),
                           tensor.data_type().bytes());
        }
        GLOG(gpuMemcpyAsync(dst, tensor_host.data(), tensor_bytes,
                            gpuMemcpyHostToDevice, copy_stream_->get()));
    }
    copy_stream_->sync();
}

Executor::Executor(int rank, int world_size, int gpu_id,
                   const std::string &name, const std::string &plan)
    : impl_(std::make_unique<Executor::Impl>(rank, world_size, gpu_id, name,
                                             plan)) {}

Executor::~Executor() = default;

int Executor::gpu_id() const { return impl_->gpu_id(); }

void Executor::compile() { impl_->compile(); }

void Executor::launch(int64_t max_spin_count) { impl_->launch(max_spin_count); }

void Executor::run(int iter) { impl_->run(iter); }

void Executor::wait(int64_t max_spin_count) { impl_->wait(max_spin_count); }

float Executor::stop(int64_t max_spin_count) {
    return impl_->stop(max_spin_count);
}

void Executor::barrier() { impl_->barrier(); }

void Executor::destroy() { impl_.reset(nullptr); }

bool Executor::destroyed() const { return impl_.get() == nullptr; }

uintptr_t Executor::tensor_address(const Tensor tensor) const {
    return impl_->tensor_address(tensor);
}

void Executor::tensor_read(const Tensor tensor, void *data, size_t bytes,
                           bool is_d2d) const {
    impl_->tensor_read(tensor, data, bytes, is_d2d);
}

void Executor::tensor_write(const Tensor tensor, const void *data, size_t bytes,
                            bool is_d2d) const {
    impl_->tensor_write(tensor, data, bytes, is_d2d);
}

DefaultExecutor::DefaultExecutor(const Model &model, int gpu_id,
                                 const std::string &name)
    : Executor(
          model.rank(), model.world_size(),
          (gpu_id < 0) ? (model.rank() % get_env().num_ranks_per_host) : gpu_id,
          name,
          DefaultPlanner(model, (gpu_id < 0) ? (model.rank() %
                                                get_env().num_ranks_per_host)
                                             : gpu_id)
              .plan()) {}

}  // namespace ark
