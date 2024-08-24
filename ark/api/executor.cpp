// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/executor.hpp"

#include <cmath>
#include <list>
#include <memory>
#include <mscclpp/core.hpp>
#include <mscclpp/proxy_channel.hpp>
#include <mscclpp/sm_channel.hpp>
#include <utility>

#include "ark/data_type.hpp"
#include "ark/model.hpp"
#include "ark/planner.hpp"
#include "buffer_registry.hpp"
#include "codegen.hpp"
#include "env.h"
#include "file_io.h"
#include "gpu/gpu.hpp"
#include "gpu/gpu_event.hpp"
#include "gpu/gpu_kernel.hpp"
#include "gpu/gpu_logging.hpp"
#include "gpu/gpu_manager.hpp"
#include "logging.hpp"
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

class CommResource {
   public:
    CommResource(int device_id, int rank, int world_size);

    int rank() const { return rank_; }

    int world_size() const { return world_size_; }

    std::shared_ptr<mscclpp::Bootstrap> bootstrap() {
        return comm_->bootstrap();
    }

    std::shared_ptr<mscclpp::Communicator> comm() { return comm_; }

    std::shared_ptr<mscclpp::ProxyService> proxy_service() {
        return proxy_service_;
    }

    struct ConnectionResource {
        std::shared_ptr<mscclpp::Connection> connection;
        std::vector<std::shared_ptr<mscclpp::SimpleProxyChannel>>
            proxy_channels;
        std::vector<std::shared_ptr<mscclpp::SmChannel>> sm_channels;
    };

    struct RankResource {
        int remote_rank;
        std::shared_ptr<ConnectionResource> ipc;
        std::shared_ptr<ConnectionResource> eth;
        std::shared_ptr<ConnectionResource> ib;
    };

    const std::shared_ptr<RankResource> resource(int rank) const {
        auto it = rank_to_resource_.find(rank);
        if (it == rank_to_resource_.end()) {
            return nullptr;
        }
        return it->second;
    }

    void connect(const PlanJson &plan_json, std::shared_ptr<GpuMemory> buffer);

   private:
    int device_id_;
    int rank_;
    int world_size_;
    std::shared_ptr<mscclpp::Communicator> comm_;
    std::shared_ptr<mscclpp::ProxyService> proxy_service_;
    std::map<int, std::shared_ptr<RankResource>> rank_to_resource_;
};

CommResource::CommResource(int device_id, int rank, int world_size)
    : device_id_(device_id), rank_(rank), world_size_(world_size) {
    auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, world_size);
    std::stringstream ip_port;
    ip_port << get_host(0) << ":" << get_env().mscclpp_port;
    bootstrap->initialize(ip_port.str());
    comm_ = std::make_shared<mscclpp::Communicator>(bootstrap);
    proxy_service_ = std::make_shared<mscclpp::ProxyService>();
}

void CommResource::connect(const PlanJson &plan_json,
                           std::shared_ptr<GpuMemory> buffer) {
    int rank = plan_json["Rank"];
    std::set<int> remote_ranks;
    for (auto &task_info : plan_json["TaskInfos"]) {
        for (auto &op : task_info["Ops"]) {
            for (auto &tns : op["ReadTensors"]) {
                auto buffer = ModelBuffer::deserialize(tns["Buffer"]);
                if (buffer->rank() != rank && buffer->rank() != -1) {
                    remote_ranks.insert(buffer->rank());
                }
            }
            for (auto &tns : op["WriteTensors"]) {
                auto buffer = ModelBuffer::deserialize(tns["Buffer"]);
                if (buffer->rank() != rank && buffer->rank() != -1) {
                    remote_ranks.insert(buffer->rank());
                }
            }
            for (auto &tns : op["ResultTensors"]) {
                auto buffer = ModelBuffer::deserialize(tns["Buffer"]);
                if (buffer->rank() != rank && buffer->rank() != -1) {
                    remote_ranks.insert(buffer->rank());
                }
            }
        }
    }
    if (remote_ranks.empty()) return;

    int num_ranks_per_node = get_env().num_ranks_per_host;
    auto rank_to_node = [&](int r) { return r / num_ranks_per_node; };
    int this_node = rank_to_node(rank);

    const mscclpp::Transport IBs[] = {
        mscclpp::Transport::IB0, mscclpp::Transport::IB1,
        mscclpp::Transport::IB2, mscclpp::Transport::IB3,
        mscclpp::Transport::IB4, mscclpp::Transport::IB5,
        mscclpp::Transport::IB6, mscclpp::Transport::IB7};

    mscclpp::TransportFlags all_transports =
        mscclpp::Transport::CudaIpc | mscclpp::Transport::Ethernet;
    if (!get_env().disable_ib) {
        all_transports |= IBs[device_id_];
    }
    mscclpp::RegisteredMemory regmem =
        comm_->registerMemory(buffer->ref(), buffer->bytes(), all_transports);

    using ConnectionFuture =
        mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>;
    std::map<int, ConnectionFuture> rank_to_ipc_connection_future;
    std::map<int, ConnectionFuture> rank_to_eth_connection_future;
    std::map<int, ConnectionFuture> rank_to_ib_connection_future;
    std::map<int, mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>>
        rank_to_remote_regmem_future;

    for (auto remote_rank : remote_ranks) {
        auto it = rank_to_resource_.find(remote_rank);
        if (it != rank_to_resource_.end()) {
            // connection already set
            continue;
        }
        auto resource = std::make_shared<RankResource>();
        rank_to_resource_[remote_rank] = resource;
        int remote_node = rank_to_node(remote_rank);
        if (remote_node == this_node) {
            rank_to_ipc_connection_future[remote_rank] = comm_->connectOnSetup(
                remote_rank, 0, mscclpp::Transport::CudaIpc);
            resource->ipc = std::make_shared<ConnectionResource>();
        }
        if ((remote_node != this_node) && get_env().disable_ib) {
            rank_to_eth_connection_future[remote_rank] = comm_->connectOnSetup(
                remote_rank, 0, mscclpp::Transport::Ethernet);
            resource->eth = std::make_shared<ConnectionResource>();
        }
        if (!get_env().disable_ib) {
            rank_to_ib_connection_future[remote_rank] =
                comm_->connectOnSetup(remote_rank, 0, IBs[device_id_]);
            resource->ib = std::make_shared<ConnectionResource>();
        }
        comm_->sendMemoryOnSetup(regmem, remote_rank, 0);
        rank_to_remote_regmem_future[remote_rank] =
            comm_->recvMemoryOnSetup(remote_rank, 0);
    }
    comm_->setup();

    for (auto &[remote_rank, future] : rank_to_ipc_connection_future) {
        rank_to_resource_[remote_rank]->ipc->connection = future.get();
    }
    for (auto &[remote_rank, future] : rank_to_eth_connection_future) {
        rank_to_resource_[remote_rank]->eth->connection = future.get();
    }
    for (auto &[remote_rank, future] : rank_to_ib_connection_future) {
        rank_to_resource_[remote_rank]->ib->connection = future.get();
    }

    mscclpp::MemoryId regmem_id = proxy_service_->addMemory(regmem);
    std::map<int, mscclpp::RegisteredMemory> rank_to_remote_regmem;
    std::map<int, mscclpp::MemoryId> rank_to_remote_regmem_id;
    for (auto &[remote_rank, future] : rank_to_remote_regmem_future) {
        rank_to_remote_regmem[remote_rank] = future.get();
        rank_to_remote_regmem_id[remote_rank] =
            proxy_service_->addMemory(rank_to_remote_regmem[remote_rank]);
    }

    for (auto &[remote_rank, resource] : rank_to_resource_) {
        auto add_proxy_channel =
            [&](std::shared_ptr<ConnectionResource> conn_resource) {
                if (!conn_resource) return;
                conn_resource->proxy_channels.push_back(
                    std::make_shared<mscclpp::SimpleProxyChannel>(
                        proxy_service_->proxyChannel(
                            proxy_service_->buildAndAddSemaphore(
                                *comm_, conn_resource->connection)),
                        rank_to_remote_regmem_id[remote_rank], regmem_id));
            };
        // NOTE: We can create multiple proxy channels here if we need in the
        // future
        add_proxy_channel(resource->ipc);
        add_proxy_channel(resource->eth);
        add_proxy_channel(resource->ib);
    }
    comm_->setup();

    std::map<int,
             std::vector<std::shared_ptr<mscclpp::SmDevice2DeviceSemaphore>>>
        sm_semaphores;
    for (auto &[remote_rank, resource] : rank_to_resource_) {
        // NOTE: We can create multiple semaphores here if we need in the future
        sm_semaphores[remote_rank].push_back(
            std::make_shared<mscclpp::SmDevice2DeviceSemaphore>(
                *comm_, resource->ipc->connection));
    }
    comm_->setup();

    for (auto &[remote_rank, resource] : rank_to_resource_) {
        // NOTE: We can create multiple sm channels here if we need in the
        // future
        resource->ipc->sm_channels.push_back(
            std::make_shared<mscclpp::SmChannel>(
                sm_semaphores[remote_rank][0],
                rank_to_remote_regmem[remote_rank], regmem.data(), nullptr));
    }
}

class PlanResourceKey {
   public:
    PlanResourceKey(const std::string &plan, int device_id,
                    const std::string &name)
        : plan_(plan), device_id_(device_id), name_(name) {}

    bool operator<(const PlanResourceKey &other) const {
        return std::tie(plan_, device_id_, name_) <
               std::tie(other.plan_, other.device_id_, other.name_);
    }

   private:
    std::string plan_;
    int device_id_;
    std::string name_;
};

class PlanResource {
   public:
    PlanResource(const PlanJson &plan_json, int device_id,
                 const std::string &name,
                 std::shared_ptr<CommResource> comm_resource);

    const PlanJson &plan_json() const { return plan_json_; }

    int device_id() const { return device_id_; }

    const std::string &name() const { return name_; }

    std::shared_ptr<GpuMemory> buffer() const { return buffer_; }

    void launch_kernel(const std::string &name, const std::vector<void *> &args,
                       gpuStream stream);

   private:
    void verify_plan();
    void init_comm_resource();
    void init_internal_buffers();
    void init_comm_connections();
    void init_kernel();

    PlanJson plan_json_;
    int device_id_;
    std::string name_;
    std::shared_ptr<CommResource> comm_resource_;

    std::shared_ptr<GpuMemory> buffer_;
    std::map<size_t, size_t> internal_buffer_id_to_offset_;
    // extra buffers: external buffers or buffers that are allocated by other
    // plans
    std::set<size_t> extra_buffer_ids_;
    std::shared_ptr<GpuKernel> kernel_;
};

PlanResource::PlanResource(const PlanJson &plan_json, int device_id,
                           const std::string &name,
                           std::shared_ptr<CommResource> comm_resource)
    : plan_json_(plan_json),
      device_id_(device_id),
      name_(name),
      comm_resource_(comm_resource) {
    if (device_id < 0) {
        ERR(InvalidUsageError, "Invalid device ID ", device_id);
    }

    // Verify if `plan_json` is describes a valid plan
    verify_plan();

    // Construct `comm_resource_` if needed
    init_comm_resource();

    // Allocate memory for internal buffers and construct
    // `internal_buffer_id_to_offset_` and `extra_buffer_ids_`.
    init_internal_buffers();

    // Create connections and channels to remote ranks
    init_comm_connections();

    // Construct `kernel_`.
    init_kernel();
}

void PlanResource::verify_plan() {
    int rank = plan_json_["Rank"];
    int world_size = plan_json_["WorldSize"];
    if (rank < 0 || rank >= world_size) {
        ERR(InvalidUsageError, "Invalid rank ", rank, " with world size ",
            world_size);
    }
    auto gpu_manager = GpuManager::get_instance(device_id_);
    if (!gpu_manager->info().arch->belongs_to(
            Arch::from_name(plan_json_.at("Architecture")))) {
        LOG(WARN, "Architecture name of the plan `",
            plan_json_.at("Architecture").get<std::string>(),
            "` is not compatible with the GPU architecture `",
            gpu_manager->info().arch->name(), "`.");
    }
}

void PlanResource::init_comm_resource() {
    int rank = plan_json_["Rank"];
    int world_size = plan_json_["WorldSize"];
    if (comm_resource_) {
        if (comm_resource_->rank() != rank) {
            ERR(InvalidUsageError,
                "Rank should be consistent across all plans. "
                "Expected ",
                rank, " but got ", comm_resource_->rank());
        }
        if (comm_resource_->world_size() != world_size) {
            ERR(InvalidUsageError,
                "World size should be consistent across all "
                "plans. Expected ",
                world_size, " but got ", comm_resource_->world_size());
        }
    } else if (world_size > 1) {
        comm_resource_ =
            std::make_shared<CommResource>(device_id_, rank, world_size);
    }
}

void PlanResource::init_internal_buffers() {
    class BufferInfo {
       public:
        BufferInfo(const std::shared_ptr<ModelBuffer> buffer)
            : buffer(buffer), bytes(0), is_input(true), is_output(true) {}

        // Underlying ModelBuffer
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

    std::map<size_t, std::shared_ptr<BufferInfo>> buffer_id_to_info;

    auto get_or_create_buffer_info = [&](const Json &buffer_json) {
        auto buffer = ModelBuffer::deserialize(buffer_json);
        auto it = buffer_id_to_info.find(buffer->id());
        if (it == buffer_id_to_info.end()) {
            auto buf_info = std::make_shared<BufferInfo>(buffer);
            buffer_id_to_info[buffer->id()] = buf_info;
            return buf_info;
        }
        return it->second;
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

    for (auto &task_info : plan_json_["TaskInfos"]) {
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
    int rank = plan_json_["Rank"];
    for (auto &[buf_id, buf_info] : buffer_id_to_info) {
        auto &buffer = buf_info->buffer;
        int r = buffer->rank();
        if (r != rank && r != -1) {
            // this is a remote buffer
            if (buffer->is_external()) {
                ERR(InvalidUsageError,
                    "Communication with external buffers is not supported");
            }
            for (const auto &tag_info : buffer->send_tags()) {
                remote_rank_to_send_tag_to_buffer_id[r][tag_info.second] =
                    buf_id;
            }
            for (const auto &tag_info : buffer->recv_tags()) {
                remote_rank_to_recv_tag_to_buffer_id[r][tag_info.second] =
                    buf_id;
            }
            continue;
        }
        auto info = BufferRegistry::get_instance().get(buf_id);
        if (info || buffer->is_external()) {
            // This buffer is external or has been already allocated by a
            // previous plan.
            extra_buffer_ids_.insert(buf_id);
        } else {
            internal_buffer_id_to_offset_[buf_id] = offset;
            for (const auto &tag_info : buffer->send_tags()) {
                remote_rank_to_send_tags_and_offsets[tag_info.first]
                    .first.push_back(tag_info.second);
                remote_rank_to_send_tags_and_offsets[tag_info.first]
                    .second.push_back(offset);
            }
            for (const auto &tag_info : buffer->recv_tags()) {
                remote_rank_to_recv_tags_and_offsets[tag_info.first]
                    .first.push_back(tag_info.second);
                remote_rank_to_recv_tags_and_offsets[tag_info.first]
                    .second.push_back(offset);
            }
            offset += buf_info->bytes;
        }
    }

    // Allocate memory for internal local buffers
    if (offset > 0) {
        buffer_ = GpuManager::get_instance(device_id_)->malloc(offset, 65536);
        for (auto &[buf_id, buf_info] : buffer_id_to_info) {
            auto &buffer = buf_info->buffer;
            if (buffer->is_external()) continue;
            int r = buffer->rank();
            if (r != rank && r != -1) continue;
            auto it = internal_buffer_id_to_offset_.find(buf_id);
            if (it == internal_buffer_id_to_offset_.end()) {
                continue;
            }
            size_t off = it->second;
            BufferRegistry::get_instance().set(buffer->id(), buffer_->ref(off),
                                               device_id_, false);
        }
    }

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
        auto bootstrap = comm_resource_->bootstrap();
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
        auto bootstrap = comm_resource_->bootstrap();
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
        auto bootstrap = comm_resource_->bootstrap();
        bootstrap->recv(&len, sizeof(int), remote_rank, 0);
        tags.resize(len);
        offsets.resize(len);
        bootstrap->recv(tags.data(), len * sizeof(int), remote_rank, 1);
        bootstrap->recv(offsets.data(), len * sizeof(size_t), remote_rank, 2);
        for (int i = 0; i < len; ++i) {
            size_t buf_id = send_tag_to_buffer_id[tags[i]];
            internal_buffer_id_to_offset_[buf_id] = offsets[i];
        }
    }
    for (auto &kv : remote_rank_to_recv_tag_to_buffer_id) {
        auto remote_rank = kv.first;
        auto &recv_tag_to_buffer_id = kv.second;
        std::vector<int> tags;
        std::vector<size_t> offsets;
        int len;
        auto bootstrap = comm_resource_->bootstrap();
        bootstrap->recv(&len, sizeof(int), remote_rank, 3);
        tags.resize(len);
        offsets.resize(len);
        bootstrap->recv(tags.data(), len * sizeof(int), remote_rank, 4);
        bootstrap->recv(offsets.data(), len * sizeof(size_t), remote_rank, 5);
        for (int i = 0; i < len; ++i) {
            size_t buf_id = recv_tag_to_buffer_id[tags[i]];
            internal_buffer_id_to_offset_[buf_id] = offsets[i];
        }
    }
}

void PlanResource::init_comm_connections() {
    if (comm_resource_ && buffer_) {
        comm_resource_->connect(plan_json_, buffer_);
    }
}

void PlanResource::init_kernel() {
    auto gpu_manager = GpuManager::get_instance(device_id_);
    auto codegen = std::make_shared<CodeGenerator>(
        plan_json_, internal_buffer_id_to_offset_, extra_buffer_ids_);
    int num_sm = static_cast<int>(codegen->num_procs());
    int threads_per_block = static_cast<int>(
        codegen->num_warps_per_proc() * gpu_manager->info().threads_per_warp);
    size_t smem_block_total =
        static_cast<size_t>(gpu_manager->info().smem_block_total);

    kernel_ = std::shared_ptr<GpuKernel>(
        new GpuKernel(device_id_, codegen->code(), {threads_per_block, 1, 1},
                      {num_sm, 1, 1}, std::max(smem_block_total, size_t(4))));
    kernel_->compile();

    int world_size = plan_json_["WorldSize"];
    if (world_size <= 1) return;

    auto get_global_rt = [&](const std::string &symbol) {
        return reinterpret_cast<void *>(kernel_->get_global(symbol));
    };
    void *proxy_chan_addr = get_global_rt("ARK_PROXY_CHANS");
    void *proxy_secondary_chan_addr =
        get_global_rt("ARK_PROXY_SECONDARY_CHANS");
    void *sm_chan_addr = get_global_rt("ARK_SM_CHANS");
    std::vector<mscclpp::SimpleProxyChannel::DeviceHandle> proxy_handles(
        world_size);
    std::vector<mscclpp::SimpleProxyChannel::DeviceHandle>
        proxy_secondary_handles(world_size);
    std::vector<mscclpp::SmChannel::DeviceHandle> sm_handles(world_size);
    for (int i = 0; i < world_size; i++) {
        auto resource = comm_resource_->resource(i);
        if (!resource) continue;
        std::vector<mscclpp::SimpleProxyChannel::DeviceHandle> p_hdls;
        if (resource->ipc) {
            sm_handles[i] = resource->ipc->sm_channels[0]->deviceHandle();
            p_hdls.push_back(resource->ipc->proxy_channels[0]->deviceHandle());
        }
        if (resource->ib) {
            p_hdls.push_back(resource->ib->proxy_channels[0]->deviceHandle());
        }
        if (resource->eth) {
            p_hdls.push_back(resource->eth->proxy_channels[0]->deviceHandle());
        }
        if (p_hdls.size() > 0) {
            proxy_handles[i] = p_hdls[0];
        }
        if (p_hdls.size() > 1) {
            proxy_secondary_handles[i] = p_hdls[1];
        }
    }
    auto tmp_stream = gpu_manager->create_stream();
    GLOG(gpuSetDevice(device_id_));
    GLOG(gpuMemcpyAsync(proxy_chan_addr, proxy_handles.data(),
                        proxy_handles.size() *
                            sizeof(mscclpp::SimpleProxyChannel::DeviceHandle),
                        gpuMemcpyHostToDevice, tmp_stream->get()));
    GLOG(gpuMemcpyAsync(proxy_secondary_chan_addr,
                        proxy_secondary_handles.data(),
                        proxy_secondary_handles.size() *
                            sizeof(mscclpp::SimpleProxyChannel::DeviceHandle),
                        gpuMemcpyHostToDevice, tmp_stream->get()));
    GLOG(gpuMemcpyAsync(
        sm_chan_addr, sm_handles.data(),
        sm_handles.size() * sizeof(mscclpp::SmChannel::DeviceHandle),
        gpuMemcpyHostToDevice, tmp_stream->get()));
    GLOG(gpuStreamSynchronize(tmp_stream->get()));
}

void PlanResource::launch_kernel(const std::string &name,
                                 const std::vector<void *> &args,
                                 gpuStream stream) {
    std::vector<void *> kernel_args = args;
    for (size_t ext_buf_id : extra_buffer_ids_) {
        auto ext_buf_addr = BufferRegistry::get_instance().get(ext_buf_id);
        if (!ext_buf_addr) {
            ERR(InternalError, "External buffer not found.");
        }
        if (ext_buf_addr->data == nullptr) {
            ERR(InvalidUsageError, "External buffer data is nullptr.");
        }
        kernel_args.push_back(&(ext_buf_addr->data));
    }
    kernel_->launch(name, stream, kernel_args);
}

class Executor::Impl {
   public:
    Impl(){};
    ~Impl();

    int device_id() const {
        return foreground_plan_resource_
                   ? foreground_plan_resource_->device_id()
                   : -1;
    }

    Stream stream() const { return reinterpret_cast<Stream>(stream_raw_); }

    std::shared_ptr<GpuMemory> buffer() const {
        return foreground_plan_resource_ ? foreground_plan_resource_->buffer()
                                         : nullptr;
    }

    std::string plan() const {
        return foreground_plan_resource_
                   ? foreground_plan_resource_->plan_json().dump_pretty()
                   : "";
    }

    std::string name() const {
        return foreground_plan_resource_ ? foreground_plan_resource_->name()
                                         : "";
    }

    void compile(const std::string &plan, int device_id,
                 const std::string &name);
    void launch(const std::unordered_map<Tensor, void *> &placeholder_data,
                Stream stream, bool loop_mode, bool record);
    void run(int iter,
             const std::unordered_map<Tensor, void *> &placeholder_data);
    void wait(int64_t max_spin_count);
    float stop(int64_t max_spin_count);
    void barrier();

    void *tensor_address(const Tensor &tensor) const;

    void tensor_read(const Tensor &tensor, void *data, size_t bytes,
                     Stream stream, bool is_d2d) const;
    void tensor_write(const Tensor &tensor, const void *data, size_t bytes,
                      Stream stream, bool is_d2d) const;

   protected:
    friend class DefaultExecutor;

    gpuStream stream_raw_;
    bool loop_mode_;

   private:
    std::map<PlanResourceKey, std::shared_ptr<PlanResource>> plan_resources_;
    std::shared_ptr<PlanResource> foreground_plan_resource_;
    std::shared_ptr<CommResource> comm_resource_;

    bool is_launched_ = false;
    bool is_recording_ = false;
    float elapsed_msec_ = -1;

    std::shared_ptr<GpuEvent> timer_begin_;
    std::shared_ptr<GpuEvent> timer_end_;
    std::shared_ptr<GpuHostMemory> flag_;
    std::shared_ptr<GpuStream> stream_;
};

Executor::Impl::~Impl() {
    if (is_launched_) stop(-1);
}

void Executor::Impl::compile(const std::string &plan, int device_id,
                             const std::string &name) {
    if (is_launched_) {
        ERR(InvalidUsageError, "Need to stop before re-compiling.");
    }
    int prev_device_id = -1;
    if (foreground_plan_resource_) {
        prev_device_id = foreground_plan_resource_->device_id();
    }
    if (prev_device_id != device_id) {
        auto gpu_manager = GpuManager::get_instance(device_id);
        timer_begin_ = gpu_manager->create_event();
        timer_end_ = gpu_manager->create_event();
        flag_ = gpu_manager->malloc_host(
            sizeof(int), gpuHostAllocMapped | gpuHostAllocWriteCombined);
        stream_ = gpu_manager->create_stream();
    }
    PlanResourceKey key(plan, device_id, name);
    auto it = plan_resources_.find(key);
    if (it == plan_resources_.end()) {
        try {
            auto plan_json = Json::parse(plan);
            auto resource = std::make_shared<PlanResource>(
                plan_json, device_id, name, comm_resource_);
            plan_resources_[key] = resource;
            foreground_plan_resource_ = resource;
        } catch (const ::nlohmann::json::parse_error &e) {
            ERR(InvalidUsageError, "Failed to parse the plan JSON: ", e.what());
        }
    } else {
        foreground_plan_resource_ = it->second;
    }
}

void Executor::Impl::launch(
    const std::unordered_map<Tensor, void *> &placeholder_data, Stream stream,
    bool loop_mode, bool record) {
    if (!foreground_plan_resource_) {
        ERR(InvalidUsageError, "Need to compile first before launch.");
    }
    if (is_launched_) {
        LOG(WARN, "Ignore launching twice.");
        return;
    }
    for (const auto &[tensor, ptr] : placeholder_data) {
        if (tensor.ref()->data(ptr) != ptr) {
            ERR(InvalidUsageError,
                "Placeholder data must be external tensors.");
        }
    }

    stream_raw_ = stream ? reinterpret_cast<gpuStream>(stream) : stream_->get();
    loop_mode_ = loop_mode;
    elapsed_msec_ = -1;

    if (record) {
        timer_begin_->record(stream_raw_);
        is_recording_ = true;
    }
    if (comm_resource_) {
        comm_resource_->proxy_service()->startProxy();
    }

    if (loop_mode_) {
        // Initialize loop flags.
        atomicStoreRelaxed(flag_->ref<int>(), 0);
        auto buffer = foreground_plan_resource_->buffer();
        void *buf_ptr = buffer ? buffer->ref() : nullptr;
        LOG(WARN, "buf_ptr: ", buf_ptr);
        void *flag_ptr = flag_->ref();
        std::vector<void *> args = {&buf_ptr, &flag_ptr};
        foreground_plan_resource_->launch_kernel("ark_loop_kernel", args,
                                                 stream_raw_);
    }
    is_launched_ = true;
}

void Executor::Impl::run(
    int iter, const std::unordered_map<Tensor, void *> &placeholder_data) {
    for (const auto &[tensor, ptr] : placeholder_data) {
        if (tensor.ref()->data(ptr) != ptr) {
            ERR(InvalidUsageError,
                "Placeholder data must be external tensors.");
        }
    }
    if (iter <= 0) return;
    if (loop_mode_) {
        while (atomicLoadRelaxed(flag_->ref<int>()) > 0) {
        }
        atomicStoreRelaxed(flag_->ref<int>(), iter);
    } else {
        auto buffer = foreground_plan_resource_->buffer();
        void *buf_ptr = buffer ? buffer->ref() : nullptr;
        int i = 0;
        std::vector<void *> args = {&buf_ptr, reinterpret_cast<void *>(&i)};
        for (; i < iter; i++) {
            foreground_plan_resource_->launch_kernel("ark_kernel", args,
                                                     stream_raw_);
        }
    }
}

void Executor::Impl::wait(int64_t max_spin_count) {
    int64_t cnt = max_spin_count;
    if (loop_mode_) {
        while (atomicLoadRelaxed(flag_->ref<int>()) > 0) {
            if (cnt-- > 0) {
                continue;
            }
            // Check if the kernel encountered an error.
            gpuError res = gpuStreamQuery(stream_raw_);
            if (res == gpuSuccess) {
                if (atomicLoadRelaxed(flag_->ref<int>()) > 0) {
                    ERR(InternalError,
                        "Stream is finished but the loop flag is still set.");
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
    } else {
        if (max_spin_count >= 0) {
            LOG(WARN, "max_spin_count is ignored in non-loop mode.");
        }
        GLOG(gpuStreamSynchronize(stream_raw_));
    }
}

float Executor::Impl::stop(int64_t max_spin_count) {
    this->wait(max_spin_count);
    if (is_recording_) {
        timer_end_->record(stream_raw_);
    }
    if (loop_mode_) {
        atomicStoreRelaxed(flag_->ref<int>(), -1);
    }
    GLOG(gpuStreamSynchronize(stream_raw_));
    if (is_recording_) {
        elapsed_msec_ = timer_end_->elapsed_msec(*timer_begin_);
        is_recording_ = false;
    }
    is_launched_ = false;
    if (comm_resource_) {
        comm_resource_->proxy_service()->stopProxy();
    }
    return elapsed_msec_;
}

void Executor::Impl::barrier() {
    if (comm_resource_) {
        comm_resource_->bootstrap()->barrier();
    }
}

void *Executor::Impl::tensor_address(const Tensor &tensor) const {
    size_t buffer_id = tensor.ref()->buffer()->id();
    auto &ext_buf_reg = BufferRegistry::get_instance();
    auto info = ext_buf_reg.get(buffer_id);
    if (info) {
        return info->data;
    }
    return nullptr;
}

void Executor::Impl::tensor_read(const Tensor &tensor, void *data, size_t bytes,
                                 Stream stream, bool is_d2d) const {
    auto buf_id = tensor.ref()->buffer()->id();
    auto info = BufferRegistry::get_instance().get(buf_id);
    if (!info) {
        ERR(InvalidUsageError, "Tensor buffer is not allocated.");
    }
    size_t device_id = info->device_id;
    GLOG(gpuSetDevice(device_id));
    std::shared_ptr<GpuStream> copy_stream;
    gpuStream copy_stream_raw;
    if (stream) {
        copy_stream_raw = reinterpret_cast<gpuStream>(stream);
        if ((stream == stream_raw_) && is_launched_) {
            LOG(WARN,
                "Reading from a tensor in the same stream of the kernel "
                "may cause a deadlock.");
        }
    } else {
        copy_stream = GpuManager::get_instance(device_id)->create_stream();
        copy_stream_raw = copy_stream->get();
    }
    size_t tensor_data_bytes =
        tensor.shape().nelems() * tensor.data_type().bytes();
    if (bytes != tensor_data_bytes) {
        ERR(InvalidUsageError, "Destination bytes (", bytes,
            ") mismatches the tensor data bytes (", tensor_data_bytes, ").");
    }
    auto kind = (is_d2d) ? gpuMemcpyDeviceToDevice : gpuMemcpyDeviceToHost;
    void *src = info->data;
    if (tensor.strides() == tensor.shape()) {
        GLOG(gpuMemcpyAsync(data, src, bytes, kind, copy_stream_raw));
    } else {
        size_t tensor_bytes =
            tensor.strides().nelems() * tensor.data_type().bytes();
        std::vector<int8_t> tensor_host(tensor_bytes);
        GLOG(gpuMemcpyAsync(tensor_host.data(), src, tensor_bytes,
                            gpuMemcpyDeviceToHost, copy_stream_raw));
        GLOG(gpuStreamSynchronize(copy_stream_raw));
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
                            gpuMemcpyHostToDevice, copy_stream_raw));
    }
    GLOG(gpuStreamSynchronize(copy_stream_raw));
}

void Executor::Impl::tensor_write(const Tensor &tensor, const void *data,
                                  size_t bytes, Stream stream,
                                  bool is_d2d) const {
    auto buf_id = tensor.ref()->buffer()->id();
    auto info = BufferRegistry::get_instance().get(buf_id);
    if (!info) {
        ERR(InvalidUsageError, "Tensor buffer is not allocated.");
    }
    size_t device_id = info->device_id;
    GLOG(gpuSetDevice(device_id));
    std::shared_ptr<GpuStream> copy_stream;
    gpuStream copy_stream_raw;
    if (stream) {
        copy_stream_raw = reinterpret_cast<gpuStream>(stream);
        if ((stream == stream_raw_) && is_launched_) {
            LOG(WARN,
                "Writing to a tensor in the same stream of the kernel "
                "may cause a deadlock.");
        }
    } else {
        copy_stream = GpuManager::get_instance(device_id)->create_stream();
        copy_stream_raw = copy_stream->get();
    }
    size_t tensor_data_bytes =
        tensor.shape().nelems() * tensor.data_type().bytes();
    if (bytes != tensor_data_bytes) {
        ERR(InvalidUsageError, "Source bytes (", bytes,
            ") mismatches the tensor data bytes (", tensor_data_bytes, ").");
    }
    size_t tensor_bytes =
        tensor.strides().nelems() * tensor.data_type().bytes();
    auto kind = (is_d2d) ? gpuMemcpyDeviceToDevice : gpuMemcpyHostToDevice;
    void *dst = info->data;
    if (tensor.strides() == tensor.shape()) {
        GLOG(gpuMemcpyAsync(dst, data, tensor_bytes, kind, copy_stream_raw));
    } else {
        std::vector<int8_t> tensor_host(tensor_bytes);
        if (!is_d2d) {
            GLOG(gpuMemcpyAsync(tensor_host.data(), dst, tensor_bytes,
                                gpuMemcpyDeviceToHost, copy_stream_raw));
            GLOG(gpuStreamSynchronize(copy_stream_raw));
            data_to_tensor(tensor_host.data(),
                           static_cast<const int8_t *>(data), tensor.shape(),
                           tensor.strides(), tensor.offsets(),
                           tensor.data_type().bytes());
        } else {
            // TODO: convert data layout on the device directly
            std::vector<int8_t> tmp(bytes);
            GLOG(gpuMemcpyAsync(tmp.data(), data, bytes, gpuMemcpyDeviceToHost,
                                copy_stream_raw));
            GLOG(gpuStreamSynchronize(copy_stream_raw));
            data_to_tensor(tensor_host.data(), tmp.data(), tensor.shape(),
                           tensor.strides(), tensor.offsets(),
                           tensor.data_type().bytes());
        }
        GLOG(gpuMemcpyAsync(dst, tensor_host.data(), tensor_bytes,
                            gpuMemcpyHostToDevice, copy_stream_raw));
    }
    GLOG(gpuStreamSynchronize(copy_stream_raw));
}

Executor::Executor() : impl_(std::make_unique<Executor::Impl>()) {}

Executor::~Executor() = default;

int Executor::device_id() const { return impl_->device_id(); }

Stream Executor::stream() const { return impl_->stream(); }

std::shared_ptr<GpuMemory> Executor::buffer() const { return impl_->buffer(); }

std::string Executor::plan() const { return impl_->plan(); }

std::string Executor::name() const { return impl_->name(); }

void Executor::compile(const std::string &plan, int device_id,
                       const std::string &name) {
    impl_->compile(plan, device_id, name);
}

void Executor::launch(
    const std::unordered_map<Tensor, void *> &placeholder_data, Stream stream,
    bool loop_mode, bool record) {
    impl_->launch(placeholder_data, stream, loop_mode, record);
}

void Executor::run(int iter,
                   const std::unordered_map<Tensor, void *> &placeholder_data) {
    impl_->run(iter, placeholder_data);
}

void Executor::wait(int64_t max_spin_count) { impl_->wait(max_spin_count); }

float Executor::stop(int64_t max_spin_count) {
    return impl_->stop(max_spin_count);
}

void Executor::barrier() { impl_->barrier(); }

void Executor::destroy() { impl_.reset(nullptr); }

bool Executor::destroyed() const { return impl_.get() == nullptr; }

void *Executor::tensor_address(const Tensor &tensor) const {
    return impl_->tensor_address(tensor);
}

void Executor::tensor_read(const Tensor &tensor, void *data, size_t bytes,
                           Stream stream, bool is_d2d) const {
    impl_->tensor_read(tensor, data, bytes, stream, is_d2d);
}

void Executor::tensor_write(const Tensor &tensor, const void *data,
                            size_t bytes, Stream stream, bool is_d2d) const {
    impl_->tensor_write(tensor, data, bytes, stream, is_d2d);
}

DefaultExecutor::DefaultExecutor(
    const Model &model, int device_id, Stream stream,
    const std::vector<Planner::ConfigRule> &config_rules,
    const std::string &name, bool loop_mode, bool record)
    : Executor(), record_(record) {
    device_id = (device_id < 0) ? (model.rank() % get_env().num_ranks_per_host)
                                : device_id;
    Planner planner(model, device_id);
    for (const auto &rule : config_rules) {
        planner.install_config_rule(rule);
    }
    compile(planner.plan(), device_id, name);
    impl_->stream_raw_ = reinterpret_cast<gpuStream>(stream);
    impl_->loop_mode_ = loop_mode;
}

void DefaultExecutor::launch(
    const std::unordered_map<Tensor, void *> &placeholder_data) {
    Executor::launch(placeholder_data,
                     reinterpret_cast<Stream>(impl_->stream_raw_),
                     impl_->loop_mode_, record_);
}

}  // namespace ark
