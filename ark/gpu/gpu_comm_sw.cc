// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_comm_sw.h"

#include <algorithm>
#include <cassert>
#include <list>
#include <mscclpp/core.hpp>
#include <mscclpp/proxy_channel.hpp>
#include <mscclpp/sm_channel.hpp>

#include "env.h"
#include "gpu/gpu_logging.h"
#include "gpu/gpu_manager.h"
#include "include/ark.h"
#include "ipc/ipc_hosts.h"
#include "ipc/ipc_socket.h"

namespace ark {

// For peer access between GPUs on the same machine.
struct GpuCommMemoryInfo {
    size_t bytes;
    uint64_t id_offsets[MAX_NUM_SID];
};

class GpuCommSw::Impl {
   public:
    Impl(const std::string &name, const int gpu_id, const int rank,
         const int world_size, std::shared_ptr<GpuMemory> data_mem);
    ~Impl();

    void configure(
        const std::vector<std::pair<int, size_t>> &export_id_offsets,
        const std::unordered_map<int, std::vector<std::shared_ptr<GpuBuffer>>>
            &import_gid_buffers);

    void launch_request_loop();
    void stop_request_loop();

    std::shared_ptr<GpuMemory> get_data_memory(const int gid);

    const void *get_proxy_channels_ref() const {
        return this->proxy_channels_.data();
    }

    int get_proxy_channels_bytes() const {
        return this->proxy_channels_.size() *
               sizeof(mscclpp::DeviceHandle<mscclpp::SimpleProxyChannel>);
    }

    int get_proxy_channels_num() const { return this->proxy_channels_.size(); }

    int get_sm_channels_num() const { return this->sm_channel_handles_.size(); }

    const void *get_sm_channels_ref() const {
        return this->sm_channel_handles_.data();
    }

    int get_sm_channels_bytes() const {
        return this->sm_channel_handles_.size() *
               sizeof(mscclpp::DeviceHandle<mscclpp::SmChannel>);
    }

   private:
    //
    const std::string name_;
    //
    const int gpu_id_;
    const int rank_;
    const int world_size_;

    std::shared_ptr<GpuManager> manager_;
    std::vector<std::shared_ptr<GpuMemory>> data_memories_;
    //
    std::unique_ptr<IpcSocket> ipc_socket_;

    std::shared_ptr<mscclpp::TcpBootstrap> bootstrap_;
    std::shared_ptr<mscclpp::Communicator> comm_;
    std::shared_ptr<mscclpp::ProxyService> proxy_service_;
    std::vector<mscclpp::DeviceHandle<mscclpp::SimpleProxyChannel>>
        proxy_channels_;
    std::vector<mscclpp::SmChannel> sm_channels_;
    std::vector<mscclpp::DeviceHandle<mscclpp::SmChannel>> sm_channel_handles_;
};

GpuCommSw::Impl::Impl(const std::string &name, const int gpu_id, const int rank,
                      const int world_size, std::shared_ptr<GpuMemory> data_mem)
    : name_{name}, gpu_id_{gpu_id}, rank_{rank}, world_size_{world_size} {
    manager_ = GpuManager::get_instance(rank_);
    // Reserve entries for GPU communication stack information.
    // Power of 2 larger than `gpu_id_` and at least 8.
    int num_entries = 8;
    while (gpu_id_ >= num_entries) {
        num_entries *= 2;
    }
    data_memories_.resize(num_entries, nullptr);

    // Create the local stack info.
    data_memories_[gpu_id_] = data_mem;

    int port = get_env().ipc_listen_port_base + gpu_id_;
    int host_id = rank_ / get_env().num_ranks_per_host;
    ipc_socket_ = std::make_unique<IpcSocket>(get_host(host_id), port);

    std::stringstream ip_port;
    ip_port << get_host(0) << ":" << get_env().mscclpp_port;

    this->bootstrap_ =
        std::make_shared<mscclpp::TcpBootstrap>(rank, world_size);
    this->bootstrap_->initialize(ip_port.str());
    this->comm_ = std::make_shared<mscclpp::Communicator>(this->bootstrap_);
    this->proxy_service_ = std::make_shared<mscclpp::ProxyService>();
}

GpuCommSw::Impl::~Impl() { this->stop_request_loop(); }

//
void GpuCommSw::Impl::configure(
    const std::vector<std::pair<int, size_t>> &export_id_offsets,
    const std::unordered_map<int, std::vector<std::shared_ptr<GpuBuffer>>>
        &import_gid_buffers) {
    //
    std::shared_ptr<GpuMemory> data_mem = this->get_data_memory(gpu_id_);
    if (data_mem->bytes() == 0) {
        LOG(INFO, "RANK ", rank_, " has no data to send");
        return;
    }
    // need to setup registered memory for the communicator
    int num_ranks_per_node = get_env().num_ranks_per_host;
    const int thisNode = rank_ / num_ranks_per_node;
    auto rankToNode = [&](int rank) { return rank / num_ranks_per_node; };

    mscclpp::Transport IBs[] = {
        mscclpp::Transport::IB0, mscclpp::Transport::IB1,
        mscclpp::Transport::IB2, mscclpp::Transport::IB3,
        mscclpp::Transport::IB4, mscclpp::Transport::IB5,
        mscclpp::Transport::IB6, mscclpp::Transport::IB7};

    const mscclpp::Transport ibTransport =
        get_env().disable_ib ? mscclpp::Transport::Unknown : IBs[gpu_id_];
    std::vector<
        mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>>
        connectionFutures;
    const mscclpp::TransportFlags all_transports =
        mscclpp::Transport::CudaIpc | ibTransport;
    mscclpp::RegisteredMemory local_reg_memory = this->comm_->registerMemory(
        (void *)(data_mem->ref()), data_mem->bytes(), all_transports);
    std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>>
        remote_reg_memories;
    for (int r = 0; r < this->world_size_; ++r) {
        if (r == rank_) {
            continue;
        }
        mscclpp::Transport transport;
        if (rankToNode(r) == thisNode) {
            transport = mscclpp::Transport::CudaIpc;
        } else {
            transport = ibTransport;
        }
        // order is matter, we need to connect first and
        // then send memory
        connectionFutures.push_back(
            this->comm_->connectOnSetup(r, 0, transport));
        this->comm_->sendMemoryOnSetup(local_reg_memory, r, 0);
        auto remote_memory = this->comm_->recvMemoryOnSetup(r, 0);
        remote_reg_memories.push_back(remote_memory);
    }
    this->comm_->setup();
    std::vector<std::shared_ptr<mscclpp::Connection>> connections;
    std::transform(connectionFutures.begin(), connectionFutures.end(),
                   std::back_inserter(connections),
                   [](const mscclpp::NonblockingFuture<
                       std::shared_ptr<mscclpp::Connection>> &future) {
                       return future.get();
                   });
    for (size_t i = 0; i < connections.size(); ++i) {
        LOG(DEBUG, "Rank ", rank_, " connected to rank ", i);
        this->proxy_channels_.push_back(
            mscclpp::deviceHandle(mscclpp::SimpleProxyChannel(
                this->proxy_service_->proxyChannel(
                    this->proxy_service_->buildAndAddSemaphore(*(this->comm_),
                                                               connections[i])),
                this->proxy_service_->addMemory(remote_reg_memories[i].get()),
                this->proxy_service_->addMemory(local_reg_memory))));
    }
    this->comm_->setup();

    // setup for sm channel
    std::unordered_map<size_t,
                       std::shared_ptr<mscclpp::SmDevice2DeviceSemaphore>>
        sm_semaphores;
    for (size_t cid = 0; cid < connections.size(); ++cid) {
        if (connections[cid]->transport() == mscclpp::Transport::CudaIpc) {
            sm_semaphores.emplace(
                cid, std::make_shared<mscclpp::SmDevice2DeviceSemaphore>(
                         *this->comm_, connections[cid]));
        }
    }
    this->comm_->setup();

    for (size_t cid = 0; cid < connections.size(); ++cid) {
        if (connections[cid]->transport() == mscclpp::Transport::CudaIpc) {
            this->sm_channels_.emplace_back(sm_semaphores[cid],
                                            remote_reg_memories[cid].get(),
                                            local_reg_memory.data(), nullptr);
        }
    }
    auto getChannelDeviceHandle =
        [](const std::vector<mscclpp::SmChannel> &in,
           std::vector<mscclpp::DeviceHandle<mscclpp::SmChannel>> &out) {
            return std::transform(in.begin(), in.end(), out.begin(),
                                  [](const mscclpp::SmChannel &smChannel) {
                                      return mscclpp::deviceHandle(smChannel);
                                  });
        };
    this->sm_channel_handles_.resize(this->sm_channels_.size());
    getChannelDeviceHandle(this->sm_channels_, this->sm_channel_handles_);

    GpuCommMemoryInfo memory_info;
    memory_info.bytes = data_mem->bytes();

    for (auto &pair : export_id_offsets) {
        int export_id = pair.first;
        size_t offset = pair.second;
        memory_info.id_offsets[export_id] = offset;
    }

    // Share comm_mem_info with other GPUs on the same machine.
    auto state = ipc_socket_->add_item("comm_mem_info", &memory_info,
                                       sizeof(memory_info));
    if (state != IpcSocket::State::SUCCESS) {
        ERR(ExecutorError, "Failed to post comm_mem_info");
    }
    int num_ranks_per_host = get_env().num_ranks_per_host;
    int my_host_id = rank_ / num_ranks_per_host;
    int port_base = get_env().ipc_listen_port_base;
    for (auto &pair : import_gid_buffers) {
        // Get comm_mem_info from other GPUs on the same
        // machine. Only from GPUs that this GPU imports.
        int gpu_id = pair.first;
        int port = port_base + gpu_id;

        assert(gpu_id != gpu_id_);
        GpuCommMemoryInfo remote_memory_info;
        state = ipc_socket_->query_item(get_host(my_host_id), port,
                                        "comm_mem_info", &remote_memory_info,
                                        sizeof(remote_memory_info), true);
        if (state != IpcSocket::State::SUCCESS) {
            ERR(ExecutorError, "Failed to query comm_mem_info from GPU ",
                gpu_id);
        }

        std::shared_ptr<GpuMemory> mem = this->get_data_memory(gpu_id);
        int channel_id = gpu_id < gpu_id_ ? gpu_id : gpu_id - 1;
        mem->resize(remote_reg_memories[channel_id].get());
        for (std::shared_ptr<GpuBuffer> buffer : pair.second) {
            int export_id = buffer->get_id();
            size_t offset = remote_memory_info.id_offsets[export_id];
            buffer->set_offset(offset);
        }
    }
    LOG(DEBUG, "RANK ", rank_, " config done");
}

//
void GpuCommSw::Impl::launch_request_loop() {
    this->proxy_service_->startProxy();
}

//
void GpuCommSw::Impl::stop_request_loop() { this->proxy_service_->stopProxy(); }

std::shared_ptr<GpuMemory> GpuCommSw::Impl::get_data_memory(const int gid) {
    int sz = (int)data_memories_.size();
    if (sz <= gid) {
        while (sz <= gid) {
            sz *= 2;
        }
        data_memories_.resize(sz, nullptr);
    }
    std::shared_ptr<GpuMemory> dm = data_memories_[gid];
    if (dm == nullptr) {
        dm = manager_->malloc(0, 1);
        data_memories_[gid] = dm;
    }
    return dm;
}

GpuCommSw::GpuCommSw(const std::string &name, const int gpu_id, const int rank,
                     const int world_size, std::shared_ptr<GpuMemory> data_mem)
    : impl{std::make_unique<GpuCommSw::Impl>(name, gpu_id, rank, world_size,
                                             data_mem)} {}

GpuCommSw::~GpuCommSw() {}

void GpuCommSw::configure(
    const std::vector<std::pair<int, size_t>> &export_sid_offs,
    const std::unordered_map<int, std::vector<std::shared_ptr<GpuBuffer>>>
        &import_gid_bufs) {
    this->impl->configure(export_sid_offs, import_gid_bufs);
}

void GpuCommSw::launch_request_loop() { this->impl->launch_request_loop(); }

void GpuCommSw::stop_request_loop() { this->impl->stop_request_loop(); }

std::shared_ptr<GpuMemory> GpuCommSw::get_data_memory(const int gid) {
    return this->impl->get_data_memory(gid);
}

const void *GpuCommSw::get_proxy_channels_ref() const {
    return this->impl->get_proxy_channels_ref();
}

int GpuCommSw::get_proxy_channels_bytes() const {
    return this->impl->get_proxy_channels_bytes();
}

int GpuCommSw::get_proxy_channels_num() const {
    return this->impl->get_proxy_channels_num();
}

const void *GpuCommSw::get_sm_channels_ref() const {
    return this->impl->get_sm_channels_ref();
}

int GpuCommSw::get_sm_channels_num() const {
    return this->impl->get_sm_channels_num();
}

int GpuCommSw::get_sm_channels_bytes() const {
    return this->impl->get_sm_channels_bytes();
}
}  // namespace ark
