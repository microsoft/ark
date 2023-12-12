// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_comm_sw.h"

#include <cassert>
#include <list>
#include <mscclpp/core.hpp>
#include <mscclpp/proxy_channel.hpp>
#include <mscclpp/sm_channel.hpp>

#include "env.h"
#include "gpu/gpu_common.h"
#include "gpu/gpu_logging.h"
#include "include/ark.h"
#include "ipc/ipc_hosts.h"
#include "ipc/ipc_socket.h"

namespace ark {

// For peer access between GPUs on the same machine.
struct GpuCommMemInfo {
    GpuMem::Info data_info;
    uint64_t sid_offs[MAX_NUM_SID];
};

class GpuCommSw::Impl {
   public:
    Impl(const std::string &name, const int gpu_id_, const int rank_,
         const int world_size_, GpuMem *data_mem);
    ~Impl();

    void configure(const std::vector<std::pair<int, size_t>> &export_sid_offs,
                   const std::map<int, std::vector<GpuBuf *>> &import_gid_bufs);

    void launch_request_loop();
    void stop_request_loop();

    GpuMem *get_data_mem(const int gid);

    const void *get_proxy_channels_ref() const {
        return this->proxy_channels.data();
    }

    int get_proxy_channels_bytes() const {
        return this->proxy_channels.size() *
               sizeof(mscclpp::DeviceHandle<mscclpp::SimpleProxyChannel>);
    }

    int get_proxy_channels_num() const { return this->proxy_channels.size(); }

    int get_sm_channels_num() const { return this->sm_channel_handles.size(); }

    const void *get_sm_channels_ref() const {
        return this->sm_channel_handles.data();
    }

    int get_sm_channels_bytes() const {
        return this->sm_channel_handles.size() *
               sizeof(mscclpp::DeviceHandle<mscclpp::SmChannel>);
    }

   private:
    //
    const std::string name_;
    //
    const int gpu_id_;
    const int rank_;
    const int world_size_;
    //
    std::list<std::unique_ptr<GpuMem>> remote_data_mems_storage_;
    //
    std::vector<GpuMem *> data_mems_;
    //
    std::unique_ptr<IpcSocket> ipc_socket_;

    std::shared_ptr<mscclpp::TcpBootstrap> bootstrap;
    std::shared_ptr<mscclpp::Communicator> comm;
    std::shared_ptr<mscclpp::ProxyService> proxy_service;
    std::vector<mscclpp::DeviceHandle<mscclpp::SimpleProxyChannel>>
        proxy_channels;
    std::vector<mscclpp::SmChannel> sm_channels;
    std::vector<mscclpp::DeviceHandle<mscclpp::SmChannel>> sm_channel_handles;
};

//
GpuCommSw::Impl::Impl(const std::string &name, const int gpu_id, const int rank,
                      const int world_size, GpuMem *data_mem)
    : name_{name}, gpu_id_{gpu_id}, rank_{rank}, world_size_{world_size} {
    // Reserve entries for GPU communication stack information.
    // Power of 2 larger than `gpu_id_` and at least 8.
    int num_entries = 8;
    while (gpu_id_ >= num_entries) {
        num_entries *= 2;
    }
    data_mems_.resize(num_entries, nullptr);

    // Create the local stack info.
    data_mems_[gpu_id_] = data_mem;

    int port = get_env().ipc_listen_port_base + gpu_id_;
    int host_id = rank_ / get_env().num_ranks_per_host;
    ipc_socket_ = std::make_unique<IpcSocket>(get_host(host_id), port);

    std::stringstream ip_port;
    ip_port << get_host(0) << ":" << get_env().mscclpp_port;

    this->bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, world_size);
    this->bootstrap->initialize(ip_port.str());
    this->comm = std::make_shared<mscclpp::Communicator>(this->bootstrap);
    this->proxy_service = std::make_shared<mscclpp::ProxyService>();
}

GpuCommSw::Impl::~Impl() { this->stop_request_loop(); }

//
void GpuCommSw::Impl::configure(
    const std::vector<std::pair<int, size_t>> &export_sid_offs,
    const std::map<int, std::vector<GpuBuf *>> &import_gid_bufs) {
    //
    GpuMem *data_mem = this->get_data_mem(gpu_id_);
    GpuCommMemInfo comm_mem_info;

    comm_mem_info.data_info.ipc_hdl = data_mem->get_info().ipc_hdl;
    comm_mem_info.data_info.bytes = data_mem->get_info().bytes;

    for (auto &p : export_sid_offs) {
        int sid = p.first;
        size_t off = p.second;
        comm_mem_info.sid_offs[sid] = off;
    }

    // Share comm_mem_info with other GPUs on the same machine.
    auto state = ipc_socket_->add_item("comm_mem_info", &comm_mem_info,
                                       sizeof(comm_mem_info));
    if (state != IpcSocket::State::SUCCESS) {
        ERR(ExecutorError, "Failed to post comm_mem_info");
    }
    int num_ranks_per_host = get_env().num_ranks_per_host;
    int my_host_id = rank_ / num_ranks_per_host;
    int port_base = get_env().ipc_listen_port_base;
    for (auto &p : import_gid_bufs) {
        // Get comm_mem_info from other GPUs on the same machine.
        // Only from GPUs that this GPU imports.
        int gpu_id = p.first;
        int port = port_base + gpu_id;

        assert(gpu_id != gpu_id_);

        GpuCommMemInfo remote_comm_mem_info;
        state = ipc_socket_->query_item(get_host(my_host_id), port,
                                        "comm_mem_info", &remote_comm_mem_info,
                                        sizeof(remote_comm_mem_info), true);
        if (state != IpcSocket::State::SUCCESS) {
            ERR(ExecutorError, "Failed to query comm_mem_info from GPU ",
                gpu_id);
        }

        // Initialize the remote GPU memory space.
        GpuMem *mem = this->get_data_mem(gpu_id);
        mem->init(remote_comm_mem_info.data_info);
        for (GpuBuf *buf : p.second) {
            int sid = buf->get_id();
            size_t off = remote_comm_mem_info.sid_offs[sid];
            buf->set_offset(off);
        }
    }

    if (data_mem->get_bytes() > 0) {
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
        mscclpp::RegisteredMemory local_reg_memory = this->comm->registerMemory(
            (void *)(data_mem->ref()), data_mem->get_bytes(), all_transports);
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
            // order is matter, we need to connect first and then send memory
            connectionFutures.push_back(
                this->comm->connectOnSetup(r, 0, transport));
            this->comm->sendMemoryOnSetup(local_reg_memory, r, 0);
            auto remote_memory = this->comm->recvMemoryOnSetup(r, 0);
            remote_reg_memories.push_back(remote_memory);
        }
        this->comm->setup();
        std::vector<std::shared_ptr<mscclpp::Connection>> connections;
        std::transform(connectionFutures.begin(), connectionFutures.end(),
                       std::back_inserter(connections),
                       [](const mscclpp::NonblockingFuture<
                           std::shared_ptr<mscclpp::Connection>> &future) {
                           return future.get();
                       });
        for (size_t i = 0; i < connections.size(); ++i) {
            LOG(INFO, "Rank ", rank_, " connected to rank ", i);
            this->proxy_channels.push_back(
                mscclpp::deviceHandle(mscclpp::SimpleProxyChannel(
                    this->proxy_service->proxyChannel(
                        this->proxy_service->buildAndAddSemaphore(
                            *(this->comm), connections[i])),
                    this->proxy_service->addMemory(
                        remote_reg_memories[i].get()),
                    this->proxy_service->addMemory(local_reg_memory))));
        }
        this->comm->setup();

        // setup for sm channel
        std::unordered_map<size_t,
                           std::shared_ptr<mscclpp::SmDevice2DeviceSemaphore>>
            sm_semaphores;
        for (size_t cid = 0; cid < connections.size(); ++cid) {
            if (connections[cid]->transport() == mscclpp::Transport::CudaIpc) {
                sm_semaphores.emplace(
                    cid, std::make_shared<mscclpp::SmDevice2DeviceSemaphore>(
                             *this->comm, connections[cid]));
            }
        }
        this->comm->setup();

        for (size_t cid = 0; cid < connections.size(); ++cid) {
            if (connections[cid]->transport() == mscclpp::Transport::CudaIpc) {
                this->sm_channels.emplace_back(
                    sm_semaphores[cid], remote_reg_memories[cid].get(),
                    local_reg_memory.data(), nullptr);
            }
        }
        auto getChannelDeviceHandle =
            [](const std::vector<mscclpp::SmChannel> &in,
               std::vector<mscclpp::DeviceHandle<mscclpp::SmChannel>> &out) {
                return std::transform(
                    in.begin(), in.end(), out.begin(),
                    [](const mscclpp::SmChannel &smChannel) {
                        return mscclpp::deviceHandle(smChannel);
                    });
            };
        this->sm_channel_handles.resize(this->sm_channels.size());
        getChannelDeviceHandle(this->sm_channels, this->sm_channel_handles);
    }

    LOG(DEBUG, "RANK ", rank_, " config done");
}

//
void GpuCommSw::Impl::launch_request_loop() {
    this->proxy_service->startProxy();
}

//
void GpuCommSw::Impl::stop_request_loop() { this->proxy_service->stopProxy(); }

//
GpuMem *GpuCommSw::Impl::get_data_mem(const int gid) {
    int sz = (int)data_mems_.size();
    if (sz <= gid) {
        while (sz <= gid) {
            sz *= 2;
        }
        data_mems_.resize(sz, nullptr);
    }
    GpuMem *dm = data_mems_[gid];
    if (dm == nullptr) {
        remote_data_mems_storage_.emplace_back(std::make_unique<GpuMem>());
        dm = remote_data_mems_storage_.back().get();
        data_mems_[gid] = dm;
    }
    return dm;
}

GpuCommSw::GpuCommSw(const std::string &name, const int gpu_id_,
                     const int rank_, const int world_size_, GpuMem *data_mem)
    : impl{std::make_unique<GpuCommSw::Impl>(name, gpu_id_, rank_, world_size_,
                                             data_mem)} {}

GpuCommSw::~GpuCommSw() {}

void GpuCommSw::configure(
    const std::vector<std::pair<int, size_t>> &export_sid_offs,
    const std::map<int, std::vector<GpuBuf *>> &import_gid_bufs) {
    this->impl->configure(export_sid_offs, import_gid_bufs);
}

void GpuCommSw::launch_request_loop() { this->impl->launch_request_loop(); }

void GpuCommSw::stop_request_loop() { this->impl->stop_request_loop(); }

GpuMem *GpuCommSw::get_data_mem(const int gid) {
    return this->impl->get_data_mem(gid);
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
