// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cassert>
#include <cerrno>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <vector>

#include "cpu_timer.h"
#include "env.h"
#include "gpu/gpu_comm_sw.h"
#include "gpu/gpu_common.h"
#include "gpu/gpu_logging.h"
#include "gpu/gpu_mgr.h"
#include "ipc/ipc_coll.h"
#include "ipc/ipc_hosts.h"
#include "ipc/ipc_socket.h"
#include "net/net_ib.h"

#ifdef ARK_USE_MSCCLPP
#include <mscclpp/core.hpp>
#include <mscclpp/proxy_channel.hpp>
#endif // ARK_USE_MSCCLPP


using namespace std;

namespace ark {

//
struct GpuCommInfo
{
    NetIbMr::Info sid_mris[MAX_NUM_SID];
    uint64_t sid_offs[MAX_NUM_SID];
    NetIbQp::Info qpi;
    uint64_t bytes;
};

//
struct GpuSendRecvInfo
{
    int sid;
    int remote_rank;
    std::size_t bytes;
    bool is_recv;
};

class GpuCommSw::Impl
{
  public:
    Impl(const std::string &name, const int gpu_id_, const int rank_,
         const int world_size_, GpuMem *data_mem, GpuMem *sc_rc_mem);
    ~Impl();

    void reg_sendrecv(int sid, int remote_rank, std::size_t bytes,
                      bool is_recv);
    void configure(std::vector<std::pair<int, size_t>> &export_sid_offs,
                   std::map<int, std::vector<GpuBuf *>> &import_gid_bufs);
    void import_buf(const int gid, GpuBuf *buf);

    void request_loop();
    void launch_request_loop();
    void stop_request_loop();

    void set_request(const Request &db);

    GpuMem *get_data_mem(const int gid);
    GpuMem *get_sc_rc_mem(const int gid);
    IpcMem *get_info(const int gid);
    GpuPtr get_request_ref() const;
    bool is_using_ib() const
    {
        return this->net_ib_mgr != nullptr;
    }

    const void *get_proxy_channels_ref() const
    {
#ifdef ARK_USE_MSCCLPP
        return this->proxy_channels.data();
#else
        return nullptr;
#endif // ARK_USE_MSCCLPP
    }

    int get_proxy_channels_bytes() const
    {
#ifdef ARK_USE_MSCCLPP
        return this->proxy_channels.size() *
               sizeof(mscclpp::DeviceHandle<mscclpp::SimpleProxyChannel>);
#else
        return 0;
#endif // ARK_USE_MSCCLPP
    }

    int get_proxy_channels_num() const
    {
#ifdef ARK_USE_MSCCLPP
        return this->proxy_channels.size();
#else
        return 0;
#endif // ARK_USE_MSCCLPP
    }

  private:
    //
    const std::string name;
    //
    const int gpu_id;
    const int rank;
    const int world_size;
    //
    std::list<std::unique_ptr<GpuMem>> remote_data_mems_storage;
    std::list<std::unique_ptr<GpuMem>> remote_sc_rc_mems_storage;
    std::list<std::unique_ptr<IpcMem>> infos_storage;
    //
    std::vector<GpuMem *> data_mems;
    //
    std::vector<GpuMem *> sc_rc_mems;
    //
    std::vector<IpcMem *> infos;
    //
    std::vector<std::vector<GpuPtr>> addr_table;
    //
    Request *request = nullptr;
    std::thread *request_loop_thread = nullptr;
    volatile bool run_request_loop_thread = false;
    //
    IpcSocket *ipc_socket = nullptr;
    //
    NetIbMgr *net_ib_mgr = nullptr;
    std::vector<NetIbMr *> sid_mrs;
    std::map<int, NetIbQp *> qps;
    std::vector<GpuSendRecvInfo> send_recv_infos;
    std::map<int, std::vector<NetIbMr::Info>> mris;

#ifdef ARK_USE_MSCCLPP
    std::shared_ptr<mscclpp::TcpBootstrap> bootstrap;
    std::shared_ptr<mscclpp::Communicator> comm;
    std::shared_ptr<mscclpp::ProxyService> proxy_service;
    std::vector<mscclpp::DeviceHandle<mscclpp::SimpleProxyChannel>>
        proxy_channels;
#endif // ARK_USE_MSCCLPP
};

//
GpuCommSw::Impl::Impl(const string &name_, const int gpu_id_, const int rank_,
                      const int world_size_, GpuMem *data_mem,
                      GpuMem *sc_rc_mem)
    : name{name_}, gpu_id{gpu_id_}, rank{rank_},
      world_size{world_size_}, request{new Request}
{
    // Register `this->request` as a mapped & pinned address.
    CULOG(cuMemHostRegister((void *)this->request, sizeof(request),
                            CU_MEMHOSTREGISTER_DEVICEMAP));

    // Reserve entries for GPU communication stack information.
    // Power of 2 larger than `gpu_id_` and at least 8.
    int num_entries = 8;
    while (gpu_id_ >= num_entries) {
        num_entries *= 2;
    }
    this->data_mems.resize(num_entries, nullptr);
    this->sc_rc_mems.resize(num_entries, nullptr);
    this->infos.resize(num_entries, nullptr);

    // Create the local stack info.
    this->data_mems[gpu_id_] = data_mem;
    this->sc_rc_mems[gpu_id_] = sc_rc_mem;

    IpcMem *ie =
        new IpcMem{ARK_GPU_INFO_NAME + name_ + to_string(gpu_id_), true};
    this->infos_storage.emplace_back(ie);
    this->infos[gpu_id_] = ie;

    int port_base = get_env().ipc_listen_port_base;
    int host_id = rank_ / get_env().num_ranks_per_host;
    this->ipc_socket = new IpcSocket{get_host(host_id), port_base + gpu_id_};

    if (!get_env().disable_ib) {
        int num_ib_dev = get_net_ib_device_num();
        // TODO: be aware of GPU-NIC interconnect topology.
        int ib_dev_id = gpu_id_ % num_ib_dev;
        this->net_ib_mgr = get_net_ib_mgr(ib_dev_id);
        this->sid_mrs.resize(MAX_NUM_SID, nullptr);
    }

#ifdef ARK_USE_MSCCLPP
    if (get_env().use_mscclpp) {
        std::stringstream ip_port;
        ip_port << get_host(0) << ":" << get_env().mscclpp_port;

        this->bootstrap =
            std::make_shared<mscclpp::TcpBootstrap>(rank, world_size);
        this->bootstrap->initialize(ip_port.str());
        this->comm = std::make_shared<mscclpp::Communicator>(this->bootstrap);
        this->proxy_service = std::make_shared<mscclpp::ProxyService>();
    }
#endif // ARK_USE_MSCCLPP
}

GpuCommSw::Impl::~Impl()
{
    this->stop_request_loop();
    if (this->request != nullptr) {
        cuMemHostUnregister((void *)this->request);
        delete this->request;
    }
    delete this->ipc_socket;
}

void GpuCommSw::Impl::reg_sendrecv(int sid, int remote_rank, size_t bytes,
                                   bool is_recv)
{
    this->send_recv_infos.emplace_back(
        GpuSendRecvInfo{sid, remote_rank, bytes, is_recv});
}

//
void GpuCommSw::Impl::configure(vector<pair<int, size_t>> &export_sid_offs,
                                map<int, vector<GpuBuf *>> &import_gid_bufs)
{
    map<int, size_t> sid_max_bytes;
    if (this->is_using_ib()) {
        // Create QPs
        for (auto &srinfo : this->send_recv_infos) {
            auto it = this->qps.find(srinfo.remote_rank);
            if (it == this->qps.end()) {
                NetIbQp *qp = this->net_ib_mgr->create_qp();
                if (qp == nullptr) {
                    LOG(ERROR, "create_qp failed");
                }
                this->qps[srinfo.remote_rank] = qp;
            }
            sid_max_bytes[srinfo.sid] =
                max(sid_max_bytes[srinfo.sid], srinfo.bytes);
        }
    }

    //
    GpuMem *data_mem = this->get_data_mem(this->gpu_id);
    for (auto &p : export_sid_offs) {
        int sid = p.first;
        size_t off = p.second;
        this->addr_table[this->gpu_id][sid] = data_mem->ref(off);
    }

    if (this->is_using_ib()) {
        // Create MRs
        for (auto &p : export_sid_offs) {
            int sid = p.first;
            if (this->sid_mrs[sid] == nullptr) {
                NetIbMr *mr = this->net_ib_mgr->reg_mr(
                    (void *)this->addr_table[this->gpu_id][sid],
                    sid_max_bytes[sid]);
                this->sid_mrs[sid] = mr;
            }
        }
    }

    //
    GpuCommInfo gi;
    for (auto &p : export_sid_offs) {
        int sid = p.first;
        size_t off = p.second;
        gi.sid_offs[sid] = off;
    }
    gi.bytes = data_mem->get_bytes();
    if (this->is_using_ib()) {
        for (size_t sid = 0; sid < this->sid_mrs.size(); ++sid) {
            if (this->sid_mrs[sid] != nullptr) {
                gi.sid_mris[sid] = this->sid_mrs[sid]->get_info();
            }
        }
    }
    IpcSocket::State s;
    for (auto &p : this->qps) {
        int remote_rank = p.first;
        NetIbQp *qp = p.second;
        if (qp == nullptr) {
            LOG(ERROR, "unexpected error");
        }
        gi.qpi = qp->get_info();
        s = this->ipc_socket->add_item(
            "gpu_comm_info_" + std::to_string(remote_rank), &gi, sizeof(gi));
        if (s != IpcSocket::State::SUCCESS) {
            LOG(ERROR, "Failed to add gpu_comm_info to ipc_socket");
        }
    }

    //
    IpcMem *info = this->get_info(this->gpu_id);
    {
        //
        IpcLockGuard lg{info->get_lock()};
        GpuCommInfo *gi = (GpuCommInfo *)info->alloc(sizeof(GpuCommInfo));
        for (auto &p : export_sid_offs) {
            int sid = p.first;
            size_t off = p.second;
            gi->sid_offs[sid] = off;
        }
        gi->bytes = data_mem->get_bytes();
    }

    //
    for (auto &p : import_gid_bufs) {
        int gid = p.first;
        IpcMem *ie = this->get_info(gid);
        if (ie->get_bytes() == 0) {
            ie->alloc(sizeof(GpuCommInfo));
        }

        GpuMem *mem = this->get_data_mem(gid);
        GpuCommInfo *info = (GpuCommInfo *)ie->get_addr();
        assert(info != nullptr);

        // Create a GPU memory mapping if it has not done yet.
        if (mem->get_bytes() == 0) {
            while (info->bytes == 0) {
                sched_yield();
            }
            mem->alloc(info->bytes);
        }

        //
        IpcLockGuard lg{ie->get_lock()};
        for (GpuBuf *buf : p.second) {
            int sid = buf->get_id();
            size_t off = info->sid_offs[sid];
            this->addr_table[gid][sid] = mem->ref(off);
            buf->set_offset(off);
        }
    }

    //
    if (this->is_using_ib()) {
        int port_base = get_env().ipc_listen_port_base;
        int ret;
        for (auto &p : this->qps) {
            int remote_rank = p.first;
            // TODO: generalize converting rank to GPU ID.
            int nrph = get_env().num_ranks_per_host;
            int remote_gpu_id = remote_rank % nrph;
            int remote_host_id = remote_rank / nrph;
            NetIbQp *qp = p.second;
            GpuCommInfo gi_remote;

            LOG(DEBUG, "querying gpu_comm_info_", this->rank, " from rank ",
                remote_rank, " (", get_host(remote_host_id), ":",
                port_base + remote_gpu_id, ")");
            s = this->ipc_socket->query_item(
                get_host(remote_host_id), port_base + remote_gpu_id,
                "gpu_comm_info_" + std::to_string(this->rank), &gi_remote,
                sizeof(gi_remote), true);
            if (s != IpcSocket::State::SUCCESS) {
                LOG(ERROR, "Failed to query gpu_comm_info from ipc_socket");
            }

            ret = qp->rtr(&gi_remote.qpi);
            if (ret != 0) {
                LOG(ERROR, "NetIbQp::rtr failed");
            }
            LOG(DEBUG, "RANK ", this->rank, " QP ", qp->get_info().qpn,
                " <--> RANK ", remote_rank, " QP ", gi_remote.qpi.qpn);
            ret = qp->rts();
            if (ret != 0) {
                LOG(ERROR, "NetIbQp::rts failed");
            }

            auto &mri_vec = this->mris[remote_rank];
            mri_vec.resize(MAX_NUM_SID);
            for (int sid = 0; sid < (int)mri_vec.size(); ++sid) {
                mri_vec[sid] = gi_remote.sid_mris[sid];
            }
        }

        // Sync with remote QPs
        int dummy_data = 42;
        s = this->ipc_socket->add_item("comm_config_done", &dummy_data,
                                       sizeof(dummy_data));
        if (s != IpcSocket::State::SUCCESS) {
            LOG(ERROR, "Failed to add comm_config_done to ipc_socket");
        }
        for (auto &p : this->qps) {
            int remote_rank = p.first;
            // TODO: generalize converting rank to GPU ID.
            int nrph = get_env().num_ranks_per_host;
            int remote_gpu_id = remote_rank % nrph;
            int remote_host_id = remote_rank / nrph;
            int remote_data;
            s = this->ipc_socket->query_item(
                get_host(remote_host_id), port_base + remote_gpu_id,
                "comm_config_done", &remote_data, sizeof(remote_data), true);
            if (s != IpcSocket::State::SUCCESS) {
                LOG(ERROR, "Failed to query gpu_comm_info from ipc_socket");
            }
            if (remote_data != dummy_data) {
                LOG(ERROR, "Failed to sync comm_config_done");
            }
        }
    }

#ifdef ARK_USE_MSCCLPP
    if (get_env().use_mscclpp) {
        // need to setup registered memory for the communicator
        int num_ranks_per_node = get_env().num_ranks_per_host;
        const int thisNode = rank / num_ranks_per_node;
        auto rankToNode = [&](int rank) { return rank / num_ranks_per_node; };

        mscclpp::Transport IBs[] = {
            mscclpp::Transport::IB0, mscclpp::Transport::IB1,
            mscclpp::Transport::IB2, mscclpp::Transport::IB3,
            mscclpp::Transport::IB4, mscclpp::Transport::IB5,
            mscclpp::Transport::IB6, mscclpp::Transport::IB7};

        const mscclpp::Transport ibTransport = IBs[gpu_id];
        std::vector<std::shared_ptr<mscclpp::Connection>> connections;
        const mscclpp::TransportFlags all_transports =
            mscclpp::Transport::CudaIpc | ibTransport;
        mscclpp::RegisteredMemory local_reg_memory = this->comm->registerMemory(
            (void *)(data_mem->ref()), data_mem->get_bytes(), all_transports);
        std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>>
            remote_reg_memories;
        for (int r = 0; r < this->world_size; ++r) {
            if (r == rank) {
                continue;
            }
            mscclpp::Transport transport;
            if (rankToNode(r) == thisNode) {
                transport = mscclpp::Transport::CudaIpc;
            } else {
                transport = ibTransport;
            }
            this->comm->sendMemoryOnSetup(local_reg_memory, r, 0);
            connections.push_back(this->comm->connectOnSetup(r, 0, transport));
            auto remote_memory = this->comm->recvMemoryOnSetup(r, 0);
            remote_reg_memories.push_back(remote_memory);
        }
        this->comm->setup();
        for (size_t i = 0; i < connections.size(); ++i) {
            LOG(INFO, "Rank ", rank, " connected to rank ", i);
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
    }
#endif // ARK_USE_MSCCLPP

    LOG(DEBUG, "RANK ", this->rank, " config done");
}

//
void GpuCommSw::Impl::import_buf(const int gid, GpuBuf *buf)
{
    IpcMem *ie = this->get_info(gid);
    if (ie->get_bytes() == 0) {
        ie->alloc(sizeof(GpuCommInfo));
    }

    GpuMem *mem = this->get_data_mem(gid);
    GpuCommInfo *info = (GpuCommInfo *)ie->get_addr();
    assert(info != nullptr);

    // Create a GPU memory mapping if it has not done yet.
    if (mem->get_bytes() == 0) {
        if (info->bytes == 0) {
            LOG(ERROR, "unexpected error");
        }
        mem->alloc(info->bytes);
    }
    //
    {
        IpcLockGuard lg{ie->get_lock()};
        int sid = buf->get_id();
        size_t off = info->sid_offs[sid];
        this->addr_table[gid][sid] = mem->ref(off);
        buf->set_offset(off);
    }
}

//
void GpuCommSw::Impl::launch_request_loop()
{
#ifdef ARK_USE_MSCCLPP
    if (get_env().use_mscclpp) {
        this->proxy_service->startProxy();
        return;
    }
#endif // ARK_USE_MSCCLPP
    if (this->request_loop_thread == nullptr) {
        this->run_request_loop_thread = true;
        this->request_loop_thread = new thread([&, gid = this->gpu_id] {
            //
            GpuState ret = get_gpu_mgr(gid)->set_current();
            if (ret == CUDA_SUCCESS) {
                //
                this->request_loop();
            } else if (ret != CUDA_ERROR_DEINITIALIZED) {
                CULOG(ret);
            }
        });
        assert(this->request_loop_thread != nullptr);
    } else {
        assert(this->run_request_loop_thread);
    }
}

//
void GpuCommSw::Impl::request_loop()
{
    const size_t sc_offset = 0;
    const size_t rc_offset = MAX_NUM_SID * sizeof(int);
    const int sid_shift = 8;

    // Get the local SC/RC host addresses.
    volatile int *sc_href =
        (volatile int *)this->get_sc_rc_mem(this->gpu_id)->href(sc_offset);
    assert(sc_href != nullptr);
    volatile int *rc_href =
        (volatile int *)this->get_sc_rc_mem(this->gpu_id)->href(rc_offset);
    assert(rc_href != nullptr);

    for (int r = 0; r < (int)this->qps.size(); ++r) {
        NetIbQp *qp = this->qps[r];
        if (qp != nullptr) {
            int ret = qp->post_recv(((uint64_t)r << sid_shift) + 1);
            if (ret != 0) {
                LOG(ERROR, "post_recv() returns ", ret);
            }
        }
    }

    //
    const bool is_using_p2p_memcpy = !get_env().disable_p2p_memcpy;
    const bool is_using_ib = this->is_using_ib();
    if (!is_using_p2p_memcpy && !is_using_ib) {
        LOG(ERROR, "no method for transport");
    }
    bool is_idle = false;
    unsigned int busy_counter = 0;
    const unsigned int max_busy_counter = 3000000000;
    // Request pointer.
    volatile uint64_t *db_val = &(this->request->value);
    // Request processing loop.
    while (this->run_request_loop_thread) {
        int wcn = 0;
        if (is_using_ib) {
            wcn = this->net_ib_mgr->poll_cq();
        }
        if (wcn > 0) {
            for (int i = 0; i < wcn; ++i) {
                int status = this->net_ib_mgr->get_wc_status(i);
                if (status != 0) {
                    LOG(ERROR, "get_wc_status() returns ", status, ": ",
                        this->net_ib_mgr->get_wc_status_str(i));
                }
                uint64_t wr_id = this->net_ib_mgr->get_wc_wr_id(i);
                if (wr_id & 0x1) {
                    // recv complete
                    unsigned int sid_dst = this->net_ib_mgr->get_wc_imm_data(i);
                    rc_href[sid_dst] = 1;
                    NetIbQp *qp = this->qps[wr_id >> sid_shift];
                    if (qp == nullptr) {
                        LOG(ERROR, "Unexpected error");
                    }
                    int ret = qp->post_recv(wr_id);
                    if (ret != 0) {
                        LOG(ERROR, "post_recv() returns ", ret);
                    }
                    LOG(DEBUG, "RC DST: ", sid_dst);
                } else {
                    // send complete
                    unsigned int sid_src = wr_id >> sid_shift;
                    sc_href[sid_src] = 1;
                    LOG(DEBUG, "SC SRC: ", sid_src);
                }
            }
            is_idle = false;
        } else if (wcn < 0) {
            LOG(ERROR, "poll_cq() returns ", wcn);
        }
        uint64_t v = *db_val;
        if (v == (uint64_t)REQUEST_INVALID) {
            if (wcn == 0) {
                if (is_idle) {
                    if (cpu_ntimer_sleep(0) != 0) {
                        LOG(WARN, "cpu_ntimer_sleep() returns errno ", errno);
                    }
                } else if (++busy_counter > max_busy_counter) {
                    is_idle = true;
                    LOG(DEBUG, "Idle.");
                }
            }
            continue;
        }
        *db_val = (uint64_t)REQUEST_INVALID;
        Request &db = (Request &)v;
        LOG(DEBUG, "Request arrived.");
        //
        GpuPtr src = this->addr_table[this->gpu_id][db.fields.src];
        if (src == 0) {
            LOG(ERROR, "Invalid SRC SID ", db.fields.src, " in GPU ",
                this->gpu_id);
        }
        LOG(DEBUG, "Request SRC: RANK ", this->rank, ", sid ", db.fields.src,
            ", ", (void *)src);
        GpuPtr dst = 0;
        // TODO: generalize converting rank to GPU ID.
        int nrph = get_env().num_ranks_per_host;
        int gid_dst = db.fields.rank % nrph;
        if ((db.fields.rank / nrph) != (this->rank / nrph)) {
            // This GPU is not in this machine.
            gid_dst = -1;
            LOG(DEBUG, "Request DST: RANK ", db.fields.rank, ", sid ",
                db.fields.dst, ", remote");
        } else {
            dst = this->addr_table[gid_dst][db.fields.dst];
            if (dst == 0) {
                LOG(ERROR, "Invalid DST SID ", db.fields.dst, " in GPU ",
                    gid_dst);
            }
            LOG(DEBUG, "Request DST: RANK ", db.fields.rank, ", sid ",
                db.fields.dst, ", ", (void *)dst);
        }
        LOG(DEBUG, "Request LEN: ", db.fields.len);

        // Transfer data.
        if (is_using_p2p_memcpy && (gid_dst != -1)) {
            CULOG(cuMemcpyDtoD(dst, src, db.fields.len));
            GpuMem *mem = this->get_sc_rc_mem(db.fields.rank);
            volatile int *rc_array = (volatile int *)mem->href(rc_offset);
            if (rc_array != nullptr) {
                rc_array[db.fields.dst] = 1;
            } else {
                GpuPtr rc_ref =
                    mem->ref(rc_offset + db.fields.dst * sizeof(int));
                CULOG(cuMemsetD32(rc_ref, 1, 1));
            }
            sc_href[db.fields.src] = 1;
        } else {
            NetIbQp *qp = this->qps[db.fields.rank];
            int ret = qp->stage_send(
                this->sid_mrs[db.fields.src],
                &this->mris[db.fields.rank][db.fields.dst], db.fields.len,
                (db.fields.src << sid_shift), db.fields.dst);
            if (ret != 1) {
                LOG(ERROR, "stage_send() returns ", ret);
            }
            ret = qp->post_send();
            if (ret != 0) {
                LOG(ERROR, "post_send() returns ", ret);
            }
        }
        LOG(DEBUG, "Request processed.");
        //
        is_idle = false;
        busy_counter = 0;
    }
}

//
void GpuCommSw::Impl::stop_request_loop()
{
#ifdef ARK_USE_MSCCLPP
    if (get_env().use_mscclpp) {
        this->proxy_service->stopProxy();
        return;
    }
#endif // ARK_USE_MSCCLPP
    this->run_request_loop_thread = false;
    if (this->request_loop_thread != nullptr) {
        if (this->request_loop_thread->joinable()) {
            this->request_loop_thread->join();
        }
        delete this->request_loop_thread;
        this->request_loop_thread = nullptr;
    }
}

//
void GpuCommSw::Impl::set_request(const Request &db)
{
    if (this->request != nullptr) {
        *(this->request) = db;
    }
}

//
GpuMem *GpuCommSw::Impl::get_data_mem(const int gid)
{
    int sz = (int)this->data_mems.size();
    assert(sz == (int)this->sc_rc_mems.size());
    if (sz <= gid) {
        while (sz <= gid) {
            sz *= 2;
        }
        this->data_mems.resize(sz, nullptr);
    }
    GpuMem *dm = this->data_mems[gid];
    if (dm == nullptr) {
        dm = new GpuMem{ARK_GPU_DATA_NAME + this->name + to_string(gid), 0,
                        false};
        assert(dm != nullptr);
        this->data_mems[gid] = dm;
        this->remote_data_mems_storage.emplace_back(dm);
    }
    //
    while (this->addr_table.size() < this->data_mems.size()) {
        this->addr_table.emplace_back();
        this->addr_table.back().resize(256, 0);
    }
    return dm;
}

//
GpuMem *GpuCommSw::Impl::get_sc_rc_mem(const int gid)
{
    int sz = (int)this->sc_rc_mems.size();
    assert(sz == (int)this->sc_rc_mems.size());
    if (sz <= gid) {
        while (sz <= gid) {
            sz *= 2;
        }
        this->sc_rc_mems.resize(sz, nullptr);
    }
    GpuMem *sm = this->sc_rc_mems[gid];
    if (sm == nullptr) {
        sm = new GpuMem{ARK_GPU_SC_RC_NAME + this->name + to_string(gid),
                        2 * MAX_NUM_SID * sizeof(int), false};
        this->sc_rc_mems[gid] = sm;
        this->remote_sc_rc_mems_storage.emplace_back(sm);
    }
    return sm;
}

//
IpcMem *GpuCommSw::Impl::get_info(const int gid)
{
    int sz = (int)this->infos.size();
    assert(sz == (int)this->infos.size());
    if (sz <= gid) {
        while (sz <= gid) {
            sz *= 2;
        }
        this->infos.resize(sz, nullptr);
    }
    IpcMem *ie = this->infos[gid];
    if (ie == nullptr) {
        ie = new IpcMem{ARK_GPU_INFO_NAME + this->name + to_string(gid), false};
        assert(ie != nullptr);
        this->infos_storage.emplace_back(ie);
        this->infos[gid] = ie;
    }
    return ie;
}

//
GpuPtr GpuCommSw::Impl::get_request_ref() const
{
    GpuPtr ref;
    CULOG(cuMemHostGetDevicePointer(&ref, this->request, 0));
    return ref;
}

GpuCommSw::GpuCommSw(const std::string &name, const int gpu_id_,
                     const int rank_, const int world_size_, GpuMem *data_mem,
                     GpuMem *sc_rc_mem)
    : impl{std::make_unique<GpuCommSw::Impl>(name, gpu_id_, rank_, world_size_,
                                             data_mem, sc_rc_mem)}
{
}

GpuCommSw::~GpuCommSw()
{
}

void GpuCommSw::reg_sendrecv(int sid, int remote_rank, std::size_t bytes,
                             bool is_recv)
{
    this->impl->reg_sendrecv(sid, remote_rank, bytes, is_recv);
}

void GpuCommSw::configure(std::vector<std::pair<int, size_t>> &export_sid_offs,
                          std::map<int, std::vector<GpuBuf *>> &import_gid_bufs)
{
    this->impl->configure(export_sid_offs, import_gid_bufs);
}

void GpuCommSw::import_buf(const int gid, GpuBuf *buf)
{
    this->impl->import_buf(gid, buf);
}

void GpuCommSw::launch_request_loop()
{
    this->impl->launch_request_loop();
}

void GpuCommSw::stop_request_loop()
{
    this->impl->stop_request_loop();
}

void GpuCommSw::set_request(const Request &db)
{
    this->impl->set_request(db);
}

GpuMem *GpuCommSw::get_data_mem(const int gid)
{
    return this->impl->get_data_mem(gid);
}

GpuPtr GpuCommSw::get_request_ref() const
{
    return this->impl->get_request_ref();
}

bool GpuCommSw::is_using_ib() const
{
    return this->impl->is_using_ib();
}

const void *GpuCommSw::get_proxy_channels_ref() const
{
    return this->impl->get_proxy_channels_ref();
}

int GpuCommSw::get_proxy_channels_bytes() const
{
    return this->impl->get_proxy_channels_bytes();
}

int GpuCommSw::get_proxy_channels_num() const
{
    return this->impl->get_proxy_channels_num();
}

} // namespace ark
