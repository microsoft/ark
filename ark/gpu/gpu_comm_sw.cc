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

using namespace std;

#define DEBUG_REQUEST 0
#define REQUEST_DEBUG(...)                                                     \
    do {                                                                       \
        if (DEBUG_REQUEST) {                                                   \
            LOG(DEBUG, __VA_ARGS__);                                           \
        }                                                                      \
    } while (0);

namespace ark {

// For peer access between GPUs on the same machine.
struct GpuCommMemInfo
{
    GpuMem::Info mem_info;
    uint64_t sid_offs[MAX_NUM_SID];
};

// For IB connection between GPUs (either inter- or intra-node).
struct GpuCommIbInfo
{
    NetIbQp::Info qp_info;
    NetIbMr::Info sid_mris[MAX_NUM_SID];
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
    void configure(const std::vector<std::pair<int, size_t>> &export_sid_offs,
                   const std::map<int, std::vector<GpuBuf *>> &import_gid_bufs);

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
    std::unique_ptr<IpcSocket> ipc_socket;
    //
    NetIbMgr *net_ib_mgr = nullptr;
    std::vector<NetIbMr *> sid_mrs;
    std::map<int, NetIbQp *> qps;
    std::vector<GpuSendRecvInfo> send_recv_infos;
    std::map<int, std::vector<NetIbMr::Info>> mris;
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

    this->infos_storage.emplace_back(std::make_unique<IpcMem>(
        ARK_GPU_INFO_NAME + name_ + to_string(gpu_id_), true));
    this->infos[gpu_id_] = this->infos_storage.back().get();

    int port = get_env().ipc_listen_port_base + gpu_id_;
    int host_id = rank_ / get_env().num_ranks_per_host;
    this->ipc_socket = std::make_unique<IpcSocket>(get_host(host_id), port);

    if (!get_env().disable_ib) {
        int num_ib_dev = get_net_ib_device_num();
        // TODO: be aware of GPU-NIC interconnect topology.
        int ib_dev_id = gpu_id_ % num_ib_dev;
        this->net_ib_mgr = get_net_ib_mgr(ib_dev_id);
        this->sid_mrs.resize(MAX_NUM_SID, nullptr);
    }
}

GpuCommSw::Impl::~Impl()
{
    this->stop_request_loop();
    if (this->request != nullptr) {
        cuMemHostUnregister((void *)this->request);
        delete this->request;
    }
}

void GpuCommSw::Impl::reg_sendrecv(int sid, int remote_rank, size_t bytes,
                                   bool is_recv)
{
    this->send_recv_infos.emplace_back(
        GpuSendRecvInfo{sid, remote_rank, bytes, is_recv});
}

//
void GpuCommSw::Impl::configure(
    const std::vector<std::pair<int, size_t>> &export_sid_offs,
    const std::map<int, std::vector<GpuBuf *>> &import_gid_bufs)
{
    // Max requested bytes for each SID. Only for IB.
    std::map<int, size_t> ib_sid_max_bytes;
    for (auto &srinfo : this->send_recv_infos) {
        ib_sid_max_bytes[srinfo.sid] =
            max(ib_sid_max_bytes[srinfo.sid], srinfo.bytes);
    }

    //
    GpuMem *data_mem = this->get_data_mem(this->gpu_id);
    GpuCommMemInfo comm_mem_info;
    GpuCommIbInfo comm_ib_info;

    comm_mem_info.mem_info.ipc_hdl = data_mem->get_info().ipc_hdl;
    comm_mem_info.mem_info.phys_addr = data_mem->get_info().phys_addr;
    comm_mem_info.mem_info.bytes = data_mem->get_info().bytes;

    for (auto &p : export_sid_offs) {
        int sid = p.first;
        size_t off = p.second;
        GpuPtr addr = data_mem->ref(off);

        this->addr_table[this->gpu_id][sid] = addr;
        comm_mem_info.sid_offs[sid] = off;

        if (this->is_using_ib() && this->sid_mrs[sid] == nullptr) {
            auto search = ib_sid_max_bytes.find(sid);
            if (search == ib_sid_max_bytes.end()) {
                // This SID is not used for IB.
                continue;
            }
            // Create an MR
            NetIbMr *mr =
                this->net_ib_mgr->reg_mr((void *)addr, ib_sid_max_bytes[sid]);
            this->sid_mrs[sid] = mr;
            comm_ib_info.sid_mris[sid] = mr->get_info();
        }
    }

    // Share comm_mem_info with other GPUs on the same machine.
    auto state = this->ipc_socket->add_item("comm_mem_info", &comm_mem_info,
                                            sizeof(comm_mem_info));
    if (state != IpcSocket::State::SUCCESS) {
        LOG(ERROR, "Failed to post comm_mem_info");
    }
    int num_ranks_per_host = get_env().num_ranks_per_host;
    int my_host_id = this->rank / num_ranks_per_host;
    int port_base = get_env().ipc_listen_port_base;
    for (auto &p : import_gid_bufs) {
        // Get comm_mem_info from other GPUs on the same machine.
        // Only from GPUs that this GPU imports.
        int gpu_id = p.first;
        int port = port_base + gpu_id;

        GpuCommMemInfo remote_comm_mem_info;
        state = this->ipc_socket->query_item(
            get_host(my_host_id), port, "comm_mem_info", &remote_comm_mem_info,
            sizeof(remote_comm_mem_info), true);
        if (state != IpcSocket::State::SUCCESS) {
            LOG(ERROR, "Failed to query comm_mem_info from GPU ", gpu_id);
        }

        // Initialize the remote GPU memory space.
        GpuMem *mem = this->get_data_mem(gpu_id);
        mem->init(remote_comm_mem_info.mem_info);
        for (GpuBuf *buf : p.second) {
            int sid = buf->get_id();
            size_t off = remote_comm_mem_info.sid_offs[sid];
            this->addr_table[gpu_id][sid] = mem->ref(off);
            buf->set_offset(off);
        }
    }

    if (!this->is_using_ib()) {
        return;
    }

    // Create QPs.
    for (auto &srinfo : this->send_recv_infos) {
        auto it = this->qps.find(srinfo.remote_rank);
        if (it == this->qps.end()) {
            NetIbQp *qp = this->net_ib_mgr->create_qp();
            if (qp == nullptr) {
                LOG(ERROR, "create_qp failed");
            }
            this->qps[srinfo.remote_rank] = qp;
        }
    }

    // Share comm_ib_info with remote ranks.

    std::set<int> remote_ranks;
    for (auto &srinfo : this->send_recv_infos) {
        remote_ranks.insert(srinfo.remote_rank);
    }
    for (int remote_rank : remote_ranks) {
        // Set the corresponding QP info.
        comm_ib_info.qp_info = this->qps[remote_rank]->get_info();

        std::string item_name = "comm_ib_info_" + std::to_string(remote_rank);
        state = this->ipc_socket->add_item(item_name, &comm_ib_info,
                                           sizeof(comm_ib_info));
        if (state != IpcSocket::State::SUCCESS) {
            LOG(ERROR, "Failed to post ", item_name);
        }
    }
    for (int remote_rank : remote_ranks) {
        int remote_gpu_id = remote_rank % num_ranks_per_host;
        int remote_host_id = remote_rank / num_ranks_per_host;
        int port = port_base + remote_gpu_id;

        std::string item_name = "comm_ib_info_" + std::to_string(this->rank);

        GpuCommIbInfo remote_comm_ib_info;

        state = this->ipc_socket->query_item(get_host(remote_host_id), port,
                                             item_name, &remote_comm_ib_info,
                                             sizeof(remote_comm_ib_info), true);
        if (state != IpcSocket::State::SUCCESS) {
            LOG(ERROR, "Failed to query ", item_name);
        }

        NetIbQp *qp = this->qps[remote_rank];
        int ret = qp->rtr(&remote_comm_ib_info.qp_info);
        if (ret != 0) {
            LOG(ERROR, "NetIbQp::rtr failed");
        }
        LOG(DEBUG, "RANK ", this->rank, " QP ", qp->get_info().qpn,
            " <--> RANK ", remote_rank, " QP ",
            remote_comm_ib_info.qp_info.qpn);
        ret = qp->rts();
        if (ret != 0) {
            LOG(ERROR, "NetIbQp::rts failed");
        }

        auto &mri_vec = this->mris[remote_rank];
        mri_vec.resize(MAX_NUM_SID);
        for (size_t sid = 0; sid < mri_vec.size(); ++sid) {
            mri_vec[sid] = remote_comm_ib_info.sid_mris[sid];
        }
    }

    // Sync with remote ranks to make sure the QP is ready.
    for (int remote_rank : remote_ranks) {
        int dummy = 0;
        std::string item_name = "comm_ib_done_" + std::to_string(remote_rank);
        state = this->ipc_socket->add_item(item_name, &dummy, sizeof(dummy));
        if (state != IpcSocket::State::SUCCESS) {
            LOG(ERROR, "Failed to add ", item_name);
        }
    }
    for (int remote_rank : remote_ranks) {
        int remote_gpu_id = remote_rank % num_ranks_per_host;
        int remote_host_id = remote_rank / num_ranks_per_host;
        int port = port_base + remote_gpu_id;

        int dummy = 0;
        std::string item_name = "comm_ib_done_" + std::to_string(this->rank);
        state = this->ipc_socket->query_item(get_host(remote_host_id), port,
                                             item_name, &dummy, sizeof(dummy),
                                             true);
        if (state != IpcSocket::State::SUCCESS) {
            LOG(ERROR, "Failed to query ", item_name);
        }
    }

    LOG(DEBUG, "RANK ", this->rank, " config done");
}

//
void GpuCommSw::Impl::launch_request_loop()
{
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
        REQUEST_DEBUG("Request arrived.");
        //
        GpuPtr src = this->addr_table[this->gpu_id][db.fields.sid];
        if (src == 0) {
            LOG(ERROR, "Invalid SRC SID ", db.fields.sid, " in GPU ",
                this->gpu_id);
        }
        REQUEST_DEBUG("Request SRC: RANK ", this->rank, ", sid ", db.fields.sid,
                      ", ", (void *)src);
        GpuPtr dst = 0;
        // TODO: generalize converting rank to GPU ID.
        int nrph = get_env().num_ranks_per_host;
        int gid_dst = db.fields.rank % nrph;
        if ((db.fields.rank / nrph) != (this->rank / nrph)) {
            // This GPU is not in this machine.
            gid_dst = -1;
            REQUEST_DEBUG("Request DST: RANK ", db.fields.rank, ", sid ",
                          db.fields.sid, ", remote");
        } else {
            dst = this->addr_table[gid_dst][db.fields.sid];
            if (dst == 0) {
                LOG(ERROR, "Invalid DST SID ", db.fields.sid, " in GPU ",
                    gid_dst);
            }
            REQUEST_DEBUG("Request DST: RANK ", db.fields.rank, ", sid ",
                          db.fields.sid, ", ", (void *)dst);
        }
        REQUEST_DEBUG("Request LEN: ", db.fields.len);

        // Transfer data.
        if (is_using_p2p_memcpy && (gid_dst != -1)) {
            CULOG(cuMemcpyDtoD(dst, src, db.fields.len));
            GpuMem *mem = this->get_sc_rc_mem(db.fields.rank);
            volatile int *rc_array = (volatile int *)mem->href(rc_offset);
            if (rc_array != nullptr) {
                rc_array[db.fields.sid] = 1;
            } else {
                GpuPtr rc_ref =
                    mem->ref(rc_offset + db.fields.sid * sizeof(int));
                CULOG(cuMemsetD32(rc_ref, 1, 1));
            }
            sc_href[db.fields.sid] = 1;
        } else {
            NetIbQp *qp = this->qps[db.fields.rank];
            int ret = qp->stage_send(
                this->sid_mrs[db.fields.sid],
                &this->mris[db.fields.rank][db.fields.sid], db.fields.len,
                (db.fields.sid << sid_shift), db.fields.sid);
            if (ret != 1) {
                LOG(ERROR, "stage_send() returns ", ret);
            }
            ret = qp->post_send();
            if (ret != 0) {
                LOG(ERROR, "post_send() returns ", ret);
            }
        }
        REQUEST_DEBUG("Request processed.");
        //
        is_idle = false;
        busy_counter = 0;
    }
}

//
void GpuCommSw::Impl::stop_request_loop()
{
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
        this->remote_data_mems_storage.emplace_back(std::make_unique<GpuMem>());
        dm = this->remote_data_mems_storage.back().get();
        this->data_mems[gid] = dm;
    }
    //
    while (this->addr_table.size() < this->data_mems.size()) {
        this->addr_table.emplace_back();
        this->addr_table.back().resize(16, 0);
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
        this->remote_sc_rc_mems_storage.emplace_back(
            std::make_unique<GpuMem>(2 * MAX_NUM_SID * sizeof(int)));
        sm = this->remote_sc_rc_mems_storage.back().get();
        this->sc_rc_mems[gid] = sm;
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
        this->infos_storage.emplace_back(std::make_unique<IpcMem>(
            ARK_GPU_INFO_NAME + this->name + to_string(gid), false));
        ie = this->infos_storage.back().get();
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

void GpuCommSw::configure(
    const std::vector<std::pair<int, size_t>> &export_sid_offs,
    const std::map<int, std::vector<GpuBuf *>> &import_gid_bufs)
{
    this->impl->configure(export_sid_offs, import_gid_bufs);
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

} // namespace ark
