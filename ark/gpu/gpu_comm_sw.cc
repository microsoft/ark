#include <cassert>
#include <cerrno>

#include "ark/cpu_timer.h"
#include "ark/env.h"
#include "ark/gpu/gpu_comm_sw.h"
#include "ark/gpu/gpu_logging.h"
#include "ark/gpu/gpu_mgr.h"
#include "ark/ipc/ipc_coll.h"
#include "ark/ipc/ipc_hosts.h"

using namespace std;

namespace ark {

//
GpuCommSw::GpuCommSw(const string &name_, const int gpu_id_, const int rank_,
                     const int world_size_, GpuMem *data_mem, GpuMem *sc_rc_mem)
    : name{name_}, gpu_id{gpu_id_}, rank{rank_},
      world_size{world_size_}, doorbell{new Doorbell}
{
    // Register `this->doorbell` as a mapped & pinned address.
    CULOG(cuMemHostRegister((void *)this->doorbell, sizeof(Doorbell),
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
}

GpuCommSw::~GpuCommSw()
{
    this->stop_doorbell_loop();
    if (this->doorbell != nullptr) {
        cuMemHostUnregister((void *)this->doorbell);
        delete this->doorbell;
    }
    delete this->ipc_socket;
}

void GpuCommSw::reg_sendrecv(int sid, int remote_rank, size_t bytes,
                             bool is_recv)
{
    this->send_recv_infos.emplace_back(
        GpuSendRecvInfo{sid, remote_rank, bytes, is_recv});
}

//
void GpuCommSw::configure(vector<pair<int, size_t>> &export_sid_offs,
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
                    LOGERR("create_qp failed");
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
            LOGERR("unexpected error");
        }
        gi.qpi = qp->get_info();
        s = this->ipc_socket->add_item(
            "gpu_comm_info_" + std::to_string(remote_rank), &gi, sizeof(gi));
        if (s != IpcSocket::State::SUCCESS) {
            LOGERR("Failed to add gpu_comm_info to ipc_socket");
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
                LOGERR("Failed to query gpu_comm_info from ipc_socket");
            }

            ret = qp->rtr(&gi_remote.qpi);
            assert(ret == 0);
            LOG(DEBUG, "RANK ", this->rank, " QP ", qp->get_info().qpn,
                " <--> RANK ", remote_rank, " QP ", gi_remote.qpi.qpn);
            ret = qp->rts();
            assert(ret == 0);

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
            LOGERR("Failed to add comm_config_done to ipc_socket");
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
                LOGERR("Failed to query gpu_comm_info from ipc_socket");
            }
            if (remote_data != dummy_data) {
                LOGERR("Failed to sync comm_config_done");
            }
        }
    }
    LOG(DEBUG, "RANK ", this->rank, " config done");
}

//
void GpuCommSw::import_buf(const int gid, GpuBuf *buf)
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
            LOGERR("unexpected error");
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
void GpuCommSw::launch_doorbell_loop()
{
    if (this->doorbell_loop_thread == nullptr) {
        this->run_doorbell_loop_thread = true;
        this->doorbell_loop_thread = new thread([&, gid = this->gpu_id] {
            //
            GpuState ret = get_gpu_mgr(gid)->set_current();
            if (ret == CUDA_SUCCESS) {
                //
                this->doorbell_loop();
            } else if (ret != CUDA_ERROR_DEINITIALIZED) {
                CULOG(ret);
            }
        });
        assert(this->doorbell_loop_thread != nullptr);
    } else {
        assert(this->run_doorbell_loop_thread);
    }
}

//
void GpuCommSw::doorbell_loop()
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
                LOGERR("post_recv() returns ", ret);
            }
        }
    }

    //
    const bool is_using_p2p_memcpy = !get_env().disable_p2p_memcpy;
    const bool is_using_ib = this->is_using_ib();
    if (!is_using_p2p_memcpy && !is_using_ib) {
        LOGERR("no method for transport");
    }
    bool is_idle = false;
    unsigned int busy_counter = 0;
    const unsigned int max_busy_counter = 3000000000;
    // Doorbell pointer.
    volatile uint64_t *db_val = &(this->doorbell->value);
    // Doorbell processing loop.
    while (this->run_doorbell_loop_thread) {
        int wcn = 0;
        if (is_using_ib) {
            wcn = this->net_ib_mgr->poll_cq();
        }
        if (wcn > 0) {
            for (int i = 0; i < wcn; ++i) {
                int status = this->net_ib_mgr->get_wc_status(i);
                if (status != 0) {
                    LOGERR("get_wc_status() returns ", status, ": ",
                           this->net_ib_mgr->get_wc_status_str(i));
                }
                uint64_t wr_id = this->net_ib_mgr->get_wc_wr_id(i);
                if (wr_id & 0x1) {
                    // recv complete
                    unsigned int sid_dst = this->net_ib_mgr->get_wc_imm_data(i);
                    rc_href[sid_dst] = 1;
                    NetIbQp *qp = this->qps[wr_id >> sid_shift];
                    if (qp == nullptr) {
                        LOGERR("Unexpected error");
                    }
                    int ret = qp->post_recv(wr_id);
                    if (ret != 0) {
                        LOGERR("post_recv() returns ", ret);
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
            LOGERR("poll_cq() returns ", wcn);
        }
        uint64_t v = *db_val;
        if (v == (uint64_t)DOORBELL_INVALID) {
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
        *db_val = (uint64_t)DOORBELL_INVALID;
        Doorbell &db = (Doorbell &)v;
        LOG(DEBUG, "Doorbell arrived.");
        //
        GpuPtr src = this->addr_table[this->gpu_id][db.fields.src];
        if (src == 0) {
            LOGERR("Invalid SRC SID ", db.fields.src, " in GPU ", this->gpu_id);
        }
        LOG(DEBUG, "Doorbell SRC: RANK ", this->rank, ", sid ", db.fields.src,
            ", ", (void *)src);
        GpuPtr dst = 0;
        // TODO: generalize converting rank to GPU ID.
        int nrph = get_env().num_ranks_per_host;
        int gid_dst = db.fields.cid % nrph;
        if ((db.fields.cid / nrph) != (this->rank / nrph)) {
            // This GPU is not in this machine.
            gid_dst = -1;
            LOG(DEBUG, "Doorbell DST: RANK ", db.fields.cid, ", sid ",
                db.fields.dst, ", remote");
        } else {
            dst = this->addr_table[gid_dst][db.fields.dst];
            if (dst == 0) {
                LOGERR("Invalid DST SID ", db.fields.dst, " in GPU ", gid_dst);
            }
            LOG(DEBUG, "Doorbell DST: RANK ", db.fields.cid, ", sid ",
                db.fields.dst, ", ", (void *)dst);
        }
        LOG(DEBUG, "Doorbell LEN: ", db.fields.len);

        // Transfer data.
        if (is_using_p2p_memcpy && (gid_dst != -1)) {
            CULOG(cuMemcpyDtoD(dst, src, db.fields.len));
            GpuMem *mem = this->get_sc_rc_mem(db.fields.cid);
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
            NetIbQp *qp = this->qps[db.fields.cid];
            int ret = qp->stage_send(
                this->sid_mrs[db.fields.src],
                &this->mris[db.fields.cid][db.fields.dst], db.fields.len,
                (db.fields.src << sid_shift), db.fields.dst);
            if (ret != 1) {
                LOGERR("stage_send() returns ", ret);
            }
            ret = qp->post_send();
            if (ret != 0) {
                LOGERR("post_send() returns ", ret);
            }
        }
        LOG(DEBUG, "Doorbell processed.");
        //
        is_idle = false;
        busy_counter = 0;
    }
}

//
void GpuCommSw::stop_doorbell_loop()
{
    this->run_doorbell_loop_thread = false;
    if (this->doorbell_loop_thread != nullptr) {
        if (this->doorbell_loop_thread->joinable()) {
            this->doorbell_loop_thread->join();
        }
        delete this->doorbell_loop_thread;
        this->doorbell_loop_thread = nullptr;
    }
}

//
void GpuCommSw::set_doorbell(const Doorbell &db)
{
    if (this->doorbell != nullptr) {
        *(this->doorbell) = db;
    }
}

//
GpuMem *GpuCommSw::get_data_mem(const int gid)
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
GpuMem *GpuCommSw::get_sc_rc_mem(const int gid)
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
IpcMem *GpuCommSw::get_info(const int gid)
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
GpuPtr GpuCommSw::get_doorbell_ref() const
{
    GpuPtr ref;
    CULOG(cuMemHostGetDevicePointer(&ref, this->doorbell, 0));
    return ref;
}

} // namespace ark
