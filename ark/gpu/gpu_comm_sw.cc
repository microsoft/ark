// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_comm_sw.h"

#include <sys/mman.h>

#include <cassert>
#include <cerrno>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "cpu_timer.h"
#include "env.h"
#include "gpu/gpu_common.h"
#include "gpu/gpu_logging.h"
#include "gpu/gpu_mgr.h"
#include "ipc/ipc_hosts.h"
#include "ipc/ipc_socket.h"
#include "net/net_ib.h"

using namespace std;

#define DEBUG_REQUEST 0
#define REQUEST_DEBUG(...)           \
    do {                             \
        if (DEBUG_REQUEST) {         \
            LOG(DEBUG, __VA_ARGS__); \
        }                            \
    } while (0);

namespace ark {

// For peer access between GPUs on the same machine.
struct GpuCommMemInfo {
    GpuMem::Info data_info;
    GpuMem::Info sc_rc_info;
    uint64_t sid_offs[MAX_NUM_SID];
};

// For IB connection between GPUs (either inter- or intra-node).
struct GpuCommIbInfo {
    NetIbQp::Info qp_info;
    NetIbMr::Info sid_mris[MAX_NUM_SID];
};

//
struct GpuSendRecvInfo {
    int sid;
    int remote_rank;
    std::size_t bytes;
    bool is_recv;
};

class GpuCommSw::Impl {
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
    GpuPtr get_request_ref() const;
    bool is_using_ib() const { return net_ib_mgr_ != nullptr; }

   private:
    //
    const std::string name_;
    //
    const int gpu_id_;
    const int rank_;
    const int world_size_;
    //
    std::list<std::unique_ptr<GpuMem>> remote_data_mems_storage_;
    std::list<std::unique_ptr<GpuMem>> remote_sc_rc_mems_storage_;
    //
    std::vector<GpuMem *> data_mems_;
    //
    std::vector<GpuMem *> sc_rc_mems_;
    //
    std::vector<std::vector<GpuPtr>> addr_table_;
    //
    Request *request_ = nullptr;
    std::thread *request_loop_thread_ = nullptr;
    volatile bool run_request_loop_thread_ = false;
    //
    std::unique_ptr<IpcSocket> ipc_socket_;
    //
    NetIbMgr *net_ib_mgr_ = nullptr;
    std::vector<NetIbMr *> sid_mrs_;
    std::map<int, NetIbQp *> qps_;
    std::vector<GpuSendRecvInfo> send_recv_infos_;
    std::map<int, std::vector<NetIbMr::Info>> mris_;
};

//
GpuCommSw::Impl::Impl(const string &name, const int gpu_id, const int rank,
                      const int world_size, GpuMem *data_mem, GpuMem *sc_rc_mem)
    : name_{name},
      gpu_id_{gpu_id},
      rank_{rank},
      world_size_{world_size},
      request_{new Request} {
    // Register `request_` as a mapped & pinned address.
    GLOG(gpuHostRegister((void *)request_, sizeof(request_),
                         gpuHostRegisterMapped));

    // Reserve entries for GPU communication stack information.
    // Power of 2 larger than `gpu_id_` and at least 8.
    int num_entries = 8;
    while (gpu_id_ >= num_entries) {
        num_entries *= 2;
    }
    data_mems_.resize(num_entries, nullptr);
    sc_rc_mems_.resize(num_entries, nullptr);

    // Create the local stack info.
    data_mems_[gpu_id_] = data_mem;
    sc_rc_mems_[gpu_id_] = sc_rc_mem;

    int port = get_env().ipc_listen_port_base + gpu_id_;
    int host_id = rank_ / get_env().num_ranks_per_host;
    ipc_socket_ = std::make_unique<IpcSocket>(get_host(host_id), port);

    if (!get_env().disable_ib) {
        int num_ib_dev = get_net_ib_device_num();
        // TODO: be aware of GPU-NIC interconnect topology.
        int ib_dev_id = gpu_id_ % num_ib_dev;
        net_ib_mgr_ = get_net_ib_mgr(ib_dev_id);
        sid_mrs_.resize(MAX_NUM_SID, nullptr);
    }
}

GpuCommSw::Impl::~Impl() {
    this->stop_request_loop();
    if (request_ != nullptr) {
        if (gpuHostUnregister((void *)request_) != gpuSuccess) {
            LOG(WARN, "gpuHostUnregister() failed.");
        }
        delete request_;
    }
}

void GpuCommSw::Impl::reg_sendrecv(int sid, int remote_rank, size_t bytes,
                                   bool is_recv) {
    send_recv_infos_.emplace_back(
        GpuSendRecvInfo{sid, remote_rank, bytes, is_recv});
}

//
void GpuCommSw::Impl::configure(
    const std::vector<std::pair<int, size_t>> &export_sid_offs,
    const std::map<int, std::vector<GpuBuf *>> &import_gid_bufs) {
    // Max requested bytes for each SID. Only for IB.
    std::map<int, size_t> ib_sid_max_bytes;
    for (auto &srinfo : send_recv_infos_) {
        ib_sid_max_bytes[srinfo.sid] =
            max(ib_sid_max_bytes[srinfo.sid], srinfo.bytes);
    }

    //
    GpuMem *data_mem = this->get_data_mem(gpu_id_);
    GpuMem *sc_rc_mem = this->get_sc_rc_mem(gpu_id_);
    GpuCommMemInfo comm_mem_info;
    GpuCommIbInfo comm_ib_info;

    comm_mem_info.data_info.ipc_hdl = data_mem->get_info().ipc_hdl;
    comm_mem_info.data_info.phys_addr = data_mem->get_info().phys_addr;
    comm_mem_info.data_info.bytes = data_mem->get_info().bytes;

    comm_mem_info.sc_rc_info.ipc_hdl = sc_rc_mem->get_info().ipc_hdl;
    comm_mem_info.sc_rc_info.phys_addr = sc_rc_mem->get_info().phys_addr;
    comm_mem_info.sc_rc_info.bytes = sc_rc_mem->get_info().bytes;

    for (auto &p : export_sid_offs) {
        int sid = p.first;
        size_t off = p.second;
        GpuPtr addr = data_mem->ref(off);

        addr_table_[gpu_id_][sid] = addr;
        comm_mem_info.sid_offs[sid] = off;

        if (this->is_using_ib() && sid_mrs_[sid] == nullptr) {
            auto search = ib_sid_max_bytes.find(sid);
            if (search == ib_sid_max_bytes.end()) {
                // This SID is not used for IB.
                continue;
            }
            // Create an MR
            NetIbMr *mr =
                net_ib_mgr_->reg_mr((void *)addr, ib_sid_max_bytes[sid]);
            sid_mrs_[sid] = mr;
            comm_ib_info.sid_mris[sid] = mr->get_info();
        }
    }

    // Share comm_mem_info with other GPUs on the same machine.
    auto state = ipc_socket_->add_item("comm_mem_info", &comm_mem_info,
                                       sizeof(comm_mem_info));
    if (state != IpcSocket::State::SUCCESS) {
        LOG(ERROR, "Failed to post comm_mem_info");
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
            LOG(ERROR, "Failed to query comm_mem_info from GPU ", gpu_id);
        }

        // Initialize the remote GPU memory space.
        GpuMem *mem = this->get_data_mem(gpu_id);
        mem->init(remote_comm_mem_info.data_info);
        for (GpuBuf *buf : p.second) {
            int sid = buf->get_id();
            size_t off = remote_comm_mem_info.sid_offs[sid];
            addr_table_[gpu_id][sid] = mem->ref(off);
            buf->set_offset(off);
        }
        mem = this->get_sc_rc_mem(gpu_id);
        mem->init(remote_comm_mem_info.sc_rc_info);
    }

    if (!this->is_using_ib()) {
        return;
    }

    // Create QPs.
    for (auto &srinfo : send_recv_infos_) {
        auto it = qps_.find(srinfo.remote_rank);
        if (it == qps_.end()) {
            NetIbQp *qp = net_ib_mgr_->create_qp();
            if (qp == nullptr) {
                LOG(ERROR, "create_qp failed");
            }
            qps_[srinfo.remote_rank] = qp;
        }
    }

    // Share comm_ib_info with remote ranks.

    std::set<int> remote_ranks;
    for (auto &srinfo : send_recv_infos_) {
        remote_ranks.insert(srinfo.remote_rank);
    }
    for (int remote_rank : remote_ranks) {
        // Set the corresponding QP info.
        comm_ib_info.qp_info = qps_[remote_rank]->get_info();

        std::string item_name = "comm_ib_info_" + std::to_string(remote_rank);
        state = ipc_socket_->add_item(item_name, &comm_ib_info,
                                      sizeof(comm_ib_info));
        if (state != IpcSocket::State::SUCCESS) {
            LOG(ERROR, "Failed to post ", item_name);
        }
    }
    for (int remote_rank : remote_ranks) {
        int remote_gpu_id = remote_rank % num_ranks_per_host;
        int remote_host_id = remote_rank / num_ranks_per_host;
        int port = port_base + remote_gpu_id;

        std::string item_name = "comm_ib_info_" + std::to_string(rank_);

        GpuCommIbInfo remote_comm_ib_info;

        state = ipc_socket_->query_item(get_host(remote_host_id), port,
                                        item_name, &remote_comm_ib_info,
                                        sizeof(remote_comm_ib_info), true);
        if (state != IpcSocket::State::SUCCESS) {
            LOG(ERROR, "Failed to query ", item_name);
        }

        NetIbQp *qp = qps_[remote_rank];
        int ret = qp->rtr(&remote_comm_ib_info.qp_info);
        if (ret != 0) {
            LOG(ERROR, "NetIbQp::rtr failed");
        }
        LOG(DEBUG, "RANK ", rank_, " QP ", qp->get_info().qpn, " <--> RANK ",
            remote_rank, " QP ", remote_comm_ib_info.qp_info.qpn);
        ret = qp->rts();
        if (ret != 0) {
            LOG(ERROR, "NetIbQp::rts failed");
        }

        auto &mri_vec = mris_[remote_rank];
        mri_vec.resize(MAX_NUM_SID);
        for (size_t sid = 0; sid < mri_vec.size(); ++sid) {
            mri_vec[sid] = remote_comm_ib_info.sid_mris[sid];
        }
    }

    // Sync with remote ranks to make sure the QP is ready.
    for (int remote_rank : remote_ranks) {
        int dummy = 0;
        std::string item_name = "comm_ib_done_" + std::to_string(remote_rank);
        state = ipc_socket_->add_item(item_name, &dummy, sizeof(dummy));
        if (state != IpcSocket::State::SUCCESS) {
            LOG(ERROR, "Failed to add ", item_name);
        }
    }
    for (int remote_rank : remote_ranks) {
        int remote_gpu_id = remote_rank % num_ranks_per_host;
        int remote_host_id = remote_rank / num_ranks_per_host;
        int port = port_base + remote_gpu_id;

        int dummy = 0;
        std::string item_name = "comm_ib_done_" + std::to_string(rank_);
        state = ipc_socket_->query_item(get_host(remote_host_id), port,
                                        item_name, &dummy, sizeof(dummy), true);
        if (state != IpcSocket::State::SUCCESS) {
            LOG(ERROR, "Failed to query ", item_name);
        }
    }

    LOG(DEBUG, "RANK ", rank_, " config done");
}

//
void GpuCommSw::Impl::launch_request_loop() {
    if (request_loop_thread_ == nullptr) {
        run_request_loop_thread_ = true;
        request_loop_thread_ = new thread([&, gid = gpu_id_] {
            //
            GpuState ret = get_gpu_mgr(gid)->set_current();
            if (ret == gpuSuccess) {
                //
                this->request_loop();
            } else if (ret != gpuErrorDeinitialized) {
                GLOG(ret);
            }
        });
        assert(request_loop_thread_ != nullptr);
    } else {
        assert(run_request_loop_thread_);
    }
}

//
void GpuCommSw::Impl::request_loop() {
    const size_t sc_offset = 0;
    const size_t rc_offset = MAX_NUM_SID * sizeof(int);

    // Get the local SC/RC host addresses.
    volatile int *sc_href =
        (volatile int *)this->get_sc_rc_mem(gpu_id_)->href(sc_offset);
    assert(sc_href != nullptr);
    volatile int *rc_href =
        (volatile int *)this->get_sc_rc_mem(gpu_id_)->href(rc_offset);
    assert(rc_href != nullptr);

    for (int r = 0; r < (int)qps_.size(); ++r) {
        NetIbQp *qp = qps_[r];
        if (qp != nullptr) {
            int ret = qp->post_recv(((uint64_t)r * MAX_NUM_SID) + 1);
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
    volatile uint64_t *db_val = &(request_->value);
    // Request processing loop.
    while (run_request_loop_thread_) {
        int wcn = 0;
        if (is_using_ib) {
            wcn = net_ib_mgr_->poll_cq();
        }
        if (wcn > 0) {
            for (int i = 0; i < wcn; ++i) {
                int status = net_ib_mgr_->get_wc_status(i);
                if (status != 0) {
                    LOG(ERROR, "get_wc_status() returns ", status, ": ",
                        net_ib_mgr_->get_wc_status_str(i));
                }
                uint64_t wr_id = net_ib_mgr_->get_wc_wr_id(i);
                if (wr_id & 0x1) {
                    // recv complete
                    unsigned int sid_dst = net_ib_mgr_->get_wc_imm_data(i);
                    rc_href[sid_dst] = 1;
                    NetIbQp *qp = qps_[wr_id / MAX_NUM_SID];
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
                    unsigned int sid_src = wr_id / MAX_NUM_SID;
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
        GpuPtr src = addr_table_[gpu_id_][db.fields.sid];
        if (src == 0) {
            LOG(ERROR, "Invalid SRC SID ", db.fields.sid, " in GPU ", gpu_id_);
        }
        REQUEST_DEBUG("Request SRC: RANK ", rank_, ", sid ", db.fields.sid,
                      ", ", (void *)src);
        GpuPtr dst = 0;
        // TODO: generalize converting rank to GPU ID.
        int nrph = get_env().num_ranks_per_host;
        int gid_dst = db.fields.rank % nrph;
        if ((db.fields.rank / nrph) != (rank_ / nrph)) {
            // This GPU is not in this machine.
            gid_dst = -1;
            REQUEST_DEBUG("Request DST: RANK ", db.fields.rank, ", sid ",
                          db.fields.sid, ", remote");
        } else {
            dst = addr_table_[gid_dst][db.fields.sid];
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
            GLOG(gpuMemcpyDtoD(dst, src, (long unsigned int)db.fields.len));
            GpuMem *mem = this->get_sc_rc_mem(db.fields.rank);
            volatile int *rc_array = (volatile int *)mem->href(rc_offset);
            if (rc_array != nullptr) {
                rc_array[db.fields.sid] = 1;
            } else {
                GpuPtr rc_ref =
                    mem->ref(rc_offset + db.fields.sid * sizeof(int));
                GLOG(gpuMemsetD32(rc_ref, 1, 1));
            }
            sc_href[db.fields.sid] = 1;
        } else {
            NetIbQp *qp = qps_[db.fields.rank];
            int ret = qp->stage_send(
                sid_mrs_[db.fields.sid], &mris_[db.fields.rank][db.fields.sid],
                db.fields.len, (db.fields.sid * MAX_NUM_SID), db.fields.sid);
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
void GpuCommSw::Impl::stop_request_loop() {
    run_request_loop_thread_ = false;
    if (request_loop_thread_ != nullptr) {
        if (request_loop_thread_->joinable()) {
            request_loop_thread_->join();
        }
        delete request_loop_thread_;
        request_loop_thread_ = nullptr;
    }
}

//
void GpuCommSw::Impl::set_request(const Request &db) {
    if (request_ != nullptr) {
        *(request_) = db;
    }
}

//
GpuMem *GpuCommSw::Impl::get_data_mem(const int gid) {
    int sz = (int)data_mems_.size();
    assert(sz == (int)sc_rc_mems_.size());
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
    //
    while (addr_table_.size() < data_mems_.size()) {
        addr_table_.emplace_back();
        addr_table_.back().resize(MAX_NUM_SID, 0);
    }
    return dm;
}

//
GpuMem *GpuCommSw::Impl::get_sc_rc_mem(const int gid) {
    int sz = (int)sc_rc_mems_.size();
    assert(sz == (int)sc_rc_mems_.size());
    if (sz <= gid) {
        while (sz <= gid) {
            sz *= 2;
        }
        sc_rc_mems_.resize(sz, nullptr);
    }
    GpuMem *sm = sc_rc_mems_[gid];
    if (sm == nullptr) {
        remote_sc_rc_mems_storage_.emplace_back(std::make_unique<GpuMem>());
        sm = remote_sc_rc_mems_storage_.back().get();
        sc_rc_mems_[gid] = sm;
    }
    return sm;
}

//
GpuPtr GpuCommSw::Impl::get_request_ref() const {
    GpuPtr ref;
    GLOG(gpuHostGetDevicePointer(&ref, request_, 0));
    return ref;
}

GpuCommSw::GpuCommSw(const std::string &name, const int gpu_id_,
                     const int rank_, const int world_size_, GpuMem *data_mem,
                     GpuMem *sc_rc_mem)
    : impl{std::make_unique<GpuCommSw::Impl>(name, gpu_id_, rank_, world_size_,
                                             data_mem, sc_rc_mem)} {}

GpuCommSw::~GpuCommSw() {}

void GpuCommSw::reg_sendrecv(int sid, int remote_rank, std::size_t bytes,
                             bool is_recv) {
    this->impl->reg_sendrecv(sid, remote_rank, bytes, is_recv);
}

void GpuCommSw::configure(
    const std::vector<std::pair<int, size_t>> &export_sid_offs,
    const std::map<int, std::vector<GpuBuf *>> &import_gid_bufs) {
    this->impl->configure(export_sid_offs, import_gid_bufs);
}

void GpuCommSw::launch_request_loop() { this->impl->launch_request_loop(); }

void GpuCommSw::stop_request_loop() { this->impl->stop_request_loop(); }

void GpuCommSw::set_request(const Request &db) { this->impl->set_request(db); }

GpuMem *GpuCommSw::get_data_mem(const int gid) {
    return this->impl->get_data_mem(gid);
}

GpuPtr GpuCommSw::get_request_ref() const {
    return this->impl->get_request_ref();
}

}  // namespace ark
