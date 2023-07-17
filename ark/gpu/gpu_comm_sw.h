// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_GPU_COMM_SW_H_
#define ARK_GPU_COMM_SW_H_

#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <vector>

#include "gpu/gpu_buf.h"
#include "gpu/gpu_common.h"
#include "ipc/ipc_socket.h"
#include "net/net_ib.h"

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

//
class GpuCommSw
{
  public:
    GpuCommSw(const std::string &name, const int gpu_id_, const int rank_,
              const int world_size_, GpuMem *data_mem, GpuMem *sc_rc_mem);
    ~GpuCommSw();

    void reg_sendrecv(int sid, int remote_rank, std::size_t bytes,
                      bool is_recv);
    void configure(std::vector<std::pair<int, size_t>> &export_sid_offs,
                   std::map<int, std::vector<GpuBuf *>> &import_gid_bufs);
    void import_buf(const int gid, GpuBuf *buf);

    void doorbell_loop();
    void launch_doorbell_loop();
    void stop_doorbell_loop();

    void set_doorbell(const Doorbell &db);

    GpuMem *get_data_mem(const int gid);
    GpuMem *get_sc_rc_mem(const int gid);
    IpcMem *get_info(const int gid);
    GpuPtr get_doorbell_ref() const;
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
    Doorbell *doorbell = nullptr;
    std::thread *doorbell_loop_thread = nullptr;
    volatile bool run_doorbell_loop_thread = false;
    //
    IpcSocket *ipc_socket = nullptr;
    //
    NetIbMgr *net_ib_mgr = nullptr;
    std::vector<NetIbMr *> sid_mrs;
    std::map<int, NetIbQp *> qps;
    std::vector<GpuSendRecvInfo> send_recv_infos;
    std::map<int, std::vector<NetIbMr::Info>> mris;
};

} // namespace ark

#endif // ARK_GPU_COMM_SW_H_
