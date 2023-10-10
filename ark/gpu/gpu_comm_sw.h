// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_GPU_COMM_SW_H_
#define ARK_GPU_COMM_SW_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "gpu_buf.h"
#include "gpu_common.h"
#include "gpu_mem.h"

namespace ark {

class GpuBuf;

//
class GpuCommSw {
   public:
    GpuCommSw(const std::string &name, const int gpu_id_, const int rank_,
              const int world_size_, GpuMem *data_mem, GpuMem *sc_rc_mem);
    ~GpuCommSw();

    void reg_sendrecv(int sid, int remote_rank, std::size_t bytes,
                      bool is_recv);
    void configure(const std::vector<std::pair<int, size_t>> &export_sid_offs,
                   const std::map<int, std::vector<GpuBuf *>> &import_gid_bufs);

    void launch_request_loop();

    void stop_request_loop();

    void set_request(const Request &db);

    GpuMem *get_data_mem(const int gid);

    GpuPtr get_request_ref() const;

    bool is_using_ib() const;

    const void *get_proxy_channels_ref() const;

    int get_proxy_channels_bytes() const;

    int get_proxy_channels_num() const;

    int get_sm_channels_num() const;

    const void *get_sm_channels_ref() const;

    int get_sm_channels_bytes() const;

  protected:
    class Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace ark

#endif  // ARK_GPU_COMM_SW_H_
