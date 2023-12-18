// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_GPU_COMM_SW_H_
#define ARK_GPU_COMM_SW_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "gpu/gpu_buffer.h"
#include "gpu/gpu_memory.h"
#include "gpu_buf.h"
#include "gpu_common.h"
#include "gpu_mem.h"

namespace ark {

class GpuBuf;

//
class GpuCommSw {
   public:
    GpuCommSw(const std::string &name, const int gpu_id_, const int rank_,
              const int world_size_, GpuMem *data_mem);
    GpuCommSw(const std::string &name, const int gpu_id, const int rank,
              const int world_size, std::shared_ptr<GpuMemory> data_mem);
    ~GpuCommSw();

    void configure(const std::vector<std::pair<int, size_t>> &export_sid_offs,
                   const std::map<int, std::vector<GpuBuf *>> &import_gid_bufs);

    void configure(
        const std::vector<std::pair<int, size_t>> &export_sid_offs,
        const std::unordered_map<int, std::vector<std::shared_ptr<GpuBuffer>>>
            &import_gid_bufs);

    void launch_request_loop();

    void stop_request_loop();

    GpuMem *get_data_mem(const int gid);

    std::shared_ptr<GpuMemory> get_data_memory(const int gid);

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
