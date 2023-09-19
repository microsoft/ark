// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_GPU_MGR_H_
#define ARK_GPU_MGR_H_

#include <cuda.h>
#include <list>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "gpu/gpu_comm_sw.h"

namespace ark {

// Types of GPU architectures.
typedef enum
{
    GPU_ARCH_UNKNOWN = -1,
    GPU_ARCH_CUDA_60,
    GPU_ARCH_CUDA_70,
    GPU_ARCH_CUDA_80,
    GPU_ARCH_CUDA_90,
} GpuArchType;

// Details of a GPU device.
struct GpuInfo
{
    // Constructor.
    void init(const int gpu_id);

    int cc_major;
    int cc_minor;
    size_t gmem_total;
    int smem_total;
    int smem_block_total;
    int num_sm;
    int clk_rate;
    GpuArchType arch;
    std::string arch_str;
    // PCIe "Domain:Bus:Slot.Function"
    std::string dbsf;
    //
    const unsigned int threads_per_warp = 32;
    const unsigned int max_registers_per_thread = 256;
    const unsigned int max_registers_per_block = 65536;
    const unsigned int min_threads_per_block =
        max_registers_per_block / max_registers_per_thread;
    const unsigned int max_threads_per_block = 1024;
    const unsigned int smem_align = 128;
};

////////////////////////////////////////////////////////////////////////////////

//
typedef CUstream GpuStream;
typedef CUevent GpuEvent;

//
class GpuMgrCtx;

//
class GpuMgr
{
  public:
    GpuMgr(const int gpu_id);
    ~GpuMgr();

    GpuMgrCtx *create_context(const std::string &name, int rank,
                              int world_size);
    void destroy_context(GpuMgrCtx *ctx);

    void validate_total_bytes();

    GpuState set_current();

    const GpuInfo &get_gpu_info() const
    {
        return gpu_info;
    }

    //
    const int gpu_id;

  private:
    //
    GpuInfo gpu_info;
    // CUDA context of this GPU.
    CUcontext cuda_ctx;
    //
    std::list<std::unique_ptr<GpuMgrCtx>> mgr_ctxs;
};

//
class GpuMgrCtx
{
  public:
    GpuMgrCtx(GpuMgr *gpu_mgr, int rank_, int world_size_,
              const std::string &name);
    ~GpuMgrCtx();

    GpuStream create_stream();
    GpuState sync_stream(const GpuStream &s);
    void destroy_stream(const GpuStream &s);
    GpuEvent create_event(bool disable_timing);

    //
    GpuBuf *mem_alloc(size_t bytes, int align = 1);
    void mem_free(GpuBuf *buf);
    void mem_export(GpuBuf *buf, size_t offset, int sid);
    GpuBuf *mem_import(size_t bytes, int sid, int gpu_id);
    void reg_sendrecv(int sid, int gpu_dst, std::size_t bytes, bool is_recv);
    void freeze(bool expose = false);
    // void send(int sid, int rank, size_t bytes);
    GpuState set_current();
    int get_world_size() const
    {
        return world_size;
    }
    int get_rank() const
    {
        return rank;
    }
    int get_gpu_id() const
    {
        return gpu_mgr->gpu_id;
    }
    const std::string &get_name() const
    {
        return name;
    }
    size_t get_total_bytes() const
    {
        return total_bytes;
    }
    // Get the host memory address of an SC flag.
    volatile int *get_sc_href(int sid) const;
    // Get the host memory address of an RC flag.
    volatile int *get_rc_href(int sid) const;
    // Get the GPU memory address of a GPU data buffer.
    GpuPtr get_data_ref(int gid = -1) const;
    // Get the GPU memory address of an SC flag.
    GpuPtr get_sc_ref(int sid) const;
    // Get the GPU memory address of an RC flag.
    GpuPtr get_rc_ref(int sid) const;
    //
    GpuPtr get_request_ref() const;
    //
    GpuCommSw *get_comm_sw() const;

  private:
    //
    struct Chunk
    {
        Chunk(size_t b_, size_t e_) : b{b_}, e{e_}
        {
        }
        size_t b;
        size_t e;
    };

    int next_id = 0;
    size_t total_bytes = 0;
    std::list<Chunk> chunks;
    std::vector<Chunk> usage;
    std::set<int> id_in_use;

    GpuMgr *gpu_mgr;
    const int rank;
    const int world_size;
    std::string name;
    GpuMem data_mem;
    GpuMem sc_rc_mem;

    //
    std::vector<GpuStream> streams;

    std::list<std::unique_ptr<GpuBuf>> bufs;

    std::vector<std::pair<int, size_t>> export_sid_offs;
    std::map<int, std::vector<GpuBuf *>> import_gid_bufs;

    std::unique_ptr<GpuCommSw> comm_sw;

    std::set<int> sids_in_use;
};

//
GpuMgr *get_gpu_mgr(const int gpu_id);

//
void gpu_memset(GpuPtr buf, int val, size_t num);
void gpu_memcpy(GpuPtr dst, const void *src, size_t bytes);
void gpu_memcpy(void *dst, const GpuPtr src, size_t bytes);
void gpu_memset(GpuBuf *buf, int val, size_t num);
void gpu_memcpy(GpuBuf *dst, const void *src, size_t bytes);
void gpu_memcpy(void *dst, const GpuBuf *src, size_t bytes);
void gpu_memcpy(GpuBuf *dst, const GpuBuf *src, size_t bytes);

} // namespace ark

#endif // ARK_GPU_MGR_H_
