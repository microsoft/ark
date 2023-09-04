// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cassert>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sys/mman.h>
#include <unistd.h>

#include <nvml.h>

#include "env.h"
#include "gpu/gpu_logging.h"
#include "gpu/gpu_mgr.h"
#include "include/ark.h"
#include "math.h"

using namespace std;

namespace ark {

// Initialize APIs.
static void gpu_init()
{
    // Initialize CUDA driver APIs.
    CULOG(cuInit(0));
    // Initialize NVML APIs.
    NVMLLOG(nvmlInit());
}

// Return the number of GPUs in the system.
static int gpu_num()
{
    int n;
    CULOG(cuDeviceGetCount(&n));
    return n;
}

//
void GpuInfo::init(const int gpu_id)
{
    CUdevice dev;
    CULOG(cuDeviceGet(&dev, gpu_id));
    //
    size_t gmem_free;
    CULOG(cuMemGetInfo(&gmem_free, &(this->gmem_total)));
    //
    CULOG(cuDeviceGetAttribute(
        &(this->cc_major), CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
    CULOG(cuDeviceGetAttribute(
        &(this->cc_minor), CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
    CULOG(cuDeviceGetAttribute(&(this->num_sm),
                               CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
    CULOG(cuDeviceGetAttribute(
        &(this->smem_total),
        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, dev));
    CULOG(cuDeviceGetAttribute(
        &(this->smem_block_total),
        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev));
    CULOG(cuDeviceGetAttribute(&(this->clk_rate),
                               CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev));

    this->arch_str = to_string(this->cc_major * 10 + this->cc_minor);
    if (this->arch_str == "60") {
        this->arch = GPU_ARCH_CUDA_60;
    } else if (this->arch_str == "70") {
        this->arch = GPU_ARCH_CUDA_70;
    } else if (this->arch_str == "80") {
        this->arch = GPU_ARCH_CUDA_80;
    } else if (this->arch_str == "90") {
        this->arch = GPU_ARCH_CUDA_90;
    } else {
        this->arch = GPU_ARCH_UNKNOWN;
    }
    // Get PCIe info.
    int pci_domain;
    int pci_bus;
    int pci_device;
    CULOG(cuDeviceGetAttribute(&pci_domain, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID,
                               dev));
    CULOG(cuDeviceGetAttribute(&pci_bus, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, dev));
    CULOG(cuDeviceGetAttribute(&pci_device, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
                               dev));
    stringstream dbsf_s;
    dbsf_s << hex << setfill('0') << setw(4) << pci_domain << ":"
           << setfill('0') << setw(2) << pci_bus << ":" << setfill('0')
           << setw(2) << pci_device << ".0";
    this->dbsf = dbsf_s.str();
}

////////////////////////////////////////////////////////////////////////////////

GpuMgr::GpuMgr(const int gpu_id_) : gpu_id{gpu_id_}
{
    // Create a CUDA context.
    CUdevice dev;
    CULOG(cuDeviceGet(&dev, gpu_id_));
    CULOG(cuCtxCreate(&(this->cuda_ctx), CU_CTX_MAP_HOST, dev));

    gpu_info.init(gpu_id_);
}

//
GpuMgr::~GpuMgr()
{
}

//
GpuMgrCtx *GpuMgr::create_context(const std::string &name, int rank,
                                  int world_size)
{
    for (auto &ctx : this->mgr_ctxs) {
        if (ctx->get_name() == name) {
            LOG(ERROR, "GpuMgrCtx ", name, " already exists.");
        }
    }
    GpuMgrCtx *ctx = new GpuMgrCtx{this, rank, world_size, name};
    this->mgr_ctxs.emplace_back(ctx);
    return ctx;
}

//
void GpuMgr::destroy_context(GpuMgrCtx *ctx)
{
    auto it = this->mgr_ctxs.begin();
    for (; it != this->mgr_ctxs.end(); ++it) {
        if (it->get() == ctx) {
            this->mgr_ctxs.erase(it);
            break;
        }
    }
}

//
void GpuMgr::validate_total_bytes()
{
    size_t total_bytes = 0;
    for (auto &mgr_ctx : this->mgr_ctxs) {
        total_bytes += mgr_ctx->get_total_bytes();
    }
    if (total_bytes > this->gpu_info.gmem_total) {
        LOG(ERROR, "out of GPU memory. Requested ", total_bytes, " bytes");
    }
    LOG(DEBUG, "Requested ", total_bytes, " bytes");
}

//
GpuState GpuMgr::set_current()
{
    return cuCtxSetCurrent(this->cuda_ctx);
}

////////////////////////////////////////////////////////////////////////////////

//
GpuMgrCtx::GpuMgrCtx(GpuMgr *gpu_mgr_, int rank_, int world_size_,
                     const std::string &name_)
    : gpu_mgr{gpu_mgr_}, rank{rank_}, world_size{world_size_}, name{name_},
      data_mem{ARK_GPU_DATA_NAME + name_ + to_string(gpu_mgr_->gpu_id), 0,
               true},
      sc_rc_mem{ARK_GPU_SC_RC_NAME + name_ + to_string(gpu_mgr_->gpu_id),
                2 * MAX_NUM_SID * sizeof(int), true}
{
    // Initialize SCs to ones.
    int *href = (int *)this->sc_rc_mem.href();
    for (int i = 0; i < MAX_NUM_SID; ++i) {
        href[i] = 1;
    }
    // Initialize RCs to zeros.
    for (int i = MAX_NUM_SID; i < 2 * MAX_NUM_SID; ++i) {
        href[i] = 0;
    }
    // Use the CPU-side software communication stack.
    this->comm_sw = std::make_unique<GpuCommSw>(
        name_, gpu_mgr_->gpu_id, rank_, world_size_, &data_mem, &sc_rc_mem);
}

//
GpuMgrCtx::~GpuMgrCtx()
{
    //
    for (GpuStream s : this->streams) {
        cuStreamDestroy(s);
    }
}

//
GpuStream GpuMgrCtx::create_stream()
{
    GpuStream s;
    this->gpu_mgr->set_current();
    CULOG(cuStreamCreate(&s, CU_STREAM_NON_BLOCKING));
    this->streams.emplace_back(s);
    return s;
}

//
GpuState GpuMgrCtx::sync_stream(const GpuStream &s)
{
    return cuStreamSynchronize(s);
}

//
void GpuMgrCtx::destroy_stream(const GpuStream &s)
{
    auto it = this->streams.begin();
    for (; it != this->streams.end(); ++it) {
        if (*it == s) {
            cuStreamDestroy(s);
            this->streams.erase(it);
            break;
        }
    }
}

//
GpuEvent GpuMgrCtx::create_event(bool disable_timing, CUipcEventHandle *handle)
{
    GpuEvent cuda_event;
    unsigned int flags = 0;
    if (disable_timing) {
        flags |= CU_EVENT_DISABLE_TIMING;
    }
    if (handle != nullptr) {
        flags |= CU_EVENT_INTERPROCESS;
    }
    CULOG(cuEventCreate(&cuda_event, flags));
    if (handle != nullptr) {
        CULOG(cuIpcGetEventHandle(handle, cuda_event));
    }
    return cuda_event;
}

//
GpuBuf *GpuMgrCtx::mem_alloc(size_t bytes, int align)
{
    if (bytes == 0) {
        return nullptr;
    }
    int al;
    if (bytes > 32768) {
        al = 65536;
    } else if (bytes > 64) {
        al = 128;
    } else {
        al = 128;
    }
    if (al < align) {
        al = align;
    }
    size_t sz = math::pad(bytes, (size_t)al);
    size_t off;
    int id = this->next_id;
    id_in_use.emplace(this->next_id++);
    //
    std::list<Chunk>::iterator it = this->chunks.begin();
    for (; it != this->chunks.end(); ++it) {
        size_t b = math::pad(it->b, al);
        if ((it->e - b) >= sz) {
            off = b;
            this->usage.emplace_back(off, off + sz);
            if (it->b != b) {
                this->chunks.emplace(it, it->b, b);
            }
            if ((it->e - off) > sz) {
                it->b = off + sz;
            } else {
                this->chunks.erase(it);
            }
            break;
        }
    }
    if (it == this->chunks.end()) {
        // No more segment available.
        // If the last byte is unused, enlarge the last segment.
        // Otherwise, create a new segment.
        if ((this->chunks.size() > 0) &&
            (this->chunks.back().e == this->total_bytes)) {
            Chunk &chunk = this->chunks.back();
            off = math::pad(chunk.b, al);
            if (off != chunk.b) {
                chunk.e = off;
            } else {
                this->chunks.pop_back();
            }
        } else {
            off = math::pad(total_bytes, al);
            if (off != total_bytes) {
                this->chunks.emplace_back(total_bytes, off);
            }
        }
        total_bytes = off + sz;
        this->usage.emplace_back(off, off + sz);
    }
    this->bufs.emplace_back(
        std::make_unique<GpuBuf>(&this->data_mem, id, off, bytes));
    LOG(DEBUG, "GPU Buffer ", this->name, " rank ", this->rank, " ID ", id,
        " off 0x", hex, off, dec, " bytes ", bytes);
    return this->bufs.back().get();
}

//
void GpuMgrCtx::mem_free(GpuBuf *buf)
{
    int id = buf->get_id();
    if ((size_t)id >= this->usage.size()) {
        LOG(ERROR, "GpuBuf ID ", id, " has never been allocated");
    }
    auto search = this->id_in_use.find(id);
    if (search == this->id_in_use.end()) {
        LOG(ERROR, "GpuBuf ID ", id, " is already freed");
    }
    this->id_in_use.erase(search);
    size_t b = this->usage[id].b;
    size_t e = this->usage[id].e;
    if (this->chunks.size() == 0) {
        this->chunks.emplace_back(b, e);
    } else {
        std::list<Chunk>::iterator it = this->chunks.begin();
        for (; it != this->chunks.end(); ++it) {
            if (it->e >= b) {
                if ((it->e == b) && (next(it)->b == e)) {
                    // Merge both sides.
                    it->e = next(it)->e;
                    this->chunks.erase(next(it));
                } else if (it->e == b) {
                    // Merge into left-side.
                    it->e = e;
                } else if (next(it)->b == e) {
                    // Merge into right-side.
                    next(it)->b = b;
                } else {
                    // No merge, just insert.
                    this->chunks.emplace(it, b, e);
                }
                break;
            }
        }
    }
}

//
void GpuMgrCtx::mem_export(GpuBuf *buf, size_t offset, int sid)
{
    // TODO: Check if `buf` is created by this context.
    this->export_sid_offs.emplace_back(sid, buf->get_offset() + offset);
    LOG(DEBUG, "Exported GPU Buffer sid ", sid, " offset ",
        buf->get_offset() + offset, " rank ", this->rank);
}

//
GpuBuf *GpuMgrCtx::mem_import(size_t bytes, int sid, int gid)
{
    GpuMem *dm = this->comm_sw->get_data_mem(gid);
    this->bufs.emplace_back(std::make_unique<GpuBuf>(dm, sid, 0, bytes));
    LOG(DEBUG, "Imported GPU Buffer from GPU ", gid, " sid ", sid, " bytes ",
        bytes);
    GpuBuf *buf = this->bufs.back().get();
    this->import_gid_bufs[gid].emplace_back(buf);

    //
    if (this->data_mem.get_bytes() > 0) {
        // Configuration is already done,
        // so we can import the buffer immediately.
        this->comm_sw->import_buf(gid, buf);
    }
    return buf;
}

void GpuMgrCtx::reg_sendrecv(int sid, int remote_gpu_id, size_t bytes,
                             bool is_recv)
{
    this->comm_sw->reg_sendrecv(sid, remote_gpu_id, bytes, is_recv);
}

//
void GpuMgrCtx::freeze()
{
    //
    this->gpu_mgr->validate_total_bytes();

    //
    if (total_bytes > 0) {
        this->data_mem.alloc(total_bytes);
        // init the data mem
        CULOG(cuMemsetD32(this->data_mem.ref(), 0, total_bytes >> 2));
    }

    //
    this->comm_sw->configure(this->export_sid_offs, this->import_gid_bufs);
    this->comm_sw->launch_request_loop();
}

//
GpuState GpuMgrCtx::set_current()
{
    return this->gpu_mgr->set_current();
}

// Get the host memory address of an SC flag.
volatile int *GpuMgrCtx::get_sc_href(int sid) const
{
    return (volatile int *)this->sc_rc_mem.href(sid * sizeof(int));
}

// Get the host memory address of an RC flag.
volatile int *GpuMgrCtx::get_rc_href(int sid) const
{
    return (volatile int *)this->sc_rc_mem.href((MAX_NUM_SID + sid) *
                                                sizeof(int));
}

//
GpuPtr GpuMgrCtx::get_data_ref(int gid) const
{
    if (gid == -1)
        return this->data_mem.ref();
    return this->comm_sw->get_data_mem(gid)->ref();
}

// Get the GPU memory address of an SC flag.
GpuPtr GpuMgrCtx::get_sc_ref(int sid) const
{
    return this->sc_rc_mem.ref(sid * sizeof(int));
}

// Get the GPU memory address of an RC flag.
GpuPtr GpuMgrCtx::get_rc_ref(int sid) const
{
    return this->sc_rc_mem.ref((MAX_NUM_SID + sid) * sizeof(int));
}

//
GpuPtr GpuMgrCtx::get_request_ref() const
{
    return this->comm_sw->get_request_ref();
}

//
GpuCommSw *GpuMgrCtx::get_comm_sw() const
{
    return this->comm_sw.get();
}

////////////////////////////////////////////////////////////////////////////////

// Global GpuMgr vector.
vector<unique_ptr<GpuMgr>> ARK_GPU_MGR_GLOBAL;

// Return a pointer to a global GpuMgr.
GpuMgr *get_gpu_mgr(const int gpu_id)
{
    if (gpu_id < 0) {
        LOG(ERROR, "invalid GPU ID ", gpu_id);
    }
    if (ARK_GPU_MGR_GLOBAL.size() == 0) {
        gpu_init();
        int ngpu = gpu_num();
        if (ngpu <= 0) {
            LOG(ERROR, "No CUDA-capable GPU is detected.");
        }
        ARK_GPU_MGR_GLOBAL.resize(ngpu);
    }
    if ((unsigned int)gpu_id >= ARK_GPU_MGR_GLOBAL.size()) {
        LOG(ERROR, "invalid GPU ID ", gpu_id);
    }
    GpuMgr *mgr = ARK_GPU_MGR_GLOBAL[gpu_id].get();
    if (mgr == nullptr) {
        mgr = new GpuMgr{gpu_id};
        assert(mgr != nullptr);
        ARK_GPU_MGR_GLOBAL[gpu_id].reset(mgr);
    }
    return mgr;
}

void gpu_memset(GpuPtr buf, int val, size_t num)
{
    CULOG(cuMemsetD32(buf, val, num));
}
void gpu_memcpy(GpuPtr dst, const void *src, size_t bytes)
{
    CULOG(cuMemcpyHtoD(dst, src, bytes));
}
void gpu_memcpy(void *dst, const GpuPtr src, size_t bytes)
{
    CULOG(cuMemcpyDtoH(dst, src, bytes));
}
void gpu_memset(GpuBuf *buf, int val, size_t num)
{
    const size_t &bytes = buf->get_bytes();
    assert(bytes >= 4);
    if ((bytes >> 2) < num) {
        LOG(ERROR,
            "memset requests too many elements. Expected <= ", bytes >> 2,
            ", given ", num);
    }
    GpuPtr pb = buf->ref();
    if (pb != 0) {
        assert((pb % 4) == 0);
        CULOG(cuMemsetD32(pb, val, num));
    } else {
        int *phb = (int *)buf->href();
        assert(phb != nullptr);
        for (size_t i = 0; i < num; ++i) {
            phb[i] = val;
        }
    }
}
void gpu_memcpy(GpuBuf *dst, const void *src, size_t bytes)
{
    CULOG(cuMemcpyHtoD(dst->ref(), src, bytes));
}
void gpu_memcpy(void *dst, const GpuBuf *src, size_t bytes)
{
    CULOG(cuMemcpyDtoH(dst, src->ref(), bytes));
}
void gpu_memcpy(GpuBuf *dst, const GpuBuf *src, size_t bytes)
{
    GpuPtr rd = dst->ref();
    GpuPtr rs = src->ref();
    if ((rd != 0) && (rs != 0)) {
        CULOG(cuMemcpyDtoD(dst->ref(), src->ref(), bytes));
    } else if (rd != 0) {
        CULOG(cuMemcpyHtoD(dst->ref(), src->href(), bytes));
    } else if (rs != 0) {
        CULOG(cuMemcpyDtoH(dst->href(), src->ref(), bytes));
    } else {
        // ::memcpy(dst->href(), src->href(), bytes);
        LOG(ERROR, "Unexpected case.");
    }
}

} // namespace ark
