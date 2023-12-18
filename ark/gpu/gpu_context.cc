#include "gpu/gpu_context.h"

#include <cassert>
#include <list>
#include <set>
#include <unordered_map>

#include "gpu/gpu_comm_sw.h"
#include "gpu/gpu_logging.h"
#include "gpu/gpu_manager.h"
#include "gpu_context.h"
#include "math.h"

namespace {
constexpr size_t GPU_PAGE_SHIFT = 16;
constexpr size_t GPU_PAGE_SIZE = (1ULL << GPU_PAGE_SHIFT);

int get_align(size_t bytes, int align) {
    if (bytes == 0) {
        return 0;
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
    return al;
}
}  // namespace

namespace ark {

struct Chunk {
    Chunk(size_t begin, size_t end) : begin(begin), end(end) {}
    size_t begin;
    size_t end;
};

class GpuContext::Impl {
   public:
    Impl(int rank, int world_size);
    ~Impl() = default;
    Impl(const Impl &) = delete;
    Impl &operator=(const Impl &) = delete;

    std::shared_ptr<GpuBuffer> allocate_buffer(size_t bytes, int align = 1);
    void free_buffer(std::shared_ptr<GpuBuffer> buffer);
    void export_buffer(std::shared_ptr<GpuBuffer> buffer, size_t offset,
                       int expose_id);
    std::shared_ptr<GpuBuffer> import_buffer(size_t bytes, int gpu_id,
                                             int expose_id);
    void freeze(bool expose);
    void memset(std::shared_ptr<GpuBuffer> buffer, size_t offset, int value,
                size_t bytes);

   private:
    friend class GpuContext;
    std::shared_ptr<GpuManager> manager_;
    std::shared_ptr<GpuCommSw> comm_sw_;
    std::shared_ptr<GpuMemory> memory_;
    std::list<Chunk> chunks_;
    std::set<int> id_in_use_;
    std::unordered_map<int, Chunk> in_use_chunks_;
    std::vector<std::pair<int, size_t>> export_id_offsets_;
    std::unordered_map<int, std::vector<std::shared_ptr<GpuBuffer>>>
        import_gid_buffers_;
    int rank_;
    int world_size_;
    unsigned int next_id_ = 0;
    size_t total_bytes_ = 0;
};

GpuContext::Impl::Impl(int rank, int world_size)
    : manager_(GpuManager::get_instance(rank)),
      rank_(rank),
      world_size_(world_size) {
    // TODO(binyli): refactor GpuCommSw later.
    memory_ = manager_->malloc(0, GPU_PAGE_SIZE);
    comm_sw_ = std::make_shared<GpuCommSw>("comm_sw", manager_->get_gpu_id(),
                                           rank_, world_size_, memory_);
}

std::shared_ptr<GpuBuffer> GpuContext::Impl::allocate_buffer(size_t bytes,
                                                             int align) {
    if (bytes == 0) {
        return nullptr;
    }
    int real_align = get_align(bytes, align);
    size_t size = math::pad(bytes, (size_t)real_align);
    size_t offset;
    int id = this->next_id_;
    id_in_use_.insert(this->next_id_++);
    std::list<Chunk>::iterator it = this->chunks_.begin();
    for (; it != this->chunks_.end(); ++it) {
        size_t begin = math::pad(it->begin, real_align);
        if ((it->end - begin) >= size) {
            offset = begin;
            this->in_use_chunks_.emplace(id, Chunk(offset, offset + size));
            if (it->begin != begin) {
                this->chunks_.emplace(it, it->begin, begin);
            }
            if ((it->end - offset) > size) {
                it->begin = offset + size;
            } else {
                this->chunks_.erase(it);
            }
            break;
        }
    }
    if (it == this->chunks_.end()) {
        // No more segment available.
        // If the last byte is unused, enlarge the last segment.
        // Otherwise, create a new segment.
        if ((this->chunks_.size() > 0) &&
            (this->chunks_.back().end == this->total_bytes_)) {
            Chunk &chunk = this->chunks_.back();
            offset = math::pad(chunk.begin, real_align);
            if (offset != chunk.begin) {
                chunk.end = offset;
            } else {
                this->chunks_.pop_back();
            }
        } else {
            offset = math::pad(total_bytes_, real_align);
            if (offset != total_bytes_) {
                this->chunks_.emplace_back(total_bytes_, offset);
            }
        }
        total_bytes_ = offset + size;
        this->in_use_chunks_.emplace(id, Chunk(offset, offset + size));
    }
    // this->bufs.emplace_back(std::make_unique<GpuBuf>(
    //     this->gpu_mgr->gpu_id, &this->data_mem, id, off, bytes));
    LOG(DEBUG, "Allocated ", bytes, " bytes of GPU memory at offset ", offset,
        " rank ", rank_);
    return std::make_shared<GpuBuffer>(manager_->get_gpu_id(), memory_, id,
                                       offset, bytes);
}

// TODO: maybe this function can be called by GpuBuffer's destructor.
void GpuContext::Impl::free_buffer(std::shared_ptr<GpuBuffer> buffer) {
    int id = buffer->get_id();
    if (id_in_use_.find(id) == id_in_use_.end()) {
        ERR(ExecutorError, "Cannot free buffer ", id,
            " because it is not in use");
    }
    id_in_use_.erase(id);
    auto it = in_use_chunks_.find(id);
    if (it == in_use_chunks_.end()) {
        ERR(ExecutorError, "Cannot free buffer ", id, " no chunk found");
    }
    Chunk &chunk = it->second;
    auto it2 = chunks_.begin();
    for (; it2 != chunks_.end(); ++it2) {
        if (it2->end >= chunk.begin) {
            if (it2->end == chunk.begin && std::next(it2) != chunks_.end() &&
                std::next(it2)->begin == chunk.end) {
                it2->end = std::next(it2)->end;
            } else if (it2->begin == chunk.end) {
                it2->begin = chunk.begin;
            } else if (it2->end == chunk.begin) {
                it2->end = chunk.end;
            } else {
                chunks_.emplace(it2, chunk.begin, chunk.end);
            }
        }
        break;
    }
    if (it2 == chunks_.end()) {
        chunks_.emplace_back(chunk.begin, chunk.end);
    }
    in_use_chunks_.erase(it);
    LOG(DEBUG, "Freed buffer ", id, " offset ", buffer->get_offset(), " rank ",
        rank_);
}

void GpuContext::Impl::export_buffer(std::shared_ptr<GpuBuffer> buffer,
                                     size_t offset, int expose_id) {
    if (expose_id < 0 || expose_id >= MAX_NUM_SID) {
        ERR(ExecutorError, "Invalid expose id ", expose_id);
    }
    this->export_id_offsets_.emplace_back(expose_id,
                                          buffer->get_offset() + offset);
}

std::shared_ptr<GpuBuffer> GpuContext::Impl::import_buffer(size_t bytes,
                                                           int gpu_id,
                                                           int expose_id) {
    if (expose_id < 0 || expose_id >= MAX_NUM_SID) {
        ERR(ExecutorError, "Invalid expose id ", expose_id);
    }

    std::shared_ptr<GpuMemory> memory = comm_sw_->get_data_memory(gpu_id);
    std::shared_ptr<GpuBuffer> buffer =
        std::make_shared<GpuBuffer>(gpu_id, memory, expose_id, 0, bytes);
    import_gid_buffers_[gpu_id].emplace_back(buffer);
    assert(memory_->bytes() == 0);
    return buffer;
}

void GpuContext::Impl::freeze(bool expose) {
    if (total_bytes_ > manager_->info().gmem_total) {
        ERR(SystemError, "out of GPU memory. Requested ", total_bytes_,
            " bytes, available ", manager_->info().gmem_total, " bytes");
    }
    if (total_bytes_ > 0) {
        LOG(INFO, "Allocating ", total_bytes_, " bytes of GPU memory");
        memory_->resize(total_bytes_, expose);
    }
    if (expose) {
        comm_sw_->configure(export_id_offsets_, import_gid_buffers_);
    }
    comm_sw_->launch_request_loop();
}

void GpuContext::Impl::memset(std::shared_ptr<GpuBuffer> buffer, size_t offset,
                              int value, size_t bytes) {
    const size_t &buffer_bytes = buffer->get_bytes();
    assert(buffer_bytes >= 4);
    if (buffer_bytes < bytes) {
        ERR(InvalidUsageError,
            "memset requests too many elements. Expected <= ", buffer_bytes,
            ", given ", bytes);
    }
    GpuPtr ptr = buffer->ref(offset);
    if (ptr == reinterpret_cast<GpuPtr>(nullptr)) {
        ERR(InvalidUsageError, "buffer is not allocated");
    }
    manager_->memset_d32_sync(reinterpret_cast<void *>(ptr), value, bytes);
}

std::shared_ptr<GpuContext> GpuContext::get_context(int rank, int world_size) {
    static std::shared_ptr<GpuContext> context =
        std::shared_ptr<GpuContext>(new GpuContext(rank, world_size));
    return context;
}

GpuContext::GpuContext(int rank, int world_size)
    : pimpl_(new Impl(rank, world_size)) {}

int GpuContext::rank() const { return pimpl_->rank_; }

int GpuContext::world_size() const { return pimpl_->world_size_; }

std::shared_ptr<GpuBuffer> GpuContext::allocate_buffer(size_t bytes,
                                                       int align) {
    return pimpl_->allocate_buffer(bytes, align);
}

void GpuContext::free_buffer(std::shared_ptr<GpuBuffer> buffer) {
    pimpl_->free_buffer(buffer);
}

void GpuContext::export_buffer(std::shared_ptr<GpuBuffer> buffer, size_t offset,
                               int expose_id) {
    pimpl_->export_buffer(buffer, offset, expose_id);
}

std::shared_ptr<GpuBuffer> GpuContext::import_buffer(size_t bytes, int gpu_id,
                                                     int expose_id) {
    return pimpl_->import_buffer(bytes, gpu_id, expose_id);
}

void GpuContext::freeze(bool expose) { pimpl_->freeze(expose); }

void GpuContext::memset(std::shared_ptr<GpuBuffer> buffer, size_t offset,
                        int value, size_t bytes) {
    pimpl_->memset(buffer, offset, value, bytes);
}

}  // namespace ark
