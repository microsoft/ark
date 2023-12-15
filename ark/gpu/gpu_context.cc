#include "gpu/gpu_context.h"

#include <list>
#include <set>
#include <unordered_map>

#include "gpu/gpu_comm_sw.h"
#include "gpu/gpu_logging.h"
#include "gpu/gpu_manager.h"
#include "gpu_context.h"
#include "math.h"

namespace {
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
    Impl(int rank, int world_size) {}
    ~Impl() = default;
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    std::shared_ptr<GpuBuffer> allocate_buffer(size_t bytes, int align = 1);
    void free_buffer(std::shared_ptr<GpuBuffer> buffer);
    void export_buffer(std::shared_ptr<GpuBuffer> buffer, size_t offset, int expose_id);
    std::shared_ptr<GpuBuffer> import_memory(size_t bytes, int gpu_id, int expose_id);
    void freeze(bool expose);

   private:
    friend class GpuContext;
    std::shared_ptr<GpuManager> manager_;
    std::shared_ptr<GpuCommSw> comm_sw_;
    std::shared_ptr<GpuMemory> memory_;
    std::list<Chunk> chunks_;
    std::set<int> id_in_use_;
    std::unordered_map<int, Chunk> in_use_chunks_;
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
    comm_sw_ = std::make_shared<GpuCommSw>("comm_sw", manager_->get_gpu_id(),
                                           rank_, world_size_);
}

std::shared_ptr<GpuBuffer> GpuContext::Impl::allocate_buffer(size_t bytes, int align) {
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
            this->in_use_chunks_[id] = Chunk(offset, offset + size);
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
        this->in_use_chunks_[id] = Chunk(offset, offset + size);
    }
    // this->bufs.emplace_back(std::make_unique<GpuBuf>(
    //     this->gpu_mgr->gpu_id, &this->data_mem, id, off, bytes));
    LOG(DEBUG, "Allocated ", bytes, " bytes of GPU memory at offset ", offset,
        " rank ", rank);
    return std::make_shared<GpuBuffer>(manager_->get_gpu_id(), memory_, id,
                                       offset, bytes);
}

// TODO: maybe this function can be called by GpuBuffer's destructor.
void GpuContext::Impl::free_buffer(std::shared_ptr<GpuBuffer> buffer) {
    int id = buffer->get_id();
    if (id_in_use_.find(id) == id_in_use_.end()) {
        LOG(ERROR, "Cannot free buffer ", id, " because it is not in use");
        return;
    }
    id_in_use_.erase(id);
    auto it = in_use_chunks_.find(id);
    if (it == in_use_chunks_.end()) {
        LOG(ERROR, "Cannot free buffer ", id, " because it is not in use");
        return;
    }
    Chunk &chunk = it->second;
    auto it2 = chunks_.begin();
    for (; it2 != chunks_.end(); ++it2) {
        if (it2->begin == chunk.end) {
            chunk.end = it2->end;
            chunks_.erase(it2);
            break;
        } else if (it2->end == chunk.begin) {
            chunk.begin = it2->begin;
            chunks_.erase(it2);
            break;
        }
    }
    if (it2 == chunks_.end()) {
        chunks_.emplace_back(chunk.begin, chunk.end);
    }
    in_use_chunks_.erase(it);
    LOG(DEBUG, "Freed buffer ", id, " rank ", rank_);
}

std::shared_ptr<GpuContext> GpuContext::get_context(int rank, int world_size) {
    static std::shared_ptr<GpuContext> context =
        std::make_shared<GpuContext>(rank, world_size);
    return context;
}

GpuContext::GpuContext(int rank, int world_size) : pimpl_(new Impl(rank, world_size)) {}

int GpuContext::rank() const { return pimpl_->rank_; }

int GpuContext::world_size() const { return pimpl_->world_size_; }


}  // namespace ark
