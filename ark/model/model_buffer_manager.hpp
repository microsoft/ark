// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_MODEL_BUFFER_MANAGER_HPP_
#define ARK_MODEL_BUFFER_MANAGER_HPP_

#include <tuple>
#include <unordered_map>

namespace ark {
// Manages externally allocated buffers not in the ARK memory space.
class ModelBufferManager {
   public:
    static ModelBufferManager& get_instance() {
        static ModelBufferManager instance;
        return instance;
    }

    void register_buffer(size_t id, void* data, size_t size) {
        buffers_[id] = std::make_tuple(data, size);
        if (external_id_map_.find(id) == external_id_map_.end()) {
            external_id_map_[id] = next_compact_id_++;
        }
    }

    void* get_buffer(size_t id) {
        auto it = buffers_.find(id);
        if (it != buffers_.end()) {
            return std::get<0>(it->second);
        }
        return nullptr;
    }

    size_t get_buffer_size(size_t id) {
        auto it = buffers_.find(id);
        if (it != buffers_.end()) {
            return std::get<1>(it->second);
        }
        return 0;
    }

    size_t get_compact_id_size() { return next_compact_id_; }

    size_t get_compact_id(size_t id) {
        auto it = external_id_map_.find(id);
        if (it != external_id_map_.end()) {
            return it->second;
        }
        return 0;
    }

    const std::unordered_map<size_t, std::tuple<void*, size_t>>& get_buffers()
        const {
        return buffers_;
    }

    void clear_buffers() {
        buffers_.clear();
        external_id_map_.clear();
        next_compact_id_ = 0;
    }

    bool is_empty() const { return buffers_.empty(); }

   private:
    std::unordered_map<size_t, std::tuple<void*, size_t>>
        buffers_;  // Maps buffer IDs to pointers and sizes.
    std::unordered_map<size_t, size_t>
        external_id_map_;  // Maps original buffer IDs to compact IDs for
                           // external buffers.
    size_t next_compact_id_ = 0;
    ModelBufferManager() {}
    ModelBufferManager(const ModelBufferManager&) = delete;
    ModelBufferManager& operator=(const ModelBufferManager&) = delete;
};
}  // namespace ark

#endif  // ARK_MODEL_BUFFER_MANAGER_HPP_
