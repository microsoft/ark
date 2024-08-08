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

    const std::unordered_map<size_t, std::tuple<void*, size_t>>& get_buffers()
        const {
        return buffers_;
    }

    void clear_buffers() { buffers_.clear(); }

    bool is_empty() const { return buffers_.empty(); }

   private:
    // Maps buffer IDs to pointers and sizes.
    std::unordered_map<size_t, std::tuple<void*, size_t>> buffers_;
    ModelBufferManager() {}
    ModelBufferManager(const ModelBufferManager&) = delete;
    ModelBufferManager& operator=(const ModelBufferManager&) = delete;
};
}  // namespace ark

#endif  // ARK_MODEL_BUFFER_MANAGER_HPP_
