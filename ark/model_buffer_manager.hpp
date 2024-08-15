// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_MODEL_BUFFER_MANAGER_HPP_
#define ARK_MODEL_BUFFER_MANAGER_HPP_

#include <tuple>
#include <unordered_map>

namespace ark {
// Manages externally allocated buffers (buffers corresponding to Tensors that
// are the output of a `placeholder` operation) outside of ARK's memory space.
class ModelBufferManager {
   public:
    static ModelBufferManager& get_instance() {
        static ModelBufferManager instance;
        return instance;
    }

    void register_buffer(const size_t id, void* const data, const size_t size) {
        buffers_[id] = std::make_tuple(data, size);
    }

    void* get_buffer(const size_t id) const {
        auto it = buffers_.find(id);
        if (it != buffers_.end()) {
            return std::get<0>(it->second);
        }
        return nullptr;
    }

    size_t get_buffer_size(const size_t id) const {
        auto it = buffers_.find(id);
        if (it != buffers_.end()) {
            return std::get<1>(it->second);
        }
        return 0;
    }

    bool is_external(const size_t id) const {
        return buffers_.find(id) != buffers_.end();
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
