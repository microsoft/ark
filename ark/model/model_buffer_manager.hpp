// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MODEL_BUFFER_MANAGER_HPP
#define MODEL_BUFFER_MANAGER_HPP

#include <tuple>
#include <unordered_map>

namespace ark {
/**
 * @brief Manages externally allocated buffers not in the ARK memory space.
 *
 * Details:
 * - `buffers_`: Maps external buffer IDs to their pointers and sizes.
 * - `externalIdMap_`: Maps external buffer IDs to their corresponding compact
 *   IDs. During the code generation phase, an array of buffer addresses
 *   (ARK_EXTERNAL_BUFFERS) is preallocated to hold addresses of external
 * buffers. Accessing an external buffer utilizes its compact ID to index into
 * this array, ensuring that each index *i* in ARK_EXTERNAL_BUFFERS corresponds
 * to the buffer with compact ID *i*, facilitating mixed allocation patterns
 *   (e.g., internal, internal, external, internal).
 */
class ModelBufferManager {
   public:
    static ModelBufferManager& getInstance() {
        static ModelBufferManager instance;
        return instance;
    }

    void registerBuffer(size_t id, void* data, size_t size) {
        buffers_[id] = std::make_tuple(data, size);
        if (externalIdMap_.find(id) == externalIdMap_.end()) {
            externalIdMap_[id] = nextCompactId_++;
        }
    }

    void* getBuffer(size_t id) {
        auto it = buffers_.find(id);
        if (it != buffers_.end()) {
            return std::get<0>(it->second);
        }
        return nullptr;
    }

    size_t getBufferSize(size_t id) {
        auto it = buffers_.find(id);
        if (it != buffers_.end()) {
            return std::get<1>(it->second);
        }
        return 0;
    }

    size_t getCompactIdSize() { return nextCompactId_; }

    size_t getCompactId(size_t id) {
        auto it = externalIdMap_.find(id);
        if (it != externalIdMap_.end()) {
            return it->second;
        }
        return 0;
    }

    const std::unordered_map<size_t, std::tuple<void*, size_t>>& getBuffers()
        const {
        return buffers_;
    }

    void clearBuffers() {
        buffers_.clear();
        externalIdMap_.clear();
        nextCompactId_ = 0;
    }

    bool isEmpty() const { return buffers_.empty(); }

   private:
    std::unordered_map<size_t, std::tuple<void*, size_t>>
        buffers_;  // Maps buffer IDs to pointers and sizes.
    std::unordered_map<size_t, size_t>
        externalIdMap_;  // Maps original buffer IDs to compact IDs for external
                         // buffers.
    size_t nextCompactId_ = 0;
    ModelBufferManager() {}
    ModelBufferManager(const ModelBufferManager&) = delete;
    ModelBufferManager& operator=(const ModelBufferManager&) = delete;
};
}  // namespace ark

#endif  // MODEL_BUFFER_MANAGER_HPP