// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_EXTERNAL_BUFFER_REGISTRY_HPP_
#define ARK_EXTERNAL_BUFFER_REGISTRY_HPP_

#include <unordered_map>

namespace ark {
// Manages externally allocated buffers (buffers corresponding to Tensors that
// are the output of a `placeholder` operation) outside of ARK's memory space.
class ExternalBufferRegistry {
   public:
    static ExternalBufferRegistry &get_instance();

    void set(const size_t id, void *data);

    void *get(const size_t id) const;

    bool has_buffer(const size_t id) const;

    void clear();

   private:
    // Maps buffer IDs to pointers and sizes.
    std::unordered_map<size_t, void *> buffers_;
    ExternalBufferRegistry() {}
    ExternalBufferRegistry(const ExternalBufferRegistry &) = delete;
    ExternalBufferRegistry &operator=(const ExternalBufferRegistry &) = delete;
};
}  // namespace ark

#endif  // ARK_EXTERNAL_BUFFER_REGISTRY_HPP_
