// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_BUFFER_REGISTRY_HPP_
#define ARK_BUFFER_REGISTRY_HPP_

#include <memory>
#include <unordered_map>

namespace ark {

/// Manages addresses of all allocated buffers including externally managed
/// buffers.
class BufferRegistry {
   public:
    struct Info {
        Info(void *data, int device_id, bool is_external)
            : data(data), device_id(device_id), is_external(is_external) {}
        void *data;
        int device_id;
        bool is_external;
    };

    ~BufferRegistry() {}

    static BufferRegistry &get_instance();

    void set(size_t id, void *data, int device_id, bool is_external);

    std::shared_ptr<Info> get(size_t id) const;

   private:
    std::unordered_map<size_t, std::shared_ptr<Info>> buffers_;
    BufferRegistry() {}
    BufferRegistry(const BufferRegistry &) = delete;
    BufferRegistry &operator=(const BufferRegistry &) = delete;
};

}  // namespace ark

#endif  // ARK_BUFFER_REGISTRY_HPP_
