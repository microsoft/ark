// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_MODEL_BUFFER_HPP_
#define ARK_MODEL_BUFFER_HPP_

#include <memory>
#include <set>
#include <vector>

#include "model_json.hpp"

namespace ark {

class ModelBuffer {
   public:
    // (remote_rank, tag)
    using TagInfo = std::pair<int, int>;

    ModelBuffer(int rank = -1);

    ModelBuffer(size_t id, int rank, const std::vector<TagInfo> &send_tags,
                const std::vector<TagInfo> &recv_tags);

    // externally managed buffer
    ModelBuffer(void *data, size_t size, int32_t device_id);
    ModelBuffer(size_t id, void *data, size_t size, int32_t device_id);

    size_t id() const { return id_; }

    int rank() const { return rank_; }

    const std::set<TagInfo> &send_tags() const { return send_tags_; }

    const std::set<TagInfo> &recv_tags() const { return recv_tags_; }

    // Identify this buffer as `tag` when sending data to `remote_rank`.
    // The same buffer can be tagged multiple times with different tags,
    // but the same tag can only be used for one sending buffer.
    void tag_send(int remote_rank, int tag);

    // Identify this buffer as `tag` when receiving from `remote_rank`.
    // The same buffer can be tagged multiple times with different tags,
    // but the same tag can only be used for one receiving buffer.
    void tag_recv(int remote_rank, int tag);

    Json serialize() const;

    static std::shared_ptr<ModelBuffer> deserialize(const Json &serialized);

    // external buffer management
    size_t external_data_size() const { return external_data_size_; }
    void *external_data() const { return external_data_; }
    int32_t device_id() const { return device_id_; }
    bool is_external() const { return is_external_; }

   private:
    static size_t curr_id;
    size_t id_;
    int rank_;
    std::set<TagInfo> send_tags_;
    std::set<TagInfo> recv_tags_;
    void *external_data_ = nullptr;
    size_t external_data_size_ = 0;
    int32_t device_id_;
    bool is_external_ = false;
};

}  // namespace ark

#endif  // ARK_MODEL_BUFFER_HPP_
