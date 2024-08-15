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

    ModelBuffer(int rank = -1, bool is_external = false);

    ModelBuffer(size_t id, int rank, bool is_external,
                const std::vector<TagInfo> &send_tags,
                const std::vector<TagInfo> &recv_tags);

    size_t id() const { return id_; }

    int rank() const { return rank_; }

    bool is_external() const { return is_external_; }

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

   private:
    static size_t curr_id;
    size_t id_;
    int rank_;
    bool is_external_;
    std::set<TagInfo> send_tags_;
    std::set<TagInfo> recv_tags_;
};

}  // namespace ark

#endif  // ARK_MODEL_BUFFER_HPP_
