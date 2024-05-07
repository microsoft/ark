// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_MODEL_BUFFER_HPP_
#define ARK_MODEL_BUFFER_HPP_

#include <memory>
#include <set>
#include <vector>

#include "json.hpp"

namespace ark {

class ModelBuffer {
   public:
    // (remote_rank, tag)
    using TagInfo = std::pair<int, int>;

    ModelBuffer(int rank = -1);

    ModelBuffer(size_t id, int rank, const std::vector<TagInfo> &send_tags,
                const std::vector<TagInfo> &recv_tags);

    size_t id() const { return id_; }

    int rank() const { return rank_; }

    const std::set<TagInfo> &send_tags() const { return send_tags_; }

    const std::set<TagInfo> &recv_tags() const { return recv_tags_; }

    void tag_send(int remote_rank, int tag);

    void tag_recv(int remote_rank, int tag);

    nlohmann::ordered_json serialize() const;

    static std::shared_ptr<ModelBuffer> deserialize(const json &serialized);

   private:
    size_t id_;
    int rank_;
    std::set<TagInfo> send_tags_;
    std::set<TagInfo> recv_tags_;
};

}  // namespace ark

#endif  // ARK_MODEL_BUFFER_HPP_
