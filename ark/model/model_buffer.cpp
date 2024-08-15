// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "model_buffer.hpp"

#include "logging.hpp"
#include "model_buffer_manager.hpp"

namespace ark {

size_t ModelBuffer::curr_id = 0;

ModelBuffer::ModelBuffer(int rank) : rank_(rank) { id_ = curr_id++; }

ModelBuffer::ModelBuffer(size_t id, int rank,
                         const std::vector<TagInfo> &send_tags,
                         const std::vector<TagInfo> &recv_tags)
    : id_(id), rank_(rank) {
    for (const auto &info : send_tags) {
        send_tags_.insert(info);
    }
    for (const auto &info : recv_tags) {
        recv_tags_.insert(info);
    }
}

void ModelBuffer::tag_send(int remote_rank, int tag) {
    send_tags_.insert(TagInfo{remote_rank, tag});
}

void ModelBuffer::tag_recv(int remote_rank, int tag) {
    recv_tags_.insert(TagInfo{remote_rank, tag});
}

Json ModelBuffer::serialize() const {
    Json j;
    j["Id"] = id_;
    j["Rank"] = rank_;
    Json send_tags = Json::array();
    Json recv_tags = Json::array();
    for (const auto &info : send_tags_) {
        send_tags.push_back({info.first, info.second});
    }
    for (const auto &info : recv_tags_) {
        recv_tags.push_back({info.first, info.second});
    }
    j["SendTags"] = send_tags;
    j["RecvTags"] = recv_tags;
    return j;
}

std::shared_ptr<ModelBuffer> ModelBuffer::deserialize(const Json &serialized) {
    if (!serialized.contains("Id")) {
        ERR(ModelError, "ModelBuffer deserialization failed: missing Id");
    } else if (!serialized.contains("Rank")) {
        ERR(ModelError, "ModelBuffer deserialization failed: missing Rank");
    } else if (!serialized.contains("SendTags")) {
        ERR(ModelError, "ModelBuffer deserialization failed: missing SendTags");
    } else if (!serialized.contains("RecvTags")) {
        ERR(ModelError, "ModelBuffer deserialization failed: missing RecvTags");
    }
    return std::make_shared<ModelBuffer>(serialized["Id"], serialized["Rank"],
                                         serialized["SendTags"],
                                         serialized["RecvTags"]);
}

}  // namespace ark
