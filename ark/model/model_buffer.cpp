// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "model_buffer.hpp"

#include "buffer_registry.hpp"
#include "logging.hpp"

namespace ark {

size_t ModelBuffer::curr_id = 0;

ModelBuffer::ModelBuffer(int rank, bool is_external)
    : rank_(rank), is_external_(is_external) {
    id_ = curr_id++;
}

ModelBuffer::ModelBuffer(size_t id, int rank, bool is_external,
                         const std::vector<TagInfo> &send_tags,
                         const std::vector<TagInfo> &recv_tags)
    : id_(id), rank_(rank), is_external_(is_external) {
    if (is_external && (!send_tags.empty() || !recv_tags.empty())) {
        ERR(ModelError, "External buffer cannot have send or receive tags");
    }
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

void *ModelBuffer::data() const {
    auto info = BufferRegistry::get_instance().get(id_);
    if (info) {
        return info->data;
    }
    return nullptr;
}

void *ModelBuffer::data(void *data) {
    if (is_external_) {
        BufferRegistry::get_instance().set(id_, data, -1, true);
        return data;
    }
    return nullptr;
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
    j["IsExternal"] = is_external_;
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
        ERR(ModelError,
            "ModelBuffer deserialization failed: missing RecvTags");
    } else if (!serialized.contains("IsExternal")) {
        ERR(ModelError,
            "ModelBuffer deserialization failed: missing IsExternal");
    }
    return std::make_shared<ModelBuffer>(
        serialized["Id"], serialized["Rank"], serialized["IsExternal"],
        serialized["SendTags"], serialized["RecvTags"]);
}

}  // namespace ark
