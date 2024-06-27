// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "model_buffer.hpp"

#include "logging.h"
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

ModelBuffer::ModelBuffer(void *data, size_t size, int32_t device_id)
    : rank_(-1),
      external_data_(data),
      external_data_size_(size),
      device_id_(device_id),
      is_external_(true) {
    id_ = curr_id++;
}

ModelBuffer::ModelBuffer(size_t id, void *data, size_t size, int32_t device_id)
    : id_(id),
      rank_(-1),
      external_data_(data),
      external_data_size_(size),
      device_id_(device_id),
      is_external_(true) {}

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
    j["IsExternal"] = is_external_;
    if (is_external_) {
        ModelBufferManager::getInstance().registerBuffer(id_, external_data_,
                                                         external_data_size_);
        j["ExternalDataSize"] = external_data_size_;
        j["DeviceId"] = device_id_;
    }
    // external_data_ptr_ is not included in JSON
    return j;
}

std::shared_ptr<ModelBuffer> ModelBuffer::deserialize(const Json &serialized) {
    if (!serialized.contains("Id")) {
        ERR(InvalidUsageError,
            "ModelBuffer deserialization failed: missing Id");
    } else if (!serialized.contains("Rank")) {
        ERR(InvalidUsageError,
            "ModelBuffer deserialization failed: missing Rank");
    } else if (!serialized.contains("SendTags")) {
        ERR(InvalidUsageError,
            "ModelBuffer deserialization failed: missing SendTags");
    } else if (!serialized.contains("RecvTags")) {
        ERR(InvalidUsageError,
            "ModelBuffer deserialization failed: missing RecvTags");
    } else if (!serialized.contains("IsExternal")) {
        ERR(InvalidUsageError,
            "ModelBuffer deserialization failed: missing IsExternal");
    }
    if (serialized["IsExternal"]) {
        if (!serialized.contains("ExternalDataSize")) {
            ERR(InvalidUsageError,
                "ModelBuffer deserialization failed: missing ExternalDataSize");
        } else if (!serialized.contains("DeviceId")) {
            ERR(InvalidUsageError,
                "ModelBuffer deserialization failed: missing DeviceId");
        }
        void *data_ptr =
            ModelBufferManager::getInstance().getBuffer(serialized["Id"]);
        if (!data_ptr) {
            ERR(InvalidUsageError,
                "ModelBuffer deserialization failed: external buffer not found "
                "in BufferManager");
        }
        return std::make_shared<ModelBuffer>(serialized["Id"], data_ptr,
                                             serialized["ExternalDataSize"],
                                             serialized["DeviceId"]);
    }
    return std::make_shared<ModelBuffer>(serialized["Id"], serialized["Rank"],
                                         serialized["SendTags"],
                                         serialized["RecvTags"]);
}

}  // namespace ark
