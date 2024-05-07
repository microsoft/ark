// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "model_tensor.hpp"

#include "ark/data_type.hpp"
#include "logging.h"
#include "model_data_type.hpp"

namespace ark {

ModelBuffer::ModelBuffer(int rank) : rank_(rank) {
    static size_t id = 0;
    id_ = id++;
}

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

nlohmann::ordered_json ModelBuffer::serialize() const {
    nlohmann::ordered_json j;
    j["Id"] = id_;
    j["Rank"] = rank_;
    nlohmann::ordered_json send_tags = json::array();
    nlohmann::ordered_json recv_tags = json::array();
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

std::shared_ptr<ModelBuffer> ModelBuffer::deserialize(const json &serialized) {
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
    }
    return std::make_shared<ModelBuffer>(serialized["Id"], serialized["Rank"],
                                         serialized["SendTags"],
                                         serialized["RecvTags"]);
}

ModelTensor::ModelTensor(ModelDataType data_type, ModelBufferRef buffer,
                         const Dims &shape, const Dims &strides,
                         const Dims &offsets, const Dims &pads)
    : data_type_(data_type), buffer_(buffer) {
    if (shape.nelems() == 0) {
        ERR(InvalidUsageError,
            "Tensor shape should consist of positive numbers. Given: ", shape);
    } else if (shape.is_no_dim()) {
        // Assume a single-element constant
        shape_ = {1};
    } else {
        shape_ = shape;
    }
    int ndims = shape_.ndims();
    if (strides.is_no_dim()) {
        strides_ = shape_;
    } else {
        if (ndims != strides.ndims()) {
            ERR(InvalidUsageError,
                "Tensor shapes and strides should have the same number of "
                "dimensions. Given: shape ",
                shape_, " strides ", strides);
        }
        strides_ = strides;
    }
    if (offsets.is_no_dim()) {
        std::vector<DimType> dims_vec;
        for (int i = 0; i < ndims; ++i) {
            dims_vec.push_back(0);
        }
        offsets_ = Dims{dims_vec};
    } else {
        if (ndims != offsets.ndims()) {
            ERR(InvalidUsageError,
                "Tensor shape and offs should have the same number of "
                "dimensions. Given: shape ",
                shape_, " offs ", offsets);
        }
        offsets_ = offsets;
    }
    if (pads.is_no_dim()) {
        std::vector<DimType> dims_vec;
        for (int i = 0; i < ndims; ++i) {
            dims_vec.push_back(1);
        }
        pads_ = Dims{dims_vec};
    } else {
        if (ndims != pads.ndims()) {
            ERR(InvalidUsageError,
                "Tensor shape and pads should have the same number of "
                "dimensions. Given: shape ",
                shape_, " pads ", pads);
        }
        pads_ = pads;
    }
    for (int i = 0; i < ndims; ++i) {
        if (strides_[i] % pads_[i] != 0) {
            ERR(InvalidUsageError,
                "Tensor strides should be a multiple of pads. strides ",
                strides_, " pads ", pads_);
        }
    }
    for (int i = 0; i < ndims; ++i) {
        if (offsets_[i] + shape_[i] > strides_[i]) {
            ERR(InvalidUsageError, "Tensor exceeds the memory boundary. offs ",
                offsets_, " shape ", shape_, " strides ", strides_);
        }
    }
    id_ = next_id();
}

ModelTensor::ModelTensor(const ModelTensor &other) {
    id_ = next_id();
    data_type_ = other.data_type_;
    buffer_ = other.buffer_;
    shape_ = other.shape_;
    strides_ = other.strides_;
    offsets_ = other.offsets_;
    pads_ = other.pads_;
}

size_t ModelTensor::shape_bytes() const {
    return shape_.nelems() * data_type_->bytes();
}

size_t ModelTensor::strides_bytes() const {
    return strides_.nelems() * data_type_->bytes();
}

bool ModelTensor::is_sequential() const {
    // Shape and strides should be the same except for the first dimension.
    for (int i = 1; i < shape_.ndims(); ++i) {
        if (shape_[i] != strides_[i]) {
            return false;
        }
    }
    return true;
}

nlohmann::ordered_json ModelTensor::serialize() const {
    nlohmann::ordered_json j;
    j["Id"] = id_;
    j["DataType"] = data_type_->type_name();
    j["Buffer"] = buffer_->serialize();
    j["Shape"] = shape_.vector();
    j["Strides"] = strides_.vector();
    j["Offsets"] = offsets_.vector();
    j["Pads"] = pads_.vector();
    return j;
}

std::shared_ptr<ModelTensor> ModelTensor::deserialize(const json &serialized) {
    if (!serialized.contains("DataType")) {
        ERR(InvalidUsageError,
            "ModelTensor deserialization failed: missing DataType");
    } else if (!serialized.contains("Buffer")) {
        ERR(InvalidUsageError,
            "ModelTensor deserialization failed: missing Buffer");
    } else if (!serialized.contains("Shape")) {
        ERR(InvalidUsageError,
            "ModelTensor deserialization failed: missing Shape");
    } else if (!serialized.contains("Strides")) {
        ERR(InvalidUsageError,
            "ModelTensor deserialization failed: missing Strides");
    } else if (!serialized.contains("Offsets")) {
        ERR(InvalidUsageError,
            "ModelTensor deserialization failed: missing Offsets");
    } else if (!serialized.contains("Pads")) {
        ERR(InvalidUsageError,
            "ModelTensor deserialization failed: missing Pads");
    } else if (!serialized.contains("Id")) {
        ERR(InvalidUsageError,
            "ModelTensor deserialization failed: missing Id");
    }
    auto ret = std::make_shared<ModelTensor>(
        DataType::from_name(serialized["DataType"]).ref(),
        ModelBuffer::deserialize(serialized["Buffer"]),
        serialized["Shape"].get<std::vector<DimType>>(),
        serialized["Strides"].get<std::vector<DimType>>(),
        serialized["Offsets"].get<std::vector<DimType>>(),
        serialized["Pads"].get<std::vector<DimType>>());
    ret->id_ = serialized["Id"];
    return ret;
}

size_t ModelTensor::next_id() {
    static size_t id = 0;
    return id++;
}

}  // namespace ark
