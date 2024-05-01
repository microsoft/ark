// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "model_tensor.hpp"

#include "ark/data_type.hpp"
#include "logging.h"
#include "model_data_type.hpp"

namespace ark {

ModelBuffer::ModelBuffer() {
    static size_t id = 0;
    id_ = id++;
}

ModelBuffer::ModelBuffer(size_t id) { id_ = id; }

ModelTensor::ModelTensor(ModelDataType data_type, ModelBufferRef buffer,
                         const Dims &shape, const Dims &strides,
                         const Dims &offsets, const Dims &pads, bool exported,
                         int imported_rank)
    : data_type_(data_type),
      buffer_(buffer),
      exported_(exported),
      imported_rank_(imported_rank) {
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
    exported_ = other.exported_;
    imported_rank_ = other.imported_rank_;
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
    j["BufferId"] = buffer_->id();
    j["Shape"] = shape_.vector();
    j["Strides"] = strides_.vector();
    j["Offsets"] = offsets_.vector();
    j["Pads"] = pads_.vector();
    j["Exported"] = exported_;
    j["ImportedRank"] = imported_rank_;
    return j;
}

std::shared_ptr<ModelTensor> ModelTensor::deserialize(
    const nlohmann::json &serialized) {
    if (!serialized.contains("DataType")) {
        ERR(InvalidUsageError,
            "ModelTensor deserialization failed: missing DataType");
    } else if (!serialized.contains("BufferId")) {
        ERR(InvalidUsageError,
            "ModelTensor deserialization failed: missing BufferId");
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
    } else if (!serialized.contains("Exported")) {
        ERR(InvalidUsageError,
            "ModelTensor deserialization failed: missing Exported");
    } else if (!serialized.contains("ImportedRank")) {
        ERR(InvalidUsageError,
            "ModelTensor deserialization failed: missing ImportedRank");
    } else if (!serialized.contains("Id")) {
        ERR(InvalidUsageError,
            "ModelTensor deserialization failed: missing Id");
    }
    auto ret = std::make_shared<ModelTensor>(
        DataType::from_name(serialized["DataType"]).ref(),
        std::make_shared<ModelBuffer>(serialized["BufferId"]),
        serialized["Shape"].get<std::vector<DimType>>(),
        serialized["Strides"].get<std::vector<DimType>>(),
        serialized["Offsets"].get<std::vector<DimType>>(),
        serialized["Pads"].get<std::vector<DimType>>(),
        serialized["Exported"].get<bool>(),
        serialized["ImportedRank"].get<int>());
    ret->id_ = serialized["Id"];
    return ret;
}

size_t ModelTensor::next_id() {
    static size_t id = 0;
    return id++;
}

}  // namespace ark
