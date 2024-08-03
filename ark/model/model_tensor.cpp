// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "model_tensor.hpp"

#include "ark/data_type.hpp"
#include "logging.hpp"
#include "model_buffer.hpp"
#include "model_data_type.hpp"

namespace ark {

ModelTensor::ModelTensor(ModelDataType data_type, ModelBufferRef buffer,
                         const Dims &shape, const Dims &strides,
                         const Dims &offsets, const Dims &padded_shape)
    : data_type_(data_type), buffer_(buffer) {
    if (shape.is_no_dim()) {
        // Assume a single-element constant
        shape_ = {1};
    } else if (shape.has_negative() || shape.nelems() == 0) {
        ERR(ModelError,
            "Tensor shape should consist of only positive numbers. Given: ",
            shape);
    } else {
        shape_ = shape;
    }
    int ndims = shape_.ndims();
    if (padded_shape.is_no_dim()) {
        padded_shape_ = shape_;
    } else {
        if (ndims != padded_shape.ndims()) {
            ERR(ModelError,
                "Tensor shape and padded shape should have the same number of "
                "dimensions. Given: shape ",
                shape_, " padded_shape ", padded_shape);
        }
        padded_shape_ = padded_shape;
    }
    for (int i = 0; i < ndims; ++i) {
        if (shape_[i] > padded_shape_[i]) {
            ERR(ModelError, "Tensor shape exceeds the padded shape. shape ",
                shape_, " padded_shape ", padded_shape_);
        }
    }
    if (strides.is_no_dim()) {
        strides_ = padded_shape_;
    } else {
        if (ndims != strides.ndims()) {
            ERR(ModelError,
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
            ERR(ModelError,
                "Tensor shape and offs should have the same number of "
                "dimensions. Given: shape ",
                shape_, " offs ", offsets);
        }
        offsets_ = offsets;
    }
    for (int i = 0; i < ndims; ++i) {
        if (offsets_[i] + padded_shape_[i] > strides_[i]) {
            ERR(ModelError, "Tensor exceeds the memory boundary. offs ",
                offsets_, " padded_shape ", padded_shape_, " strides ",
                strides_);
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
    padded_shape_ = other.padded_shape_;
}

size_t ModelTensor::shape_bytes() const {
    return shape_.nelems() * data_type_->bytes();
}

Json ModelTensor::serialize() const {
    Json j;
    j["Id"] = id_;
    j["DataType"] = data_type_->type_name();
    j["Shape"] = shape_.vector();
    j["Strides"] = strides_.vector();
    j["Offsets"] = offsets_.vector();
    j["PaddedShape"] = padded_shape_.vector();
    j["Buffer"] = buffer_->serialize();
    return j;
}

std::shared_ptr<ModelTensor> ModelTensor::deserialize(const Json &serialized) {
    if (!serialized.contains("DataType")) {
        ERR(ModelError, "ModelTensor deserialization failed: missing DataType");
    } else if (!serialized.contains("Buffer")) {
        ERR(ModelError, "ModelTensor deserialization failed: missing Buffer");
    } else if (!serialized.contains("Shape")) {
        ERR(ModelError, "ModelTensor deserialization failed: missing Shape");
    } else if (!serialized.contains("Strides")) {
        ERR(ModelError, "ModelTensor deserialization failed: missing Strides");
    } else if (!serialized.contains("Offsets")) {
        ERR(ModelError, "ModelTensor deserialization failed: missing Offsets");
    } else if (!serialized.contains("PaddedShape")) {
        ERR(ModelError,
            "ModelTensor deserialization failed: missing PaddedShape");
    } else if (!serialized.contains("Id")) {
        ERR(ModelError, "ModelTensor deserialization failed: missing Id");
    }
    auto ret = std::make_shared<ModelTensor>(
        DataType::from_name(serialized["DataType"]).ref(),
        ModelBuffer::deserialize(serialized["Buffer"]),
        serialized["Shape"].get<std::vector<DimType>>(),
        serialized["Strides"].get<std::vector<DimType>>(),
        serialized["Offsets"].get<std::vector<DimType>>(),
        serialized["PaddedShape"].get<std::vector<DimType>>());
    ret->id_ = serialized["Id"];
    return ret;
}

size_t ModelTensor::next_id() {
    static size_t id = 0;
    return id++;
}

}  // namespace ark
