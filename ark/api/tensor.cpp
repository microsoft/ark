// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/tensor.hpp"

#include <ark/tensor.hpp>

#include "ark/dims.hpp"
#include "logging.h"
#include "model/model_buffer.hpp"
#include "model/model_data_type.hpp"
#include "model/model_tensor.hpp"

namespace ark {

Tensor::Tensor(void* data_ptr, int32_t device_id, int8_t dtype_bytes,
               const std::vector<int64_t>& shape,
               const std::string& ark_type_str) {
    size_t external_data_size = std::accumulate(shape.begin(), shape.end(), 1,
                                                std::multiplies<int64_t>()) *
                                dtype_bytes;
    auto buffer =
        std::make_shared<ModelBuffer>(data_ptr, external_data_size, device_id);
    ark::ModelDataType dtype = DataType::from_name(ark_type_str).ref();
    auto tensor = std::make_shared<ModelTensor>(dtype, buffer, Dims(shape),
                                                Dims(shape), Dims(), Dims());
    ref_ = tensor;
}

size_t Tensor::id() const {
    if (ref_) {
        return ref_->id();
    }
    return 0;
}

Dims Tensor::shape() const {
    if (ref_) {
        return ref_->shape();
    }
    return Dims();
}

Dims Tensor::strides() const {
    if (ref_) {
        return ref_->strides();
    }
    return Dims();
}

Dims Tensor::offsets() const {
    if (ref_) {
        return ref_->offsets();
    }
    return Dims();
}

Dims Tensor::padded_shape() const {
    if (ref_) {
        return ref_->padded_shape();
    }
    return Dims();
}

const DataType& Tensor::data_type() const {
    if (ref_) {
        return DataType::from_name(ref_->data_type()->type_name());
    }
    return NONE;
}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    if (tensor.is_null()) {
        os << "null";
    } else {
        os << tensor.ref()->serialize().dump();
    }
    return os;
}

}  // namespace ark
