// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/tensor.hpp"

#include "model/model_tensor.hpp"

namespace ark {

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

Dims Tensor::pads() const {
    if (ref_) {
        return ref_->pads();
    }
    return Dims();
}

ModelDataType Tensor::data_type() const {
    if (ref_) {
        return ref_->data_type();
    }
    return nullptr;
}

std::ostream &operator<<(std::ostream &os, const Tensor &tensor) {
    if (tensor.is_none()) {
        os << "null";
    } else {
        os << tensor.ref()->serialize().dump();
    }
    return os;
}

}  // namespace ark
