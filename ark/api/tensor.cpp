// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/tensor.hpp"

#include "model/model_buffer.hpp"
#include "model/model_data_type.hpp"
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

Dims Tensor::padded_shape() const {
    if (ref_) {
        return ref_->padded_shape();
    }
    return Dims();
}

const DataType &Tensor::data_type() const {
    if (ref_) {
        return DataType::from_name(ref_->data_type()->type_name());
    }
    return NONE;
}

Dims Tensor::torch_strides() const {
    if (ref_) {
        Dims st = ref_->strides();
        int ndims = st.ndims();
        std::vector<DimType> tmp;
        for (int i = 1; i < ndims; ++i) {
            tmp.push_back(st[i]);
        }
        tmp.push_back(1);
        for (int i = ndims - 2; i >= 0; --i) {
            tmp[i] *= tmp[i + 1];
        }
        return Dims(tmp);
    }
    return Dims();
}

std::ostream &operator<<(std::ostream &os, const Tensor &tensor) {
    if (tensor.is_null()) {
        os << "null";
    } else {
        os << tensor.ref()->serialize().dump();
    }
    return os;
}

}  // namespace ark
