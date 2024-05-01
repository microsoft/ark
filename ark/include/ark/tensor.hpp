// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_TENSOR_HPP
#define ARK_TENSOR_HPP

#include <ostream>

#include "dims.hpp"
#include "model_ref.hpp"

namespace ark {

class ModelDataT;
using ModelDataType = std::shared_ptr<ModelDataT>;

class Tensor {
   protected:
    friend class Model;
    ModelTensorRef ref_;

   public:
    Tensor() = default;
    Tensor(ModelTensorRef ref) : ref_(ref) {}
    Tensor(const Tensor &other) = default;
    Tensor &operator=(const Tensor &other) = default;

    bool operator==(const Tensor &other) const { return ref_ == other.ref_; }
    bool operator!=(const Tensor &other) const { return ref_ != other.ref_; }

    bool is_none() const { return !ref_; }

    ModelTensorRef ref() const { return ref_; }

    size_t id() const;

    Dims shape() const;

    Dims strides() const;

    Dims offsets() const;

    Dims pads() const;

    ModelDataType data_type() const;
};

const Tensor NoneTensor;

std::ostream &operator<<(std::ostream &os, const Tensor &tensor);

}  // namespace ark

#endif  // ARK_TENSOR_HPP
