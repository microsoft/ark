// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_MODEL_TENSOR_HPP_
#define ARK_MODEL_TENSOR_HPP_

#include "ark/dims.hpp"
#include "ark/model_ref.hpp"
#include "model_json.hpp"

namespace ark {

class ModelDataT;
using ModelDataType = std::shared_ptr<ModelDataT>;

class ModelTensor {
   public:
    ModelTensor(ModelDataType data_type, ModelBufferRef buffer,
                const Dims &shape, const Dims &strides = {},
                const Dims &offsets = {}, const Dims &pads = {});

    ModelTensor(const ModelTensor &other);

    size_t id() const { return id_; }

    ModelDataType data_type() const { return data_type_; }

    ModelBufferRef buffer() const { return buffer_; }

    const Dims &shape() const { return shape_; }

    const Dims &strides() const { return strides_; }

    const Dims &offsets() const { return offsets_; }

    const Dims &pads() const { return pads_; }

    size_t shape_bytes() const;

    size_t strides_bytes() const;

    bool is_sequential() const;

    Json serialize() const;

    static std::shared_ptr<ModelTensor> deserialize(const Json &serialized);

   private:
    static size_t next_id();

    size_t id_;
    ModelDataType data_type_;
    ModelBufferRef buffer_;
    Dims shape_;
    Dims strides_;
    Dims offsets_;
    Dims pads_;
};

}  // namespace ark

#endif  // ARK_MODEL_TENSOR_HPP_
