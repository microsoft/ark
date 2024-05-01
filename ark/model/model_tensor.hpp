// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_MODEL_TENSOR_HPP_
#define ARK_MODEL_TENSOR_HPP_

#include "ark/dims.hpp"
#include "ark/model_ref.hpp"
#include "nlohmann/json.hpp"

namespace ark {

class ModelBuffer {
   public:
    ModelBuffer();

    ModelBuffer(size_t id);

    size_t id() const { return id_; }

   private:
    size_t id_;
};

class ModelDataT;
using ModelDataType = std::shared_ptr<ModelDataT>;

/// Tensor is a view of a TensorBuf.
///
/// Illustration of a single axis of a tensor:
///
/// 0           off                                                       stride
/// |------------|-------------shape-------------|---------------------------|
///       ^       <----------------------------->                ^
///       |          data range of this tensor                   |
///       +------------------------------------------+-----------+
///                                                  |
///                                        We call these "padding".
///
class ModelTensor {
   public:
    ModelTensor(ModelDataType data_type, ModelBufferRef buffer,
                const Dims &shape, const Dims &strides = {},
                const Dims &offsets = {}, const Dims &pads = {},
                bool exported = false, int imported_rank = -1);

    ModelTensor(const ModelTensor &other);

    size_t id() const { return id_; }

    ModelDataType data_type() const { return data_type_; }

    const ModelBufferRef buffer() const { return buffer_; }

    const Dims &shape() const { return shape_; }

    const Dims &strides() const { return strides_; }

    const Dims &offsets() const { return offsets_; }

    const Dims &pads() const { return pads_; }

    size_t shape_bytes() const;

    size_t strides_bytes() const;

    bool exported() const { return exported_; }

    int imported_rank() const { return imported_rank_; }

    bool is_sequential() const;

    void set_exported() { exported_ = true; }

    void set_imported_rank(int rank) { imported_rank_ = rank; }

    nlohmann::ordered_json serialize() const;

    static std::shared_ptr<ModelTensor> deserialize(
        const nlohmann::json &serialized);

   private:
    static size_t next_id();

    size_t id_;
    ModelDataType data_type_;
    ModelBufferRef buffer_;
    Dims shape_;
    Dims strides_;
    Dims offsets_;
    Dims pads_;
    bool exported_;
    int imported_rank_;
};

}  // namespace ark

#endif  // ARK_MODEL_TENSOR_HPP_
