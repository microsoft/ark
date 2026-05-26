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

/// Location of tensor data in the memory hierarchy.
enum class TensorLocation {
    GLOBAL,    // GPU global memory (HBM) — default, current behavior
    SHARED,    // Shared memory (SMEM) — scoped to one thread block
    REGISTER,  // Register file — scoped to one warp group (no buffer allocation)
               // TODO: Register-level fusion is not yet implemented.
               // Planner and buffer allocator do not yet skip global
               // allocation for REGISTER tensors. See ModelOpMma/ModelOpStore.
};

class ModelTensor {
   public:
    ModelTensor(ModelDataType data_type, ModelBufferRef buffer,
                const Dims &shape, const Dims &strides = {},
                const Dims &offsets = {}, const Dims &padded_shape = {},
                TensorLocation location = TensorLocation::GLOBAL);

    ModelTensor(const ModelTensor &other);

    size_t id() const { return id_; }

    ModelDataType data_type() const { return data_type_; }

    ModelBufferRef buffer() const { return buffer_; }

    const Dims &shape() const { return shape_; }

    const Dims &strides() const { return strides_; }

    const Dims &offsets() const { return offsets_; }

    const Dims &padded_shape() const { return padded_shape_; }

    size_t shape_bytes() const;

    void *data() const;

    void *data(void *data);

    bool is_external() const;

    TensorLocation location() const { return location_; }
    void set_location(TensorLocation loc) { location_ = loc; }

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
    Dims padded_shape_;
    TensorLocation location_ = TensorLocation::GLOBAL;
};

}  // namespace ark

#endif  // ARK_MODEL_TENSOR_HPP_
