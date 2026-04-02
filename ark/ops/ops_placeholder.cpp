// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_placeholder.hpp"

#include "buffer_registry.hpp"
#include "logging.hpp"
#include "ops_common.hpp"

namespace ark {

ModelOpPlaceholder::ModelOpPlaceholder(ModelBufferRef buffer, const Dims &shape,
                                       ModelDataType data_type,
                                       const Dims &strides, const Dims &offsets,
                                       const Dims &padded_shape, void *data)
    : ModelOp("Placeholder", true) {
    if (!buffer) {
        buffer = std::make_shared<ModelBuffer>(-1, true);
    }

    BufferRegistry::get_instance().set(buffer->id(), data, -1, true);

    ModelTensorRef tensor = std::make_shared<ModelTensor>(
        data_type, buffer, shape, strides, offsets, padded_shape);

    result_tensors_.emplace_back(tensor);

    verify();
}

Tensor Model::placeholder(const Dims &shape, const DataType &data_type,
                          const Dims &strides, const Dims &offsets,
                          const Dims &padded_shape, int rank, void *data,
                          const std::string &name) {
    if (rank != -1) {
        if (rank == this->rank()) {
            rank = -1;
        } else if (rank < 0 || rank >= this->world_size()) {
            ERR(ModelError, "Invalid rank %d", rank);
        }
    }
    return impl_
        ->create_op<ModelOpPlaceholder>(
            name, std::make_shared<ModelBuffer>(rank, true), shape,
            data_type.ref(), strides, offsets, padded_shape, data)
        ->result_tensors()[0];
}

}  // namespace ark
