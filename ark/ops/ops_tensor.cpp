// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_tensor.hpp"

#include "logging.hpp"
#include "ops_common.hpp"

namespace ark {

ModelOpTensor::ModelOpTensor(ModelBufferRef buffer, const Dims &shape,
                             ModelDataType data_type, const Dims &strides,
                             const Dims &offsets, const Dims &padded_shape)
    : ModelOp("Tensor", true) {
    if (!buffer) {
        buffer = std::make_shared<ModelBuffer>();
    }

    ModelTensorRef tensor = std::make_shared<ModelTensor>(
        data_type, buffer, shape, strides, offsets, padded_shape);

    result_tensors_.emplace_back(tensor);

    verify();
}

Tensor Model::tensor(const Dims &shape, const DataType &data_type,
                     const Dims &strides, const Dims &offsets,
                     const Dims &padded_shape, int rank,
                     const std::string &name) {
    if (rank != -1) {
        if (rank == this->rank()) {
            rank = -1;
        } else if (rank < 0 || rank >= this->world_size()) {
            ERR(ModelError, "Invalid rank %d", rank);
        }
    }
    return impl_
        ->create_op<ModelOpTensor>(name, std::make_shared<ModelBuffer>(rank),
                                   shape, data_type.ref(), strides, offsets,
                                   padded_shape)
        ->result_tensors()[0];
}

}  // namespace ark
