// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_tensor.hpp"

#include "ops_common.hpp"

namespace ark {

ModelOpTensor::ModelOpTensor(ModelBufferRef buffer, const Dims &shape,
                             ModelDataType data_type, const Dims &strides,
                             const Dims &offsets, const Dims &pads,
                             bool exported, int imported_rank)
    : ModelOp("Tensor", true) {
    if (!buffer) {
        buffer = std::make_shared<ModelBuffer>();
    }

    ModelTensorRef tensor =
        std::make_shared<ModelTensor>(data_type, buffer, shape, strides,
                                      offsets, pads, exported, imported_rank);

    result_tensors_.emplace_back(tensor);

    verify();
}

ModelTensorRef Model::tensor(const Dims &shape, ModelDataType data_type,
                             const Dims &strides, const Dims &offsets,
                             const Dims &pads, bool exported, int imported_rank,
                             const std::string &name) {
    return impl_
        ->create_op<ModelOpTensor>(name, nullptr, shape, data_type, strides,
                                   offsets, pads, exported, imported_rank)
        ->result_tensors()[0];
}

}  // namespace ark
