// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_refer.hpp"

#include "ops_common.hpp"

namespace ark {

ModelOpRefer::ModelOpRefer(ModelTensorRef input, const Dims &shape,
                           const Dims &strides, const Dims &offsets,
                           const Dims &padded_shape)
    : ModelOpTensor(input->buffer(), shape, input->data_type(), strides,
                    offsets, padded_shape) {
    read_tensors_ = {input};
    verify();
}

Tensor Model::refer(Tensor input, const Dims &shape, const Dims &strides,
                    const Dims &offsets, const Dims &padded_shape,
                    const std::string &name) {
    return impl_
        ->create_op<ModelOpRefer>(name, input.ref_, shape, strides, offsets,
                                  padded_shape)
        ->result_tensors()[0];
}

}  // namespace ark
