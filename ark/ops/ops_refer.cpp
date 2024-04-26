// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_refer.hpp"

#include <set>

#include "ops_common.hpp"

namespace ark {

ModelOpRefer::ModelOpRefer(ModelTensorRef input, const Dims &shape,
                           const Dims &strides, const Dims &offsets,
                           const Dims &pads)
    : ModelOpTensor(input->buffer(), shape, input->data_type(), strides,
                    offsets, pads, input->exported(), input->imported_rank()) {
    read_tensors_ = {input};
    verify();
}

ModelTensorRef Model::refer(ModelTensorRef input, const Dims &shape,
                            const Dims &strides, const Dims &offsets,
                            const Dims &pads, const std::string &name) {
    return impl_
        ->create_op<ModelOpRefer>(name, input, shape, strides, offsets, pads)
        ->result_tensors()[0];
}

}  // namespace ark
