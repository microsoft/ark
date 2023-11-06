// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cassert>

#include "logging.h"
#include "model.h"

namespace ark {

MaxPoolOp::MaxPoolOp(const std::string &prec_type, Tensor *input,
                     Tensor *output, DimType kernel_size, DimType stride,
                     const std::string &name)
    : Op{OP_MAX_POOL, prec_type, {input}, {output}, {{kernel_size, stride}},
         name,        nullptr,   -1} {}

// TODO: implement
Tensor *Model::max_pool(Tensor *input, DimType kernel_size, DimType stride,
                        Tensor *output, const std::string &name) {
    assert(input != nullptr);
    if (output != nullptr && input->type != output->type) {
        ERR(InvalidUsageError, "invalid output data type: ", output->type);
    }
    const Dims &is = input->shape;
    Dims os{{is[0], (is[1] + stride - 1) / stride,
             (is[2] + stride - 1) / stride, is[3]}};
    if (output == nullptr) {
        output = this->tensor(os, input->type);
    }
    MaxPoolOp op{output->type.name(), input, output, kernel_size, stride, name};
    return this->impl->add_op(op)[0];
}

}  // namespace ark
