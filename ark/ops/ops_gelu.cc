// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/logging.h"
#include "ark/model_io.h"

using namespace std;

namespace ark {

// Multiply `input` by `val`.
// TODO: make it into a general element-wise operation
Tensor *Model::gelu(Tensor *input, Tensor *output, const string &name)
{
    assert(input != nullptr);
    OpPrecType pt;
    if (input->type == FP16) {
        pt = OP_PREC_FP16;
    } else if (input->type == FP32) {
        pt = OP_PREC_FP32;
    } else {
        LOGERR("unsupported input data type: ", type_str(input->type));
    }
    if (output != nullptr && input->type != output->type) {
        LOGERR("invalid output data type: ", type_str(output->type));
    }
    if (output == nullptr) {
        output = this->tensor(input->shape, input->type, input->buf);
    } else if (output->shape != input->shape) {
        LOGERR("invalid output shape: ", output->shape);
    }
    this->create_op(OP_GELU, pt, {input}, {output}, {}, name);
    return output;
}

} // namespace ark
