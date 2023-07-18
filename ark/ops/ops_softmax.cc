// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "logging.h"
#include "model.h"
#include "tensor.h"

using namespace std;

namespace ark {

Tensor *Model::softmax(Tensor *input, Tensor *output, const string &name)
{
    assert(input != nullptr);
    LOG(DEBUG, "softmax ", input->shape, " ", input->ldims, " ");
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
        output = this->tensor(input->shape, input->type);
    } else if (output == input) {
        output = this->identity(output);
    }
    this->impl->add_op(OP_SOFTMAX, pt, {input}, {output}, {}, name);
    return output;
}

} // namespace ark
