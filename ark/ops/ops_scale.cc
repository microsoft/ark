// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "logging.h"
#include "model.h"
#include "tensor.h"

using namespace std;

namespace ark {

class ScaleOp : public Op
{
  public:
    ScaleOp::ScaleOp(OpPrecType prec_type, Tensor *input, Tensor *output,
                     float val, const string &name);
};

ScaleOp::ScaleOp(OpPrecType prec_type, Tensor *input, Tensor *output,
                 float val, const string &name)
    : Op{OP_SCALE, prec_type, {input}, {output}, {{val}}, name, -1}
{
}

// Multiply `input` by `val`.
// TODO: make it into a general element-wise operation
Tensor *Model::scale(Tensor *input, float val, Tensor *output,
                     const string &name)
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
    ScaleOp op{pt, input, output, val, name};
    this->impl->add_op(op);
    return output;
}

} // namespace ark
