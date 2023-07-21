// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "logging.h"
#include "model.h"
#include "tensor.h"

using namespace std;

namespace ark {

class MaxPoolOp : public Op
{
  public:
    MaxPoolOp::MaxPoolOp(OpPrecType prec_type, Tensor *input, Tensor *output,
                         DimType kernel_size, DimType stride, const string &name);
};

MaxPoolOp::MaxPoolOp(OpPrecType prec_type, Tensor *input, Tensor *output,
                     DimType kernel_size, DimType stride, const string &name)
    : Op{OP_MAX_POOL, prec_type, {input}, {output}, {{kernel_size, stride}}, name, -1}
{
}

Tensor *Model::max_pool(Tensor *input, DimType kernel_size, DimType stride,
                        Tensor *output, const string &name)
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
    const Dims &is = input->shape;
    Dims os{{is[0], (is[1] + stride - 1) / stride,
             (is[2] + stride - 1) / stride, is[3]}};
    if (output == nullptr) {
        output = this->tensor(os, input->type);
    }
    MaxPoolOp op{pt, input, output, kernel_size, stride, name};
    this->impl->add_op(op);
    return output;
}

} // namespace ark
