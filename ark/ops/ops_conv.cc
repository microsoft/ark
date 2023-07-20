// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "logging.h"
#include "model.h"
#include "tensor.h"

using namespace std;

namespace ark {

Tensor *Model::conv2d(Tensor *input, DimType in_channels, DimType out_channels,
                      DimType kernel_size, DimType stride, DimType padding,
                      bool bias, Tensor *output, const string &name)
{
    LOG(DEBUG, "conv2d ", in_channels, " ", out_channels, " ", kernel_size, " ",
        stride, " ", padding, " ", bias);
    assert(input != nullptr);
    OpPrecType pt;
    if (input->type == FP16) {
        pt = OP_PREC_FP16;
    } else if (input->type == FP32) {
        pt = OP_PREC_FP32;
    } else {
        LOGERR("unsupported input data type: ", type_str(input->type));
    }
    if ((output != nullptr) && (input->type != output->type)) {
        LOGERR("invalid output data type: ", type_str(output->type));
    }
    // Shape verification.
    const Dims &is = input->shape;
    if (is[3] != in_channels) {
        LOGERR("Input tensor shape mismatches with the given `in_channels`.");
    }
    Dims os{{is[0], (is[1] + 2 * padding - kernel_size) / stride + 1,
             (is[2] + 2 * padding - kernel_size) / stride + 1, out_channels}};
    DimType in_dim = in_channels * kernel_size * kernel_size;
    // TODO: im2col
    Tensor *i2c;
    // if (kernel_size == 1) {
    //     assert(padding == 0);
    //     i2c = this->reshape(input, {is[0], os[1] * os[2], 1, in_dim});
    // } else {
    i2c = this->tensor({is[0], os[1] * os[2], 1, in_dim}, input->type);
    this->impl->add_op(OP_IM2COL, pt, {input}, {i2c}, {}, name + "/im2col");
    // }
    Tensor *weight = this->tensor({1, out_channels, 1, in_dim}, input->type);
    Tensor *conv = this->matmul(i2c, weight, output, 1, false, true, false,
                                name + "/matmul");
    return this->reshape(conv, os, false, nullptr, name + "/reshape");
}

} // namespace ark
