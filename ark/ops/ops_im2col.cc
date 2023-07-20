// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "logging.h"
#include "model.h"
#include "tensor.h"

using namespace std;

namespace ark {

Tensor *Model::im2col(Tensor *input, DimType kernel_height,
                      DimType kernel_width, DimType stride_height,
                      DimType stride_width, DimType pad_height,
                      DimType pad_width, DimType dilation_height,
                      DimType dilation_width, Tensor *output,
                      const string &name)
{
    assert(input != nullptr);
    DimType n, c, h, w;
    int input_ndims = input->ndims();
    if (input_ndims == 2) {
        n = 1;
        c = 1;
        h = input->shape[0];
        w = input->shape[1];
    } else if (input_ndims == 3) {
        n = 1;
        c = input->shape[0];
        h = input->shape[1];
        w = input->shape[2];
    } else if (input_ndims == 4) {
        n = input->shape[0];
        c = input->shape[1];
        h = input->shape[2];
        w = input->shape[3];
    } else {
        LOGERR("invalid # of input dimensions. Expected 2, 3, or 4, but given ",
               input_ndims);
    }
    OpPrecType pt;
    if (input->type == FP16) {
        pt = OP_PREC_FP16;
    } else if (input->type == FP32) {
        pt = OP_PREC_FP32;
    } else {
        LOGERR("unsupported input data type: ", type_str(input->type));
    }
    DimType out_h = (h + 2 * pad_height - kernel_height) / stride_height + 1;
    DimType out_w = (w + 2 * pad_width - kernel_width) / stride_width + 1;
    assert((out_h > 0) && (out_w > 0));
    DimType out_m = out_h * out_w;
    DimType inner_dim = c * kernel_height * kernel_width;
    Dims out_shape;
    if (input_ndims <= 3) {
        out_shape = {inner_dim, out_m};
    } else {
        out_shape = {n, inner_dim, out_m};
    }
    if (output == nullptr) {
        output = this->tensor(out_shape, input->type);
    } else {
        assert(output->shape == out_shape);
    }
    this->impl->add_op(OP_IM2COL, pt, {input}, {output},
                       {kernel_height, kernel_width, stride_height,
                        stride_width, pad_height, pad_width, dilation_height,
                        dilation_width},
                       name);
    return output;
}

} // namespace ark
