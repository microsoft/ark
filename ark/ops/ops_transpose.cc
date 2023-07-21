// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "logging.h"
#include "model.h"
#include "tensor.h"

using namespace std;

namespace ark {

class TransposeOp : public Op
{
  public:
    TransposeOp::TransposeOp(OpPrecType prec_type, Tensor *input, Tensor *output,
                             int tp_type, const string &name);
};

TransposeOp::TransposeOp(OpPrecType prec_type, Tensor *input, Tensor *output,
                             int tp_type, const string &name)
    : Op{OP_TRANSPOSE, prec_type, {input}, {output}, {{tp_type}}, name, -1}
{
}

Tensor *Model::transpose(Tensor *input, Dims perm, Tensor *output,
                         const std::string &name)
{
    OpPrecType pt;
    if (input->type == FP16) {
        pt = OP_PREC_FP16;
    } else if (input->type == FP32) {
        pt = OP_PREC_FP32;
    } else {
        LOGERR("unsupported input data type: ", type_str(input->type));
    }
    int input_ndims = input->ndims();
    Dims in_shape;
    if (input_ndims == 2) {
        in_shape[0] = 1;
        in_shape[1] = 1;
        in_shape[2] = input->shape[0];
        in_shape[3] = input->shape[1];
    } else if (input_ndims == 3) {
        in_shape[0] = 1;
        in_shape[1] = input->shape[0];
        in_shape[2] = input->shape[1];
        in_shape[3] = input->shape[2];
    } else if (input_ndims == 4) {
        in_shape[0] = input->shape[0];
        in_shape[1] = input->shape[1];
        in_shape[2] = input->shape[2];
        in_shape[3] = input->shape[3];
    } else {
        LOGERR("Invalid # of input dimensions. Expected 2, 3, or 4, but given ",
               input_ndims);
    }
    if (perm.ndims() != input_ndims) {
        LOGERR("Permutation should have the same number of dimensions as the "
               "one of input. Given input shape: ",
               input->shape, ", permutation: ", perm);
    }
    int count[DIMS_LEN];
    for (int i = 0; i < input_ndims; ++i) {
        count[i] = 0;
    }
    for (int i = 0; i < input_ndims; ++i) {
        if (perm[i] >= input_ndims) {
            LOGERR("Each value in permutation should be less than the number "
                   "of input dimensions. Given permutation: ",
                   perm);
        }
        if (count[perm[i]] > 0) {
            LOGERR("Each value in permutation should be unique. Given "
                   "permutation: ",
                   perm);
        }
        count[perm[i]]++;
    }
    int tp_type = perm[0] * 1000 + perm[1] * 100 + perm[2] * 10 + perm[3];
    Dims out_shape;
    out_shape[0] = in_shape[perm[0]];
    out_shape[1] = in_shape[perm[1]];
    out_shape[2] = in_shape[perm[2]];
    out_shape[3] = in_shape[perm[3]];
    if (output == nullptr) {
        output = this->tensor(out_shape, input->type);
    } else {
        assert(output->shape == out_shape);
    }
    TransposeOp op{pt, input, output, tp_type, name};
    this->impl->add_op(op);
    return output;
}

} // namespace ark
