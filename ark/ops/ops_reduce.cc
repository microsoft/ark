// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/logging.h"
#include "ark/model_io.h"

using namespace std;

namespace ark {

Tensor *Model::reduce_sum(Tensor *input, int axis, Tensor *output,
                          const string &name)
{
    assert(input != nullptr);
    LOG(DEBUG, "reduce_sum ", input->shape, " ", input->ldims, " ", axis);
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
        Dims reduced_shape{input->shape};
        reduced_shape[axis] = 1;
        output = this->tensor(reduced_shape, input->type);
    } else if (output == input) {
        LOGERR("output tensor cannot be the same as input tensor for "
               "reduce_sum op");
    }
    if (axis == input->shape.ndims() - 1) {
        this->create_op(OP_REDUCE_W_SUM, pt, {input}, {output}, {axis}, name);
    } else {
        this->create_op(OP_REDUCE_E_SUM, pt, {input}, {output}, {axis}, name);
    }
    return output;
}

Tensor *Model::reduce_mean(Tensor *input, int axis, Tensor *output,
                           const string &name)
{
    assert(input != nullptr);
    LOG(DEBUG, "reduce_mean ", input->shape, " ", input->ldims, " ", axis);
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
        Dims reduced_shape{input->shape};
        reduced_shape[axis] = 1;
        output = this->tensor(reduced_shape, input->type);
    } else if (output == input) {
        LOGERR("output tensor cannot be the same as input tensor for "
               "reduce_mean op");
    }
    if (axis == input->shape.ndims() - 1) {
        this->create_op(OP_REDUCE_W_MEAN, pt, {input}, {output}, {axis}, name);
    } else {
        this->create_op(OP_REDUCE_E_MEAN, pt, {input}, {output}, {axis}, name);
    }
    return output;
}

Tensor *Model::reduce_max(Tensor *input, int axis, Tensor *output,
                          const string &name)
{
    assert(input != nullptr);
    LOG(DEBUG, "reduce_max ", input->shape, " ", input->ldims, " ", axis);
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
        Dims reduced_shape{input->shape};
        reduced_shape[axis] = 1;
        output = this->tensor(reduced_shape, input->type);
    } else if (output == input) {
        LOGERR("output tensor cannot be the same as input tensor for "
               "reduce_max op");
    }
    if (axis == input->shape.ndims() - 1) {
        this->create_op(OP_REDUCE_W_MAX, pt, {input}, {output}, {axis}, name);
    } else {
        this->create_op(OP_REDUCE_E_MAX, pt, {input}, {output}, {axis}, name);
    }
    return output;
}

} // namespace ark
