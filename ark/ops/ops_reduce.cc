// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "logging.h"
#include "model.h"
#include "tensor.h"

using namespace std;

namespace ark {

class ReduceWSumOp : public Op
{
  public:
    ReduceWSumOp::ReduceWSumOp(OpPrecType prec_type, Tensor *input, Tensor *output,
                             int axis, const string &name);
    std::string ReduceWSumOp::function_string(const OpConfig &cfg) const;
};

class ReduceESumOp : public Op
{
  public:
    ReduceESumOp::ReduceESumOp(OpPrecType prec_type, Tensor *input, Tensor *output,
                             int axis, const string &name);
    std::string ReduceESumOp::function_string(const OpConfig &cfg) const;
};

class ReduceWMaxOp : public Op
{
  public:
    ReduceWMaxOp::ReduceWMaxOp(OpPrecType prec_type, Tensor *input, Tensor *output,
                             int axis, const string &name);
    std::string ReduceWMaxOp::function_string(const OpConfig &cfg) const;
};

class ReduceEMaxOp : public Op
{
  public:
    ReduceEMaxOp::ReduceEMaxOp(OpPrecType prec_type, Tensor *input, Tensor *output,
                             int axis, const string &name);
    std::string ReduceEMaxOp::function_string(const OpConfig &cfg) const;
};

class ReduceWMeanOp : public Op
{
  public:
    ReduceWMeanOp::ReduceWMeanOp(OpPrecType prec_type, Tensor *input, Tensor *output,
                             int axis, const string &name);
    std::string ReduceWMeanOp::function_string(const OpConfig &cfg) const;
};

class ReduceEMeanOp : public Op
{
  public:
    ReduceEMeanOp::ReduceEMeanOp(OpPrecType prec_type, Tensor *input, Tensor *output,
                             int axis, const string &name);
    std::string ReduceEMeanOp::function_string(const OpConfig &cfg) const;
};

ReduceWSumOp::ReduceWSumOp(OpPrecType prec_type, Tensor *input, Tensor *output,
                     int axis, const string &name)
    : Op{OP_REDUCE_W_SUM, prec_type, {input}, {output}, {{axis}}, name, -1}
{
}

ReduceESumOp::ReduceESumOp(OpPrecType prec_type, Tensor *input, Tensor *output,
                     int axis, const string &name)
    : Op{OP_REDUCE_E_SUM, prec_type, {input}, {output}, {{axis}}, name, -1}
{
}

ReduceWMaxOp::ReduceWMaxOp(OpPrecType prec_type, Tensor *input, Tensor *output,
                     int axis, const string &name)
    : Op{OP_REDUCE_W_MAX, prec_type, {input}, {output}, {{axis}}, name, -1}
{
}

ReduceEMaxOp::ReduceEMaxOp(OpPrecType prec_type, Tensor *input, Tensor *output,
                     int axis, const string &name)
    : Op{OP_REDUCE_E_MAX, prec_type, {input}, {output}, {{axis}}, name, -1}
{
}

ReduceWMeanOp::ReduceWMeanOp(OpPrecType prec_type, Tensor *input, Tensor *output,
                     int axis, const string &name)
    : Op{OP_REDUCE_W_MEAN, prec_type, {input}, {output}, {{axis}}, name, -1}
{
}

ReduceEMeanOp::ReduceEMeanOp(OpPrecType prec_type, Tensor *input, Tensor *output,
                     int axis, const string &name)
    : Op{OP_REDUCE_E_MEAN, prec_type, {input}, {output}, {{axis}}, name, -1}
{
}

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
        ReduceWSumOp op{pt, input, output, axis, name};
        this->impl->add_op(op);
    } else {
        ReduceESumOp op{pt, input, output, axis, name};
        this->impl->add_op(op);
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
        ReduceWMeanOp op{pt, input, output, axis, name};
        this->impl->add_op(op);
    } else {
        ReduceEMeanOp op{pt, input, output, axis, name};
        this->impl->add_op(op);
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
        ReduceWMaxOp op{pt, input, output, axis, name};
        this->impl->add_op(op);
    } else {
        ReduceEMaxOp op{pt, input, output, axis, name};
        this->impl->add_op(op);
    }
    return output;
}

} // namespace ark
