// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "logging.h"
#include "model.h"
#include "tensor.h"

using namespace std;

namespace ark {

ReduceOp::ReduceOp(const OpType &type, const OpPrecType &prec_type,
                   const std::vector<Tensor *> &in_deps,
                   const std::vector<Tensor *> &out_deps, const OpArgs &args,
                   const std::string &name, int gran_lev)
    : Op{type, prec_type, in_deps, out_deps, args, name, gran_lev, true}
{
}

///
/// @param cfg
/// @param type "[w|e]_[sum|max|mean]"
/// @return
std::string ReduceOp::function_name(const OpConfig &cfg,
                                    const std::string &type) const
{
    Tensor *input = this->in_deps[0];
    Tensor *output = this->out_deps[0];

    int ndims = output->shape.ndims();
    const OpTile &tile_out = cfg.out_deps_tiles[0];
    CHECK(output->ldims[ndims - 1] % tile_out.y == 0);
    if (ndims > 1) {
        CHECK(output->ldims[ndims - 2] % tile_out.x == 0);
    } else {
        CHECK(tile_out.x == 1);
    }

    Dims shp_in = input->shape;
    int axis;
    this->args.get(&axis, 0);

    // Translate the axis value into 4D representation.
    axis += 4 - shp_in.ndims();

    if (type[0] == 'w') {
        // Warp-wise reduction is supported only for the last axis.
        CHECK(axis == 3);
    }

    Dims unit_out_shape{1, 1, tile_out.x, tile_out.y};
    return Op::function_name("ark::reduce_" + type,
                             {{
                                 input->ldims.dims4(),  // InDims
                                 input->shape.dims4(),  // InShape
                                 output->ldims.dims4(), // OutDims
                                 output->shape.dims4(), // OutShape
                                 unit_out_shape,        // UnitOutShape
                                 cfg.num_warps * 32,    // ThreadsNum
                                 cfg.smem_bytes,        // SmemBytes
                                 axis,                  // Axis
                             }});
}

ReduceWSumOp::ReduceWSumOp(OpPrecType prec_type, Tensor *input, Tensor *output,
                           int axis, const string &name)
    : ReduceOp{OP_REDUCE_W_SUM, prec_type, {input}, {output},
               {{axis}},        name,      -1}
{
}

std::string ReduceWSumOp::function_name(const OpConfig &cfg) const
{
    return ReduceOp::function_name(cfg, "w_sum");
}

ReduceESumOp::ReduceESumOp(OpPrecType prec_type, Tensor *input, Tensor *output,
                           int axis, const string &name)
    : ReduceOp{OP_REDUCE_E_SUM, prec_type, {input}, {output},
               {{axis}},        name,      -1}
{
}

std::string ReduceESumOp::function_name(const OpConfig &cfg) const
{
    return ReduceOp::function_name(cfg, "e_sum");
}

ReduceWMaxOp::ReduceWMaxOp(OpPrecType prec_type, Tensor *input, Tensor *output,
                           int axis, const string &name)
    : ReduceOp{OP_REDUCE_W_MAX, prec_type, {input}, {output},
               {{axis}},        name,      -1}
{
}

std::string ReduceWMaxOp::function_name(const OpConfig &cfg) const
{
    return ReduceOp::function_name(cfg, "w_max");
}

ReduceEMaxOp::ReduceEMaxOp(OpPrecType prec_type, Tensor *input, Tensor *output,
                           int axis, const string &name)
    : ReduceOp{OP_REDUCE_E_MAX, prec_type, {input}, {output},
               {{axis}},        name,      -1}
{
}

std::string ReduceEMaxOp::function_name(const OpConfig &cfg) const
{
    return ReduceOp::function_name(cfg, "e_max");
}

ReduceWMeanOp::ReduceWMeanOp(OpPrecType prec_type, Tensor *input,
                             Tensor *output, int axis, const string &name)
    : ReduceOp{OP_REDUCE_W_MEAN, prec_type, {input}, {output},
               {{axis}},         name,      -1}
{
}

std::string ReduceWMeanOp::function_name(const OpConfig &cfg) const
{
    return ReduceOp::function_name(cfg, "w_mean");
}

ReduceEMeanOp::ReduceEMeanOp(OpPrecType prec_type, Tensor *input,
                             Tensor *output, int axis, const string &name)
    : ReduceOp{OP_REDUCE_E_MEAN, prec_type, {input}, {output},
               {{axis}},         name,      -1}
{
}

std::string ReduceEMeanOp::function_name(const OpConfig &cfg) const
{
    return ReduceOp::function_name(cfg, "e_mean");
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
