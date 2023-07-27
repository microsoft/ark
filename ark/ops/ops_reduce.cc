// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "logging.h"
#include "model.h"
#include "tensor.h"

using namespace std;

namespace ark {

ReduceOp::ReduceOp(const OpType &type, const OpPrecType &prec_type,
                   const std::vector<Tensor *> &inputs,
                   const std::vector<Tensor *> &outputs, const OpArgs &args,
                   const std::string &name, const OpConfigMap *cfg_map,
                   int gran_lev)
    : Op{type, prec_type, inputs, outputs, args, name, cfg_map, gran_lev, true}
{
}

///
/// @param cfg
/// @param type "[w|e]_[sum|max|mean]"
/// @return
std::string ReduceOp::function_name(const OpConfig &cfg,
                                    const std::string &type) const
{
    Tensor *input = this->inputs[0];
    Tensor *output = this->outputs[0];

    int ndims = output->shape.ndims();
    const OpTile &tile_out = cfg.output_tiles[0];
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

    Dims unit_out_dims{1, 1, tile_out.x, tile_out.y};
    return Op::function_name("ark::reduce_" + type,
                             {{
                                 input->ldims.dims4(),  // InDims
                                 input->shape.dims4(),  // InShape
                                 output->ldims.dims4(), // OutDims
                                 output->shape.dims4(), // OutShape
                                 unit_out_dims,         // UnitOutDims
                                 cfg.num_warps * 32,    // NumThreads
                                 cfg.smem_bytes,        // SmemBytes
                                 axis,                  // Axis
                             }});
}

extern const OpConfigMap ReduceWConfigMap;
extern const OpConfigMap ReduceEConfigMap;

ReduceWSumOp::ReduceWSumOp(OpPrecType prec_type, Tensor *input, Tensor *output,
                           int axis, const string &name)
    : ReduceOp{OP_REDUCE_W_SUM, prec_type, {input},           {output},
               {{axis}},        name,      &ReduceWConfigMap, -1}
{
}

std::string ReduceWSumOp::function_name(const OpConfig &cfg) const
{
    return ReduceOp::function_name(cfg, "w_sum");
}

ReduceESumOp::ReduceESumOp(OpPrecType prec_type, Tensor *input, Tensor *output,
                           int axis, const string &name)
    : ReduceOp{OP_REDUCE_E_SUM, prec_type, {input},           {output},
               {{axis}},        name,      &ReduceEConfigMap, -1}
{
}

std::string ReduceESumOp::function_name(const OpConfig &cfg) const
{
    return ReduceOp::function_name(cfg, "e_sum");
}

ReduceWMaxOp::ReduceWMaxOp(OpPrecType prec_type, Tensor *input, Tensor *output,
                           int axis, const string &name)
    : ReduceOp{OP_REDUCE_W_MAX, prec_type, {input},           {output},
               {{axis}},        name,      &ReduceWConfigMap, -1}
{
}

std::string ReduceWMaxOp::function_name(const OpConfig &cfg) const
{
    return ReduceOp::function_name(cfg, "w_max");
}

ReduceEMaxOp::ReduceEMaxOp(OpPrecType prec_type, Tensor *input, Tensor *output,
                           int axis, const string &name)
    : ReduceOp{OP_REDUCE_E_MAX, prec_type, {input},           {output},
               {{axis}},        name,      &ReduceEConfigMap, -1}
{
}

std::string ReduceEMaxOp::function_name(const OpConfig &cfg) const
{
    return ReduceOp::function_name(cfg, "e_max");
}

ReduceWMeanOp::ReduceWMeanOp(OpPrecType prec_type, Tensor *input,
                             Tensor *output, int axis, const string &name)
    : ReduceOp{OP_REDUCE_W_MEAN, prec_type, {input},           {output},
               {{axis}},         name,      &ReduceWConfigMap, -1}
{
}

std::string ReduceWMeanOp::function_name(const OpConfig &cfg) const
{
    return ReduceOp::function_name(cfg, "w_mean");
}

ReduceEMeanOp::ReduceEMeanOp(OpPrecType prec_type, Tensor *input,
                             Tensor *output, int axis, const string &name)
    : ReduceOp{OP_REDUCE_E_MEAN, prec_type, {input},           {output},
               {{axis}},         name,      &ReduceEConfigMap, -1}
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

const OpConfigMap ReduceEConfigMap = {
    {{OP_ARCH_CUDA_80, OP_PREC_FP16},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {8, 0, {{128, 256}, {128, 256}}, {{128, 256}}, true, false},
         {8, 0, {{256, 128}, {256, 128}}, {{256, 128}}, true, false},
         {8, 0, {{128, 128}, {128, 128}}, {{128, 128}}, true, false},
         {4, 0, {{64, 64}, {64, 64}}, {{64, 64}}, true, false},
         {2, 0, {{32, 64}, {32, 64}}, {{32, 64}}, true, false},
         {1, 0, {{16, 64}, {16, 64}}, {{16, 64}}, true, false},
         {1, 0, {{8, 64}, {8, 64}}, {{8, 64}}, true, false},
         {1, 0, {{2, 128}, {2, 128}}, {{2, 128}}, true, false},
         {1, 0, {{4, 64}, {4, 64}}, {{4, 64}}, true, false},
         {1, 0, {{2, 64}, {2, 64}}, {{2, 64}}, true, false},
         {1, 0, {{1, 64}, {1, 64}}, {{1, 64}}, true, false},
         {1, 0, {{1, 32}, {1, 32}}, {{1, 32}}, true, false},
     }},
};

const OpConfigMap ReduceWConfigMap = {
    {{OP_ARCH_CUDA_70, OP_PREC_FP16},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {1, 128, {{32, 1}}, {{32, 1}}, true, false},
         {1, 128, {{16, 1}}, {{16, 1}}, true, false},
         {1, 128, {{8, 1}}, {{8, 1}}, true, false},
         {1, 128, {{4, 1}}, {{4, 1}}, true, false},
         {1, 128, {{2, 1}}, {{2, 1}}, true, false},
         {1, 128, {{1, 1}}, {{1, 1}}, true, false},
         {4, 128, {{1, 1}}, {{1, 1}}, true, false},
         {8, 128, {{1, 1}}, {{1, 1}}, true, false},
     }},
    {{OP_ARCH_CUDA_80, OP_PREC_FP16},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {1, 128, {{32, 1}}, {{32, 1}}, true, false},
         {1, 128, {{16, 1}}, {{16, 1}}, true, false},
         {1, 128, {{8, 1}}, {{8, 1}}, true, false},
         {1, 128, {{4, 1}}, {{4, 1}}, true, false},
         {1, 128, {{2, 1}}, {{2, 1}}, true, false},
         {1, 128, {{1, 1}}, {{1, 1}}, true, false},
         {4, 128, {{1, 1}}, {{1, 1}}, true, false},
         {8, 128, {{1, 1}}, {{1, 1}}, true, false},
     }},
    {{OP_ARCH_CUDA_70, OP_PREC_FP32},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {1, 128, {{32, 1}}, {{32, 1}}, true, false},
         {1, 128, {{16, 1}}, {{16, 1}}, true, false},
         {1, 128, {{8, 1}}, {{8, 1}}, true, false},
         {1, 128, {{4, 1}}, {{4, 1}}, true, false},
         {1, 128, {{2, 1}}, {{2, 1}}, true, false},
         {1, 128, {{1, 1}}, {{1, 1}}, true, false},
         {4, 128, {{1, 1}}, {{1, 1}}, true, false},
         {8, 128, {{1, 1}}, {{1, 1}}, true, false},
     }},
    {{OP_ARCH_CUDA_80, OP_PREC_FP32},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {1, 128, {{32, 1}}, {{32, 1}}, true, false},
         {1, 128, {{16, 1}}, {{16, 1}}, true, false},
         {1, 128, {{8, 1}}, {{8, 1}}, true, false},
         {1, 128, {{4, 1}}, {{4, 1}}, true, false},
         {1, 128, {{2, 1}}, {{2, 1}}, true, false},
         {1, 128, {{1, 1}}, {{1, 1}}, true, false},
         {4, 128, {{1, 1}}, {{1, 1}}, true, false},
         {8, 128, {{1, 1}}, {{1, 1}}, true, false},
     }},
};

} // namespace ark
