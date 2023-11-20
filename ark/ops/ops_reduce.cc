// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cassert>

#include "logging.h"
#include "model.h"

namespace ark {

ReduceOp::ReduceOp(const OpType &type, const std::string &prec_type,
                   const std::vector<Tensor *> &inputs,
                   const std::vector<Tensor *> &outputs, const OpArgs &args,
                   const std::string &name, const OpConfigMap *cfg_map,
                   int gran_lev)
    : Op{type, prec_type, inputs,   outputs, args,
         name, cfg_map,   gran_lev, true} {}

///
/// @param cfg
/// @param type "[w|e]_[sum|max|mean]"
/// @return
std::string ReduceOp::function_name(const OpConfig &cfg,
                                    const std::string &type) const {
    Tensor *input = this->inputs[0];
    Tensor *output = this->outputs[0];

    int axis;
    bool keepdims;
    this->args.get(&axis, 0);
    this->args.get(&keepdims, 1);

    int ndims = output->shape.ndims();
    OpTile tile_out = cfg.output_tiles[0];
    if (tile_out.x < 0) tile_out.x = output->ldims.dims4()[2];
    if (tile_out.y < 0) tile_out.y = output->ldims.dims4()[3];
    CHECK(output->ldims[ndims - 1] % tile_out.y == 0);
    if (ndims > 1) {
        CHECK(output->ldims[ndims - 2] % tile_out.x == 0);
    } else {
        CHECK(tile_out.x == 1);
    }

    Dims shp_in = input->shape;

    // Translate the axis value into 4D representation.
    axis += 4 - shp_in.ndims();

    if (type[0] == 'w') {
        // Warp-wise reduction is supported only for the last axis.
        CHECK(axis == 3);
    }
    Dims outdims = output->ldims;
    Dims outshape = output->shape;
    if (!keepdims) {
        outdims.insert(axis, 1);
        outshape.insert(axis, 1);
    }

    Dims unit_out_dims{1, 1, tile_out.x, tile_out.y};
    return Op::function_name("ark::reduce_" + type,
                             {{
                                 input->ldims.dims4(),  // InDims
                                 input->shape.dims4(),  // InShape
                                 outdims.dims4(),       // OutDims
                                 outshape.dims4(),      // OutShape
                                 unit_out_dims,         // UnitOutDims
                                 cfg.num_warps,         // NumWarps
                                 cfg.smem_bytes,        // SmemBytes
                                 axis,                  // Axis
                             }});
}

extern const OpConfigMap ReduceWConfigMap;
extern const OpConfigMap ReduceEConfigMap;

ReduceWSumOp::ReduceWSumOp(const std::string &prec_type, Tensor *input,
                           Tensor *output, int axis, bool keepdims,
                           const std::string &name)
    : ReduceOp{OP_REDUCE_W_SUM,    prec_type, {input},           {output},
               {{axis, keepdims}}, name,      &ReduceWConfigMap, -1} {}

std::string ReduceWSumOp::function_name(const OpConfig &cfg) const {
    return ReduceOp::function_name(cfg, "w_sum");
}

ReduceESumOp::ReduceESumOp(const std::string &prec_type, Tensor *input,
                           Tensor *output, int axis, bool keepdims,
                           const std::string &name)
    : ReduceOp{OP_REDUCE_E_SUM,    prec_type, {input},           {output},
               {{axis, keepdims}}, name,      &ReduceEConfigMap, -1} {}

std::string ReduceESumOp::function_name(const OpConfig &cfg) const {
    return ReduceOp::function_name(cfg, "e_sum");
}

ReduceWMaxOp::ReduceWMaxOp(const std::string &prec_type, Tensor *input,
                           Tensor *output, int axis, bool keepdims,
                           const std::string &name)
    : ReduceOp{OP_REDUCE_W_MAX,    prec_type, {input},           {output},
               {{axis, keepdims}}, name,      &ReduceWConfigMap, -1} {}

std::string ReduceWMaxOp::function_name(const OpConfig &cfg) const {
    return ReduceOp::function_name(cfg, "w_max");
}

ReduceEMaxOp::ReduceEMaxOp(const std::string &prec_type, Tensor *input,
                           Tensor *output, int axis, bool keepdims,
                           const std::string &name)
    : ReduceOp{OP_REDUCE_E_MAX,    prec_type, {input},           {output},
               {{axis, keepdims}}, name,      &ReduceEConfigMap, -1} {}

std::string ReduceEMaxOp::function_name(const OpConfig &cfg) const {
    return ReduceOp::function_name(cfg, "e_max");
}

ReduceWMeanOp::ReduceWMeanOp(const std::string &prec_type, Tensor *input,
                             Tensor *output, int axis, bool keepdims,
                             const std::string &name)
    : ReduceOp{OP_REDUCE_W_MEAN,   prec_type, {input},           {output},
               {{axis, keepdims}}, name,      &ReduceWConfigMap, -1} {}

std::string ReduceWMeanOp::function_name(const OpConfig &cfg) const {
    return ReduceOp::function_name(cfg, "w_mean");
}

ReduceEMeanOp::ReduceEMeanOp(const std::string &prec_type, Tensor *input,
                             Tensor *output, int axis, bool keepdims,
                             const std::string &name)
    : ReduceOp{OP_REDUCE_E_MEAN,   prec_type, {input},           {output},
               {{axis, keepdims}}, name,      &ReduceEConfigMap, -1} {}

std::string ReduceEMeanOp::function_name(const OpConfig &cfg) const {
    return ReduceOp::function_name(cfg, "e_mean");
}

template <typename ReduceOpType>
Tensor *Model::reduce(Tensor *input, int axis, bool keepdims, Tensor *output,
                      const std::string &name) {
    assert(input != nullptr);
    if (output != nullptr && input->type != output->type) {
        ERR(InvalidUsageError, "invalid output data type: ", output->type);
    }
    Dims reduced_shape{input->shape};
    if (axis < 0) {
        axis += reduced_shape.ndims();
    }
    if (axis < 0 || axis >= reduced_shape.ndims()) {
        ERR(InvalidUsageError, "invalid reduction axis ", axis);
    }
    if (keepdims) {
        reduced_shape[axis] = 1;
    } else {
        reduced_shape.erase(axis);
    }
    if (output == nullptr) {
        output = this->tensor(reduced_shape, input->type);
    } else {
        if (output->shape != reduced_shape) {
            ERR(InvalidUsageError, "invalid output shape ", output->shape,
                " with input shape ", input->shape, ", reduction axis ", axis,
                ", and keepdims ", keepdims);
        }
        if (output == input) {
            ERR(InvalidUsageError,
                "output tensor cannot be the same as input tensor for "
                "reduce_sum op");
        }
    }
    ReduceOpType op{output->type.name(), input, output, axis, keepdims, name};
    return this->impl->add_op(op)[0];
}

Tensor *Model::reduce_sum(Tensor *input, int axis, bool keepdims,
                          Tensor *output, const std::string &name) {
    if (axis == input->shape.ndims() - 1 || axis == -1) {
        return reduce<ReduceWSumOp>(input, axis, keepdims, output, name);
    } else {
        return reduce<ReduceESumOp>(input, axis, keepdims, output, name);
    }
}

Tensor *Model::reduce_mean(Tensor *input, int axis, bool keepdims,
                           Tensor *output, const std::string &name) {
    if (axis == input->shape.ndims() - 1 || axis == -1) {
        return reduce<ReduceWMeanOp>(input, axis, keepdims, output, name);
    } else {
        return reduce<ReduceEMeanOp>(input, axis, keepdims, output, name);
    }
}

Tensor *Model::reduce_max(Tensor *input, int axis, bool keepdims,
                          Tensor *output, const std::string &name) {
    if (axis == input->shape.ndims() - 1 || axis == -1) {
        return reduce<ReduceWMaxOp>(input, axis, keepdims, output, name);
    } else {
        return reduce<ReduceEMaxOp>(input, axis, keepdims, output, name);
    }
}

const OpConfigMap ReduceEConfigMap = {
    {{OP_ARCH_ANY, "any"},
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
    {{OP_ARCH_CUDA_ANY, "any"},
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
    {{OP_ARCH_ROCM_ANY, "any"},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {1, 256, {{32, 1}}, {{32, 1}}, true, false},
         {1, 256, {{16, 1}}, {{16, 1}}, true, false},
         {1, 256, {{8, 1}}, {{8, 1}}, true, false},
         {1, 256, {{4, 1}}, {{4, 1}}, true, false},
         {1, 256, {{2, 1}}, {{2, 1}}, true, false},
         {1, 256, {{1, 1}}, {{1, 1}}, true, false},
         {4, 256, {{1, 1}}, {{1, 1}}, true, false},
         {8, 256, {{1, 1}}, {{1, 1}}, true, false},
     }},
};

}  // namespace ark
