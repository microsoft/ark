// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "logging.h"
#include "model.h"

namespace ark {

extern const OpConfigMap Broadcast2ConfigMap;

ArithmeticOp::ArithmeticOp(const OpType &type, const std::string &prec_type,
                           Tensor *input, Tensor *other, Tensor *output,
                           const std::string &name)
    : Op{type, prec_type, {input, other},       {output},
         {},   name,      &Broadcast2ConfigMap, -1,
         true} {}

std::string ArithmeticOp::function_name(const OpConfig &cfg,
                                        const std::string &type) const {
    Tensor *input = this->inputs[0];
    Tensor *other = this->inputs[1];
    Tensor *output = this->outputs[0];

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

    Dims unit_out_dims{1, 1, tile_out.x, tile_out.y};
    return Op::function_name("ark::" + type,
                             {{
                                 input->ldims.dims4(),   // In0Dims
                                 input->shape.dims4(),   // In0Shape
                                 other->ldims.dims4(),   // In1Dims
                                 other->shape.dims4(),   // In1Shape
                                 output->ldims.dims4(),  // OutDims
                                 output->shape.dims4(),  // OutShape
                                 unit_out_dims,          // UnitOutDims
                                 cfg.num_warps,          // NumWarps
                                 cfg.smem_bytes,         // SmemBytes
                             }});
}

template <typename ArithmeticOpType>
Tensor *Model::arithmetic(Tensor *input, Tensor *other, Tensor *output,
                          const std::string &name) {
    CHECK(input != nullptr);
    CHECK(other != nullptr);
    if (input->type != other->type) {
        ERR(InvalidUsageError, "input data types mismatch: ", input->type, ", ",
            other->type);
    }
    if (output != nullptr && input->type != output->type) {
        ERR(InvalidUsageError, "invalid output data type: ", output->type);
    }
    Dims output_shape = broadcast(input->shape, other->shape);
    if (output == nullptr) {
        output = this->tensor(output_shape, input->type);
    } else if (output->shape != output_shape) {
        ERR(InvalidUsageError, "invalid output shape: ", output->shape);
    } else if (output == input) {
        output = this->identity(output);
    }
    ArithmeticOpType op{output->type.name(), input, other, output, name};
    return this->impl->add_op(op)[0];
}

////////////////////////////////////////////////////////////////////////////////

AddOp::AddOp(const std::string &prec_type, Tensor *input, Tensor *other,
             Tensor *output, const std::string &name)
    : ArithmeticOp{OP_ADD, prec_type, input, other, output, name} {}

std::string AddOp::function_name(const OpConfig &cfg) const {
    return ArithmeticOp::function_name(cfg, "add");
}

Tensor *Model::add(Tensor *input, Tensor *other, Tensor *output,
                   const std::string &name) {
    return arithmetic<AddOp>(input, other, output, name);
}

////////////////////////////////////////////////////////////////////////////////

SubOp::SubOp(const std::string &prec_type, Tensor *input, Tensor *other,
             Tensor *output, const std::string &name)
    : ArithmeticOp{OP_SUB, prec_type, input, other, output, name} {}

std::string SubOp::function_name(const OpConfig &cfg) const {
    return ArithmeticOp::function_name(cfg, "sub");
}

Tensor *Model::sub(Tensor *input, Tensor *other, Tensor *output,
                   const std::string &name) {
    return arithmetic<SubOp>(input, other, output, name);
}

////////////////////////////////////////////////////////////////////////////////

MulOp::MulOp(const std::string &prec_type, Tensor *input, Tensor *other,
             Tensor *output, const std::string &name)
    : ArithmeticOp{OP_MUL, prec_type, input, other, output, name} {}

std::string MulOp::function_name(const OpConfig &cfg) const {
    return ArithmeticOp::function_name(cfg, "mul");
}

Tensor *Model::mul(Tensor *input, Tensor *other, Tensor *output,
                   const std::string &name) {
    return arithmetic<MulOp>(input, other, output, name);
}

////////////////////////////////////////////////////////////////////////////////

DivOp::DivOp(const std::string &prec_type, Tensor *input, Tensor *other,
             Tensor *output, const std::string &name)
    : ArithmeticOp{OP_DIV, prec_type, input, other, output, name} {}

std::string DivOp::function_name(const OpConfig &cfg) const {
    return ArithmeticOp::function_name(cfg, "div");
}

Tensor *Model::div(Tensor *input, Tensor *other, Tensor *output,
                   const std::string &name) {
    return arithmetic<DivOp>(input, other, output, name);
}

////////////////////////////////////////////////////////////////////////////////

const OpConfigMap Broadcast2ConfigMap = {
    {{OP_ARCH_ANY, "any"},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {1, 0, {{1, 8192}, {1, 8192}}, {{1, 8192}}, false, false},
         {1, 0, {{8192, 1}, {8192, 1}}, {{8192, 1}}, false, false},
         {1, 0, {{1, 512}, {1, 512}}, {{1, 512}}, false, false},
         {1, 0, {{512, 1}, {512, 1}}, {{512, 1}}, false, false},
         {1, 0, {{1, 256}, {1, 256}}, {{1, 256}}, false, false},
         {1, 0, {{256, 1}, {256, 1}}, {{256, 1}}, false, false},
         {1, 0, {{1, 128}, {1, 128}}, {{1, 128}}, false, false},
         {1, 0, {{128, 1}, {128, 1}}, {{128, 1}}, false, false},
         {1, 0, {{1, 64}, {1, 64}}, {{1, 64}}, false, false},
         {1, 0, {{64, 1}, {64, 1}}, {{64, 1}}, false, false},
     }},
};

}  // namespace ark
