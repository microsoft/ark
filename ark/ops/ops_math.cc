// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cassert>

#include "logging.h"
#include "model.h"

using namespace std;

namespace ark {

extern const OpConfigMap MathConfigMap;

MathOp::MathOp(const OpType &type, const std::string &prec_type, Tensor *input,
               Tensor *output, const std::string &name)
    : Op{type, prec_type,      {input}, {output}, {},
         name, &MathConfigMap, -1,      true} {}

std::string MathOp::function_name(const OpConfig &cfg,
                                  const std::string &type) const {
    Tensor *input = this->inputs[0];
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
                                 input->ldims.dims4(),   // InDims
                                 input->shape.dims4(),   // InShape
                                 output->ldims.dims4(),  // OutDims
                                 output->shape.dims4(),  // OutShape
                                 unit_out_dims,          // UnitOutDims
                                 cfg.num_warps,          // NumWarps
                                 cfg.smem_bytes,         // SmemBytes
                             }});
}

template <typename MathOpType>
Tensor *Model::math(Tensor *input, Tensor *output, const string &name) {
    assert(input != nullptr);
    if (output != nullptr && input->type != output->type) {
        ERR(InvalidUsageError, "invalid output data type: ", output->type);
    }
    if (output == nullptr) {
        output = this->tensor(input->shape, input->type);
    } else if (output->shape != input->shape) {
        ERR(InvalidUsageError, "invalid output shape: ", output->shape);
    }
    MathOpType op{output->type.name(), input, output, name};
    return this->impl->add_op(op)[0];
}

////////////////////////////////////////////////////////////////////////////////

SqrtOp::SqrtOp(const std::string &prec_type, Tensor *input, Tensor *output,
               const string &name)
    : MathOp{OP_SQRT, prec_type, input, output, name} {}

std::string SqrtOp::function_name(const OpConfig &cfg) const {
    return MathOp::function_name(cfg, "sqrt");
}

Tensor *Model::sqrt(Tensor *input, Tensor *output, const string &name) {
    return math<SqrtOp>(input, output, name);
}

////////////////////////////////////////////////////////////////////////////////

RsqrtOp::RsqrtOp(const std::string &prec_type, Tensor *input, Tensor *output,
                 const string &name)
    : MathOp{OP_RSQRT, prec_type, input, output, name} {}

std::string RsqrtOp::function_name(const OpConfig &cfg) const {
    return MathOp::function_name(cfg, "rsqrt");
}

Tensor *Model::rsqrt(Tensor *input, Tensor *output, const string &name) {
    return math<RsqrtOp>(input, output, name);
}

const OpConfigMap MathConfigMap = {
    {{OP_ARCH_ANY, "any"},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {1, 0, {{256, 1}}, {{256, 1}}, false, false},
         {1, 0, {{1, 256}}, {{1, 256}}, false, false},
         {1, 0, {{128, 1}}, {{128, 1}}, false, false},
         {1, 0, {{1, 128}}, {{1, 128}}, false, false},
         {1, 0, {{64, 1}}, {{64, 1}}, false, false},
         {1, 0, {{1, 64}}, {{1, 64}}, false, false},
     }},
};

}  // namespace ark
