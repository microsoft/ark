// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "logging.h"
#include "model.h"
#include <cassert>

namespace ark {

extern const OpConfigMap CastConfigMap;

CastOp::CastOp(Tensor *input, Tensor *output, const std::string &name)
    : Op{OP_CAST, OP_PREC_NONE,   {input}, {output}, {},
         name,    &CastConfigMap, -1,      true}
{
}

std::string CastOp::function_name(const OpConfig &cfg) const
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

    Dims unit_out_dims{1, 1, tile_out.x, tile_out.y};
    return Op::function_name("ark::cast",
                             {{
                                 input->ldims.dims4(),  // InDims
                                 input->shape.dims4(),  // InShape
                                 output->ldims.dims4(), // OutDims
                                 output->shape.dims4(), // OutShape
                                 unit_out_dims,         // UnitOutDims
                                 cfg.num_warps * 32,    // NumThreads
                             }});
}

Tensor *Model::cast(Tensor *input, TensorType ttype, Tensor *output,
                    const std::string &name)
{
    assert(input != nullptr);
    if (output != nullptr && ttype != output->type) {
        LOG(ERROR, "invalid output data type: ", output->type);
    }
    if (output == nullptr) {
        output = this->tensor(input->shape, ttype);
    } else if (output->shape != input->shape) {
        LOG(ERROR, "invalid output shape: ", output->shape);
    }
    CastOp op{input, output, name};
    return this->impl->add_op(op)[0];
}

const OpConfigMap CastConfigMap = {
    {{OP_ARCH_CUDA_ANY, OP_PREC_NONE},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {8, 0, {{128, 256}}, {{128, 256}}, false, false},
         {8, 0, {{256, 128}}, {{256, 128}}, false, false},
         {8, 0, {{128, 128}}, {{128, 128}}, false, false},
         {4, 0, {{64, 64}}, {{64, 64}}, false, false},
         {2, 0, {{32, 64}}, {{32, 64}}, false, false},
         {1, 0, {{16, 64}}, {{16, 64}}, false, false},
         {1, 0, {{8, 64}}, {{8, 64}}, false, false},
         {1, 0, {{2, 128}}, {{2, 128}}, false, false},
         {1, 0, {{4, 64}}, {{4, 64}}, false, false},
         {1, 0, {{2, 64}}, {{2, 64}}, false, false},
         {1, 0, {{1, 128}}, {{1, 128}}, false, false},
         {1, 0, {{1, 64}}, {{1, 64}}, false, false},
         {1, 0, {{1, 32}}, {{1, 32}}, false, false},
     }},
};

} // namespace ark
