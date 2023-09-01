// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "logging.h"
#include "model.h"
#include <cassert>

namespace ark {

extern const OpConfigMap LayernormConfigMap;

RMSnormOp::RMSnormOp(OpPrecType prec_type, Tensor *input, Tensor *output,
                     const std::string &name)
    : Op{OP_RMSNORM, prec_type,           {input}, {output}, {},
         name,       &LayernormConfigMap, -1}
{
}

std::string RMSnormOp::function_name(const OpConfig &cfg) const
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
    return Op::function_name("ark::rmsnorm",
                             {{
                                 input->ldims.dims4(),  // InDims
                                 input->shape.dims4(),  // InShape
                                 output->ldims.dims4(), // OutDims
                                 output->shape.dims4(), // OutShape
                                 unit_out_dims,         // UnitOutDims
                                 cfg.num_warps * 32,    // NumThreads
                                 cfg.smem_bytes,        // SmemBytes
                             }});
}

Tensor *Model::rmsnorm(Tensor *input, Tensor *output, const std::string &name)
{
    assert(input != nullptr);
    OpPrecType pt;
    if (input->type == FP16) {
        pt = OP_PREC_FP16;
    } else if (input->type == FP32) {
        pt = OP_PREC_FP32;
    } else {
        LOG(ERROR, "unsupported input data type: ", input->type);
    }
    if (output != nullptr && input->type != output->type) {
        LOG(ERROR, "invalid output data type: ", output->type);
    }
    if (output == nullptr) {
        output = this->tensor(input->shape, input->type);
    } else if (output == input) {
        output = this->identity(output);
    }
    RMSnormOp op{pt, input, output, name};
    return this->impl->add_op(op)[0];
};

} // namespace ark
