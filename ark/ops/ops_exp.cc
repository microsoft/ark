// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cassert>

#include "logging.h"
#include "model.h"

using namespace std;

namespace ark {

extern const OpConfigMap MathConfigMap;

ExpOp::ExpOp(const std::string &prec_type, Tensor *input, Tensor *output,
             const string &name)
    : Op{OP_EXP, prec_type,      {input}, {output}, {},
         name,   &MathConfigMap, -1,      true} {}

std::string ExpOp::function_name(const OpConfig &cfg) const {
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
    return Op::function_name("ark::exp", {{
                                             input->ldims.dims4(),   // InDims
                                             input->shape.dims4(),   // InShape
                                             output->ldims.dims4(),  // OutDims
                                             output->shape.dims4(),  // OutShape
                                             unit_out_dims,   // UnitOutDims
                                             cfg.num_warps,   // NumWarps
                                             cfg.smem_bytes,  // SmemBytes
                                         }});
}

Tensor *Model::exp(Tensor *input, Tensor *output, const string &name) {
    assert(input != nullptr);
    if (output != nullptr && input->type != output->type) {
        LOG(ERROR, "invalid output data type: ", output->type);
    }
    if (output == nullptr) {
        output = this->tensor(input->shape, input->type, input->buf);
    } else if (output->shape != input->shape) {
        LOG(ERROR, "invalid output shape: ", output->shape);
    }
    ExpOp op{output->type.name(), input, output, name};
    return this->impl->add_op(op)[0];
}

}  // namespace ark
