// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "logging.h"
#include "model.h"
#include "tensor.h"

using namespace std;

namespace ark {

LayernormOp::LayernormOp(OpPrecType prec_type, Tensor *input, Tensor *output,
                         const string &name)
    : Op{OP_LAYERNORM, prec_type, {input}, {output}, {}, name, -1}
{
}

std::string LayernormOp::function_name(const OpConfig &cfg) const
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

    Dims unit_out_shape{1, 1, tile_out.x, tile_out.y};
    return Op::function_name("ark::layernorm",
                             {{
                                 input->ldims.dims4(),  // InDims
                                 input->shape.dims4(),  // InShape
                                 output->ldims.dims4(), // OutDims
                                 output->shape.dims4(), // OutShape
                                 unit_out_shape,        // UnitOutShape
                                 cfg.num_warps * 32,    // ThreadsNum
                                 cfg.smem_bytes,        // SmemBytes
                             }});
}

Tensor *Model::layernorm(Tensor *input, Tensor *output, const string &name)
{
    assert(input != nullptr);
    LOG(DEBUG, "layernorm ", input->shape, " ", input->ldims, " ");
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
        output = this->tensor(input->shape, input->type);
    } else if (output == input) {
        output = this->identity(output);
    }
    LayernormOp op{pt, input, output, name};
    this->impl->add_op(op);
    return output;
}

} // namespace ark
