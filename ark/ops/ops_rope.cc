// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "logging.h"
#include "model.h"
#include <cassert>

namespace ark {

extern const OpConfigMap ArithmeticConfigMap;

RopeOp::RopeOp(OpPrecType prec_type, Tensor *input, Tensor *other,
               Tensor *output, const std::string &name)
    : Op{OP_ROPE, prec_type, {input, other},       {output},
         {},      name,      &ArithmeticConfigMap, -1,
         true}
{
}

std::string RopeOp::function_name(const OpConfig &cfg) const
{
    Tensor *input = this->inputs[0];
    Tensor *other = this->inputs[1];
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
    return Op::function_name("ark::rope", {{
                                              input->ldims.dims4(),  // In0Dims
                                              input->shape.dims4(),  // In0Shape
                                              other->ldims.dims4(),  // In1Dims
                                              other->shape.dims4(),  // In1Shape
                                              output->ldims.dims4(), // OutDims
                                              output->shape.dims4(), // OutShape
                                              unit_out_dims,      // UnitOutDims
                                              cfg.num_warps * 32, // NumThreads
                                              cfg.smem_bytes,     // SmemBytes
                                          }});
}

Tensor *Model::rope(Tensor *input, Tensor *other, Tensor *output,
                    const std::string &name)
{
    LOG(DEBUG, "rope ", input->shape, " ", other->shape);
    assert(input != nullptr);
    assert(other != nullptr);
    OpPrecType pt;
    if (input->type == FP16) {
        pt = OP_PREC_FP16;
    } else if (input->type == FP32) {
        pt = OP_PREC_FP32;
    } else {
        LOG(ERROR, "unsupported input data type: ", input->type);
    }
    if (input->type != other->type) {
        LOG(ERROR, "input data types mismatch: ", input->type, ", ",
            other->type);
    }
    if (output != nullptr && input->type != output->type) {
        LOG(ERROR, "invalid output data type: ", output->type);
    }
    Dims output_shape = broadcast(input->shape, other->shape);
    if (output == nullptr) {
        output = this->tensor(output_shape, input->type);
    } else if (output->shape != output_shape) {
        LOG(ERROR, "invalid output shape: ", output->shape);
    } else if (output == input) {
        output = this->identity(output);
    }
    RopeOp op{pt, input, other, output, name};
    return this->impl->add_op(op)[0];
}

} // namespace ark