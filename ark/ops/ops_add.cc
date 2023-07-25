// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "logging.h"
#include "model.h"
#include "tensor.h"

using namespace std;

namespace ark {

AddOp::AddOp(OpPrecType prec_type, Tensor *input, Tensor *other, Tensor *output,
             const string &name)
    : Op{OP_ADD, prec_type, {input, other}, {output}, {}, name, -1, true}
{
}

std::string AddOp::function_name(const OpConfig &cfg) const
{
    Tensor *input = this->in_deps[0];
    Tensor *other = this->in_deps[1];
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
    return Op::function_name("ark::add", {{
                                             input->ldims.dims4(),  // In0Dims
                                             input->shape.dims4(),  // In0Shape
                                             other->ldims.dims4(),  // In1Dims
                                             other->shape.dims4(),  // In1Shape
                                             output->ldims.dims4(), // OutDims
                                             output->shape.dims4(), // OutShape
                                             unit_out_shape,     // UnitOutShape
                                             cfg.num_warps * 32, // ThreadsNum
                                             cfg.smem_bytes,     // SmemBytes
                                         }});
}

Tensor *Model::add(Tensor *input, Tensor *other, Tensor *output,
                   const string &name)
{
    LOG(DEBUG, "add ", input->shape, " ", other->shape);
    CHECK(input != nullptr);
    CHECK(other != nullptr);
    OpPrecType pt;
    if (input->type == FP16) {
        pt = OP_PREC_FP16;
    } else if (input->type == FP32) {
        pt = OP_PREC_FP32;
    } else {
        LOGERR("unsupported input data type: ", type_str(input->type));
    }
    if (input->type != other->type) {
        LOGERR("input data types mismatch: ", type_str(input->type), ", ",
               type_str(other->type));
    }
    if (output != nullptr && input->type != output->type) {
        LOGERR("invalid output data type: ", type_str(output->type));
    }
    Dims output_shape = broadcast(input->shape, other->shape);
    if (output == nullptr) {
        output = this->tensor(output_shape, input->type);
    } else if (output->shape != output_shape) {
        LOGERR("invalid output shape: ", output->shape);
    } else if (output == input) {
        output = this->identity(output);
    }
    AddOp op{pt, input, other, output, name};
    this->impl->add_op(op);
    return output;
}

} // namespace ark
