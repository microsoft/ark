// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "logging.h"
#include "model.h"
#include "tensor.h"

using namespace std;

namespace ark {

SoftmaxOp::SoftmaxOp(OpPrecType prec_type, Tensor *input, Tensor *output,
                     const string &name)
    : Op{OP_SOFTMAX, prec_type, {input}, {output}, {}, name, -1}
{
}

std::string SoftmaxOp::function_name(const OpConfig &cfg) const
{
    Tensor *input = this->in_deps[0];
    Tensor *output = this->out_deps[0];

    Dims shp_out = output->shape;
    int ndims = shp_out.ndims();
    CHECK(ndims < 4);

    const OpTile &tile_out = cfg.out_deps_tiles[0];
    Dims unit_out_shape{1, 1, tile_out.x, tile_out.y};

    return Op::function_name("ark::softmax",
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

Tensor *Model::softmax(Tensor *input, Tensor *output, const string &name)
{
    assert(input != nullptr);
    LOG(DEBUG, "softmax ", input->shape, " ", input->ldims, " ");
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
    SoftmaxOp op{pt, input, output, name};
    this->impl->add_op(op);
    return output;
}

} // namespace ark
