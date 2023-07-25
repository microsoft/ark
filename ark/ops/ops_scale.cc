// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "logging.h"
#include "model.h"
#include "tensor.h"

using namespace std;

namespace ark {

ScaleOp::ScaleOp(OpPrecType prec_type, Tensor *input, Tensor *output, float val,
                 const string &name)
    : Op{OP_SCALE, prec_type, {input}, {output}, {{val}}, name, -1, true}
{
}

std::string ScaleOp::function_name(const OpConfig &cfg) const
{
    Tensor *output = this->out_deps[0];

    int ndims = output->shape.ndims();
    const OpTile &tile_out = cfg.out_deps_tiles[0];
    CHECK(output->ldims[ndims - 1] % tile_out.y == 0);
    if (ndims > 1) {
        CHECK(output->ldims[ndims - 2] % tile_out.x == 0);
    } else {
        CHECK(tile_out.x == 1);
    }

    DimType ldm = output->ldims[ndims - 1];
    DimType ldn = (ndims > 1) ? output->ldims[ndims - 2] : 1;

    return Op::function_name("ark::scale", {{
                                               ldm,                // M
                                               ldn,                // N
                                               cfg.num_warps * 32, // TN
                                               cfg.smem_bytes,     // SB
                                               tile_out.y,         // TDM
                                               tile_out.x,         // TDN
                                               1,                  // TDK
                                           }});
}

OpArgs ScaleOp::function_call_args(const OpConfig &) const
{
    OpArgs opargs;
    std::vector<Tensor *> deps = this->out_deps;
    deps.insert(deps.end(), this->in_deps.begin(), this->in_deps.end());
    for (Tensor *tns : deps) {
        opargs.put(tns);
    }
    float val;
    this->args.get(&val, 0);
    opargs.put(val);
    return opargs;
}

// Multiply `input` by `val`.
// TODO: make it into a general element-wise operation
Tensor *Model::scale(Tensor *input, float val, Tensor *output,
                     const string &name)
{
    assert(input != nullptr);
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
        output = this->tensor(input->shape, input->type, input->buf);
    } else if (output->shape != input->shape) {
        LOGERR("invalid output shape: ", output->shape);
    }
    ScaleOp op{pt, input, output, val, name};
    this->impl->add_op(op);
    return output;
}

} // namespace ark
