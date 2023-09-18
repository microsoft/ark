// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "logging.h"
#include "model.h"
#include <cassert>

namespace ark {

extern const OpConfigMap SoftmaxConfigMap;

SoftmaxOp::SoftmaxOp(OpPrecType prec_type, Tensor *input, Tensor *output,
                     const std::string &name)
    : Op{OP_SOFTMAX, prec_type,         {input}, {output}, {},
         name,       &SoftmaxConfigMap, -1,      true}
{
}

std::string SoftmaxOp::function_name(const OpConfig &cfg) const
{
    Tensor *input = this->inputs[0];
    Tensor *output = this->outputs[0];

    const OpTile &tile_out = cfg.output_tiles[0];
    Dims unit_out_dims{1, 1, tile_out.x, tile_out.y};

    return Op::function_name("ark::softmax",
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

Tensor *Model::softmax(Tensor *input, Tensor *output, const std::string &name)
{
    assert(input != nullptr);
    OpPrecType pt = OP_PREC_NONE;
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
    SoftmaxOp op{pt, input, output, name};
    return this->impl->add_op(op)[0];
}

const OpConfigMap SoftmaxConfigMap = {
    {{OP_ARCH_CUDA_ANY, OP_PREC_ANY},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {1, 128, {{32, -1}}, {{32, -1}}, true, false},
         {1, 128, {{16, -1}}, {{16, -1}}, true, false},
         {1, 128, {{8, -1}}, {{8, -1}}, true, false},
         {1, 128, {{4, -1}}, {{4, -1}}, true, false},
         {1, 128, {{2, -1}}, {{2, -1}}, true, false},
         {1, 128, {{1, -1}}, {{1, -1}}, true, false},
         {4, 128, {{1, -1}}, {{1, -1}}, true, false},
         {8, 128, {{1, -1}}, {{1, -1}}, true, false},
     }},
};

} // namespace ark
