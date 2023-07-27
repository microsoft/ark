// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "logging.h"
#include "model.h"
#include "tensor.h"

using namespace std;

namespace ark {

extern const OpConfigMap ActivationConfigMap;

ReluOp::ReluOp(OpPrecType prec_type, Tensor *input, Tensor *output,
               const string &name)
    : Op{OP_RELU, prec_type, {input}, {output}, {}, name, &ActivationConfigMap,
         -1,      true}
{
}

std::string ReluOp::function_name(const OpConfig &cfg) const
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
    return Op::function_name("ark::relu", {{
                                              input->ldims.dims4(),  // InDims
                                              input->shape.dims4(),  // InShape
                                              output->ldims.dims4(), // OutDims
                                              output->shape.dims4(), // OutShape
                                              unit_out_dims,      // UnitOutDims
                                              cfg.num_warps * 32, // NumThreads
                                              cfg.smem_bytes,     // SmemBytes
                                          }});
}

Tensor *Model::relu(Tensor *input, Tensor *output, const string &name)
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
    ReluOp op{pt, input, output, name};
    this->impl->add_op(op);
    return output;
}

const OpConfigMap ActivationConfigMap = {
    {{OP_ARCH_CUDA_70, OP_PREC_FP16},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {4, 0, {{64, 64}}, {{64, 64}}, false, false},
         {2, 0, {{32, 64}}, {{32, 64}}, false, false},
         {1, 0, {{16, 64}}, {{16, 64}}, false, false},
         {1, 0, {{8, 64}}, {{8, 64}}, false, false},
         {1, 0, {{2, 128}}, {{2, 128}}, false, false},
         {1, 0, {{4, 64}}, {{4, 64}}, false, false},
         {1, 0, {{2, 64}}, {{2, 64}}, false, false},
         {1, 0, {{1, 64}}, {{1, 64}}, false, false},
     }},
    {{OP_ARCH_CUDA_80, OP_PREC_FP16},
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
         {1, 0, {{1, 64}}, {{1, 64}}, false, false},
     }},
};

} // namespace ark
