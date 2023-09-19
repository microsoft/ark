// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "logging.h"
#include "model.h"
#include <cassert>

namespace ark {

extern const OpConfigMap TransposeConfigMap;

TransposeOp::TransposeOp(OpPrecType prec_type, Tensor *input, Tensor *output,
                         int tp_type, const std::string &name)
    : Op{OP_TRANSPOSE, prec_type,           {input}, {output}, {{tp_type}},
         name,         &TransposeConfigMap, -1,      true}
{
}

std::string TransposeOp::function_name(const OpConfig &cfg) const
{
    int tp_type;
    this->args.get(&tp_type, 0);

    std::string tp_type_str = std::to_string(tp_type);
    if (tp_type_str.size() == DIMS_LEN - 1) {
        tp_type_str = "0" + tp_type_str;
    }
    if (tp_type_str.size() != DIMS_LEN) {
        LOG(ERROR, "Unexpected error");
    }

    Tensor *input = this->inputs[0];
    Tensor *output = this->outputs[0];
    const OpTile &tile_out = cfg.output_tiles[0];
    Dims unit_out_dims{1, 1, tile_out.x, tile_out.y};

    return Op::function_name("ark::transpose" + tp_type_str,
                             {{
                                 input->ldims.dims4(),  // InDims
                                 output->ldims.dims4(), // OutDims
                                 output->shape.dims4(), // OutShape
                                 unit_out_dims,         // UnitOutDims
                                 cfg.num_warps * 32,    // NumThreads
                                 cfg.smem_bytes,        // SmemBytes
                             }});
}

Tensor *Model::transpose(Tensor *input, Dims perm, Tensor *output,
                         const std::string &name)
{
    OpPrecType pt = OP_PREC_NONE;
    if (input->type == FP16) {
        pt = OP_PREC_FP16;
    } else if (input->type == FP32) {
        pt = OP_PREC_FP32;
    } else {
        LOG(ERROR, "unsupported input data type: ", input->type);
    }
    int input_ndims = input->ndims();
    Dims in_shape{1, 1, 1, 1};
    if (input_ndims < 2 || input_ndims > 4) {
        LOG(ERROR,
            "Invalid # of input dimensions. Expected 2, 3, or 4, but given ",
            input_ndims);
    }
    for (int i = 0; i < input_ndims; ++i) {
        in_shape[4 - input_ndims + i] = input->shape[i];
    }
    if (perm.ndims() != input_ndims) {
        LOG(ERROR,
            "Permutation should have the same number of dimensions as the "
            "one of input. Given input shape: ",
            input->shape, ", permutation: ", perm);
    }
    int count[DIMS_LEN];
    for (int i = 0; i < input_ndims; ++i) {
        count[i] = 0;
    }
    for (int i = 0; i < input_ndims; ++i) {
        if (perm[i] >= input_ndims) {
            LOG(ERROR,
                "Each value in permutation should be less than the number "
                "of input dimensions. Given permutation: ",
                perm);
        }
        if (count[perm[i]] > 0) {
            LOG(ERROR,
                "Each value in permutation should be unique. Given "
                "permutation: ",
                perm);
        }
        count[perm[i]]++;
    }
    int tp_type = perm[0] * 1000 + perm[1] * 100 + perm[2] * 10 + perm[3];
    Dims out_shape{in_shape[perm[0]], in_shape[perm[1]], in_shape[perm[2]],
                   in_shape[perm[3]]};
    if (output == nullptr) {
        output = this->tensor(out_shape, input->type);
    } else {
        assert(output->shape == out_shape);
    }
    TransposeOp op{pt, input, output, tp_type, name};
    return this->impl->add_op(op)[0];
}

const OpConfigMap TransposeConfigMap = {
    {{OP_ARCH_CUDA_ANY, OP_PREC_FP32},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {8, 0, {{1, 1}}, {{128, 128}}, true, false},
         {4, 0, {{1, 1}}, {{64, 128}}, true, false},
         {4, 0, {{1, 1}}, {{128, 64}}, true, false},
         {4, 0, {{1, 1}}, {{64, 64}}, true, false},
         {2, 0, {{1, 1}}, {{32, 32}}, true, false},
     }},
    {{OP_ARCH_CUDA_ANY, OP_PREC_FP16},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {8, 0, {{1, 1}}, {{128, 128}}, true, false},
         {4, 0, {{1, 1}}, {{64, 128}}, true, false},
         {4, 0, {{1, 1}}, {{128, 64}}, true, false},
         {4, 0, {{1, 1}}, {{64, 64}}, true, false},
         {2, 0, {{1, 1}}, {{32, 32}}, true, false},
         {1, 0, {{1, 1}}, {{16, 16}}, true, false},
         {1, 0, {{1, 1}}, {{8, 16}}, true, false},
         {1, 0, {{1, 1}}, {{4, 8}}, true, false},
     }},
};

} // namespace ark
