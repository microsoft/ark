// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cassert>

#include "logging.h"
#include "model.h"

namespace ark {

extern const OpConfigMap EmbeddingConfigMap;

EmbeddingOp::EmbeddingOp(const std::string &prec_type, Tensor *input,
                         Tensor *weight, Tensor *output,
                         const std::string &name)
    : Op{OP_EMBEDDING, prec_type, {input, weight},     {output},
         {},           name,      &EmbeddingConfigMap, -1,
         true} {}

std::string EmbeddingOp::function_name(const OpConfig &cfg) const {
    Tensor *input = this->inputs[0];
    Tensor *weight = this->inputs[1];
    Tensor *output = this->outputs[0];

    auto in_dims = input->ldims.dims4();
    auto in_shape = input->shape.dims4();

    assert(in_dims[0] == 1);
    assert(in_shape[0] == 1);

    Dims new_in_dims{in_dims[1], in_dims[2], in_dims[3], 1};
    Dims new_in_shape{in_shape[1], in_shape[2], in_shape[3], 1};

    int emb_dim = weight->shape[-1];
    return Op::function_name("ark::embedding",
                             {{
                                 new_in_dims,            // InDims
                                 new_in_shape,           // InShape
                                 weight->ldims.dims4(),  // WeightDims
                                 weight->shape.dims4(),  // WeightShape
                                 output->ldims.dims4(),  // OutDims
                                 output->shape.dims4(),  // OutShape
                                 emb_dim,                // EmbeddingDim
                                 cfg.num_warps,          // NumWarps
                             }});
}

Tensor *Model::embedding(Tensor *input, Tensor *weight, Tensor *output,
                         const std::string &name) {
    assert(input != nullptr);
    assert(weight != nullptr);
    if (input->shape.ndims() > 3) {
        ERR(InvalidUsageError, "input shape ndims > 3: ", input->shape);
    }
    if (weight->shape.ndims() != 2) {
        ERR(InvalidUsageError, "weight shape ndims != 2: ", weight->shape);
    }
    auto emb_dim = weight->shape[-1];

    std::vector<DimType> output_dims;
    for (int i = 0; i < input->shape.ndims(); ++i) {
        output_dims.push_back(input->shape[i]);
    }
    output_dims.push_back(emb_dim);
    Dims out_shape{output_dims};
    if (output == nullptr) {
        output = this->tensor(out_shape, weight->type);
    }
    EmbeddingOp op{output->type.name(), input, weight, output, name};
    return this->impl->add_op(op)[0];
}

const OpConfigMap EmbeddingConfigMap = {
    {{OP_ARCH_ANY, "any"},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {1, 0, {{1, 1}, {1, -1}}, {{1, -1}}, true, false},
         {2, 0, {{1, 1}, {1, -1}}, {{1, -1}}, true, false},
         {4, 0, {{1, 1}, {1, -1}}, {{1, -1}}, true, false},
         {8, 0, {{1, 1}, {1, -1}}, {{1, -1}}, true, false},
     }},
};

}  // namespace ark
