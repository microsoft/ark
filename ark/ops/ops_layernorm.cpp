// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_layernorm.hpp"

#include "logging.hpp"
#include "ops_common.hpp"

namespace ark {

ModelOpLayerNorm::ModelOpLayerNorm(ModelTensorRef input, ModelTensorRef gamma,
                                   ModelTensorRef beta, ModelTensorRef output)
    : ModelOp("LayerNorm") {
    check_null(input);
    check_null(gamma);
    check_null(beta);

    // gamma and beta must be 1-D tensors matching the last dimension of input
    DimType norm_dim = input->shape()[-1];
    if (gamma->shape().nelems() != norm_dim) {
        ERR(ModelError, "gamma size ", gamma->shape().nelems(),
            " does not match last dimension of input ", norm_dim);
    }
    if (beta->shape().nelems() != norm_dim) {
        ERR(ModelError, "beta size ", beta->shape().nelems(),
            " does not match last dimension of input ", norm_dim);
    }
    check_match_data_type(input, gamma);
    check_match_data_type(input, beta);

    if (output) {
        check_match_data_type(input, output);
        check_match_shape(output, input->shape());
    } else {
        output = std::make_shared<ModelTensor>(
            input->data_type(), std::make_shared<ModelBuffer>(),
            input->shape());
    }
    ModelTensorRef result = std::make_shared<ModelTensor>(*output);
    read_tensors_ = {input, gamma, beta};
    write_tensors_ = {output};
    result_tensors_ = {result};
    verify();
}

std::string ModelOpLayerNorm::impl_name(const Json &config) const {
    check_fields_config(config, {"NumWarps", "SramBytes", "Tile"});
    int num_warps = config.at("NumWarps");
    int sram_bytes = config.at("SramBytes");
    Dims unit_out_dims(config.at("Tile").get<std::vector<DimType>>());

    std::vector<std::string> template_args = {
        vec_string(read_tensors_[0]->strides().dims4()),
        vec_string(read_tensors_[0]->shape().dims4()),
        vec_string(write_tensors_[0]->strides().dims4()),
        vec_string(write_tensors_[0]->shape().dims4()),
        vec_string(unit_out_dims.dims4()),
        std::to_string(num_warps),
        std::to_string(sram_bytes),
    };

    // Add NelemPerThread if specified and > 1
    if (config.contains("NelemPerThread")) {
        int nelem = config.at("NelemPerThread");
        if (nelem > 1) {
            template_args.push_back(std::to_string(nelem));
        }
    }

    return function_name_string("layernorm_affine", template_args);
}

std::vector<ModelOpArg> ModelOpLayerNorm::impl_args(
    [[maybe_unused]] const Json &config) const {
    // Order must match kernel function signature:
    // layernorm_affine(out, in, gamma, beta, uop_idx, smem_per_warp)
    return {result_tensors_[0], read_tensors_[0], read_tensors_[1],
            read_tensors_[2]};
}

Json ModelOpLayerNorm::default_config(
    [[maybe_unused]] const ArchRef arch) const {
    Json config;
    config["NumWarps"] = 1;
    config["SramBytes"] = 256;
    // The tile must cover the entire W dimension since layernorm reduces along W.
    // Each task processes one complete row.
    auto shape = result_tensors_[0]->shape().dims4();
    config["Tile"] = {1, 1, 1, static_cast<int64_t>(shape[3])};
    // One task per row (N * C * H)
    config["NumTasks"] = shape[0] * shape[1] * shape[2];
    return config;
}

Tensor Model::layernorm(Tensor input, Tensor gamma, Tensor beta, Tensor output,
                        const std::string &name) {
    return impl_
        ->create_op<ModelOpLayerNorm>(name, input.ref_, gamma.ref_, beta.ref_,
                                      output.ref_)
        ->result_tensors()[0];
}

}  // namespace ark
