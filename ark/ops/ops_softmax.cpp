// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_softmax.hpp"

#include "logging.hpp"
#include "ops_common.hpp"

namespace ark {

ModelOpSoftmax::ModelOpSoftmax(ModelTensorRef input, ModelTensorRef output)
    : ModelOp("Softmax") {
    check_null(input);

    if (output) {
        check_match_data_type(input, output);
        check_match_shape(output, input->shape());
    } else {
        output = std::make_shared<ModelTensor>(input->data_type(),
                                               std::make_shared<ModelBuffer>(),
                                               input->shape());
    }
    ModelTensorRef result = std::make_shared<ModelTensor>(*output);
    read_tensors_ = {input};
    write_tensors_ = {output};
    result_tensors_ = {result};
    verify();
}

std::string ModelOpSoftmax::impl_name(const Json &config) const {
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

    return function_name_string("softmax", template_args);
}

std::vector<ModelOpArg> ModelOpSoftmax::impl_args(
    [[maybe_unused]] const Json &config) const {
    // Order: out, in
    return {result_tensors_[0], read_tensors_[0]};
}

Json ModelOpSoftmax::default_config([[maybe_unused]] const ArchRef arch) const {
    Json config;
    config["NumWarps"] = 1;
    config["SramBytes"] = 256;
    auto shape = result_tensors_[0]->shape().dims4();
    config["Tile"] = {1, 1, 1, static_cast<int64_t>(shape[3])};
    config["NumTasks"] = shape[0] * shape[1] * shape[2];
    return config;
}

Tensor Model::softmax(Tensor input, Tensor output, const std::string &name) {
    return impl_->create_op<ModelOpSoftmax>(name, input.ref_, output.ref_)
        ->result_tensors()[0];
}

}  // namespace ark
