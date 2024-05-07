// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_embedding.hpp"

#include "ops_common.hpp"

namespace ark {

ModelOpEmbedding::ModelOpEmbedding(ModelTensorRef input, ModelTensorRef weight,
                                   ModelTensorRef output)
    : ModelOp("Embedding") {
    check_null(input);
    check_null(weight);
    if (input->shape().ndims() > 3) {
        ERR(InvalidUsageError, "input shape ndims > 3: ", input->shape());
    }
    if (weight->shape().ndims() != 2) {
        ERR(InvalidUsageError, "weight shape ndims != 2: ", weight->shape());
    }
    if (output) {
        check_match_data_type(weight, output);
    } else {
        Dims input_shape = input->shape().dims4();
        Dims output_shape(input_shape[1], input_shape[2], input_shape[3],
                          weight->shape()[-1]);
        output = std::make_shared<ModelTensor>(
            weight->data_type(), std::make_shared<ModelBuffer>(), output_shape);
    }
    ModelTensorRef result = std::make_shared<ModelTensor>(*output);
    read_tensors_ = {input, weight};
    write_tensors_ = {output};
    result_tensors_ = {result};
    verify();
}

std::string ModelOpEmbedding::impl_name(const json &config) const {
    int num_warps = config.at("NumWarps");

    auto in_strides = read_tensors_[0]->strides().dims4();
    auto in_shape = read_tensors_[0]->shape().dims4();

    return function_name_string(
        "embedding",
        {
            vec_string(Dims(in_strides[1], in_strides[2], in_strides[3], 1)),
            vec_string(Dims(in_shape[1], in_shape[2], in_shape[3], 1)),
            vec_string(read_tensors_[1]->strides().dims4()),
            vec_string(read_tensors_[1]->shape().dims4()),
            vec_string(write_tensors_[0]->strides().dims4()),
            vec_string(write_tensors_[0]->shape().dims4()),
            std::to_string(num_warps),
        });
}

std::vector<ModelOpArg> ModelOpEmbedding::impl_args([
    [maybe_unused]] const json &config) const {
    return {result_tensors_[0], read_tensors_[0], read_tensors_[1]};
}

nlohmann::ordered_json ModelOpEmbedding::default_config() const {
    nlohmann::ordered_json config;
    config["NumWarps"] = 1;
    config["SramBytes"] = 0;
    const auto &shape = result_tensors_[0]->shape().dims4();
    config["NumTasks"] = shape[0] * shape[1] * shape[2];
    return config;
}

Tensor Model::embedding(Tensor input, Tensor weight, Tensor output,
                        const std::string &name) {
    return impl_
        ->create_op<ModelOpEmbedding>(name, input.ref_, weight.ref_,
                                      output.ref_)
        ->result_tensors()[0];
}

}  // namespace ark
