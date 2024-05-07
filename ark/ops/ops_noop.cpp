// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_noop.hpp"

#include "ops_common.hpp"

namespace ark {

ModelOpNoop::ModelOpNoop(ModelTensorRef input) : ModelOp("Noop") {
    read_tensors_ = {input};
    verify();
}

std::string ModelOpNoop::impl_name([[maybe_unused]] const json &config) const {
    return function_name_string("noop");
}

std::vector<ModelOpArg> ModelOpNoop::impl_args([
    [maybe_unused]] const json &config) const {
    return {};
}

ordered_json ModelOpNoop::default_config() const {
    ordered_json config;
    config["NumWarps"] = 1;
    config["SramBytes"] = 0;
    config["NumTasks"] = 0;
    return config;
}

void Model::noop(Tensor input, const std::string &name) {
    impl_->create_op<ModelOpNoop>(name, input.ref_);
}

}  // namespace ark
