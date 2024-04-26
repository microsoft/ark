// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_OPS_ARITHMETIC_HPP_
#define ARK_OPS_ARITHMETIC_HPP_

#include "ark/dims.hpp"
#include "ark/model.hpp"
#include "model/model_op.hpp"

namespace ark {

class ModelOpArithmetic : public ModelOp {
   public:
    ModelOpArithmetic() = default;
    ModelOpArithmetic(const std::string &type_name, ModelTensorRef input,
                      ModelTensorRef other, ModelTensorRef output);

    std::string impl_name(const nlohmann::json &config) const override;

    std::vector<ModelOpArg> impl_args(
        [[maybe_unused]] const nlohmann::json &config) const override;

    nlohmann::ordered_json default_config() const override;
};

class ModelOpAdd : public ModelOpArithmetic {
   public:
    ModelOpAdd() = default;
    ModelOpAdd(ModelTensorRef input, ModelTensorRef other,
               ModelTensorRef output);
};

class ModelOpMul : public ModelOpArithmetic {
   public:
    ModelOpMul() = default;
    ModelOpMul(ModelTensorRef input, ModelTensorRef other,
               ModelTensorRef output);
};

class ModelOpSub : public ModelOpArithmetic {
   public:
    ModelOpSub() = default;
    ModelOpSub(ModelTensorRef input, ModelTensorRef other,
               ModelTensorRef output);
};

class ModelOpDiv : public ModelOpArithmetic {
   public:
    ModelOpDiv() = default;
    ModelOpDiv(ModelTensorRef input, ModelTensorRef other,
               ModelTensorRef output);
};

}  // namespace ark

#endif  // ARK_OPS_ARITHMETIC_HPP_
