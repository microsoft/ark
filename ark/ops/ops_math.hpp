// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_OPS_MATH_HPP_
#define ARK_OPS_MATH_HPP_

#include "ark/dims.hpp"
#include "ark/model.hpp"
#include "model/model_op.hpp"

namespace ark {

class ModelOpMath : public ModelOp {
   public:
    ModelOpMath() = default;
    ModelOpMath(const std::string &type_name, ModelTensorRef input,
                ModelTensorRef output);

    std::string impl_name(const nlohmann::json &config) const override;

    std::vector<ModelOpArg> impl_args(
        [[maybe_unused]] const nlohmann::json &config) const override;

    nlohmann::ordered_json default_config() const override;
};

class ModelOpExp : public ModelOpMath {
   public:
    ModelOpExp() = default;
    ModelOpExp(ModelTensorRef input, ModelTensorRef output);
};

class ModelOpGelu : public ModelOpMath {
   public:
    ModelOpGelu() = default;
    ModelOpGelu(ModelTensorRef input, ModelTensorRef output);
};

class ModelOpRelu : public ModelOpMath {
   public:
    ModelOpRelu() = default;
    ModelOpRelu(ModelTensorRef input, ModelTensorRef output);
};

class ModelOpRsqrt : public ModelOpMath {
   public:
    ModelOpRsqrt() = default;
    ModelOpRsqrt(ModelTensorRef input, ModelTensorRef output);
};

class ModelOpSigmoid : public ModelOpMath {
   public:
    ModelOpSigmoid() = default;
    ModelOpSigmoid(ModelTensorRef input, ModelTensorRef output);
};

class ModelOpSqrt : public ModelOpMath {
   public:
    ModelOpSqrt() = default;
    ModelOpSqrt(ModelTensorRef input, ModelTensorRef output);
};

}  // namespace ark

#endif  // ARK_OPS_MATH_HPP_
